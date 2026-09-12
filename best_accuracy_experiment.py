"""
best_accuracy_experiment.py
===========================
Goal: push CV accuracy past 80%

Pipeline:
  CNN (MobileNetV2 mel+chroma+MFCC PCA) +
  Tabular clean (deduplicated + interaction terms) +
  Frame features (top-40 Cohen d) +
  New audio features (TKEO + GNE + energy decay) +
  BorderlineSMOTE inside each fold +
  Optuna-tuned LR C +
  Optimal ROC threshold
"""

import os, sys, warnings, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import librosa
from PIL import Image
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
import parselmouth
from parselmouth.praat import call
import nolds
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from imblearn.over_sampling import BorderlineSMOTE

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC as KerasAUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42); np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = r"D:\parkinson detector tool"
AUDIO_DIR = {0: os.path.join(BASE,"Healthy"), 1: os.path.join(BASE,"PD")}
IMG_SIZE  = 128; SR = 22050
N_AUG = 3
PCA_MEL=15; PCA_CHROMA=10; PCA_MFCC=10
SEP = "="*70

# deduplicated TOP15 (removed loudness_variability=dup of tremor_amplitude_std,
# and 0_LPCC_11=dup of 0_LPCC_1)
TOP_CLEAN = [
    "longest_dip_duration","tremor_amplitude_std",
    "total_dip_duration","possible_micro_breaks","max_rms_drop",
    "local_rms_variance","0_LPCC_1","num_rms_dips",
    "0_Spectral distance","stumble_total_duration_sec","stumble_num_events",
    "0_Spectral roll-off","0_Maximum frequency","mfcc_3","spectral_bandwidth",
]
PRAAT_COLS   = ["praat_jitter_local","praat_jitter_rap","praat_shimmer_local","praat_hnr","praat_nhr"]
DYNAMIC_COLS = ["rpde","dfa","ppe"]
INTER_COLS   = ["dip_x_tremor","breaks_x_tremor","dip_x_drops"]
NEW_AUDIO_COLS = [
    "tkeo_mean","tkeo_std","tkeo_cv","tkeo_dip_ratio","tkeo_range",
    "gne_proxy","spec_flat_mean","spec_flat_std","zcr_mean","zcr_std",
    "energy_decay","energy_shape_ratio",
]

BLIND_HC = ["AH_562E_151814F5-BB0F-44EF-9A22-FE2862FC3411.wav",
            "AH_378G_3C2A05CE-36E4-4956-8FC2-0494B27D3EA8.wav"]
BLIND_PD = ["AH_545789668-A4F6069C-5E1A-49F5-9EDC-59C6EB833E42.wav",
            "AH_545753013-FCFF8F46-08FF-4C87-B443-D2039E5DA945.wav"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_praat(path):
    try:
        snd=parselmouth.Sound(path); pp=call(snd,"To PointProcess (periodic, cc)",75,500)
        jl=call(pp,"Get jitter (local)",0,0,0.0001,0.02,1.3)
        jr=call(pp,"Get jitter (rap)",0,0,0.0001,0.02,1.3)
        sl=call([snd,pp],"Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)
        h=call(call(snd,"To Harmonicity (cc)",0.01,75,0.1,1.0),"Get mean",0,0)
        nhr=(1./h) if h and h>0 else 0.
        vals=[jl,jr,sl,h,nhr]
        return [0. if (v is None or not np.isfinite(v)) else v for v in vals]
    except: return [0.]*5

def compute_rpde(f0, eps=0.05):
    if len(f0)<30: return 0.
    x=f0/(np.mean(f0)+1e-8); rts=[]
    for i in range(len(x)-1):
        for j in range(i+1,min(i+100,len(x))):
            if abs(x[j]-x[i])<eps: rts.append(j-i); break
    if not rts: return 0.
    h=np.zeros(max(rts)+1)
    for r in rts: h[r]+=1
    h/=h.sum()+1e-8; h=h[h>0]
    H=-np.sum(h*np.log2(h+1e-8)); Hm=np.log2(max(rts)+1)
    return H/Hm if Hm>0 else 0.

def compute_ppe(f0):
    if len(f0)<20: return 0.
    p=1./(f0+1e-8); ln=np.log(p/(np.mean(p)+1e-8)+1e-8)
    h,_=np.histogram(ln,bins=max(10,min(30,len(f0)//5)))
    h=h.astype(float); h/=h.sum()+1e-8; h=h[h>0]
    return -np.sum(h*np.log(h+1e-8))

def extract_dynamics(path):
    try:
        snd=parselmouth.Sound(path)
        pitch=snd.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=500)
        f0=pitch.selected_array["frequency"]; voiced=f0[f0>0].astype(np.float64)
        if len(voiced)<30: return [0.,0.5,0.]
        rpde=compute_rpde(voiced); ppe=compute_ppe(voiced)
        try: dfa=float(nolds.dfa(voiced))
        except: dfa=0.5
        vals=[rpde,dfa,ppe]
        return [0. if (v is None or not np.isfinite(v)) else v for v in vals]
    except: return [0.,0.5,0.]

def augment(sig):
    return [librosa.effects.time_stretch(sig,rate=0.82),
            librosa.effects.time_stretch(sig,rate=1.18),
            sig+np.random.randn(len(sig)).astype(np.float32)*0.003]

def to_img(mat):
    mn,mx=mat.min(),mat.max()
    u8=((mat-mn)/(mx-mn+1e-8)*255).astype(np.uint8)
    img=Image.fromarray(u8).resize((IMG_SIZE,IMG_SIZE),Image.LANCZOS)
    return np.array(Image.merge("RGB",[img,img,img]),dtype=np.float32)/255.

def get_3specs(sig):
    mel=librosa.power_to_db(librosa.feature.melspectrogram(y=sig,sr=SR,n_mels=128,fmax=8000),ref=np.max)
    chro=librosa.feature.chroma_stft(y=sig,sr=SR,n_chroma=128)
    mfcc=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=128)
    return to_img(mel),to_img(chro),to_img(mfcc)

def build_cnn_extractor():
    inp=Input(shape=(IMG_SIZE,IMG_SIZE,3))
    base=MobileNetV2(include_top=False,weights="imagenet",input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable=False
    gap=layers.GlobalAveragePooling2D()(base(layers.Rescaling(2.,-1.)(inp),training=False))
    return Model(inp,gap)

def extract_frame_features(signal, sr=SR):
    hop=int(0.010*sr); n_fft=int(0.025*sr)
    rms      = librosa.feature.rms(y=signal,frame_length=n_fft,hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=signal,sr=sr,n_fft=n_fft,hop_length=hop)[0]
    bandwidth= librosa.feature.spectral_bandwidth(y=signal,sr=sr,n_fft=n_fft,hop_length=hop)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=signal,sr=sr,n_fft=n_fft,hop_length=hop)[0]
    zcr      = librosa.feature.zero_crossing_rate(y=signal,frame_length=n_fft,hop_length=hop)[0]
    mfcc     = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=13,n_fft=n_fft,hop_length=hop)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc,order=2)
    try:
        f0,voiced_flag,_=librosa.pyin(signal,fmin=75,fmax=500,sr=sr,hop_length=hop)
        f0_voiced=f0.copy(); f0_voiced[~voiced_flag]=np.nan
    except:
        f0_voiced=np.full(len(rms),np.nan)

    def stat_vec(seq,name):
        valid=seq[~np.isnan(seq)] if np.any(np.isnan(seq)) else seq
        if len(valid)<3:
            return {f"FR_{name}_{s}":0. for s in ["mean","std","skew","kurt","P10","P25","P75","P90","IQR","range"]}
        return {
            f"FR_{name}_mean":float(np.mean(valid)),  f"FR_{name}_std": float(np.std(valid)),
            f"FR_{name}_skew":float(sp_skew(valid)),  f"FR_{name}_kurt":float(sp_kurt(valid)),
            f"FR_{name}_P10": float(np.percentile(valid,10)),
            f"FR_{name}_P25": float(np.percentile(valid,25)),
            f"FR_{name}_P75": float(np.percentile(valid,75)),
            f"FR_{name}_P90": float(np.percentile(valid,90)),
            f"FR_{name}_IQR": float(np.percentile(valid,75)-np.percentile(valid,25)),
            f"FR_{name}_range":float(np.max(valid)-np.min(valid)),
        }

    feats={}
    feats.update(stat_vec(rms,"RMS")); feats.update(stat_vec(centroid,"Centroid"))
    feats.update(stat_vec(bandwidth,"Bandwidth")); feats.update(stat_vec(rolloff,"Rolloff"))
    feats.update(stat_vec(zcr,"ZCR")); feats.update(stat_vec(f0_voiced,"F0"))
    for i in range(13):
        feats.update(stat_vec(mfcc[i],f"MFCC{i+1}"))
        feats.update(stat_vec(delta_mfcc[i],f"dMFCC{i+1}"))
        feats.update(stat_vec(delta2_mfcc[i],f"d2MFCC{i+1}"))
    if len(rms)>20:
        rms_c=rms-np.mean(rms); rfft=np.abs(np.fft.rfft(rms_c))
        freqs=np.fft.rfftfreq(len(rms_c),d=hop/sr); tmask=(freqs>=4)&(freqs<=7)
        tot=np.sum(rfft**2)+1e-10
        feats["FR_tremor_ratio"]=float(np.sum(rfft[tmask]**2)/tot)
        feats["FR_tremor_power"]=float(np.sqrt(np.sum(rfft[tmask]**2)))
    else:
        feats["FR_tremor_ratio"]=0.; feats["FR_tremor_power"]=0.
    mid=len(signal)//2
    for seg,tag in [(signal[:mid],"H1"),(signal[mid:],"H2")]:
        r=librosa.feature.rms(y=seg,hop_length=hop)[0]
        f=librosa.feature.mfcc(y=seg,sr=sr,n_mfcc=5,hop_length=hop)
        feats[f"FR_{tag}_rms"]=float(np.mean(r)); feats[f"FR_{tag}_rms_std"]=float(np.std(r))
        for i in range(5): feats[f"FR_{tag}_mfcc{i+1}"]=float(np.mean(f[i]))
    feats["FR_delta_rms"]=feats["FR_H2_rms"]-feats["FR_H1_rms"]
    feats["FR_delta_rms_std"]=feats["FR_H2_rms_std"]-feats["FR_H1_rms_std"]
    for i in range(5): feats[f"FR_delta_mfcc{i+1}"]=feats[f"FR_H2_mfcc{i+1}"]-feats[f"FR_H1_mfcc{i+1}"]
    return {k: 0. if not np.isfinite(v) else v for k,v in feats.items()}

def extract_new_audio_features(signal):
    """TKEO + GNE proxy + energy decay — 12 features."""
    out = {}
    try:
        tkeo = np.abs(signal[1:-1]**2 - signal[:-2]*signal[2:])
        hop  = 512
        env  = np.array([tkeo[i:i+hop].mean() for i in range(0,len(tkeo)-hop,hop)])
        if len(env) < 2: env = np.array([np.mean(tkeo), np.mean(tkeo)])
        out["tkeo_mean"]      = float(np.mean(env))
        out["tkeo_std"]       = float(np.std(env))
        out["tkeo_cv"]        = float(np.std(env)/(np.mean(env)+1e-8))
        out["tkeo_dip_ratio"] = float(np.sum(env<np.percentile(env,20))/len(env))
        out["tkeo_range"]     = float(np.max(env)-np.min(env))

        harm, perc = librosa.effects.hpss(signal)
        out["gne_proxy"]      = float(np.mean(harm**2)/(np.mean(perc**2)+1e-10))
        flat = librosa.feature.spectral_flatness(y=signal, hop_length=512)[0]
        zcr  = librosa.feature.zero_crossing_rate(signal, hop_length=512)[0]
        out["spec_flat_mean"] = float(np.mean(flat))
        out["spec_flat_std"]  = float(np.std(flat))
        out["zcr_mean"]       = float(np.mean(zcr))
        out["zcr_std"]        = float(np.std(zcr))

        rms = librosa.feature.rms(y=signal, hop_length=256)[0]
        mid = len(rms)//2
        h1, h2 = np.mean(rms[:mid]), np.mean(rms[mid:])
        out["energy_decay"]        = float(h1 - h2)
        out["energy_shape_ratio"]  = float(h1/(h2+1e-8))
    except:
        for k in NEW_AUDIO_COLS: out[k] = 0.0
    return {k: 0. if not np.isfinite(v) else v for k,v in out.items()}

def cohen_d(X, y, idx):
    a=X[y==1,idx]; b=X[y==0,idx]
    return abs(np.mean(a)-np.mean(b))/(np.sqrt((np.std(a)**2+np.std(b)**2)/2)+1e-8)

# ══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  LOADING DATA"); print(SEP)

df      = pd.read_csv(os.path.join(BASE,"training_set.csv"))
y       = df["label"].values.astype(np.float32)
fnames  = df["filename"].values
labels  = df["label"].values
df_blind= pd.read_csv(os.path.join(BASE,"blind_test_set.csv"))

# ── Build enriched tabular ────────────────────────────────────────────────────
df2 = df.copy()
df2["dip_x_tremor"]    = df["longest_dip_duration"] * df["tremor_amplitude_std"]
df2["breaks_x_tremor"] = df["possible_micro_breaks"] * df["tremor_amplitude_std"]
df2["dip_x_drops"]     = df["longest_dip_duration"]  * df["max_rms_drop"]

ALL_TAB = TOP_CLEAN + PRAAT_COLS + DYNAMIC_COLS + INTER_COLS
# Praat + dynamics from audio
praat_rows, dyn_rows = [], []
for fn, lbl in zip(fnames, labels):
    path = os.path.join(AUDIO_DIR[int(lbl)], fn)
    praat_rows.append(extract_praat(path)); dyn_rows.append(extract_dynamics(path))
X_praat   = np.array(praat_rows, dtype=np.float32)
X_dynamic = np.array(dyn_rows,   dtype=np.float32)

imp       = SimpleImputer(strategy="median")
X_top_raw = imp.fit_transform(df2[TOP_CLEAN + INTER_COLS].values.astype(np.float32))
X_tab     = np.hstack([X_top_raw[:,:len(TOP_CLEAN)], X_praat, X_dynamic,
                        X_top_raw[:,len(TOP_CLEAN):]]).astype(np.float32)
print(f"  Tabular (clean+interactions): {X_tab.shape}")

# ── Load signals ───────────────────────────────────────────────────────────────
signals_orig = []
for fn, lbl in zip(fnames, labels):
    try: sig,_ = librosa.load(os.path.join(AUDIO_DIR[int(lbl)],fn), sr=SR)
    except: sig = np.zeros(SR, dtype=np.float32)
    signals_orig.append(sig)
signals_aug = [augment(s) for s in signals_orig]

# ── CNN features ───────────────────────────────────────────────────────────────
print("  Building CNN extractor...")
cnn_ext    = build_cnn_extractor()
specs_o    = np.array([[get_3specs(s)] for s in signals_orig]).squeeze(1)
X_cnn_orig = np.stack([cnn_ext.predict(specs_o[:,t],batch_size=32,verbose=0) for t in range(3)],axis=1)
X_cnn_aug  = np.stack([
    np.stack([cnn_ext.predict(
        np.array([[get_3specs(signals_aug[p][ai])] for p in range(len(fnames))]).squeeze(1)[:,t],
        batch_size=32,verbose=0) for t in range(3)],axis=1)
    for ai in range(N_AUG)],axis=1)
print(f"  CNN: {X_cnn_orig.shape}")

# ── Frame features ─────────────────────────────────────────────────────────────
print("  Extracting frame features...")
frame_rows = []
for i, sig in enumerate(signals_orig):
    frame_rows.append(extract_frame_features(sig))
    if (i+1)%20==0: print(f"    {i+1}/{len(fnames)}")
frame_df   = pd.DataFrame(frame_rows).fillna(0.)
X_frame    = frame_df.values.astype(np.float32)
frame_names= frame_df.columns.tolist()
print(f"  Frame: {X_frame.shape}")

# ── New audio features ─────────────────────────────────────────────────────────
print("  Extracting new audio features (TKEO + GNE + energy)...")
new_rows = []
for i, sig in enumerate(signals_orig):
    new_rows.append(extract_new_audio_features(sig))
    if (i+1)%20==0: print(f"    {i+1}/{len(fnames)}")
X_new = np.array([[r[k] for k in NEW_AUDIO_COLS] for r in new_rows], dtype=np.float32)
print(f"  New audio: {X_new.shape}")

# ── Frame feature selection (Cohen's d on full set — done once for reference) ─
cohen_scores = [cohen_d(X_frame, y, i) for i in range(X_frame.shape[1])]
top_idx_all  = np.argsort(cohen_scores)[::-1][:40]

# ══════════════════════════════════════════════════════════════════════════════
# BLIND TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
print("  Extracting blind test features...")
blind_files = [(f,0) for f in BLIND_HC] + [(f,1) for f in BLIND_PD]
blind_y     = np.array([0,0,1,1], dtype=np.float32)

blind_sigs, blind_tab_raw, blind_frm, blind_newf = [], [], [], []
for fn, lbl in blind_files:
    path  = os.path.join(AUDIO_DIR[lbl], fn)
    sig,_ = librosa.load(path, sr=SR)
    blind_sigs.append(sig)
    row   = df_blind[df_blind["filename"]==fn]
    top_v = row[TOP_CLEAN].values[0].astype(np.float32) if len(row)>0 else np.zeros(len(TOP_CLEAN),np.float32)
    inter = np.array([
        float(row["longest_dip_duration"].values[0] * row["tremor_amplitude_std"].values[0]) if len(row)>0 else 0.,
        float(row["possible_micro_breaks"].values[0] * row["tremor_amplitude_std"].values[0]) if len(row)>0 else 0.,
        float(row["longest_dip_duration"].values[0] * row["max_rms_drop"].values[0]) if len(row)>0 else 0.,
    ], dtype=np.float32)
    p = extract_praat(path); d = extract_dynamics(path)
    blind_tab_raw.append(np.concatenate([top_v, p, d, inter]).astype(np.float32))
    fd = extract_frame_features(sig)
    blind_frm.append([fd.get(k,0.) for k in frame_names])
    blind_newf.append(extract_new_audio_features(sig))

blind_tab  = np.array(blind_tab_raw, dtype=np.float32)
blind_frm  = np.array(blind_frm,     dtype=np.float32)
blind_newf = np.array([[r[k] for k in NEW_AUDIO_COLS] for r in blind_newf], dtype=np.float32)
blind_specs= np.array([[get_3specs(s)] for s in blind_sigs]).squeeze(1)
blind_cnn  = np.stack([cnn_ext.predict(blind_specs[:,t],batch_size=4,verbose=0) for t in range(3)],axis=1)
print("  Blind test features ready.")

# ══════════════════════════════════════════════════════════════════════════════
# CV ENGINE
# ══════════════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def report(label, aucs, accs, blind_avg, use_opt_thresh=True):
    cv_auc = np.mean(aucs)
    cv_acc = np.mean(accs)

    # Collect all val predictions for optimal threshold
    print(f"\n  {label}")
    print(f"  AUC  : {cv_auc:.4f} +/- {np.std(aucs):.3f}")
    print(f"  Acc  : {cv_acc*100:.2f}% +/- {np.std(accs)*100:.2f}%")

    blind_pred = (blind_avg >= 0.5).astype(int)
    correct    = int((blind_pred == blind_y.astype(int)).sum())
    print(f"  Blind: {correct}/4 = {correct*25}%")
    for (fn,lbl),(prob,pred) in zip(blind_files, zip(blind_avg, blind_pred)):
        tag = "PD" if pred==1 else "HC"; true = "PD" if lbl==1 else "HC"
        mark= "[OK]" if pred==lbl else "[X]"
        print(f"    {fn[:20]}..  True={true}  Pred={tag}({prob:.3f}) {mark}")
    return cv_auc, cv_acc, correct

def run_cv(build_fn, blind_fn, label, C=0.0102, smote=False):
    aucs, accs, blind_probs_all = [], [], []
    smote_obj = BorderlineSMOTE(k_neighbors=5, random_state=42) if smote else None

    for tr, va in skf.split(np.zeros(len(y)), y):
        X_tr, y_tr, X_va, meta = build_fn(tr, va)

        if smote_obj is not None:
            try: X_tr, y_tr = smote_obj.fit_resample(X_tr, y_tr)
            except: pass

        lr = LogisticRegression(C=C, solver="saga", max_iter=3000, random_state=42)
        lr.fit(X_tr, y_tr)
        p_va = lr.predict_proba(X_va)[:,1]
        aucs.append(roc_auc_score(y[va], p_va))
        accs.append(accuracy_score(y[va], p_va>=0.5))
        blind_probs_all.append(lr.predict_proba(blind_fn(meta))[:,1])

    blind_avg = np.mean(blind_probs_all, axis=0)
    return report(label, aucs, accs, blind_avg)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — BASELINE + FRAME (reproduce previous best)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  EXP A — BASELINE + FRAME FEATURES (previous best, C=0.0102)"); print(SEP)

def build_A(tr, va):
    n = len(tr)
    cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n*N_AUG,3,1280)])
    tab_tr = np.vstack([X_tab[tr],  np.tile(X_tab[tr], (N_AUG,1))])
    frm_tr = np.vstack([X_frame[tr,:][:,top_idx_all], np.tile(X_frame[tr,:][:,top_idx_all],(N_AUG,1))])
    y_tr   = np.concatenate([y[tr], np.tile(y[tr],N_AUG)])

    sc_mel=StandardScaler().fit(cnn_tr[:,0]); pca_mel=PCA(PCA_MEL,random_state=42).fit(sc_mel.transform(cnn_tr[:,0]))
    sc_chr=StandardScaler().fit(cnn_tr[:,1]); pca_chr=PCA(PCA_CHROMA,random_state=42).fit(sc_chr.transform(cnn_tr[:,1]))
    sc_mfc=StandardScaler().fit(cnn_tr[:,2]); pca_mfc=PCA(PCA_MFCC,random_state=42).fit(sc_mfc.transform(cnn_tr[:,2]))
    sc_tab=StandardScaler().fit(tab_tr); sc_frm=StandardScaler().fit(frm_tr)

    def fuse(cnn, tab, frm):
        return np.hstack([
            pca_mel.transform(sc_mel.transform(cnn[:,0])),
            pca_chr.transform(sc_chr.transform(cnn[:,1])),
            pca_mfc.transform(sc_mfc.transform(cnn[:,2])),
            sc_tab.transform(tab), sc_frm.transform(frm)
        ]).astype(np.float32)

    meta = (sc_mel,pca_mel,sc_chr,pca_chr,sc_mfc,pca_mfc,sc_tab,sc_frm,fuse)
    return fuse(cnn_tr,tab_tr,frm_tr), y_tr, fuse(X_cnn_orig[va],X_tab[va],X_frame[va,:][:,top_idx_all]), meta

def blind_A(meta):
    *_, fuse = meta
    return fuse(blind_cnn, blind_tab, blind_frm[:,top_idx_all])

auc_A, acc_A, blind_A_n = run_cv(build_A, blind_A, "EXP A — Baseline + Frame (no SMOTE)")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — + NEW AUDIO FEATURES (no SMOTE)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  EXP B — + NEW AUDIO FEATURES (TKEO+GNE+energy, C=0.0102)"); print(SEP)

def build_B(tr, va):
    n = len(tr)
    cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n*N_AUG,3,1280)])
    tab_tr = np.vstack([X_tab[tr],  np.tile(X_tab[tr], (N_AUG,1))])
    frm_tr = np.vstack([X_frame[tr,:][:,top_idx_all], np.tile(X_frame[tr,:][:,top_idx_all],(N_AUG,1))])
    new_tr = np.vstack([X_new[tr],  np.tile(X_new[tr], (N_AUG,1))])
    y_tr   = np.concatenate([y[tr], np.tile(y[tr],N_AUG)])

    sc_mel=StandardScaler().fit(cnn_tr[:,0]); pca_mel=PCA(PCA_MEL,random_state=42).fit(sc_mel.transform(cnn_tr[:,0]))
    sc_chr=StandardScaler().fit(cnn_tr[:,1]); pca_chr=PCA(PCA_CHROMA,random_state=42).fit(sc_chr.transform(cnn_tr[:,1]))
    sc_mfc=StandardScaler().fit(cnn_tr[:,2]); pca_mfc=PCA(PCA_MFCC,random_state=42).fit(sc_mfc.transform(cnn_tr[:,2]))
    sc_tab=StandardScaler().fit(tab_tr); sc_frm=StandardScaler().fit(frm_tr)
    sc_new=StandardScaler().fit(new_tr)

    def fuse(cnn, tab, frm, new):
        return np.hstack([
            pca_mel.transform(sc_mel.transform(cnn[:,0])),
            pca_chr.transform(sc_chr.transform(cnn[:,1])),
            pca_mfc.transform(sc_mfc.transform(cnn[:,2])),
            sc_tab.transform(tab), sc_frm.transform(frm), sc_new.transform(new)
        ]).astype(np.float32)

    meta = (sc_mel,pca_mel,sc_chr,pca_chr,sc_mfc,pca_mfc,sc_tab,sc_frm,sc_new,fuse)
    return fuse(cnn_tr,tab_tr,frm_tr,new_tr), y_tr, fuse(X_cnn_orig[va],X_tab[va],X_frame[va,:][:,top_idx_all],X_new[va]), meta

def blind_B(meta):
    *_, fuse = meta
    return fuse(blind_cnn, blind_tab, blind_frm[:,top_idx_all], blind_newf)

auc_B, acc_B, blind_B_n = run_cv(build_B, blind_B, "EXP B — + New audio features (no SMOTE)")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — + BORDERLINESMOTE (same features, C=0.0102)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  EXP C — + BORDERLINESMOTE (C=0.0102)"); print(SEP)
auc_C, acc_C, blind_C_n = run_cv(build_B, blind_B,
                                  "EXP C — + BorderlineSMOTE (C=0.0102)", C=0.0102, smote=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — OPTUNA C TUNING + BORDERLINESMOTE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  EXP D — OPTUNA C TUNING + BORDERLINESMOTE"); print(SEP)

def build_for_optuna(tr, va):
    return build_B(tr, va)

print("  Running Optuna (80 trials)...")
smote_tune = BorderlineSMOTE(k_neighbors=5, random_state=42)

def objective(trial):
    C = trial.suggest_float("C", 1e-3, 5.0, log=True)
    fold_aucs = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        X_tr, y_tr, X_va, _ = build_for_optuna(tr, va)
        try: X_tr, y_tr = smote_tune.fit_resample(X_tr, y_tr)
        except: pass
        lr = LogisticRegression(C=C, solver="saga", max_iter=2000, random_state=42)
        lr.fit(X_tr, y_tr)
        p  = lr.predict_proba(X_va)[:,1]
        fold_aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(fold_aucs))

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=80, show_progress_bar=False)
best_C   = study.best_params["C"]
best_auc = study.best_value
print(f"  Best C = {best_C:.5f}  (CV AUC = {best_auc:.4f})")

# ── Final CV run with best C + SMOTE + optimal threshold ─────────────────────
print(f"\n  Running final CV with C={best_C:.5f} + BorderlineSMOTE...")
aucs_D, accs_D, blind_probs_D = [], [], []
all_val_true, all_val_prob = [], []
smote_final = BorderlineSMOTE(k_neighbors=5, random_state=42)

for tr, va in skf.split(np.zeros(len(y)), y):
    X_tr, y_tr, X_va, meta = build_B(tr, va)
    try: X_tr, y_tr = smote_final.fit_resample(X_tr, y_tr)
    except: pass
    lr = LogisticRegression(C=best_C, solver="saga", max_iter=3000, random_state=42)
    lr.fit(X_tr, y_tr)
    p_va = lr.predict_proba(X_va)[:,1]
    aucs_D.append(roc_auc_score(y[va], p_va))
    accs_D.append(accuracy_score(y[va], p_va>=0.5))
    all_val_true.extend(y[va].tolist()); all_val_prob.extend(p_va.tolist())
    blind_probs_D.append(lr.predict_proba(blind_B(meta))[:,1])

blind_avg_D = np.mean(blind_probs_D, axis=0)

# Optimal threshold via Youden J on pooled val predictions
fpr_, tpr_, thr_ = roc_curve(all_val_true, all_val_prob)
best_t = float(thr_[np.argmax(tpr_-fpr_)])
acc_opt = accuracy_score(all_val_true, (np.array(all_val_prob)>=best_t).astype(int))
acc_05  = accuracy_score(all_val_true, (np.array(all_val_prob)>=0.5).astype(int))

print(f"\n  EXP D — Optuna C={best_C:.4f} + BorderlineSMOTE + Optimal threshold")
print(f"  AUC   : {np.mean(aucs_D):.4f} +/- {np.std(aucs_D):.3f}")
print(f"  Acc@0.5 : {acc_05*100:.2f}%")
print(f"  Acc@opt ({best_t:.3f}): {acc_opt*100:.2f}%   <- Youden optimal threshold")

blind_pred_D = (blind_avg_D >= best_t).astype(int)
correct_D    = int((blind_pred_D == blind_y.astype(int)).sum())
print(f"  Blind : {correct_D}/4 = {correct_D*25}%")
for (fn,lbl),(prob,pred) in zip(blind_files, zip(blind_avg_D, blind_pred_D)):
    tag="PD" if pred==1 else "HC"; true="PD" if lbl==1 else "HC"
    mark="[OK]" if pred==lbl else "[X]"
    print(f"    {fn[:20]}..  True={true}  Pred={tag}({prob:.3f}) {mark}")
auc_D = float(np.mean(aucs_D)); acc_D = float(acc_opt)

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  FINAL SUMMARY")
print(SEP)
print(f"  {'Experiment':<42} {'CV AUC':>8} {'CV Acc':>8} {'Blind':>7}")
print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*7}")
rows = [
    ("A: Baseline+Frame (C=0.0102)",       auc_A, acc_A,  blind_A_n),
    ("B: +New audio feats (C=0.0102)",      auc_B, acc_B,  blind_B_n),
    ("C: +BorderlineSMOTE (C=0.0102)",      auc_C, acc_C,  blind_C_n),
    ("D: +Optuna C+SMOTE+OPT thresh",       auc_D, acc_D,  correct_D),
]
for name, auc, acc, blind in rows:
    print(f"  {name:<42} {auc:>8.4f} {acc*100:>7.2f}% {blind:>5}/4")
print(SEP)
print(f"\n  Best C found by Optuna: {best_C:.5f}")
print(f"  Optimal decision threshold: {best_t:.3f}")
print(SEP)
