"""
parkinson_app.py  —  Parkinson's Disease Voice Analysis System
==============================================================
streamlit run parkinson_app.py
"""

import os, sys, io, warnings, pickle, time, tempfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import shap
from PIL import Image
from scipy.stats import kurtosis as sp_kurt, skew as sp_skew
import parselmouth
from parselmouth.praat import call
import nolds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import streamlit as st
from raw_recorder import raw_recorder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.applications import MobileNetV2

tf.random.set_seed(42); np.random.seed(42)

BASE     = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE, "model_artifacts")
IMG_SIZE = 128; SR = 22050

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
[data-testid="stAppViewContainer"]  { background:#0D1117; }
[data-testid="stSidebar"]           { background:#161B22; border-right:1px solid #30363D; }
[data-testid="stSidebar"] *         { color:#E6EDF3 !important; }
h1,h2,h3,h4                        { color:#58A6FF !important; font-family:'Segoe UI',sans-serif; }
p, li, label, span, div            { color:#C9D1D9; font-family:'Segoe UI',sans-serif; }
.card {
    background:#161B22; border:1px solid #30363D; border-radius:12px;
    padding:20px 24px; margin-bottom:16px;
}
.metric-card {
    background:#161B22; border:1px solid #30363D; border-radius:10px;
    padding:16px; text-align:center;
}
.pred-pd   { background:#2D0A0A; border:2px solid #FF4444; border-radius:14px; padding:24px; }
.pred-hc   { background:#0A2D0A; border:2px solid #44FF88; border-radius:14px; padding:24px; }
.pred-unk  { background:#1A1A0A; border:2px solid #FFAA44; border-radius:14px; padding:24px; }
.badge-pd  { background:#FF4444; color:#000; padding:4px 12px; border-radius:20px; font-weight:700; font-size:12px; }
.badge-hc  { background:#44FF88; color:#000; padding:4px 12px; border-radius:20px; font-weight:700; font-size:12px; }
.feat-label{ color:#8B949E; font-size:13px; }
.feat-value{ color:#E6EDF3; font-weight:600; font-size:22px; }
.feat-unit { color:#58A6FF; font-size:12px; }
.feat-delta-up   { color:#FF6B6B; font-size:12px; }
.feat-delta-down { color:#56D364; font-size:12px; }
.section-header { color:#58A6FF; font-size:18px; font-weight:600; margin:20px 0 10px; border-bottom:1px solid #30363D; padding-bottom:8px; }
div[data-testid="metric-container"] { background:#161B22; border:1px solid #30363D; border-radius:10px; padding:16px; }
div[data-testid="metric-container"] label { color:#8B949E !important; font-size:13px !important; }
div[data-testid="metric-container"] div  { color:#E6EDF3 !important; font-size:24px !important; }
.stButton>button { background:#238636; color:#fff; border:none; border-radius:8px; font-weight:600; }
.stButton>button:hover { background:#2EA043; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ── Feature display metadata ───────────────────────────────────────────────────
FEATURE_META = {
    "FR_tremor_ratio":       ("Tremor Power 4–7 Hz", "",   "High in PD (voice shaking)"),
    "longest_dip_duration":  ("Longest Voice Dip",   "s",  "Long dips common in PD"),
    "possible_micro_breaks": ("Voice Micro-Breaks",  "",   "More breaks → PD pattern"),
    "tremor_amplitude_std":  ("Energy Variability",  "",   "High variability in PD"),
    "praat_jitter_local":    ("Pitch Jitter",         "%",  "Pitch irregularity (PD higher)"),
    "praat_shimmer_local":   ("Amplitude Shimmer",    "%",  "Amplitude irregularity (PD higher)"),
    "praat_hnr":             ("Harmonic-Noise Ratio", "dB", "Lower in PD (noisier voice)"),
    "tkeo_cv":               ("TKEO Modulation",      "",   "Voice energy fluctuation (PD higher)"),
    "gne_proxy":             ("Glottal Clarity",      "",   "Higher = cleaner vocal fold vibration"),
    "energy_decay":          ("Energy Decay",         "",   "First-half vs second-half RMS diff"),
}

MODEL_LABELS = {
    "Model A: Baseline + Frame":          {"label":"Baseline + Frame Features (fixed, untuned)",   "color":"#636EFA","rank":3},
    "Model B: + New Audio + SMOTE":       {"label":"+ TKEO + GNE + SMOTE (fixed, untuned)",        "color":"#EF553B","rank":2},
    "Model C: + Optuna + Optimal Thresh": {"label":"Full Pipeline — Nested CV (honest estimate)",  "color":"#00CC96","rank":1},
}

# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model...")
def load_cnn():
    inp  = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    base = MobileNetV2(include_top=False,weights="imagenet",input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    gap  = layers.GlobalAveragePooling2D()(base(layers.Rescaling(2.,-1.)(inp),training=False))
    return Model(inp,gap)

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    pl  = pickle.load(open(os.path.join(ART_DIR,"pipeline.pkl"),"rb"))
    cvs = pickle.load(open(os.path.join(ART_DIR,"cv_stats.pkl"),"rb"))
    td  = pickle.load(open(os.path.join(ART_DIR,"training_data.pkl"),"rb"))
    return pl, cvs, td

# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (all from audio — no CSV needed)
# ══════════════════════════════════════════════════════════════════════════════
def extract_praat(sig, sr):
    try:
        snd = parselmouth.Sound(sig.astype(float), sr)
        pp  = call(snd,"To PointProcess (periodic, cc)",75,500)
        jl  = call(pp,"Get jitter (local)",0,0,0.0001,0.02,1.3)
        jr  = call(pp,"Get jitter (rap)",0,0,0.0001,0.02,1.3)
        sl  = call([snd,pp],"Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)
        h   = call(call(snd,"To Harmonicity (cc)",0.01,75,0.1,1.0),"Get mean",0,0)
        nhr = (1./h) if h and h>0 else 0.
        vals= [jl,jr,sl,h,nhr]
        return [0. if (v is None or not np.isfinite(v)) else v for v in vals]
    except: return [0.]*5

def extract_dynamics(sig, sr):
    try:
        snd   = parselmouth.Sound(sig.astype(float), sr)
        pitch = snd.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=500)
        f0    = pitch.selected_array["frequency"]
        voiced= f0[f0>0].astype(np.float64)
        if len(voiced)<30: return [0.,0.5,0.]
        # RPDE
        x=voiced/(np.mean(voiced)+1e-8); rts=[]
        for i in range(len(x)-1):
            for j in range(i+1,min(i+100,len(x))):
                if abs(x[j]-x[i])<0.05: rts.append(j-i); break
        if rts:
            h=np.zeros(max(rts)+1)
            for r in rts: h[r]+=1
            h/=h.sum()+1e-8; h=h[h>0]
            H=-np.sum(h*np.log2(h+1e-8)); Hm=np.log2(max(rts)+1)
            rpde= H/Hm if Hm>0 else 0.
        else: rpde=0.
        # PPE
        p=1./(voiced+1e-8); ln=np.log(p/(np.mean(p)+1e-8)+1e-8)
        hh,_=np.histogram(ln,bins=max(10,min(30,len(voiced)//5)))
        hh=hh.astype(float); hh/=hh.sum()+1e-8; hh=hh[hh>0]
        ppe=-np.sum(hh*np.log(hh+1e-8))
        try: dfa=float(nolds.dfa(voiced))
        except: dfa=0.5
        vals=[rpde,dfa,ppe]
        return [0. if (v is None or not np.isfinite(v)) else v for v in vals]
    except: return [0.,0.5,0.]

def compute_tabular(sig, sr=SR):
    hop=256; n_fft=2048
    rms=librosa.feature.rms(y=sig,hop_length=hop)[0]
    fd=hop/sr; thresh=np.percentile(rms,25)
    in_dip=rms<thresh; dip_evs=[]; start=None
    for i,d in enumerate(in_dip):
        if d and start is None: start=i
        elif not d and start is not None:
            dip_evs.append((i-start)*fd); start=None
    if start is not None: dip_evs.append((len(in_dip)-start)*fd)
    drops=np.abs(np.diff(rms)); wf=int(0.5*sr/hop)
    lvars=[float(np.var(rms[i:i+wf])) for i in range(0,max(1,len(rms)-wf),wf//2)]
    mfcc=librosa.feature.mfcc(y=sig,sr=sr,n_mfcc=13,hop_length=hop)
    rolloff=librosa.feature.spectral_rolloff(y=sig,sr=sr,hop_length=hop)[0]
    bw=librosa.feature.spectral_bandwidth(y=sig,sr=sr,hop_length=hop)[0]
    S=np.abs(librosa.stft(sig,n_fft=n_fft,hop_length=hop))
    freqs=librosa.fft_frequencies(sr=sr,n_fft=n_fft)
    mfi=int(np.argmax(np.mean(S,axis=1)))
    sf=float(np.mean(np.sum(np.diff(S,axis=1)**2,axis=0)))
    try:
        a=librosa.lpc(sig,order=12); c=np.zeros(13)
        for m in range(1,13):
            c[m]=-a[m] if m<len(a) else 0.
            for k in range(1,m):
                if k<len(a): c[m]-=(k/m)*c[k]*a[m-k] if (m-k)<len(a) else 0.
        l1=float(c[1])
    except: l1=0.
    tab={
        "longest_dip_duration":    max(dip_evs) if dip_evs else 0.,
        "tremor_amplitude_std":    float(np.std(rms)),
        "total_dip_duration":      sum(dip_evs),
        "possible_micro_breaks":   float(len(dip_evs)),
        "max_rms_drop":            float(np.max(drops)) if len(drops)>0 else 0.,
        "local_rms_variance":      float(np.mean(lvars)) if lvars else 0.,
        "num_rms_dips":            float(len([d for d in dip_evs if d>=0.05])),
        "stumble_total_duration_sec": sum(d for d in dip_evs if d>=0.10),
        "stumble_num_events":      float(len([d for d in dip_evs if d>=0.10])),
        "mfcc_3":                  float(np.mean(mfcc[2])),
        "spectral_bandwidth":      float(np.mean(bw)),
        "spectral_rolloff":        float(np.mean(rolloff)),
        "max_frequency":           float(freqs[mfi]),
        "spectral_distance":       float(np.clip(sf,0,1e6)),
        "lpcc_1":                  l1,
    }
    # interaction features
    tab["dip_x_tremor"]    = tab["longest_dip_duration"] * tab["tremor_amplitude_std"]
    tab["breaks_x_tremor"] = tab["possible_micro_breaks"] * tab["tremor_amplitude_std"]
    tab["dip_x_drops"]     = tab["longest_dip_duration"] * tab["max_rms_drop"]
    return {k:0. if(v is None or not np.isfinite(v)) else v for k,v in tab.items()}

def sv(seq,name):
    valid=seq[~np.isnan(seq)] if np.any(np.isnan(seq)) else seq
    if len(valid)<3:
        return {f"FR_{name}_{s}":0. for s in ["mean","std","skew","kurt","P10","P25","P75","P90","IQR","range"]}
    return {f"FR_{name}_mean":float(np.mean(valid)),f"FR_{name}_std":float(np.std(valid)),
            f"FR_{name}_skew":float(sp_skew(valid)),f"FR_{name}_kurt":float(sp_kurt(valid)),
            f"FR_{name}_P10":float(np.percentile(valid,10)),f"FR_{name}_P25":float(np.percentile(valid,25)),
            f"FR_{name}_P75":float(np.percentile(valid,75)),f"FR_{name}_P90":float(np.percentile(valid,90)),
            f"FR_{name}_IQR":float(np.percentile(valid,75)-np.percentile(valid,25)),
            f"FR_{name}_range":float(np.max(valid)-np.min(valid))}

def extract_frame(sig, sr=SR):
    hop=int(0.010*sr); nf=int(0.025*sr)
    rms=librosa.feature.rms(y=sig,frame_length=nf,hop_length=hop)[0]
    ce=librosa.feature.spectral_centroid(y=sig,sr=sr,n_fft=nf,hop_length=hop)[0]
    bw=librosa.feature.spectral_bandwidth(y=sig,sr=sr,n_fft=nf,hop_length=hop)[0]
    ro=librosa.feature.spectral_rolloff(y=sig,sr=sr,n_fft=nf,hop_length=hop)[0]
    zc=librosa.feature.zero_crossing_rate(y=sig,frame_length=nf,hop_length=hop)[0]
    mf=librosa.feature.mfcc(y=sig,sr=sr,n_mfcc=13,n_fft=nf,hop_length=hop)
    dm=librosa.feature.delta(mf); d2m=librosa.feature.delta(mf,order=2)
    try:
        f0,vf,_=librosa.pyin(sig,fmin=75,fmax=500,sr=sr,hop_length=hop)
        f0v=f0.copy(); f0v[~vf]=np.nan
    except: f0v=np.full(len(rms),np.nan)
    feats={}
    for arr,nm in [(rms,"RMS"),(ce,"Centroid"),(bw,"Bandwidth"),(ro,"Rolloff"),(zc,"ZCR"),(f0v,"F0")]:
        feats.update(sv(arr,nm))
    for i in range(13):
        feats.update(sv(mf[i],f"MFCC{i+1}"))
        feats.update(sv(dm[i],f"dMFCC{i+1}"))
        feats.update(sv(d2m[i],f"d2MFCC{i+1}"))
    if len(rms)>20:
        rc=rms-np.mean(rms); rf=np.abs(np.fft.rfft(rc))
        fr=np.fft.rfftfreq(len(rc),d=hop/sr); tm=(fr>=4)&(fr<=7)
        tot=np.sum(rf**2)+1e-10
        feats["FR_tremor_ratio"]=float(np.sum(rf[tm]**2)/tot)
        feats["FR_tremor_power"]=float(np.sqrt(np.sum(rf[tm]**2)))
    else: feats["FR_tremor_ratio"]=0.; feats["FR_tremor_power"]=0.
    mid=len(sig)//2
    for seg,tag in [(sig[:mid],"H1"),(sig[mid:],"H2")]:
        r=librosa.feature.rms(y=seg,hop_length=hop)[0]
        f=librosa.feature.mfcc(y=seg,sr=sr,n_mfcc=5,hop_length=hop)
        feats[f"FR_{tag}_rms"]=float(np.mean(r)); feats[f"FR_{tag}_rms_std"]=float(np.std(r))
        for i in range(5): feats[f"FR_{tag}_mfcc{i+1}"]=float(np.mean(f[i]))
    feats["FR_delta_rms"]=feats["FR_H2_rms"]-feats["FR_H1_rms"]
    feats["FR_delta_rms_std"]=feats["FR_H2_rms_std"]-feats["FR_H1_rms_std"]
    for i in range(5): feats[f"FR_delta_mfcc{i+1}"]=feats[f"FR_H2_mfcc{i+1}"]-feats[f"FR_H1_mfcc{i+1}"]
    return {k:0. if not np.isfinite(v) else v for k,v in feats.items()}

def extract_new_audio(sig):
    out={}
    try:
        tkeo=np.abs(sig[1:-1]**2-sig[:-2]*sig[2:]); hop=512
        env=np.array([tkeo[i:i+hop].mean() for i in range(0,len(tkeo)-hop,hop)])
        if len(env)<2: env=np.array([np.mean(tkeo),np.mean(tkeo)])
        out["tkeo_mean"]=float(np.mean(env)); out["tkeo_std"]=float(np.std(env))
        out["tkeo_cv"]=float(np.std(env)/(np.mean(env)+1e-8))
        out["tkeo_dip_ratio"]=float(np.sum(env<np.percentile(env,20))/len(env))
        out["tkeo_range"]=float(np.max(env)-np.min(env))
        harm,perc=librosa.effects.hpss(sig)
        out["gne_proxy"]=float(np.mean(harm**2)/(np.mean(perc**2)+1e-10))
        flat=librosa.feature.spectral_flatness(y=sig,hop_length=512)[0]
        zcr=librosa.feature.zero_crossing_rate(sig,hop_length=512)[0]
        out["spec_flat_mean"]=float(np.mean(flat)); out["spec_flat_std"]=float(np.std(flat))
        out["zcr_mean"]=float(np.mean(zcr)); out["zcr_std"]=float(np.std(zcr))
        rms=librosa.feature.rms(y=sig,hop_length=256)[0]; mid=len(rms)//2
        h1,h2=np.mean(rms[:mid]),np.mean(rms[mid:])
        out["energy_decay"]=float(h1-h2); out["energy_shape_ratio"]=float(h1/(h2+1e-8))
    except:
        for k in ["tkeo_mean","tkeo_std","tkeo_cv","tkeo_dip_ratio","tkeo_range",
                  "gne_proxy","spec_flat_mean","spec_flat_std","zcr_mean","zcr_std",
                  "energy_decay","energy_shape_ratio"]: out[k]=0.
    return {k:0. if not np.isfinite(v) else v for k,v in out.items()}

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

@st.cache_data(show_spinner=False)
def full_extract(_sig_bytes, sr):
    sig = np.frombuffer(_sig_bytes, dtype=np.float32)
    return _do_extract(sig, sr)

def _do_extract(sig, sr):
    pl,_,_ = load_artifacts()
    cnn    = load_cnn()

    tab_d   = compute_tabular(sig, sr)
    tab_arr = np.array([[tab_d.get(c,0.) for c in pl["tab_cols"]]], dtype=np.float32)

    frm_d   = extract_frame(sig, sr)
    frm_arr = np.array([[frm_d.get(n,0.) for n in pl["frame_names"]]], dtype=np.float32)
    frm_top = frm_arr[:,pl["top_frame_idx"]]

    new_d   = extract_new_audio(sig)
    new_arr = np.array([[new_d.get(k,0.) for k in pl["new_audio_cols"]]], dtype=np.float32)

    praat_v = extract_praat(sig, sr)
    dyn_v   = extract_dynamics(sig, sr)

    specs   = np.array([[get_3specs(sig)]]).squeeze(1)
    cnn_f   = np.stack([cnn.predict(specs[:,t],batch_size=1,verbose=0) for t in range(3)],axis=1)

    # Update tab_arr with correct praat/dynamic values
    praat_cols = ["praat_jitter_local","praat_jitter_rap","praat_shimmer_local","praat_hnr","praat_nhr"]
    dyn_cols   = ["rpde","dfa","ppe"]
    for i,c in enumerate(praat_cols):
        idx = pl["tab_cols"].index(c) if c in pl["tab_cols"] else -1
        if idx>=0: tab_arr[0,idx] = praat_v[i]
    for i,c in enumerate(dyn_cols):
        idx = pl["tab_cols"].index(c) if c in pl["tab_cols"] else -1
        if idx>=0: tab_arr[0,idx] = dyn_v[i]

    def fuse(cnn_f2, tab2, frm2, new2):
        return np.hstack([
            pl["pca_mel"].transform(pl["sc_mel"].transform(cnn_f2[:,0])),
            pl["pca_chr"].transform(pl["sc_chr"].transform(cnn_f2[:,1])),
            pl["pca_mfc"].transform(pl["sc_mfc"].transform(cnn_f2[:,2])),
            pl["sc_tab"].transform(tab2),
            pl["sc_frm"].transform(frm2),
            pl["sc_new"].transform(new2),
        ]).astype(np.float32)

    X = fuse(cnn_f, tab_arr, frm_top, new_arr)
    prob   = float(pl["lr_model"].predict_proba(X)[0,1])
    s_vals = pl["explainer"].shap_values(X)[0]

    # Interpretable raw values for radar chart
    radar_raw = {}
    for feat in pl["radar_features"]:
        if feat in tab_d:    radar_raw[feat]=tab_d[feat]
        elif feat in frm_d:  radar_raw[feat]=frm_d[feat]
        elif feat in new_d:  radar_raw[feat]=new_d[feat]
        elif feat=="praat_jitter_local":   radar_raw[feat]=praat_v[0]
        elif feat=="praat_shimmer_local":  radar_raw[feat]=praat_v[2]
        elif feat=="praat_hnr":            radar_raw[feat]=praat_v[3]
        else: radar_raw[feat]=0.

    mel_img  = librosa.power_to_db(librosa.feature.melspectrogram(y=sig,sr=sr,n_mels=128,fmax=8000),ref=np.max)
    chro_img = librosa.feature.chroma_stft(y=sig,sr=sr,n_chroma=128)
    mfcc_img = librosa.feature.mfcc(y=sig,sr=sr,n_mfcc=128)

    return {
        "prob":prob,"X":X,"shap_vals":s_vals,
        "radar_raw":radar_raw,"tab_d":tab_d,"frm_d":frm_d,
        "mel_img":mel_img,"chro_img":chro_img,"mfcc_img":mfcc_img,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ══════════════════════════════════════════════════════════════════════════════
def gauge_chart(prob, threshold):
    pct = prob*100
    if prob >= threshold: color,label = "#FF4444","ABOVE THRESHOLD"
    elif prob >= threshold-0.1: color,label = "#FFAA44","NEAR THRESHOLD"
    else: color,label = "#44FF88","BELOW THRESHOLD"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"font":{"size":36,"color":"#E6EDF3"}},
        delta={"reference":threshold*100,"increasing":{"color":"#FF4444"},"decreasing":{"color":"#44FF88"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#8B949E","tickfont":{"color":"#8B949E"}},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"#161B22",
            "bordercolor":"#30363D",
            "steps":[
                {"range":[0,threshold*100],"color":"#0D2818"},
                {"range":[threshold*100,100],"color":"#2D0A0A"},
            ],
            "threshold":{"line":{"color":"#FFFFFF","width":2},"thickness":0.8,"value":threshold*100},
        },
        title={"text":f"Uncalibrated PD Score (not a probability)  |  {label}","font":{"size":14,"color":"#8B949E"}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"#E6EDF3"},
                      height=260,margin=dict(l=20,r=20,t=40,b=10))
    return fig

def shap_chart(shap_vals, feature_names, n=15):
    order   = np.argsort(np.abs(shap_vals))[::-1][:n]
    vals    = shap_vals[order]
    names   = [feature_names[i] for i in order]
    # Clean up names
    clean   = []
    for nm in names:
        nm2 = nm.replace("FR_","").replace("_mean","(μ)").replace("_std","(σ)").replace("_"," ")
        clean.append(nm2)
    colors  = ["#FF6B6B" if v>0 else "#56D364" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=clean[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in vals[::-1]],
        textposition="outside", textfont={"color":"#E6EDF3","size":11},
    ))
    fig.add_vline(x=0, line_color="#8B949E", line_width=1)
    fig.update_layout(
        title={"text":"Feature Contributions (SHAP)  🔴 toward PD  🟢 toward HC",
               "font":{"color":"#58A6FF","size":14}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title":"SHAP Value","color":"#8B949E","gridcolor":"#21262D","zeroline":False},
        yaxis={"color":"#C9D1D9","gridcolor":"#21262D"},
        height=max(350, n*28), margin=dict(l=180,r=80,t=50,b=30),
    )
    return fig

def radar_chart(radar_raw, tab_stats, radar_features):
    labels=[FEATURE_META.get(f,(f,"",""))[0] for f in radar_features]
    def norm(val,mn,mx): return float(np.clip((val-mn)/(mx-mn+1e-8),0,1))*100

    patient_n=[norm(radar_raw.get(f,0),tab_stats[f]["global_min"],tab_stats[f]["global_max"]) for f in radar_features]
    hc_n     =[norm(tab_stats[f]["HC_mean"],tab_stats[f]["global_min"],tab_stats[f]["global_max"]) for f in radar_features]
    pd_n     =[norm(tab_stats[f]["PD_mean"],tab_stats[f]["global_min"],tab_stats[f]["global_max"]) for f in radar_features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=hc_n+[hc_n[0]],  theta=labels+[labels[0]],name="HC Mean", fill="toself",
                                   line_color="#44FF88",fillcolor="rgba(68,255,136,0.1)",mode="lines"))
    fig.add_trace(go.Scatterpolar(r=pd_n+[pd_n[0]],  theta=labels+[labels[0]],name="PD Mean", fill="toself",
                                   line_color="#FF4444",fillcolor="rgba(255,68,68,0.1)",mode="lines"))
    fig.add_trace(go.Scatterpolar(r=patient_n+[patient_n[0]],theta=labels+[labels[0]],name="This Patient",fill="toself",
                                   line_color="#FFAA44",fillcolor="rgba(255,170,68,0.25)",
                                   line=dict(width=2.5),marker=dict(size=6),mode="lines+markers"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,100],color="#8B949E",gridcolor="#21262D"),
                   angularaxis=dict(color="#C9D1D9"),bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",legend=dict(font=dict(color="#C9D1D9"),bgcolor="rgba(0,0,0,0)"),
        title={"text":"Voice Biomarker Profile (0–100 normalised to dataset range)",
               "font":{"color":"#58A6FF","size":13}},
        height=420, margin=dict(l=60,r=60,t=50,b=30),
    )
    return fig

def spectrogram_fig(mat, title, cmap="magma"):
    fig,ax=plt.subplots(figsize=(5,3))
    fig.patch.set_facecolor("#161B22"); ax.set_facecolor("#161B22")
    ax.imshow(mat,aspect="auto",origin="lower",cmap=cmap)
    ax.set_title(title,color="#58A6FF",fontsize=10,pad=4)
    ax.set_xlabel("Time",color="#8B949E",fontsize=8); ax.set_ylabel("Frequency / Feature",color="#8B949E",fontsize=8)
    ax.tick_params(colors="#8B949E",labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
    plt.tight_layout(pad=0.5)
    buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=100,facecolor="#161B22"); plt.close(); buf.seek(0)
    return buf

def roc_figure(cv_stats):
    fig=go.Figure()
    fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,
                  line=dict(color="#30363D",width=1,dash="dash"))
    for cfg_name,meta in MODEL_LABELS.items():
        s=cv_stats.get(cfg_name,{});
        if not s: continue
        fig.add_trace(go.Scatter(
            x=s["fpr"],y=s["tpr"],mode="lines",name=f"{meta['label']} (AUC={s['mean_auc']:.3f})",
            line=dict(color=meta["color"],width=2.5),
        ))
    fig.update_layout(
        title={"text":"ROC Curves — 5-Fold Cross-Validation","font":{"color":"#58A6FF","size":14}},
        xaxis={"title":"False Positive Rate","color":"#8B949E","gridcolor":"#21262D"},
        yaxis={"title":"True Positive Rate","color":"#8B949E","gridcolor":"#21262D"},
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#C9D1D9"),bgcolor="rgba(0,0,0,0)",x=0.55,y=0.1),
        height=400,margin=dict(l=50,r=20,t=50,b=50),
    )
    return fig

def confusion_fig(cm_data, model_name):
    z  = np.array(cm_data)
    txt= [[f"TN={z[0,0]}",f"FP={z[0,1]}"],[f"FN={z[1,0]}",f"TP={z[1,1]}"]]
    fig= go.Figure(go.Heatmap(
        z=z,x=["Pred HC","Pred PD"],y=["True HC","True PD"],
        colorscale=[[0,"#0D2818"],[0.5,"#1F4F39"],[1,"#238636"]],
        showscale=False,text=txt,texttemplate="%{text}",
        textfont={"size":16,"color":"#E6EDF3"},
    ))
    spec=any(l==model_name for l in MODEL_LABELS.keys())
    fig.update_layout(
        title={"text":f"Confusion Matrix — {MODEL_LABELS.get(model_name,{}).get('label',model_name)}",
               "font":{"color":"#58A6FF","size":13}},
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"color":"#C9D1D9"},yaxis={"color":"#C9D1D9","autorange":"reversed"},
        height=280,margin=dict(l=60,r=20,t=60,b=50),
    )
    return fig

def global_importance_fig(lr_model, feature_names):
    coefs = lr_model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:20]
    vals  = coefs[order]
    names = []
    for i in order:
        n=feature_names[i]
        n=n.replace("FR_","").replace("_mean","(μ)").replace("_std","(σ)").replace("_"," ")
        names.append(n)
    colors=["#FF6B6B" if v>0 else "#56D364" for v in vals]
    fig=go.Figure(go.Bar(x=vals[::-1],y=names[::-1],orientation="h",
                          marker_color=colors[::-1],
                          text=[f"{v:+.3f}" for v in vals[::-1]],
                          textposition="outside",textfont={"color":"#E6EDF3","size":10}))
    fig.add_vline(x=0,line_color="#8B949E",line_width=1)
    fig.update_layout(
        title={"text":"Global Feature Importance (LR Coefficients)","font":{"color":"#58A6FF","size":14}},
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title":"Coefficient","color":"#8B949E","gridcolor":"#21262D"},
        yaxis={"color":"#C9D1D9","gridcolor":"#21262D"},
        height=550,margin=dict(l=200,r=80,t=50,b=30),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 PD Voice Analysis")
    st.markdown("*AI-powered Parkinson's Disease detection from sustained vowel recordings*")
    st.divider()
    page = st.radio("Navigation", ["🔬 Analyze Patient","📊 Model Performance","ℹ️ About"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("77 patients · 39 HC · 38 PD")
    st.markdown("**Nested CV Result** (not shown to outperform a fixed baseline — see paper Table 2b)")
    if os.path.exists(os.path.join(ART_DIR,"cv_stats.pkl")):
        _cvs = pickle.load(open(os.path.join(ART_DIR,"cv_stats.pkl"),"rb"))
        _best = _cvs.get("Model C: + Optuna + Optimal Thresh", {})
        _auc  = _best.get("mean_auc", 0)
        _acc  = _best.get("opt_acc", 0)
        st.markdown(f"AUC: **{_auc:.3f}** · Acc: **{_acc*100:.1f}%**")
    else:
        st.markdown("AUC: **–** · Acc: **–**")
    st.markdown("**Task**: Sustained 'Aaah' vowel")
    st.divider()
    st.markdown("<span style='color:#8B949E;font-size:11px'>⚠️ For research use only.<br>Not a clinical diagnostic tool.</span>",
                unsafe_allow_html=True)

# Check artifacts exist
if not os.path.exists(os.path.join(ART_DIR,"pipeline.pkl")):
    st.error("**Model artifacts not found.** Please run `python save_model_artifacts.py` first.")
    st.stop()

pl, cv_stats, td = load_artifacts()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyze Patient
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Analyze Patient":
    st.markdown("# 🔬 Patient Voice Analysis")
    st.markdown("Upload a sustained **'Aaah'** vowel recording (.wav) to analyze for Parkinson's Disease biomarkers.")

    col_up, col_info = st.columns([2,1])
    with col_up:
        input_mode = st.radio("Input method", ["📁 Upload file", "🎙️ Record live"],
                               horizontal=True, label_visibility="collapsed")
        if input_mode == "📁 Upload file":
            audio_file = st.file_uploader(
                "Upload voice recording", type=["wav","mp3","m4a","ogg","flac","aac","webm","opus"],
                help="Sustained 'Aaah' vowel, 3–10 seconds, mono or stereo. Any common audio "
                     "format works — WAV, MP3, M4A, OGG, FLAC, AAC, WebM, Opus.")
        else:
            st.caption("Click **Start recording**, take a breath, hold a steady **'Aaah'** for "
                       "3–10 seconds, then click **Stop recording**.")
            st.info("⚠️ Having trouble with live recording? Use **Upload file** instead — record a "
                    "voice memo with your phone's built-in recorder app, then upload it here. That "
                    "path is fully tested and works on any device.")
            mic_result = raw_recorder(key="mic")
            audio_file = None
            if mic_result is not None and mic_result.get("bytes"):
                n_bytes = len(mic_result["bytes"])
                st.caption(f"✅ Recording received: {n_bytes:,} bytes, mime={mic_result.get('mime_type','?')}")
                if n_bytes < 1000:
                    st.warning("This recording looks too small to contain real audio — it may be "
                               "empty or corrupted. Please try again or use Upload file instead.")
                else:
                    ext = ".webm"
                    mime = mic_result.get("mime_type", "")
                    if "mp4" in mime: ext = ".mp4"
                    elif "ogg" in mime: ext = ".ogg"
                    elif "wav" in mime: ext = ".wav"
                    audio_file = io.BytesIO(mic_result["bytes"])
                    audio_file.name = f"recording{ext}"
    with col_info:
        st.markdown("""<div class='card'>
        <b style='color:#58A6FF'>Recording Guidelines</b><br><br>
        • Sustained 'Aaah' vowel<br>
        • 3–10 seconds duration<br>
        • Quiet environment<br>
        • Consistent volume<br>
        • .wav format preferred
        </div>""", unsafe_allow_html=True)

    if audio_file is not None:
        try:
            with st.spinner("Extracting voice biomarkers... (30–60 sec for CNN features)"):
                # Route through a real temp file rather than an in-memory buffer:
                # compressed formats (MP3/M4A/AAC) fall back from soundfile to the
                # ffmpeg-based audioread backend, which needs an actual file path
                # to hand to the ffmpeg subprocess, not a BytesIO object.
                suffix = os.path.splitext(getattr(audio_file, "name", "") or "")[1] or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                try:
                    wav_bytes,sr = librosa.load(tmp_path, sr=SR, mono=True)
                finally:
                    os.remove(tmp_path)
                duration = len(wav_bytes) / sr

                # Re-encode the decoded PCM to a clean WAV for playback, instead of
                # replaying the browser's raw recording blob directly: st.audio_input's
                # webm/ogg output has known duration/speed metadata bugs on Firefox
                # (streamlit/streamlit#9799) that make correct audio sound sped up.
                # subtype="PCM_16" matters: librosa returns float32 samples, and
                # soundfile defaults to writing 32-bit FLOAT wav for float input --
                # a format many mobile browsers (esp. Safari/iOS) only partially
                # support, playing back a few seconds and then going silent.
                # 16-bit PCM WAV is universally supported everywhere.
                pcm16 = np.clip(wav_bytes, -1.0, 1.0)
                clean_wav_buf = io.BytesIO()
                sf.write(clean_wav_buf, pcm16, sr, format="WAV", subtype="PCM_16")
                st.audio(clean_wav_buf.getvalue(), format="audio/wav")
                st.caption(f"🎧 Playing back the {duration:.1f}s recording as analyzed by the model "
                           "(re-encoded server-side to avoid a browser playback-speed bug).")

                if duration < 1.0:
                    st.warning(f"Recording is only {duration:.1f}s long. Please provide at least "
                               "1–2 seconds of sustained vowel sound and try again.")
                    st.stop()
                if np.abs(wav_bytes).max() < 1e-4:
                    st.warning("This recording appears to be silent or the volume is too low. "
                               "Please check your microphone and try again.")
                    st.stop()
                results = _do_extract(wav_bytes, sr)
        except Exception as e:
            st.error(f"Could not process this audio file ({type(e).__name__}). "
                     "Please make sure it's a valid, uncorrupted audio recording and try again.")
            st.stop()

        prob      = results["prob"]
        threshold = pl["threshold"]
        shap_v    = results["shap_vals"]
        feat_n    = pl["feature_names"]

        # ── Screening-estimate banner (not a diagnosis) ─────────────────────
        st.markdown("---")
        is_pd  = prob >= threshold

        if is_pd:
            st.markdown(f"""<div class='pred-pd'>
            <h2 style='color:#FF4444;margin:0'>⚠️ ELEVATED PD SCREENING ESTIMATE — SUGGEST CLINICAL FOLLOW-UP</h2>
            <p style='color:#FFAAAA;margin:8px 0 0'>This is a research screening estimate, not a diagnosis. Voice features resemble the pattern seen in the Parkinson's disease group of this study's training cohort.<br>
            Uncalibrated model score: <b>{prob*100:.1f}</b> &nbsp;|&nbsp; Decision threshold: {threshold*100:.0f} &nbsp;|&nbsp; This score has not been validated as a calibrated probability — see the paper's Section 3.6.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='pred-hc'>
            <h2 style='color:#44FF88;margin:0'>✅ LOW PD SCREENING ESTIMATE</h2>
            <p style='color:#AAFFCC;margin:8px 0 0'>This is a research screening estimate, not a diagnosis. Voice features resemble the pattern seen in the healthy-control group of this study's training cohort.<br>
            Uncalibrated model score: <b>{prob*100:.1f}</b> &nbsp;|&nbsp; Decision threshold: {threshold*100:.0f} &nbsp;|&nbsp; This score has not been validated as a calibrated probability — see the paper's Section 3.6.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gauge + Key metrics ────────────────────────────────────────────
        col_g, col_m = st.columns([1,2])
        with col_g:
            st.plotly_chart(gauge_chart(prob, threshold), use_container_width=True)
        with col_m:
            st.markdown("#### Key Voice Biomarkers")
            rr  = results["radar_raw"]
            ts  = pl["tab_stats"]
            rf  = pl["radar_features"]

            cols_m = st.columns(3)
            display_feats = rf[:6]
            for i,feat in enumerate(display_feats):
                meta   = FEATURE_META.get(feat,(feat,"",""))
                val    = rr.get(feat,0.)
                hc_m   = ts[feat]["HC_mean"]
                pd_m   = ts[feat]["PD_mean"]
                # Determine direction: is higher val more PD or less PD?
                higher_is_pd = pd_m > hc_m
                if higher_is_pd: abnormal = val > hc_m + ts[feat]["HC_std"]
                else:            abnormal = val < hc_m - ts[feat]["HC_std"]
                arrow = "▲" if (higher_is_pd and val>hc_m) else ("▼" if (not higher_is_pd and val<hc_m) else "–")
                delta_class = "feat-delta-up" if abnormal else "feat-delta-down"
                with cols_m[i%3]:
                    st.markdown(f"""<div class='metric-card'>
                    <div class='feat-label'>{meta[0]}</div>
                    <div class='feat-value'>{val:.4f}</div>
                    <div class='feat-unit'>{meta[1]}</div>
                    <div class='{delta_class}'>{arrow} HC avg: {hc_m:.4f} | PD avg: {pd_m:.4f}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── SHAP ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Why This Prediction? (SHAP Explanation)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class='card'>
        Base rate: mean uncalibrated model score across all training patients = <b>{td['expected_value']*100:.1f}</b><br>
        This recording's uncalibrated model score: <b>{prob*100:.1f}</b> (not a calibrated probability — see paper Section 3.6)<br>
        The chart below shows which voice features moved this prediction up (🔴 toward PD) or down (🟢 toward HC).
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(shap_chart(shap_v, feat_n, n=15), use_container_width=True)

        # ── Radar chart ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📡 Voice Feature Profile vs Training Set</div>',
                    unsafe_allow_html=True)
        col_r, col_t = st.columns([3,2])
        with col_r:
            st.plotly_chart(radar_chart(rr, ts, rf), use_container_width=True)
        with col_t:
            st.markdown("#### Feature Reference Table")
            rows=[]
            for feat in rf:
                meta=FEATURE_META.get(feat,(feat,"",""))
                val =rr.get(feat,0.)
                hcm =ts[feat]["HC_mean"]; hcs=ts[feat]["HC_std"]
                pdm =ts[feat]["PD_mean"]; pds=ts[feat]["PD_std"]
                z   =(val-hcm)/(hcs+1e-8)
                rows.append({"Feature":meta[0],"Patient":f"{val:.4f}",
                             "HC Mean":f"{hcm:.4f}","PD Mean":f"{pdm:.4f}",
                             "Z-score vs HC":f"{z:+.2f}","Note":meta[2]})
            df_ref=pd.DataFrame(rows)
            st.dataframe(df_ref,use_container_width=True,hide_index=True,height=340)

        # ── Spectrograms ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🎵 Spectrogram Analysis</div>', unsafe_allow_html=True)
        st.markdown("The CNN model extracts patterns from these three time-frequency representations.")
        sc1,sc2,sc3=st.columns(3)
        with sc1: st.image(spectrogram_fig(results["mel_img"],"Mel Spectrogram","magma"),use_container_width=True)
        with sc2: st.image(spectrogram_fig(results["chro_img"],"Chroma Features","plasma"),use_container_width=True)
        with sc3: st.image(spectrogram_fig(results["mfcc_img"],"MFCC Coefficients","viridis"),use_container_width=True)
        with st.expander("What do these spectrograms show?"):
            st.markdown("""
| Spectrogram | What it shows | PD Pattern |
|---|---|---|
| **Mel Spectrogram** | Energy across frequency over time | Irregular tremor bands at 4–7 Hz |
| **Chroma Features** | Harmonic content (pitch classes) | Reduced harmonic richness |
| **MFCC** | Vocal tract shape (timbre) | More variable coefficients |
""")

        # ── Research notes (not a clinical interpretation) ──────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Research Notes</div>', unsafe_allow_html=True)
        if is_pd:
            st.warning(f"""**Research pipeline output: features resemble this study's PD training group (uncalibrated score {prob*100:.1f})**

This is a pattern-matching result from a single-cohort, unvalidated research pipeline (see the accompanying paper). It is **not** a clinical finding: it does not establish tremor severity, motor impairment, or phonation abnormality, and no individual feature listed here has been validated as a clinical marker.

This tool does not replace clinical diagnosis, DaTscan, or examination by a clinician, and its score has not been shown to be calibrated or externally validated.""")
        else:
            st.success(f"""**Research pipeline output: features resemble this study's healthy-control training group (uncalibrated score {(1-prob)*100:.1f})**

This is a pattern-matching result from a single-cohort, unvalidated research pipeline (see the accompanying paper). It is **not** a clinical finding, and a low score here does not rule out Parkinson's disease, especially early or atypical presentations, which this single-cohort pipeline was never validated to detect.

This tool does not replace clinical diagnosis, DaTscan, or examination by a clinician.""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance Dashboard")

    # ── Top 3 comparison table ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Top 3 Model Comparison (5-Fold Cross-Validation)</div>',
                unsafe_allow_html=True)
    model_rows=[]
    for cfg_name,meta in MODEL_LABELS.items():
        s=cv_stats.get(cfg_name,{});
        if not s: continue
        model_rows.append({
            "Rank":f"#{meta['rank']}","Model":meta["label"],
            "CV AUC":f"{s['mean_auc']:.4f} ± {s['std_auc']:.3f}",
            "Acc @0.5":f"{s['mean_acc']*100:.1f}%",
            "Acc @Opt":f"{s['opt_acc']*100:.1f}%",
            "Opt Threshold":f"{s['threshold']:.3f}",
            "C":f"{s['C']:.4f}",
            "SMOTE":"Yes" if s.get("use_smote") else "No",
        })
    df_m=pd.DataFrame(model_rows)
    st.dataframe(df_m,use_container_width=True,hide_index=True)

    # ── Visual metrics ─────────────────────────────────────────────────────
    m1,m2,m3=st.columns(3)
    best_s=cv_stats.get("Model C: + Optuna + Optimal Thresh",{})
    m1.metric("Best AUC",f"{best_s.get('mean_auc',0):.4f}","+15% over random chance")
    m2.metric("Best CV Accuracy",f"{best_s.get('opt_acc',0)*100:.1f}%","with optimal threshold")
    m3.metric("Training Patients","77","39 HC + 38 PD")

    st.markdown("---")

    # ── ROC curves + Confusion matrix ──────────────────────────────────────
    col_roc, col_cm = st.columns([3,2])
    with col_roc:
        st.plotly_chart(roc_figure(cv_stats), use_container_width=True)
    with col_cm:
        best_cm = cv_stats.get("Model C: + Optuna + Optimal Thresh",{})
        if "confusion_matrix" in best_cm:
            st.plotly_chart(
                confusion_fig(best_cm["confusion_matrix"],"Model C: + Optuna + Optimal Thresh"),
                use_container_width=True)
        # Sensitivity / Specificity
        if "confusion_matrix" in best_cm:
            cm=np.array(best_cm["confusion_matrix"])
            tn,fp,fn,tp=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
            sens=tp/(tp+fn+1e-8); spec=tn/(tn+fp+1e-8)
            ppv =tp/(tp+fp+1e-8); npv=tn/(tn+fn+1e-8)
            st.markdown(f"""<div class='card'>
            <b>Nested CV Clinical Metrics</b><br><br>
            Sensitivity (TPR): <b>{sens*100:.1f}%</b><br>
            Specificity (TNR): <b>{spec*100:.1f}%</b><br>
            Positive Predictive Value: <b>{ppv*100:.1f}%</b><br>
            Negative Predictive Value: <b>{npv*100:.1f}%</b><br><br>
            <span style='color:#8B949E;font-size:12px'>Evaluated on 5-fold CV pooled predictions</span>
            </div>""", unsafe_allow_html=True)

    # ── Global feature importance ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Global Feature Importance (Illustrative Configuration)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(global_importance_fig(pl["lr_model"],pl["feature_names"]),use_container_width=True)
    st.caption("Positive coefficients push prediction toward PD, negative toward HC. "
               "Coefficients shown after internal standardization.")

    # ── SHAP on training set ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 SHAP — Training Set Overview (77 Patients)</div>',
                unsafe_allow_html=True)
    shap_all = td["shap_values"]   # (77, n_features)
    shap_mean= np.mean(np.abs(shap_all), axis=0)
    order    = np.argsort(shap_mean)[::-1][:20]
    feat_n   = pl["feature_names"]
    clean_n  = [feat_n[i].replace("FR_","").replace("_mean","(μ)").replace("_std","(σ)").replace("_"," ")
                for i in order]
    fig_sv = go.Figure(go.Bar(
        x=shap_mean[order][::-1], y=clean_n[::-1], orientation="h",
        marker_color="#58A6FF",
        text=[f"{v:.4f}" for v in shap_mean[order][::-1]],
        textposition="outside", textfont={"color":"#E6EDF3","size":10},
    ))
    fig_sv.update_layout(
        title={"text":"Mean |SHAP| per Feature (all 77 patients)","font":{"color":"#58A6FF","size":14}},
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title":"Mean |SHAP Value|","color":"#8B949E","gridcolor":"#21262D"},
        yaxis={"color":"#C9D1D9","gridcolor":"#21262D"},
        height=520,margin=dict(l=200,r=80,t=50,b=30),
    )
    st.plotly_chart(fig_sv, use_container_width=True)

    # ── Probability distribution ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 PD Probability Distribution — Training Patients</div>',
                unsafe_allow_html=True)
    probs_hc  = td["train_probs"][td["y"]==0]
    probs_pd  = td["train_probs"][td["y"]==1]
    fig_dist  = go.Figure()
    fig_dist.add_trace(go.Violin(y=probs_hc,name="Healthy Controls",side="negative",
                                  line_color="#44FF88",fillcolor="rgba(68,255,136,0.3)",
                                  meanline_visible=True))
    fig_dist.add_trace(go.Violin(y=probs_pd,name="Parkinson's Disease",side="positive",
                                  line_color="#FF4444",fillcolor="rgba(255,68,68,0.3)",
                                  meanline_visible=True))
    fig_dist.add_hline(y=pl["threshold"],line_color="#FFAA44",line_width=2,line_dash="dash",
                        annotation_text=f"Decision Threshold ({pl['threshold']:.3f})",
                        annotation_font_color="#FFAA44")
    fig_dist.update_layout(
        title={"text":"PD Probability by Group","font":{"color":"#58A6FF","size":14}},
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"title":"PD Probability","color":"#8B949E","gridcolor":"#21262D"},
        xaxis={"color":"#8B949E"},
        legend=dict(font=dict(color="#C9D1D9"),bgcolor="rgba(0,0,0,0)"),
        height=380,margin=dict(l=60,r=30,t=50,b=30),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Overlap zone: 32/38 PD patients have features within HC range — the fundamental challenge of this dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This System")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='card'>
        <h3 style='color:#58A6FF'>How It Works</h3>
        <p>This system analyzes a sustained 'Aaah' vowel recording to detect Parkinson's Disease
        acoustic biomarkers using a multi-stage AI pipeline:</p>
        <ol>
        <li><b>Spectrogram extraction</b>: Mel, Chroma, and MFCC time-frequency images</li>
        <li><b>CNN feature extraction</b>: MobileNetV2 extracts 1280-dim visual patterns from each spectrogram</li>
        <li><b>PCA compression</b>: Reduced to 35 compact CNN features</li>
        <li><b>Frame-level analysis</b>: 473 per-frame acoustic statistics → top 40 selected by Cohen's d</li>
        <li><b>Voice biomarkers</b>: TKEO energy modulation, GNE glottal noise, energy decay (12 features)</li>
        <li><b>Praat/Dynamics</b>: Jitter, shimmer, HNR, RPDE, DFA, PPE (8 features)</li>
        <li><b>Tabular features</b>: Dip patterns, micro-breaks, spectral features (26 total)</li>
        <li><b>Logistic Regression</b>: Calibrated LR classifier with optimal threshold</li>
        </ol>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
        <h3 style='color:#58A6FF'>Key Voice Biomarkers</h3>
        <table style='width:100%;border-collapse:collapse;color:#C9D1D9;font-size:13px'>
        <tr style='border-bottom:1px solid #30363D'><th>Biomarker</th><th>Description</th><th>PD Pattern</th></tr>
        <tr><td>Tremor (4–7 Hz)</td><td>Voice energy oscillation</td><td>Elevated</td></tr>
        <tr><td>Jitter</td><td>Cycle-to-cycle pitch variation</td><td>Higher</td></tr>
        <tr><td>Shimmer</td><td>Amplitude variation between cycles</td><td>Higher</td></tr>
        <tr><td>HNR</td><td>Harmonic-to-noise ratio</td><td>Lower</td></tr>
        <tr><td>TKEO</td><td>Teager-Kaiser energy operator</td><td>More variable</td></tr>
        <tr><td>GNE proxy</td><td>Glottal noise estimate</td><td>Lower clarity</td></tr>
        <tr><td>Voice dips</td><td>Breaks/drops in vocal energy</td><td>More frequent, longer</td></tr>
        <tr><td>RPDE</td><td>Recurrence period density entropy</td><td>Higher</td></tr>
        <tr><td>DFA</td><td>Detrended fluctuation analysis</td><td>Different scaling</td></tr>
        </table>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class='card'>
        <h3 style='color:#58A6FF'>Dataset & Performance</h3>
        <table style='width:100%;border-collapse:collapse;color:#C9D1D9;font-size:13px'>
        <tr style='border-bottom:1px solid #30363D'><th>Property</th><th>Value</th></tr>
        <tr><td>Total patients</td><td>77 (39 HC, 38 PD)</td></tr>
        <tr><td>Task</td><td>Sustained 'Aaah' vowel</td></tr>
        <tr><td>Sample rate</td><td>22,050 Hz</td></tr>
        <tr><td>Evaluation</td><td>5-Fold Stratified CV</td></tr>
        <tr><td>Best AUC</td><td>0.8906</td></tr>
        <tr><td>Best Accuracy</td><td>80.5% (optimal threshold)</td></tr>
        <tr><td>Data ceiling</td><td>~82% (32/38 PD overlap HC)</td></tr>
        </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
        <h3 style='color:#58A6FF'>Model Architecture</h3>
        <b>Feature vector: 113 dimensions</b><br><br>
        • CNN Mel PCA: 15 dims<br>
        • CNN Chroma PCA: 10 dims<br>
        • CNN MFCC PCA: 10 dims<br>
        • Tabular (dip + spectral + praat + dynamics + interactions): 26 dims<br>
        • Frame features (top 40 by Cohen's d): 40 dims<br>
        • New audio (TKEO + GNE + energy): 12 dims<br><br>
        <b>Training:</b> BorderlineSMOTE → Optuna-tuned LR (C=0.024)<br>
        <b>Threshold:</b> 0.274 (Youden J optimal)<br>
        <b>Augmentation:</b> 4× per-patient (time-stretch × 2 + noise)
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
        <h3 style='color:#FFAA44'>⚠️ Important Limitations</h3>
        <ul style='color:#C9D1D9'>
        <li>Small dataset (n=77) — results may not generalise</li>
        <li>Sustained vowel only — running speech captures more PD-related deficits</li>
        <li>84% of PD patients have features overlapping with healthy controls</li>
        <li>Model is most sensitive for moderate-to-advanced PD</li>
        <li>Not validated for clinical use — research tool only</li>
        <li>Recording quality significantly affects results</li>
        </ul>
        </div>""", unsafe_allow_html=True)
