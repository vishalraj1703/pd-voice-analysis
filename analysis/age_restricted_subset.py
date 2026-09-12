# -*- coding: utf-8 -*-
"""Age-restricted overlap-band evaluation (follow-up to age_confound_analysis.py part F)."""
import sys, io, os, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

SCRATCH = os.path.dirname(os.path.abspath(__file__)) + "/results"
with open(os.path.dirname(os.path.abspath(__file__)) + "/data/raw_features_cache.pkl", "rb") as f:
    C = pickle.load(f)
X_cnn_orig=C["X_cnn_orig"]; X_cnn_aug=C["X_cnn_aug"]; X_tab=C["X_tab"]
X_frame=C["X_frame"]; X_new=C["X_new"]; y=C["y"]; fnames=list(C["fnames"])
N_AUG=3; PCA_MEL=15; PCA_CHROMA=10; PCA_MFCC=10; FIXED_K=20; FIXED_C=0.0102

demo = pd.read_excel(os.path.dirname(os.path.abspath(__file__)) + "/data/Demographics_age_sex.xlsx")
demo["Sample ID"] = demo["Sample ID"].astype(str)
demo_map = demo.set_index("Sample ID")[["Age","Sex"]].to_dict("index")
age = np.array([demo_map[f.replace(".wav","")]["Age"] for f in fnames], dtype=float)

overlap_lo = max(age[y==0].min(), age[y==1].min())
overlap_hi = min(age[y==0].max(), age[y==1].max())
mask = (age >= overlap_lo) & (age <= overlap_hi)
idx_overlap = np.where(mask)[0]
print(f"Age-restricted subset: [{overlap_lo:.0f},{overlap_hi:.0f}]y -> "
      f"n={len(idx_overlap)} ({np.sum(y[idx_overlap]==0)} HC, {np.sum(y[idx_overlap]==1)} PD)")
print(f"Residual age gap within band: HC mean={age[idx_overlap][y[idx_overlap]==0].mean():.1f}, "
      f"PD mean={age[idx_overlap][y[idx_overlap]==1].mean():.1f}")

def cohen_d(X, yy, idx):
    a=X[yy==1,idx]; b=X[yy==0,idx]
    return abs(np.mean(a)-np.mean(b))/(np.sqrt((np.std(a)**2+np.std(b)**2)/2)+1e-8)

def top_frame_idx_for(idx_subset, k):
    scores=[cohen_d(X_frame[idx_subset], y[idx_subset], i) for i in range(X_frame.shape[1])]
    return np.argsort(scores)[::-1][:k]

def build(tr_global, va_global, k_frame=FIXED_K):
    n=len(tr_global)
    top_idx_tr = top_frame_idx_for(tr_global, k_frame)
    cnn_tr = np.vstack([X_cnn_orig[tr_global], X_cnn_aug[tr_global].reshape(n*N_AUG,3,1280)])
    tab_tr = np.vstack([X_tab[tr_global], np.tile(X_tab[tr_global],(N_AUG,1))])
    frm_tr = np.vstack([X_frame[tr_global,:][:,top_idx_tr], np.tile(X_frame[tr_global,:][:,top_idx_tr],(N_AUG,1))])
    new_tr = np.vstack([X_new[tr_global], np.tile(X_new[tr_global],(N_AUG,1))])
    y_tr = np.concatenate([y[tr_global], np.tile(y[tr_global],N_AUG)])
    sc_mel=StandardScaler().fit(cnn_tr[:,0]); pca_mel=PCA(PCA_MEL,random_state=42).fit(sc_mel.transform(cnn_tr[:,0]))
    sc_chr=StandardScaler().fit(cnn_tr[:,1]); pca_chr=PCA(PCA_CHROMA,random_state=42).fit(sc_chr.transform(cnn_tr[:,1]))
    sc_mfc=StandardScaler().fit(cnn_tr[:,2]); pca_mfc=PCA(PCA_MFCC,random_state=42).fit(sc_mfc.transform(cnn_tr[:,2]))
    sc_tab=StandardScaler().fit(tab_tr); sc_frm=StandardScaler().fit(frm_tr); sc_new=StandardScaler().fit(new_tr)
    def fuse(cnn,tab,frm,new):
        return np.hstack([pca_mel.transform(sc_mel.transform(cnn[:,0])), pca_chr.transform(sc_chr.transform(cnn[:,1])),
                           pca_mfc.transform(sc_mfc.transform(cnn[:,2])), sc_tab.transform(tab), sc_frm.transform(frm), sc_new.transform(new)]).astype(np.float32)
    X_tr = fuse(cnn_tr,tab_tr,frm_tr,new_tr)
    X_va = fuse(X_cnn_orig[va_global], X_tab[va_global], X_frame[va_global,:][:,top_idx_tr], X_new[va_global])
    return X_tr, y_tr, X_va

y_sub = y[idx_overlap]
n_splits = min(5, np.sum(y_sub==0), np.sum(y_sub==1))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
all_true, all_prob, all_pred = [], [], []
for tr_rel, va_rel in skf.split(np.zeros(len(idx_overlap)), y_sub):
    tr_global = idx_overlap[tr_rel]; va_global = idx_overlap[va_rel]
    X_tr, y_tr, X_va = build(tr_global, va_global)
    lr = LogisticRegression(C=FIXED_C, solver="saga", max_iter=5000, random_state=42)
    lr.fit(X_tr, y_tr)
    p = lr.predict_proba(X_va)[:,1]
    all_true.extend(y[va_global].tolist()); all_prob.extend(p.tolist()); all_pred.extend((p>=0.5).astype(int).tolist())

all_true=np.array(all_true); all_prob=np.array(all_prob); all_pred=np.array(all_pred)
auc = roc_auc_score(all_true, all_prob); acc = accuracy_score(all_true, all_pred)
print(f"\n[Age-restricted subset, n={len(idx_overlap)}, {n_splits}-fold CV] AUC={auc:.4f}  Acc={acc*100:.1f}%")

# Age-only classifier WITHIN the same restricted subset, for direct comparison
all_true2, all_prob2 = [], []
for tr_rel, va_rel in skf.split(np.zeros(len(idx_overlap)), y_sub):
    tr_global = idx_overlap[tr_rel]; va_global = idx_overlap[va_rel]
    sc = StandardScaler().fit(age[tr_global].reshape(-1,1))
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(sc.transform(age[tr_global].reshape(-1,1)), y[tr_global])
    p = lr.predict_proba(sc.transform(age[va_global].reshape(-1,1)))[:,1]
    all_true2.extend(y[va_global].tolist()); all_prob2.extend(p.tolist())
auc_age_restricted = roc_auc_score(all_true2, all_prob2)
print(f"[Age-only, SAME restricted subset]                     AUC={auc_age_restricted:.4f}")

with open(f"{SCRATCH}/age_restricted_result.pkl","wb") as f:
    pickle.dump(dict(n=len(idx_overlap), n_hc=int(np.sum(y_sub==0)), n_pd=int(np.sum(y_sub==1)),
                      band=(float(overlap_lo),float(overlap_hi)), auc_full=float(auc), acc_full=float(acc),
                      auc_age_only=float(auc_age_restricted)), f)
print(f"\nSaved -> {SCRATCH}/age_restricted_result.pkl")
