# -*- coding: utf-8 -*-
"""
Fold-wise SHAP recomputation responding to peer-review Major Comment 8.

For each of the 5 outer folds (same split, seed=42, as nested_cv_final.py):
  - Rebuild features using that fold's own selected K (from nested_cv_result.pkl)
  - Refit logistic regression with that fold's own selected C, with SMOTE applied
    to the outer-training data only (matching nested_cv_final.py exactly)
  - Use shap.LinearExplainer with the outer TRAINING fold as background
  - Explain only the held-out OUTER TEST fold samples (never the training data)

Attribution is aggregated at the feature-GROUP level (CNN / clinical-acoustic /
frame-level / engineered-acoustic), not individual PCA components, and rank
stability across the 5 folds is reported -- directly answering the reviewer's
request rather than presenting one all-data refit as if it were the nested
estimate's explanation.
"""
import sys, io, os, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE

SCRATCH = os.path.dirname(os.path.abspath(__file__)) + "/results"
with open(os.path.dirname(os.path.abspath(__file__)) + "/data/raw_features_cache.pkl", "rb") as f:
    C = pickle.load(f)
with open(f"{SCRATCH}/nested_cv_result.pkl", "rb") as f:
    NR = pickle.load(f)

X_cnn_orig=C["X_cnn_orig"]; X_cnn_aug=C["X_cnn_aug"]; X_tab=C["X_tab"]
X_frame=C["X_frame"]; X_new=C["X_new"]; y=C["y"]
N_AUG=3; PCA_MEL=15; PCA_CHROMA=10; PCA_MFCC=10
fold_summaries = NR["fold_summaries"]

def cohen_d(X, yy, idx):
    a=X[yy==1,idx]; b=X[yy==0,idx]
    return abs(np.mean(a)-np.mean(b))/(np.sqrt((np.std(a)**2+np.std(b)**2)/2)+1e-8)

def top_frame_idx_for(idx_subset, k):
    scores=[cohen_d(X_frame[idx_subset], y[idx_subset], i) for i in range(X_frame.shape[1])]
    return np.argsort(scores)[::-1][:k]

def build(tr, va, k_frame):
    n=len(tr)
    top_idx_tr = top_frame_idx_for(tr, k_frame)
    cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n*N_AUG,3,1280)])
    tab_tr = np.vstack([X_tab[tr], np.tile(X_tab[tr],(N_AUG,1))])
    frm_tr = np.vstack([X_frame[tr,:][:,top_idx_tr], np.tile(X_frame[tr,:][:,top_idx_tr],(N_AUG,1))])
    new_tr = np.vstack([X_new[tr], np.tile(X_new[tr],(N_AUG,1))])
    y_tr = np.concatenate([y[tr], np.tile(y[tr],N_AUG)])
    sc_mel=StandardScaler().fit(cnn_tr[:,0]); pca_mel=PCA(PCA_MEL,random_state=42).fit(sc_mel.transform(cnn_tr[:,0]))
    sc_chr=StandardScaler().fit(cnn_tr[:,1]); pca_chr=PCA(PCA_CHROMA,random_state=42).fit(sc_chr.transform(cnn_tr[:,1]))
    sc_mfc=StandardScaler().fit(cnn_tr[:,2]); pca_mfc=PCA(PCA_MFCC,random_state=42).fit(sc_mfc.transform(cnn_tr[:,2]))
    sc_tab=StandardScaler().fit(tab_tr); sc_frm=StandardScaler().fit(frm_tr); sc_new=StandardScaler().fit(new_tr)
    def fuse(cnn,tab,frm,new):
        return np.hstack([pca_mel.transform(sc_mel.transform(cnn[:,0])), pca_chr.transform(sc_chr.transform(cnn[:,1])),
                           pca_mfc.transform(sc_mfc.transform(cnn[:,2])), sc_tab.transform(tab), sc_frm.transform(frm), sc_new.transform(new)]).astype(np.float32)
    X_tr = fuse(cnn_tr,tab_tr,frm_tr,new_tr)
    X_va = fuse(X_cnn_orig[va], X_tab[va], X_frame[va,:][:,top_idx_tr], X_new[va])
    # group boundaries for THIS fold's feature vector
    n_cnn = PCA_MEL+PCA_CHROMA+PCA_MFCC
    n_tab = X_tab.shape[1]
    n_frm = k_frame
    n_new = X_new.shape[1]
    bounds = dict(cnn=(0,n_cnn), tab=(n_cnn,n_cnn+n_tab), frm=(n_cnn+n_tab,n_cnn+n_tab+n_frm),
                  new=(n_cnn+n_tab+n_frm, n_cnn+n_tab+n_frm+n_new))
    return X_tr, y_tr, X_va, bounds

skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_splits = list(skf_outer.split(np.zeros(len(y)), y))

group_names = ["CNN (Mel+Chroma+MFCC PCA)", "Clinical-acoustic (tabular)", "Frame-level (Cohen's d top-K)", "Engineered acoustic (novel)"]
group_keys = ["cnn", "tab", "frm", "new"]
per_fold_group_importance = []  # list of dicts: fold -> {group: mean|SHAP|}

for fi, (outer_tr, outer_va) in enumerate(outer_splits):
    fs = fold_summaries[fi]
    k_sel, c_sel = fs["k"], fs["C"]
    X_tr, y_tr, X_va, bounds = build(outer_tr, outer_va, k_sel)

    # background for SHAP = outer TRAINING fold, BEFORE SMOTE (real samples only)
    background = X_tr.copy()

    sm = BorderlineSMOTE(k_neighbors=5, random_state=42)
    try:
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    except Exception:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    lr = LogisticRegression(C=c_sel, solver="saga", max_iter=5000, random_state=42)
    lr.fit(X_tr_sm, y_tr_sm)

    explainer = shap.LinearExplainer(lr, background)
    shap_values = explainer.shap_values(X_va)  # shape (n_outer_va, n_features)

    group_imp = {}
    for gname, gkey in zip(group_names, group_keys):
        lo, hi = bounds[gkey]
        group_imp[gkey] = float(np.mean(np.abs(shap_values[:, lo:hi])))
    per_fold_group_importance.append(group_imp)
    ranked = sorted(group_imp.items(), key=lambda kv: -kv[1])
    print(f"  Fold {fi} (K={k_sel}, C={c_sel:.5f}): group mean|SHAP| ranking: "
          + ", ".join(f"{k}={v:.4f}" for k, v in ranked))

print("\n" + "="*90)
print("RANK STABILITY ACROSS OUTER FOLDS (1 = most important that fold)")
print("="*90)
ranks = {k: [] for k in group_keys}
for gi in per_fold_group_importance:
    order = sorted(gi.items(), key=lambda kv: -kv[1])
    for rank, (k, _) in enumerate(order, start=1):
        ranks[k].append(rank)
for gname, gkey in zip(group_names, group_keys):
    r = ranks[gkey]
    vals = [gi[gkey] for gi in per_fold_group_importance]
    print(f"  {gname:32s}  ranks across folds={r}  mean|SHAP| mean={np.mean(vals):.4f} sd={np.std(vals):.4f} "
          f"(CV={np.std(vals)/ (np.mean(vals)+1e-9):.2f})")

with open(f"{SCRATCH}/foldwise_shap_result.pkl", "wb") as f:
    pickle.dump(dict(per_fold_group_importance=per_fold_group_importance, ranks=ranks,
                      group_names=group_names, group_keys=group_keys), f)
print(f"\nSaved -> {SCRATCH}/foldwise_shap_result.pkl")
