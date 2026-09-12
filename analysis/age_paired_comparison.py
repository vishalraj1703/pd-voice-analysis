# -*- coding: utf-8 -*-
"""
Paired comparisons responding to peer-review Major Comment 4 (round 2):
"statistically indistinguishable" and "reduces AUC" claims need an actual
paired difference with a confidence interval, not two marginal AUCs eyeballed
against each other.

Comparison A: Full unadjusted model (Baseline+NewFeatures, fixed K=20/C=0.0102,
no SMOTE -- same as controlled_ablation.py row "newfeat") vs Age-only model.
Both evaluated on the SAME outer 5-fold split (seed=42), predictions paired
sample-for-sample, difference bootstrapped (same resampled indices applied to
both before differencing).

Comparison B: Full unadjusted model vs Full model with age regressed out of
every CNN PCA dimension (fold-confined). Same paired-bootstrap procedure.
"""
import sys, io, os, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
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

def cohen_d(X, yy, idx):
    a=X[yy==1,idx]; b=X[yy==0,idx]
    return abs(np.mean(a)-np.mean(b))/(np.sqrt((np.std(a)**2+np.std(b)**2)/2)+1e-8)
def top_frame_idx_for(idx_subset, k):
    scores=[cohen_d(X_frame[idx_subset], y[idx_subset], i) for i in range(X_frame.shape[1])]
    return np.argsort(scores)[::-1][:k]

def build_full(tr, va, age_deconfound=False):
    n=len(tr)
    top_idx_tr = top_frame_idx_for(tr, FIXED_K)
    cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n*N_AUG,3,1280)])
    tab_tr = np.vstack([X_tab[tr], np.tile(X_tab[tr],(N_AUG,1))])
    frm_tr = np.vstack([X_frame[tr,:][:,top_idx_tr], np.tile(X_frame[tr,:][:,top_idx_tr],(N_AUG,1))])
    new_tr = np.vstack([X_new[tr], np.tile(X_new[tr],(N_AUG,1))])
    age_tr_aug = np.tile(age[tr], N_AUG+1)
    y_tr = np.concatenate([y[tr], np.tile(y[tr],N_AUG)])
    sc_mel=StandardScaler().fit(cnn_tr[:,0]); pca_mel=PCA(PCA_MEL,random_state=42).fit(sc_mel.transform(cnn_tr[:,0]))
    sc_chr=StandardScaler().fit(cnn_tr[:,1]); pca_chr=PCA(PCA_CHROMA,random_state=42).fit(sc_chr.transform(cnn_tr[:,1]))
    sc_mfc=StandardScaler().fit(cnn_tr[:,2]); pca_mfc=PCA(PCA_MFCC,random_state=42).fit(sc_mfc.transform(cnn_tr[:,2]))
    sc_tab=StandardScaler().fit(tab_tr); sc_frm=StandardScaler().fit(frm_tr); sc_new=StandardScaler().fit(new_tr)
    def cnn_block(cnn):
        return np.hstack([pca_mel.transform(sc_mel.transform(cnn[:,0])), pca_chr.transform(sc_chr.transform(cnn[:,1])),
                           pca_mfc.transform(sc_mfc.transform(cnn[:,2]))])
    cb_tr = cnn_block(cnn_tr); cb_va = cnn_block(X_cnn_orig[va])
    if age_deconfound:
        reg = LinearRegression().fit(age_tr_aug.reshape(-1,1), cb_tr)
        cb_tr = cb_tr - reg.predict(age_tr_aug.reshape(-1,1))
        cb_va = cb_va - reg.predict(age[va].reshape(-1,1))
    X_tr = np.hstack([cb_tr, sc_tab.transform(tab_tr), sc_frm.transform(frm_tr), sc_new.transform(new_tr)]).astype(np.float32)
    X_va = np.hstack([cb_va, sc_tab.transform(X_tab[va]), sc_frm.transform(X_frame[va,:][:,top_idx_tr]), sc_new.transform(X_new[va])]).astype(np.float32)
    return X_tr, y_tr, X_va

skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_splits = list(skf_outer.split(np.zeros(len(y)), y))

def eval_rows():
    out = {"full": {"true":[], "prob":[], "pred":[]}, "age_only": {"true":[], "prob":[], "pred":[]},
           "full_deconf": {"true":[], "prob":[], "pred":[]}}
    for outer_tr, outer_va in outer_splits:
        # full (unadjusted)
        X_tr, y_tr, X_va = build_full(outer_tr, outer_va, age_deconfound=False)
        lr = LogisticRegression(C=FIXED_C, solver="saga", max_iter=5000, random_state=42).fit(X_tr, y_tr)
        p = lr.predict_proba(X_va)[:,1]
        out["full"]["true"].extend(y[outer_va].tolist()); out["full"]["prob"].extend(p.tolist()); out["full"]["pred"].extend((p>=0.5).astype(int).tolist())
        # age-only
        sc = StandardScaler().fit(age[outer_tr].reshape(-1,1))
        lr2 = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(age[outer_tr].reshape(-1,1)), y[outer_tr])
        p2 = lr2.predict_proba(sc.transform(age[outer_va].reshape(-1,1)))[:,1]
        out["age_only"]["true"].extend(y[outer_va].tolist()); out["age_only"]["prob"].extend(p2.tolist()); out["age_only"]["pred"].extend((p2>=0.5).astype(int).tolist())
        # full, CNN age-deconfounded
        X_tr3, y_tr3, X_va3 = build_full(outer_tr, outer_va, age_deconfound=True)
        lr3 = LogisticRegression(C=FIXED_C, solver="saga", max_iter=5000, random_state=42).fit(X_tr3, y_tr3)
        p3 = lr3.predict_proba(X_va3)[:,1]
        out["full_deconf"]["true"].extend(y[outer_va].tolist()); out["full_deconf"]["prob"].extend(p3.tolist()); out["full_deconf"]["pred"].extend((p3>=0.5).astype(int).tolist())
    for k in out:
        for kk in out[k]:
            out[k][kk] = np.array(out[k][kk])
    return out

results = eval_rows()
for name in ["full", "age_only", "full_deconf"]:
    r = results[name]
    auc = roc_auc_score(r["true"], r["prob"]); acc = accuracy_score(r["true"], r["pred"])
    print(f"[{name}] AUC={auc:.4f} Acc={acc*100:.1f}%")

# Paired bootstrap: same resampled indices applied to BOTH rows in each comparison
rng = np.random.RandomState(42)
n_boot = 2000
n = len(y)

def paired_diff(rowA, rowB, label):
    d_auc = []
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        bt = rowA["true"][idx]
        if len(set(bt.tolist())) < 2:
            continue
        auc_a = roc_auc_score(bt, rowA["prob"][idx])
        auc_b = roc_auc_score(bt, rowB["prob"][idx])
        d_auc.append(auc_a - auc_b)
    lo, hi = np.percentile(d_auc, [2.5, 97.5])
    print(f"{label}: mean paired dAUC={np.mean(d_auc):+.4f}, 95% paired bootstrap CI [{lo:+.4f}, {hi:+.4f}]"
          f"  {'(includes 0 -> no evidence of a difference)' if lo<=0<=hi else '(excludes 0)'}")
    return np.mean(d_auc), (lo, hi)

print()
paired_diff(results["full"], results["age_only"], "Full (unadjusted) minus Age-only")
paired_diff(results["full"], results["full_deconf"], "Full (unadjusted) minus Full (CNN age-deconfounded)")

with open(f"{SCRATCH}/age_paired_result.pkl", "wb") as f:
    pickle.dump(dict(full_auc=float(roc_auc_score(results['full']['true'], results['full']['prob'])),
                      age_only_auc=float(roc_auc_score(results['age_only']['true'], results['age_only']['prob'])),
                      full_deconf_auc=float(roc_auc_score(results['full_deconf']['true'], results['full_deconf']['prob']))), f)
print(f"\nSaved -> {SCRATCH}/age_paired_result.pkl")
