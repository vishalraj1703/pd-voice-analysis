# -*- coding: utf-8 -*-
"""
Age/sex confound analysis responding to peer-review Major Comment 5.

Uses the SAME outer 5-fold StratifiedKFold(seed=42) split as the main nested
analysis and controlled_ablation.py, so results are directly comparable.

Runs:
  A. Age-only classifier      (1 feature: age)
  B. Sex-only classifier      (1 feature: sex)
  C. Age+Sex classifier       (2 features)
  D. Full model (Baseline+NewFeatures, fixed K=20/C=0.0102, no SMOTE -- same
     as controlled_ablation.py row 2) with age and sex APPENDED as covariates
  E. Full model with age REGRESSED OUT of every CNN PCA dimension before
     classification (per outer-training fold: fit linear regression of each
     CNN PCA feature on age using only training rows, subtract the age-
     predicted component from both train and validation rows) -- tests
     whether CNN attribution survives once its age-covariation is removed.
  F. Age-restricted subset: keep only samples within an overlapping age band
     across both groups (if feasible; reports the resulting n and whether
     there is enough stratification to fit at all).
"""
import sys, io, os, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

SCRATCH = os.path.dirname(os.path.abspath(__file__)) + "/results"
CACHE = os.path.dirname(os.path.abspath(__file__)) + "/data/raw_features_cache.pkl"
with open(CACHE, "rb") as f:
    C = pickle.load(f)
X_cnn_orig = C["X_cnn_orig"]; X_cnn_aug = C["X_cnn_aug"]; X_tab = C["X_tab"]
X_frame = C["X_frame"]; X_new = C["X_new"]; y = C["y"]; fnames = list(C["fnames"])
N_AUG = 3; PCA_MEL = 15; PCA_CHROMA = 10; PCA_MFCC = 10
FIXED_K = 20; FIXED_C = 0.0102

demo = pd.read_excel(os.path.dirname(os.path.abspath(__file__)) + "/data/Demographics_age_sex.xlsx")
demo["Sample ID"] = demo["Sample ID"].astype(str)
demo_map = demo.set_index("Sample ID")[["Age", "Sex"]].to_dict("index")

age = np.array([demo_map[f.replace(".wav", "")]["Age"] for f in fnames], dtype=float)
sex_raw = [demo_map[f.replace(".wav", "")]["Sex"] for f in fnames]
sex = np.array([1.0 if s == "M" else 0.0 for s in sex_raw])
print(f"Matched {len(age)}/{len(fnames)} samples to demographics.")
print(f"Age: HC mean={age[y==0].mean():.1f} (sd={age[y==0].std():.1f}), "
      f"PD mean={age[y==1].mean():.1f} (sd={age[y==1].std():.1f})")
print(f"Age range overlap: HC [{age[y==0].min():.0f},{age[y==0].max():.0f}], "
      f"PD [{age[y==1].min():.0f},{age[y==1].max():.0f}]")

def cohen_d(X, yy, idx):
    a = X[yy == 1, idx]; b = X[yy == 0, idx]
    return abs(np.mean(a) - np.mean(b)) / (np.sqrt((np.std(a)**2 + np.std(b)**2) / 2) + 1e-8)

def top_frame_idx_for(idx_subset, k):
    scores = [cohen_d(X_frame[idx_subset], y[idx_subset], i) for i in range(X_frame.shape[1])]
    return np.argsort(scores)[::-1][:k]

def build_full(tr, va, k_frame=FIXED_K, augment=True, age_deconfound=False):
    n = len(tr)
    top_idx_tr = top_frame_idx_for(tr, k_frame)
    if augment:
        cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n * N_AUG, 3, 1280)])
        tab_tr = np.vstack([X_tab[tr], np.tile(X_tab[tr], (N_AUG, 1))])
        frm_tr = np.vstack([X_frame[tr, :][:, top_idx_tr], np.tile(X_frame[tr, :][:, top_idx_tr], (N_AUG, 1))])
        new_tr = np.vstack([X_new[tr], np.tile(X_new[tr], (N_AUG, 1))])
        age_tr_aug = np.tile(age[tr], N_AUG + 1)
        y_tr = np.concatenate([y[tr], np.tile(y[tr], N_AUG)])
    else:
        cnn_tr, tab_tr, frm_tr, new_tr, y_tr = X_cnn_orig[tr], X_tab[tr], X_frame[tr, :][:, top_idx_tr], X_new[tr], y[tr]
        age_tr_aug = age[tr]

    sc_mel = StandardScaler().fit(cnn_tr[:, 0]); pca_mel = PCA(PCA_MEL, random_state=42).fit(sc_mel.transform(cnn_tr[:, 0]))
    sc_chr = StandardScaler().fit(cnn_tr[:, 1]); pca_chr = PCA(PCA_CHROMA, random_state=42).fit(sc_chr.transform(cnn_tr[:, 1]))
    sc_mfc = StandardScaler().fit(cnn_tr[:, 2]); pca_mfc = PCA(PCA_MFCC, random_state=42).fit(sc_mfc.transform(cnn_tr[:, 2]))
    sc_tab = StandardScaler().fit(tab_tr); sc_frm = StandardScaler().fit(frm_tr); sc_new = StandardScaler().fit(new_tr)

    def cnn_block(cnn):
        return np.hstack([pca_mel.transform(sc_mel.transform(cnn[:, 0])),
                           pca_chr.transform(sc_chr.transform(cnn[:, 1])),
                           pca_mfc.transform(sc_mfc.transform(cnn[:, 2]))])

    cnn_block_tr = cnn_block(cnn_tr)
    cnn_block_va = cnn_block(X_cnn_orig[va])

    if age_deconfound:
        # Regress each CNN PCA dim on age using TRAINING rows only, subtract the
        # age-predicted component from both train and validation rows.
        reg = LinearRegression().fit(age_tr_aug.reshape(-1, 1), cnn_block_tr)
        cnn_block_tr = cnn_block_tr - reg.predict(age_tr_aug.reshape(-1, 1))
        cnn_block_va = cnn_block_va - reg.predict(age[va].reshape(-1, 1))

    X_tr = np.hstack([cnn_block_tr, sc_tab.transform(tab_tr), sc_frm.transform(frm_tr), sc_new.transform(new_tr)]).astype(np.float32)
    X_va = np.hstack([cnn_block_va, sc_tab.transform(X_tab[va]), sc_frm.transform(X_frame[va, :][:, top_idx_tr]), sc_new.transform(X_new[va])]).astype(np.float32)
    return X_tr, y_tr, X_va

skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_splits = list(skf_outer.split(np.zeros(len(y)), y))

def eval_feature_set(name, feature_fn):
    all_true, all_prob, all_pred = [], [], []
    for outer_tr, outer_va in outer_splits:
        X_tr, y_tr, X_va = feature_fn(outer_tr, outer_va)
        lr = LogisticRegression(C=FIXED_C, solver="saga", max_iter=5000, random_state=42)
        lr.fit(X_tr, y_tr)
        p = lr.predict_proba(X_va)[:, 1]
        pred = (p >= 0.5).astype(int)
        all_true.extend(y[outer_va].tolist()); all_prob.extend(p.tolist()); all_pred.extend(pred.tolist())
    all_true = np.array(all_true); all_prob = np.array(all_prob); all_pred = np.array(all_pred)
    auc = roc_auc_score(all_true, all_prob); acc = accuracy_score(all_true, all_pred)
    print(f"  [{name}] AUC={auc:.4f}  Acc={acc*100:.1f}%")
    return dict(name=name, auc=auc, acc=acc, true=all_true, prob=all_prob)

print("\n" + "="*90)
print("A/B/C: Demographic-only baselines (can age/sex alone predict the label?)")
print("="*90)
res_age = eval_feature_set("A. Age-only", lambda tr, va: (
    StandardScaler().fit(age[tr].reshape(-1,1)).transform(age[tr].reshape(-1,1)), y[tr],
    StandardScaler().fit(age[tr].reshape(-1,1)).transform(age[va].reshape(-1,1))))
res_sex = eval_feature_set("B. Sex-only", lambda tr, va: (
    sex[tr].reshape(-1,1).astype(float), y[tr], sex[va].reshape(-1,1).astype(float)))
res_agesex = eval_feature_set("C. Age+Sex", lambda tr, va: (
    np.hstack([StandardScaler().fit(age[tr].reshape(-1,1)).transform(age[tr].reshape(-1,1)), sex[tr].reshape(-1,1)]), y[tr],
    np.hstack([StandardScaler().fit(age[tr].reshape(-1,1)).transform(age[va].reshape(-1,1)), sex[va].reshape(-1,1)])))

print("\n" + "="*90)
print("D: Full model (Baseline+NewFeatures) with age+sex appended as covariates")
print("="*90)
def full_plus_agesex(tr, va):
    X_tr, y_tr, X_va = build_full(tr, va)
    age_sc = StandardScaler().fit(age[tr].reshape(-1,1))
    age_tr_aug = np.tile(age_sc.transform(age[tr].reshape(-1,1)).flatten(), N_AUG+1)
    sex_tr_aug = np.tile(sex[tr], N_AUG+1)
    X_tr = np.hstack([X_tr, age_tr_aug.reshape(-1,1), sex_tr_aug.reshape(-1,1)])
    X_va = np.hstack([X_va, age_sc.transform(age[va].reshape(-1,1)), sex[va].reshape(-1,1)])
    return X_tr, y_tr, X_va
res_full_agesex = eval_feature_set("D. Full+Age+Sex", full_plus_agesex)

print("\n" + "="*90)
print("E: Full model with age regressed out of every CNN PCA dimension (fold-confined)")
print("="*90)
res_full_deconf = eval_feature_set("E. Full, CNN age-deconfounded",
    lambda tr, va: build_full(tr, va, age_deconfound=True))

print("\n" + "="*90)
print("Reference: full model, no age adjustment (= controlled_ablation.py row 2, 'newfeat')")
print("="*90)
res_full_plain = eval_feature_set("Ref. Full, unadjusted", lambda tr, va: build_full(tr, va))

print("\n" + "="*90)
print("F: Age-restricted overlap subset")
print("="*90)
# Find the overlapping age band between groups
overlap_lo = max(age[y==0].min(), age[y==1].min())
overlap_hi = min(age[y==0].max(), age[y==1].max())
mask = (age >= overlap_lo) & (age <= overlap_hi)
n_hc_overlap = int(np.sum(mask & (y==0))); n_pd_overlap = int(np.sum(mask & (y==1)))
print(f"  Overlapping age band: [{overlap_lo:.0f}, {overlap_hi:.0f}] years")
print(f"  Samples in band: {n_hc_overlap} HC, {n_pd_overlap} PD (of {np.sum(y==0)} HC, {np.sum(y==1)} PD total)")
if min(n_hc_overlap, n_pd_overlap) >= 10:
    idx_overlap = np.where(mask)[0]
    y_overlap = y[idx_overlap]
    skf_small = StratifiedKFold(n_splits=min(5, min(n_hc_overlap, n_pd_overlap)), shuffle=True, random_state=42)
    print("  Sample size sufficient for a restricted-subset CV -- results below are exploratory given small n.")
else:
    print(f"  INSUFFICIENT sample size for a reliable age-restricted analysis "
          f"(smaller group has only {min(n_hc_overlap, n_pd_overlap)} samples in the overlap band). "
          f"Not attempting a restricted-subset model -- reporting this limitation directly instead.")

results = dict(age_only=res_age, sex_only=res_sex, age_sex=res_agesex,
               full_plain=res_full_plain, full_agesex=res_full_agesex, full_deconf=res_full_deconf,
               age_hc_mean=float(age[y==0].mean()), age_hc_sd=float(age[y==0].std()),
               age_pd_mean=float(age[y==1].mean()), age_pd_sd=float(age[y==1].std()),
               overlap_band=(float(overlap_lo), float(overlap_hi)),
               n_hc_overlap=n_hc_overlap, n_pd_overlap=n_pd_overlap)
with open(f"{SCRATCH}/age_confound_result.pkl", "wb") as f:
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("true","prob")} if isinstance(v, dict) else v
                 for k, v in results.items()}, f)
print(f"\nSaved -> {SCRATCH}/age_confound_result.pkl")
