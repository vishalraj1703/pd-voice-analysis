# -*- coding: utf-8 -*-
"""
Proper NESTED cross-validation, fixing the reviewer-identified leak:
previously, Optuna's C search, the frame-feature-count K sweep, and the
Youden-J threshold were all selected using the SAME outer folds that
performance was then reported on. This script fixes that: an inner CV loop
inside each outer training fold selects K, C, and the threshold; the outer
test fold is touched exactly once, after all tuning is frozen.
"""
import sys, io, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from imblearn.over_sampling import BorderlineSMOTE

CACHE = "C:/Users/VISHAL~1/AppData/Local/Temp/claude/D--parkinson-the-finale-dataset-and-audio-file/be45d466-5b03-4ef8-947b-b56f2a827e3b/scratchpad/raw_features_cache.pkl"
with open(CACHE, "rb") as f:
    C = pickle.load(f)

X_cnn_orig=C["X_cnn_orig"]; X_cnn_aug=C["X_cnn_aug"]; X_tab=C["X_tab"]
X_frame=C["X_frame"]; X_new=C["X_new"]; y=C["y"]; frame_names=C["frame_names"]
ALL_TAB=C["ALL_TAB"]; NEW_AUDIO_COLS=C["NEW_AUDIO_COLS"]; fnames=list(C["fnames"])
N_AUG=3; PCA_MEL=15; PCA_CHROMA=10; PCA_MFCC=10
K_CANDIDATES = [20, 30, 40]

def cohen_d(X, yy, idx):
    a=X[yy==1,idx]; b=X[yy==0,idx]
    return abs(np.mean(a)-np.mean(b))/(np.sqrt((np.std(a)**2+np.std(b)**2)/2)+1e-8)

def top_frame_idx_for(idx_subset, k):
    scores = [cohen_d(X_frame[idx_subset], y[idx_subset], i) for i in range(X_frame.shape[1])]
    return np.argsort(scores)[::-1][:k]

def build(tr, va, k_frame):
    """tr, va are index arrays into the GLOBAL 77-sample arrays."""
    n = len(tr)
    top_idx_tr = top_frame_idx_for(tr, k_frame)
    cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n*N_AUG,3,1280)])
    tab_tr = np.vstack([X_tab[tr],  np.tile(X_tab[tr], (N_AUG,1))])
    frm_tr = np.vstack([X_frame[tr,:][:,top_idx_tr], np.tile(X_frame[tr,:][:,top_idx_tr],(N_AUG,1))])
    new_tr = np.vstack([X_new[tr],  np.tile(X_new[tr], (N_AUG,1))])
    y_tr   = np.concatenate([y[tr], np.tile(y[tr],N_AUG)])

    sc_mel=StandardScaler().fit(cnn_tr[:,0]); pca_mel=PCA(PCA_MEL,random_state=42).fit(sc_mel.transform(cnn_tr[:,0]))
    sc_chr=StandardScaler().fit(cnn_tr[:,1]); pca_chr=PCA(PCA_CHROMA,random_state=42).fit(sc_chr.transform(cnn_tr[:,1]))
    sc_mfc=StandardScaler().fit(cnn_tr[:,2]); pca_mfc=PCA(PCA_MFCC,random_state=42).fit(sc_mfc.transform(cnn_tr[:,2]))
    sc_tab=StandardScaler().fit(tab_tr); sc_frm=StandardScaler().fit(frm_tr); sc_new=StandardScaler().fit(new_tr)

    def fuse(cnn, tab, frm, new):
        return np.hstack([
            pca_mel.transform(sc_mel.transform(cnn[:,0])), pca_chr.transform(sc_chr.transform(cnn[:,1])),
            pca_mfc.transform(sc_mfc.transform(cnn[:,2])), sc_tab.transform(tab), sc_frm.transform(frm), sc_new.transform(new)
        ]).astype(np.float32)

    X_tr = fuse(cnn_tr,tab_tr,frm_tr,new_tr)
    X_va = fuse(X_cnn_orig[va],X_tab[va],X_frame[va,:][:,top_idx_tr],X_new[va])
    return X_tr, y_tr, X_va, top_idx_tr

def inner_cv_auc(outer_tr_idx, k_frame, C_val, n_inner=5):
    """Runs inner CV strictly within outer_tr_idx; returns mean AUC and pooled (true,prob)."""
    y_sub = y[outer_tr_idx]
    skf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=123)
    aucs = []
    all_true, all_prob = [], []
    for itr_rel, iva_rel in skf_inner.split(np.zeros(len(outer_tr_idx)), y_sub):
        itr = outer_tr_idx[itr_rel]; iva = outer_tr_idx[iva_rel]
        X_tr, y_tr, X_va, _ = build(itr, iva, k_frame)
        sm = BorderlineSMOTE(k_neighbors=5, random_state=42)
        try: X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except: pass
        lr = LogisticRegression(C=C_val, solver="saga", max_iter=3000, random_state=42)
        lr.fit(X_tr, y_tr)
        p = lr.predict_proba(X_va)[:,1]
        aucs.append(roc_auc_score(y[iva], p))
        all_true.extend(y[iva].tolist()); all_prob.extend(p.tolist())
    return float(np.mean(aucs)), all_true, all_prob

def select_k_and_c(outer_tr_idx, n_optuna_trials=25):
    """Inner-loop model selection: pick K and C using ONLY outer_tr_idx data."""
    best = {"auc": -1, "k": None, "c": None, "true": None, "prob": None}
    for k in K_CANDIDATES:
        def objective(trial):
            Cc = trial.suggest_float("C", 1e-3, 5.0, log=True)
            auc, _, _ = inner_cv_auc(outer_tr_idx, k, Cc, n_inner=5)
            return auc
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
        best_c_for_k = study.best_params["C"]
        auc_for_k, true_k, prob_k = inner_cv_auc(outer_tr_idx, k, best_c_for_k, n_inner=5)
        if auc_for_k > best["auc"]:
            best.update(auc=auc_for_k, k=k, c=best_c_for_k, true=true_k, prob=prob_k)
    return best  # contains inner out-of-fold (true, prob) under the WINNING (k, c) for threshold selection

print("="*90)
print("  NESTED CROSS-VALIDATION (outer test folds never touch tuning decisions)")
print("="*90)
t0 = time.time()

skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_true, outer_prob, outer_idx_order = [], [], []
fold_summaries = []

for fold_i, (outer_tr, outer_va) in enumerate(skf_outer.split(np.zeros(len(y)), y)):
    sel = select_k_and_c(outer_tr, n_optuna_trials=25)
    k_sel, c_sel = sel["k"], sel["c"]
    # Threshold from INNER out-of-fold predictions only (never touches outer_va)
    fpr_i, tpr_i, thr_i = roc_curve(sel["true"], sel["prob"])
    t_sel = float(thr_i[np.argmax(tpr_i - fpr_i)])

    # Refit on the FULL outer training set with the selected (k, C), then predict outer_va ONCE
    X_tr, y_tr, X_va, _ = build(outer_tr, outer_va, k_sel)
    sm = BorderlineSMOTE(k_neighbors=5, random_state=42)
    try: X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    except: pass
    lr = LogisticRegression(C=c_sel, solver="saga", max_iter=5000, random_state=42)
    lr.fit(X_tr, y_tr)
    p_va = lr.predict_proba(X_va)[:,1]

    outer_true.extend(y[outer_va].tolist())
    outer_prob.extend(p_va.tolist())
    outer_idx_order.extend(outer_va.tolist())
    fold_auc = roc_auc_score(y[outer_va], p_va) if len(set(y[outer_va].tolist()))>1 else float("nan")
    fold_summaries.append(dict(fold=fold_i, k=k_sel, C=c_sel, inner_auc=sel["auc"], threshold=t_sel, outer_auc=fold_auc))
    print(f"  Outer fold {fold_i}: selected K={k_sel}, C={c_sel:.5f} (inner AUC={sel['auc']:.4f}), "
          f"threshold={t_sel:.3f}, outer-fold AUC={fold_auc:.4f}  [{time.time()-t0:.0f}s elapsed]")

# ══════════════════════════════════════════════════════════════════════════════
# Aggregate outer predictions -> final, genuinely unbiased performance estimate
# ══════════════════════════════════════════════════════════════════════════════
outer_true_arr = np.array(outer_true); outer_prob_arr = np.array(outer_prob)
overall_auc = roc_auc_score(outer_true_arr, outer_prob_arr)

# Apply each sample's own fold-specific (inner-selected) threshold
outer_pred = np.zeros(len(outer_true_arr), dtype=int)
ptr = 0
for fs in fold_summaries:
    pass  # thresholds already applied per-fold below using recorded order

# Re-walk folds to apply per-fold thresholds correctly (order matches outer_idx_order construction)
pos = 0
outer_pred_list = []
fold_ptr = 0
sizes = []
# recompute sizes per fold from outer splits (same seed/order as above)
skf_outer2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for (_, outer_va) in skf_outer2.split(np.zeros(len(y)), y):
    sizes.append(len(outer_va))
for fi, sz in enumerate(sizes):
    seg_prob = outer_prob_arr[pos:pos+sz]
    seg_pred = (seg_prob >= fold_summaries[fi]["threshold"]).astype(int)
    outer_pred_list.extend(seg_pred.tolist())
    pos += sz
outer_pred = np.array(outer_pred_list)

acc = accuracy_score(outer_true_arr, outer_pred)
tn = int(np.sum((outer_true_arr==0)&(outer_pred==0))); fp = int(np.sum((outer_true_arr==0)&(outer_pred==1)))
fn = int(np.sum((outer_true_arr==1)&(outer_pred==0))); tp = int(np.sum((outer_true_arr==1)&(outer_pred==1)))
sens = tp/(tp+fn+1e-10); spec = tn/(tn+fp+1e-10); ppv = tp/(tp+fp+1e-10); npv = tn/(tn+fn+1e-10)

print("\n" + "="*90)
print("  FINAL NESTED RESULT (outer test folds untouched by any tuning decision)")
print("="*90)
print(f"  AUC: {overall_auc:.4f}")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"  Sensitivity={sens*100:.1f}%  Specificity={spec*100:.1f}%  PPV={ppv*100:.1f}%  NPV={npv*100:.1f}%")
print(f"  Per-fold selections: {fold_summaries}")

# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap 95% CIs on the pooled outer-of-fold predictions
# ══════════════════════════════════════════════════════════════════════════════
rng = np.random.RandomState(42)
n_boot = 2000
n = len(outer_true_arr)
boot_auc, boot_acc, boot_sens, boot_spec = [], [], [], []
for b in range(n_boot):
    idx = rng.randint(0, n, n)
    bt, bp, bpred = outer_true_arr[idx], outer_prob_arr[idx], outer_pred[idx]
    if len(set(bt.tolist())) < 2:
        continue
    boot_auc.append(roc_auc_score(bt, bp))
    boot_acc.append(accuracy_score(bt, bpred))
    btp = np.sum((bt==1)&(bpred==1)); bfn = np.sum((bt==1)&(bpred==0))
    btn = np.sum((bt==0)&(bpred==0)); bfp = np.sum((bt==0)&(bpred==1))
    boot_sens.append(btp/(btp+bfn+1e-10)); boot_spec.append(btn/(btn+bfp+1e-10))

def ci(arr): return (float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5)))
print("\n  95% Bootstrap CIs (2000 resamples of pooled outer predictions):")
print(f"    AUC:         {overall_auc:.3f}  CI {ci(boot_auc)}")
print(f"    Accuracy:    {acc:.3f}  CI {ci(boot_acc)}")
print(f"    Sensitivity: {sens:.3f}  CI {ci(boot_sens)}")
print(f"    Specificity: {spec:.3f}  CI {ci(boot_spec)}")

OUT = "C:/Users/VISHAL~1/AppData/Local/Temp/claude/D--parkinson-the-finale-dataset-and-audio-file/be45d466-5b03-4ef8-947b-b56f2a827e3b/scratchpad"
with open(f"{OUT}/nested_cv_result.pkl", "wb") as f:
    pickle.dump(dict(
        outer_true=outer_true_arr, outer_prob=outer_prob_arr, outer_pred=outer_pred,
        auc=overall_auc, acc=acc, sens=sens, spec=spec, ppv=ppv, npv=npv,
        tn=tn, fp=fp, fn=fn, tp=tp, fold_summaries=fold_summaries,
        ci_auc=ci(boot_auc), ci_acc=ci(boot_acc), ci_sens=ci(boot_sens), ci_spec=ci(boot_spec),
    ), f)
print(f"\nSaved -> {OUT}/nested_cv_result.pkl")
print(f"Total time: {time.time()-t0:.0f}s")
