# -*- coding: utf-8 -*-
"""
Controlled ablation study responding to peer-review Major Comment 1.

Every row below uses the SAME outer 5-fold StratifiedKFold(seed=42) split as
nested_cv_final.py, so predictions are paired sample-for-sample across rows.
Only one factor changes at a time from the Baseline row. Paired bootstrap CIs
(same resampled index set applied to both rows before differencing) are
reported for each row's difference from Baseline -- not independent marginal
CIs, which the reviewer correctly notes can overlap while a paired difference
would not.

Rows:
  0  Baseline            : CNN(PCA)+tabular+frame(K=20 fixed), no new feats, no SMOTE, C=0.0102, thr=0.5
  1  +SMOTE              : Baseline + BorderlineSMOTE
  2  +NewFeatures        : Baseline + 12 engineered features, no SMOTE
  3  +NewFeatures+SMOTE  : both of the above together
  4  Tuned K             : like row 3, but K chosen per outer fold via inner CV (C fixed=0.0102)
  5  Tuned C             : like row 3, K fixed=20, C chosen per outer fold via inner CV (Optuna)
  6  Tuned threshold     : like row 3, K=20 fixed, C=0.0102 fixed, threshold from inner-fold Youden
  7  Fully nested (all tuned + SMOTE) : reuses nested_cv_result.pkl computed separately
"""
import sys, io, os, pickle, warnings, time
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

SCRATCH = os.path.dirname(os.path.abspath(__file__)) + "/results"
CACHE = os.path.dirname(os.path.abspath(__file__)) + "/data/raw_features_cache.pkl"
with open(CACHE, "rb") as f:
    C = pickle.load(f)

X_cnn_orig = C["X_cnn_orig"]; X_cnn_aug = C["X_cnn_aug"]; X_tab = C["X_tab"]
X_frame = C["X_frame"]; X_new = C["X_new"]; y = C["y"]
N_AUG = 3; PCA_MEL = 15; PCA_CHROMA = 10; PCA_MFCC = 10
FIXED_K = 20; FIXED_C = 0.0102; FIXED_THR = 0.5
K_CANDIDATES = [20, 30, 40]
N_OUTER = 5

def cohen_d(X, yy, idx):
    a = X[yy == 1, idx]; b = X[yy == 0, idx]
    return abs(np.mean(a) - np.mean(b)) / (np.sqrt((np.std(a)**2 + np.std(b)**2) / 2) + 1e-8)

def top_frame_idx_for(idx_subset, k):
    scores = [cohen_d(X_frame[idx_subset], y[idx_subset], i) for i in range(X_frame.shape[1])]
    return np.argsort(scores)[::-1][:k]

def build(tr, va, k_frame, use_new_features, augment=True):
    """Fold-confined feature construction. tr/va index into the global 77 samples."""
    n = len(tr)
    top_idx_tr = top_frame_idx_for(tr, k_frame)
    if augment:
        cnn_tr = np.vstack([X_cnn_orig[tr], X_cnn_aug[tr].reshape(n * N_AUG, 3, 1280)])
        tab_tr = np.vstack([X_tab[tr], np.tile(X_tab[tr], (N_AUG, 1))])
        frm_tr = np.vstack([X_frame[tr, :][:, top_idx_tr], np.tile(X_frame[tr, :][:, top_idx_tr], (N_AUG, 1))])
        new_tr = np.vstack([X_new[tr], np.tile(X_new[tr], (N_AUG, 1))])
        y_tr = np.concatenate([y[tr], np.tile(y[tr], N_AUG)])
    else:
        cnn_tr, tab_tr, frm_tr, new_tr, y_tr = X_cnn_orig[tr], X_tab[tr], X_frame[tr, :][:, top_idx_tr], X_new[tr], y[tr]

    sc_mel = StandardScaler().fit(cnn_tr[:, 0]); pca_mel = PCA(PCA_MEL, random_state=42).fit(sc_mel.transform(cnn_tr[:, 0]))
    sc_chr = StandardScaler().fit(cnn_tr[:, 1]); pca_chr = PCA(PCA_CHROMA, random_state=42).fit(sc_chr.transform(cnn_tr[:, 1]))
    sc_mfc = StandardScaler().fit(cnn_tr[:, 2]); pca_mfc = PCA(PCA_MFCC, random_state=42).fit(sc_mfc.transform(cnn_tr[:, 2]))
    sc_tab = StandardScaler().fit(tab_tr); sc_frm = StandardScaler().fit(frm_tr)
    sc_new = StandardScaler().fit(new_tr) if use_new_features else None

    def fuse(cnn, tab, frm, new):
        parts = [pca_mel.transform(sc_mel.transform(cnn[:, 0])), pca_chr.transform(sc_chr.transform(cnn[:, 1])),
                 pca_mfc.transform(sc_mfc.transform(cnn[:, 2])), sc_tab.transform(tab), sc_frm.transform(frm)]
        if use_new_features:
            parts.append(sc_new.transform(new))
        return np.hstack(parts).astype(np.float32)

    X_tr = fuse(cnn_tr, tab_tr, frm_tr, new_tr)
    X_va = fuse(X_cnn_orig[va], X_tab[va], X_frame[va, :][:, top_idx_tr], X_new[va])
    return X_tr, y_tr, X_va, top_idx_tr

def fit_predict(tr, va, k, c_val, use_smote, use_new_features):
    X_tr, y_tr, X_va, _ = build(tr, va, k, use_new_features)
    if use_smote:
        sm = BorderlineSMOTE(k_neighbors=5, random_state=42)
        try: X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception: pass
    lr = LogisticRegression(C=c_val, solver="saga", max_iter=5000, random_state=42)
    lr.fit(X_tr, y_tr)
    return lr.predict_proba(X_va)[:, 1]

def inner_cv_metric(outer_tr_idx, k_frame, c_val, use_smote, use_new_features, n_inner=5):
    y_sub = y[outer_tr_idx]
    skf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=123)
    aucs = []
    all_true, all_prob = [], []
    for itr_rel, iva_rel in skf_inner.split(np.zeros(len(outer_tr_idx)), y_sub):
        itr = outer_tr_idx[itr_rel]; iva = outer_tr_idx[iva_rel]
        p = fit_predict(itr, iva, k_frame, c_val, use_smote, use_new_features)
        aucs.append(roc_auc_score(y[iva], p))
        all_true.extend(y[iva].tolist()); all_prob.extend(p.tolist())
    return float(np.mean(aucs)), all_true, all_prob

skf_outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
outer_splits = list(skf_outer.split(np.zeros(len(y)), y))
fold_sizes = [len(va) for _, va in outer_splits]

def run_row(name, k_mode, c_mode, use_smote, use_new_features, thr_mode):
    """k_mode/c_mode: 'fixed' or 'tuned'. thr_mode: 'fixed' or 'youden'."""
    t0 = time.time()
    all_true, all_prob, all_pred = [], [], []
    per_fold_info = []
    for fi, (outer_tr, outer_va) in enumerate(outer_splits):
        if k_mode == "tuned":
            best = {"auc": -1, "k": None}
            for k in K_CANDIDATES:
                auc_k, _, _ = inner_cv_metric(outer_tr, k, FIXED_C, use_smote, use_new_features)
                if auc_k > best["auc"]:
                    best.update(auc=auc_k, k=k)
            k_sel = best["k"]
        else:
            k_sel = FIXED_K

        if c_mode == "tuned":
            def objective(trial):
                cc = trial.suggest_float("C", 1e-3, 5.0, log=True)
                auc_c, _, _ = inner_cv_metric(outer_tr, k_sel, cc, use_smote, use_new_features)
                return auc_c
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=25, show_progress_bar=False)
            c_sel = study.best_params["C"]
        else:
            c_sel = FIXED_C

        if thr_mode == "youden":
            _, inner_true, inner_prob = inner_cv_metric(outer_tr, k_sel, c_sel, use_smote, use_new_features)
            fpr_i, tpr_i, thr_i = roc_curve(inner_true, inner_prob)
            t_sel = float(thr_i[np.argmax(tpr_i - fpr_i)])
        else:
            t_sel = FIXED_THR

        p_va = fit_predict(outer_tr, outer_va, k_sel, c_sel, use_smote, use_new_features)
        pred_va = (p_va >= t_sel).astype(int)
        all_true.extend(y[outer_va].tolist()); all_prob.extend(p_va.tolist()); all_pred.extend(pred_va.tolist())
        per_fold_info.append(dict(fold=fi, k=k_sel, C=round(c_sel,5), threshold=round(t_sel,3),
                                   auc=round(roc_auc_score(y[outer_va], p_va),4) if len(set(y[outer_va]))>1 else float("nan")))

    all_true = np.array(all_true); all_prob = np.array(all_prob); all_pred = np.array(all_pred)
    auc = roc_auc_score(all_true, all_prob); acc = accuracy_score(all_true, all_pred)
    tp = np.sum((all_true==1)&(all_pred==1)); fn = np.sum((all_true==1)&(all_pred==0))
    tn = np.sum((all_true==0)&(all_pred==0)); fp = np.sum((all_true==0)&(all_pred==1))
    sens = tp/(tp+fn+1e-10); spec = tn/(tn+fp+1e-10)
    print(f"  [{name}] AUC={auc:.4f} Acc={acc*100:.1f}% Sens={sens*100:.1f}% Spec={spec*100:.1f}% "
          f"({time.time()-t0:.0f}s)  folds={per_fold_info}")
    return dict(name=name, true=all_true, prob=all_prob, pred=all_pred,
                auc=auc, acc=acc, sens=sens, spec=spec, per_fold=per_fold_info)

print("="*100)
print("CONTROLLED ABLATION -- same outer folds (seed=42) throughout, one factor varied at a time")
print("="*100)

rows = {}
# Every non-baseline row below changes EXACTLY ONE factor relative to "baseline"
# (fixed K=20, fixed C=0.0102, no engineered features, no SMOTE, threshold=0.5).
# None of them are chained off each other -- each independently reverts every
# other factor to the baseline value. This corrects an earlier version of this
# script where the tuned-K/C/threshold rows also had engineered features and
# SMOTE turned on (inherited from row 3), which a reviewer correctly pointed
# out meant those rows could not isolate the tuning effect alone.
rows["baseline"]     = run_row("0 Baseline (fixed K=20, fixed C=0.0102, no feats, no SMOTE, thr=0.5)", "fixed","fixed",False,False,"fixed")
rows["smote"]        = run_row("1 Baseline + SMOTE only",                                              "fixed","fixed",True, False,"fixed")
rows["newfeat"]      = run_row("2 Baseline + engineered features only",                                "fixed","fixed",False,True, "fixed")
rows["newfeat_smote"]= run_row("3 Baseline + engineered features + SMOTE",                             "fixed","fixed",True, True, "fixed")
rows["tuned_k"]      = run_row("4 Baseline + tuned K only",                                            "tuned","fixed",False,False,"fixed")
rows["tuned_c"]      = run_row("5 Baseline + tuned C only",                                            "fixed","tuned",False,False,"fixed")
rows["tuned_thr"]    = run_row("6 Baseline + tuned threshold only (inner Youden)",                     "fixed","fixed",False,False,"youden")

# ---- Paired bootstrap CI on the DIFFERENCE from baseline, for every non-baseline row ----
# Same resampled index set is applied to both rows before differencing -- a true paired
# comparison, not two overlapping marginal intervals.
rng = np.random.RandomState(42)
n_boot = 2000
n = len(y)
base = rows["baseline"]

def paired_ci(row):
    d_auc, d_acc, d_sens, d_spec = [], [], [], []
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        bt = base["true"][idx]
        if len(set(bt.tolist())) < 2:
            continue
        bp_base, bpred_base = base["prob"][idx], base["pred"][idx]
        bp_row,  bpred_row  = row["prob"][idx],  row["pred"][idx]
        auc_base = roc_auc_score(bt, bp_base); auc_row = roc_auc_score(bt, bp_row)
        acc_base = accuracy_score(bt, bpred_base); acc_row = accuracy_score(bt, bpred_row)
        def sens_spec(bt_, bpred_):
            tp=np.sum((bt_==1)&(bpred_==1)); fn=np.sum((bt_==1)&(bpred_==0))
            tn=np.sum((bt_==0)&(bpred_==0)); fp=np.sum((bt_==0)&(bpred_==1))
            return tp/(tp+fn+1e-10), tn/(tn+fp+1e-10)
        sens_base, spec_base = sens_spec(bt, bpred_base)
        sens_row,  spec_row  = sens_spec(bt, bpred_row)
        d_auc.append(auc_row-auc_base); d_acc.append(acc_row-acc_base)
        d_sens.append(sens_row-sens_base); d_spec.append(spec_row-spec_base)
    def ci(arr): return (float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5)))
    return dict(d_auc=(np.mean(d_auc), ci(d_auc)), d_acc=(np.mean(d_acc), ci(d_acc)),
                d_sens=(np.mean(d_sens), ci(d_sens)), d_spec=(np.mean(d_spec), ci(d_spec)))

print("\n" + "="*100)
print("PAIRED DIFFERENCES FROM BASELINE (mean [95% CI]) -- CI excluding 0 = statistically supported effect")
print("="*100)
diffs = {}
for key in ["smote","newfeat","newfeat_smote","tuned_k","tuned_c","tuned_thr"]:
    d = paired_ci(rows[key])
    diffs[key] = d
    print(f"  {key:16s}  dAUC={d['d_auc'][0]:+.4f} {d['d_auc'][1]}   dAcc={d['d_acc'][0]:+.4f} {d['d_acc'][1]}   "
          f"dSens={d['d_sens'][0]:+.4f} {d['d_sens'][1]}   dSpec={d['d_spec'][0]:+.4f} {d['d_spec'][1]}")

with open(f"{SCRATCH}/controlled_ablation_result.pkl", "wb") as f:
    pickle.dump(dict(rows={k: {kk: vv for kk, vv in v.items() if kk not in ("true","prob","pred")} for k,v in rows.items()},
                      diffs=diffs), f)
print(f"\nSaved -> {SCRATCH}/controlled_ablation_result.pkl")
