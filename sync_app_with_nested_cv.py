# -*- coding: utf-8 -*-
"""
Syncs model_artifacts/cv_stats.pkl's "Model C" entry with the genuine nested
cross-validation result (nested_cv_result.pkl), so the app's Model Performance
page matches the numbers now reported in the paper (AUC 0.828, Acc 72.7%),
instead of the earlier non-nested numbers (AUC 0.877, Acc 80.5%).
"""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, accuracy_score

NESTED = "C:/Users/VISHAL~1/AppData/Local/Temp/claude/D--parkinson-the-finale-dataset-and-audio-file/be45d466-5b03-4ef8-947b-b56f2a827e3b/scratchpad/nested_cv_result.pkl"
CACHE  = "C:/Users/VISHAL~1/AppData/Local/Temp/claude/D--parkinson-the-finale-dataset-and-audio-file/be45d466-5b03-4ef8-947b-b56f2a827e3b/scratchpad/raw_features_cache.pkl"
ARTDIR = "D:/parkinson detector tool/model_artifacts"

with open(NESTED, "rb") as f:
    R = pickle.load(f)
with open(CACHE, "rb") as f:
    C = pickle.load(f)
y = C["y"]

outer_true = R["outer_true"]; outer_prob = R["outer_prob"]; outer_pred = R["outer_pred"]
fold_summaries = R["fold_summaries"]

# Recompute per-fold accuracy (same outer split/order used in nested_cv_final.py)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sizes = [len(va) for _, va in skf.split(np.zeros(len(y)), y)]
per_fold_accs, per_fold_aucs = [], []
pos = 0
for fi, sz in enumerate(sizes):
    seg_true = outer_true[pos:pos+sz]; seg_pred = outer_pred[pos:pos+sz]
    per_fold_accs.append(accuracy_score(seg_true, seg_pred))
    per_fold_aucs.append(fold_summaries[fi]["outer_auc"])
    pos += sz

fpr, tpr, thr = roc_curve(outer_true, outer_prob)
mean_C = float(np.mean([fs["C"] for fs in fold_summaries]))
mean_threshold = float(np.mean([fs["threshold"] for fs in fold_summaries]))
acc_at_05 = accuracy_score(outer_true, (outer_prob >= 0.5).astype(int))

model_C_nested = dict(
    aucs=[float(a) for a in per_fold_aucs],
    accs=[float(a) for a in per_fold_accs],
    mean_auc=float(R["auc"]),
    std_auc=float(np.std(per_fold_aucs)),
    mean_acc=float(acc_at_05),
    opt_acc=float(R["acc"]),
    threshold=mean_threshold,
    val_true=outer_true.tolist(),
    val_prob=outer_prob.tolist(),
    fpr=fpr.tolist(), tpr=tpr.tolist(), thr=thr.tolist(),
    confusion_matrix=[[int(R["tn"]), int(R["fp"])], [int(R["fn"]), int(R["tp"])]],
    C=mean_C,
    use_new=True, use_smote=True,
    # Extra fields for transparency (ignored by older app code, available for future use)
    nested=True,
    ci_auc=R["ci_auc"], ci_acc=R["ci_acc"], ci_sens=R["ci_sens"], ci_spec=R["ci_spec"],
    per_fold_selections=fold_summaries,
)

with open(f"{ARTDIR}/cv_stats.pkl", "rb") as f:
    cv_stats = pickle.load(f)

import shutil
shutil.copy(f"{ARTDIR}/cv_stats.pkl", f"{ARTDIR}/cv_stats.pkl.PRE_NESTED_BACKUP")

cv_stats["Model C: + Optuna + Optimal Thresh"] = model_C_nested
with open(f"{ARTDIR}/cv_stats.pkl", "wb") as f:
    pickle.dump(cv_stats, f)

print("Synced. Model C now shows:")
print(f"  AUC: {model_C_nested['mean_auc']:.4f}  (std across outer folds: {model_C_nested['std_auc']:.4f})")
print(f"  Acc@0.5: {model_C_nested['mean_acc']*100:.1f}%   Acc@opt(nested): {model_C_nested['opt_acc']*100:.1f}%")
print(f"  Mean threshold across folds: {model_C_nested['threshold']:.3f}")
print(f"  Mean C across folds: {model_C_nested['C']:.5f}")
print(f"  Confusion: {model_C_nested['confusion_matrix']}")
print(f"Old cv_stats.pkl backed up -> {ARTDIR}/cv_stats.pkl.PRE_NESTED_BACKUP")
