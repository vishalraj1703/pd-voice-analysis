# Reproducibility Package

This document accompanies the code release for "Voice-Based Detection of
Parkinson's Disease Using Fused CNN–Acoustic Features and SHAP-Interpretable
Logistic Regression."

## Environment

| Package | Version |
|---|---|
| Python | 3.12 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.14.1 |
| shap | 0.50.0 |
| optuna | 4.8.0 |
| tensorflow | 2.18.0 |
| librosa | 0.11.0 |
| parselmouth (Praat) | see `pip freeze` in this repo |
| nolds | see `pip freeze` in this repo |

Run `pip freeze > requirements-frozen.txt` in the project environment before
release and include it alongside this file.

## Random seeds

| Component | Seed |
|---|---|
| Outer `StratifiedKFold` (5 folds) | `random_state=42` |
| Inner `StratifiedKFold` (5 folds, within each outer training fold) | `random_state=123` |
| `PCA` (Mel/Chroma/MFCC) | `random_state=42` |
| `BorderlineSMOTE` | `random_state=42`, `k_neighbors=5` |
| `LogisticRegression` | `random_state=42`, `solver="saga"` |
| Optuna `TPESampler` | `seed=42` |
| TensorFlow / NumPy global seeds | `tf.random.set_seed(42)`, `np.random.seed(42)` |

## Nested cross-validation design (fixes the leakage identified in peer review)

An earlier version of this pipeline selected the retained frame-feature count
using Cohen's d computed on the full 77-sample dataset before any
cross-validation split, and separately selected the regularisation strength
(via Optuna) and the Youden-J decision threshold using the same outer folds
that performance was then reported on. Both are leakage: outer test-fold
information influenced modelling decisions before the outer fold was scored.

The corrected design (`nested_cv_final.py`) is:

1. **Outer loop**: 5-fold stratified CV over the 77 samples. Each outer test
   fold is touched exactly once, at the very end of that fold's iteration.
2. **Inner loop**: within each outer *training* partition only, a further
   5-fold stratified CV selects:
   - the retained frame-feature count `K` from candidates `{20, 30, 40}`
     (Cohen's d ranking recomputed on inner-training rows for every inner
     split — never touches inner-validation or outer-test rows),
   - the logistic-regression regularisation strength `C`, via Optuna
     (25 trials per `K` candidate, objective = mean inner-fold AUC),
   - the Youden-J decision threshold, computed from the pooled
     out-of-fold predictions of the *winning* `(K, C)` configuration's inner
     CV — this threshold is fixed before the outer test fold is ever scored.
3. The winning `(K, C)` is refit once on the *full* outer training partition
   (with BorderlineSMOTE applied to that refit only), and used to predict the
   outer test fold exactly once.
4. Predictions from all 5 outer folds are pooled to compute the final AUC,
   accuracy, sensitivity, specificity, PPV, and NPV.
5. 95% confidence intervals are obtained by percentile bootstrap (2,000
   resamples) over the pooled outer-fold (true label, predicted probability)
   pairs — a standard, though approximate, approach for cross-validated
   predictions; it does not fully account for the correlation induced by
   repeated model refitting, and should be read as indicative rather than
   exact.

See `nested_cv_final.py` for the complete implementation and
`analysis/results/nested_cv_result.pkl` for the exact per-fold `(K, C,
threshold)` selections and pooled predictions underlying the numbers reported
in the paper. Note: uncertainty is now reported in the paper as a **percentile
bootstrap interval over pooled out-of-fold predictions**, not as an ordinary
confidence interval — see `analysis/controlled_ablation.py` and the paper's
Section 3.3 for why that distinction matters (the bootstrap resamples fixed
predictions rather than repeating feature selection, tuning, and refitting).

## Exact pipeline order (every script in `analysis/`)

Every analysis in this package follows this order, and every step marked
"fold-confined" is fit **only** on the data available at that point (inner-
training rows for inner-loop steps, outer-training rows for outer-loop
steps) and never sees the corresponding validation/test rows before scoring:

1. **Split**: outer `StratifiedKFold(5, shuffle=True, random_state=42)` on
   the 77-sample analysis set. Fixed indices are saved once and reused by
   every script — see `analysis/results/outer_fold_manifest.json` (created
   by re-running the outer split with the same seed; contains each fold's
   train/test sample indices and filenames).
2. **Frame-feature ranking** (fold-confined): Cohen's d computed on
   training rows only, top-K retained (K ∈ {20,30,40} for nested/tuned runs,
   K=20 or K=40 fixed for ablation/fixed runs).
3. **Scaling**: a separate `StandardScaler` fit per feature block (CNN-Mel,
   CNN-Chroma, CNN-MFCC, tabular, frame, engineered) on training rows only.
4. **PCA** (CNN blocks only): fit on the scaled training rows only
   (Mel→15, Chroma→10, MFCC→10 components), then applied to validation/test
   rows using the training-fitted transform.
5. **Augmentation** (nested/nested-adjacent runs only): each training
   recording's 3 pre-computed spectrogram augmentations are added to the
   training fold only, never to validation/test rows.
6. **BorderlineSMOTE** (`k_neighbors=5`, fold-confined): applied to the
   fused, scaled training rows only, immediately before fitting the
   classifier; never applied to validation/test rows, and the pre-SMOTE
   training rows are what SHAP's background distribution uses
   (`analysis/foldwise_shap.py`).
7. **Logistic regression** (`solver="saga"`, `random_state=42`): fit on the
   (possibly SMOTE-resampled) training rows.
8. **Score generation**: `predict_proba` on validation/test rows — this is
   the classifier's own sigmoid output, not a separately fitted calibrator
   (Section 3.6 corrects an earlier draft that incorrectly described a
   Platt-scaling step).
9. **Threshold selection** (nested runs only, fold-confined): Youden-J
   computed from the *inner* out-of-fold predictions of the winning (K, C)
   configuration; applied unmodified to that outer fold's test predictions.

## Missing-value and failed-extraction handling

Praat/Parselmouth perturbation measures, DFA, and the 12 engineered
features are wrapped in per-feature `try/except` blocks in the extraction
code (`parkinson_app.py`): a failed jitter/shimmer/HNR extraction defaults to
`0.0`, a failed DFA computation defaults to `0.5`, and a failed nonlinear-
dynamics extraction (RPDE/DFA/PPE) defaults to `[0., 0.5, 0.]`. Unvoiced
frames in the F0 contour are marked `NaN` and excluded before computing
frame-level F0 statistics. As a final safeguard, every returned engineered
feature is checked with `np.isfinite`; any non-finite value (`NaN`/`Inf`) is
replaced with `0.0` before the feature reaches the classifier. No recording
in the 77-sample analysis set or the 4 reserved recordings required this
fallback during the original feature-cache build, but the fallback exists
and is exercised by unusual or very short inputs (e.g. the "recording too
short/too quiet" guard added to the live application, Section 3.8).

## Reserved recordings (Section 3.1)

The 4 reserved recordings (2 HC, 2 PwPD; see `BLIND_HC`/`BLIND_PD` in
`best_accuracy_experiment.py`) are present in `raw_features_cache.pkl` as
`blind_cnn`/`blind_tab`/`blind_frm`/`blind_newf`/`blind_y`, but **no script
in `analysis/` scores them**: they are not used for model selection, feature
engineering, threshold selection, or application development, and no
performance number for them appears anywhere in the paper.

## Known limitations not resolved by this release

- The dataset (Prior et al., 2023, Figshare) is a single cohort with a
  substantial age difference between groups (HC mean 47.7y vs PD mean 67.0y
  in the full 81-participant collection); `analysis/age_confound_analysis.py`,
  `analysis/age_restricted_subset.py`, and `analysis/age_paired_comparison.py`
  quantify this but do not resolve it — see the paper's Section 3.1 and 5.2.
- Per-recording device/telephone metadata beyond "participant's own
  telephone" was not available and could not be audited for a device-based
  confound.
- The age-overlap subset analysis (n=59) uses a separate cross-validation
  split from every other analysis in this package and has not been paired-
  bootstrapped against its own age-only baseline; it is exploratory, not
  confirmatory (paper Section 3.1, 5.2).
- The controlled ablation (`analysis/controlled_ablation.py`) tests one
  factor at a time from a single fixed baseline; it does not test every
  possible combination of factors (a full factorial design), only the
  combinations reported in Table 2b.
