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
`nested_cv_result.pkl` for the exact per-fold `(K, C, threshold)` selections
and pooled predictions underlying the numbers reported in the paper.

## Known limitations not resolved by this release

- The dataset (Prior et al., 2023, Figshare) is a single cohort with a
  substantial age difference between groups (HC mean 47.7y vs PD mean 67.0y);
  this package does not include an age-matched re-analysis.
- Per-recording device/telephone metadata beyond "participant's own
  telephone" was not available and could not be audited for a device-based
  confound.
- A full factorial ablation across all feature-group combinations was not
  run; the three reported configurations (A/B/C) change more than one factor
  at a time and cannot isolate individual component contributions.
