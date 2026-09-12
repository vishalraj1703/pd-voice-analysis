# Analysis scripts (peer-review response, round 2)

These scripts and their outputs (`results/`) support every number in Section 4
and Section 5 of the manuscript beyond the original nested cross-validation
result (`nested_cv_final.py`, in the repo root).

- **controlled_ablation.py** — Table 2b. Seven configurations, each changing
  exactly one factor from a common fixed baseline (K=20, C=0.0102, no
  engineered features, no SMOTE, threshold=0.5), all on the identical outer
  5-fold split (seed=42). Paired bootstrap CIs on the difference from baseline
  for every row.
- **age_confound_analysis.py** — age-only, sex-only, age+sex, full+age+sex,
  and CNN-age-deconfounded models, all on the identical outer folds.
- **age_restricted_subset.py** — full model vs. age-only model restricted to
  the age band where the two groups overlap (n=59).
- **age_paired_comparison.py** — the paired bootstrap comparisons behind the
  "no evidence of a difference" claims: full model vs. age-only, and full
  model vs. CNN-age-deconfounded, both evaluated on the same outer-fold
  predictions with the same resampled indices applied to both sides of each
  comparison before differencing.
- **foldwise_shap.py** — Table 4. SHAP recomputed independently within each
  outer fold, using only that fold's own training-fold background and that
  fold's own fitted model, explaining only that fold's held-out outer-test
  samples.

All scripts read `raw_features_cache.pkl` (raw MobileNetV2 embeddings, tabular
features, frame-level statistics, engineered features, and labels for all 77
analysis-sample recordings plus the 4 reserved recordings) and
`Demographics_age_sex.xlsx` (age/sex for the full 81-participant collection).
Every script fixes `random_state=42` for the outer split and reports its own
seeds for any inner search.
