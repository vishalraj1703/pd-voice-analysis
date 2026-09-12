# PD Voice Analysis

Streamlit app for research screening of Parkinson's disease voice biomarkers
from a sustained "Aaah" vowel recording. Companion tool to "Voice-Based
Detection of Parkinson's Disease Using Fused CNN–Acoustic Features and
SHAP-Interpretable Logistic Regression."

**Research use only — not a clinical diagnostic tool.**

## Run locally

```bash
pip install -r requirements.txt
streamlit run parkinson_app.py
```

Requires `model_artifacts/pipeline.pkl`, `model_artifacts/cv_stats.pkl`, and
`model_artifacts/training_data.pkl` to be present (already included in this
repo). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for how these were
generated, including the nested cross-validation design.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app".
3. Select this repo, branch `main`, main file `parkinson_app.py`.
4. Deploy. `requirements.txt` and `packages.txt` are picked up automatically.

## Model performance (nested 5-fold cross-validation)

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.828 | 0.724 – 0.912 |
| Accuracy | 72.7% | 62.3% – 81.8% |
| Sensitivity | 65.8% | 50.0% – 80.0% |
| Specificity | 79.5% | 65.9% – 91.7% |

See `REPRODUCIBILITY.md` for the full nested CV methodology and known
limitations.
