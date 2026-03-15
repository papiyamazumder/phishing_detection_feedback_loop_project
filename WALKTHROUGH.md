# Technical Walkthrough

This document explains the key technical decisions in the PhishGuard AI project. It is a companion to the README — the README covers what was built, this covers why.

---

## Detection architecture

The system combines a DistilBERT model with a rule-based keyword scanner. Neither layer works well alone:

- **ML alone** gets fooled by spear phishing. When 85% of the tokens are clean business language, the model's average confidence leans towards "legitimate" even if there is a malicious `.biz` link in the text.
- **Rules alone** miss novel attack patterns and cannot understand semantic context.

The hybrid approach gives each layer what it is good at. Rules catch high-confidence signals (suspicious URLs, known BEC templates) with guaranteed recall. The ML model handles everything else using contextual understanding.

---

## Preprocessing choices

Two separate pipelines. This is not standard practice — most tutorials use one pipeline for everything.

**Why:** DistilBERT performs worse when you lemmatize and strip stop words. Its attention mechanism needs natural sentence structure to work properly. TF-IDF vectors, on the other hand, need clean normalized tokens to produce meaningful features. Separate pipelines let each model receive the input format it actually needs.

One detail worth noting: NLTK's default stop word list removes words like "urgent", "verify", "click". These are important phishing signals, so I kept them in.

---

## Data sources

| Source | Why included | Size |
|---|---|---|
| `naserabdullahalam/phishing-email-dataset` | 6 real corpora (Enron + Nazario + 4 others). Combines real corporate email with real phishing — not synthetic templates. | ~82k emails, 2000–2008 |
| `subhajournal/phishingemails` | Closes the era gap. Phishing tactics changed significantly after 2008 (COVID lures, cloud impersonation). | ~18k emails, 2019–2021 |

Final dataset: 30,000 rows after dedup and balancing (15,000 per class). The cap at 15k is intentional — beyond this, accuracy gains are marginal but training time doubles.

The downloader has a 4-tier fallback so it works even without Kaggle credentials.

---

## Feature scaling

I used `MaxAbsScaler` instead of `MinMaxScaler`. TF-IDF produces sparse matrices where zeros mean "word not present." MinMaxScaler shifts the mean above zero, which destroys that sparsity. MaxAbsScaler scales without shifting, preserving both sparsity and meaning.

---

## Production extras

These were not required but solve practical problems:

| Feature | What it does | Where |
|---|---|---|
| Unicode NFKD normalization | Strips homoglyph attacks (Greek 'ο' swapped for Latin 'o') | `src/preprocess.py` |
| INT8 quantization | Halves model memory, 2–4x faster CPU inference | `app.py` |
| SHAP explanations | Shows which words drove each prediction | `src/model_comparison.py` |
| Feedback endpoint | Saves user corrections for retraining | `app.py`, `data/feedback.csv` |

---

## Deliverable checklist

| Requirement | Status | Location |
|---|---|---|
| Python project or Jupyter notebook | Done | `src/`, `notebook/` |
| Trained model file (.pkl) | Done | `models/phishing_model.pkl` |
| Data preprocessing code | Done | `src/preprocess.py` |
| Model training code | Done | `src/train.py` |
| Evaluation metrics | Done | `models/metrics.json` |
| Confusion matrix | Done | `models/confusion_matrix.png` |
| README explaining approach | Done | `README.md` |
| UI (optional) | Done | `frontend/` (React dashboard) |
| API endpoints (optional) | Done | `app.py` (6 endpoints) |
