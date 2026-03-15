# PhishGuard AI

Phishing detection system for emails and sms. Uses a fine-tuned DistilBERT transformer combined with a rule-based keyword scanner. Includes a React dashboard and Flask API.

> Assignment submission for AI Engineer role at QMSMART.  
> Candidate: Papiya Mazumder

---

## Approach

A bag-of-words classifier handles obvious phishing ("URGENT: Your account is SUSPENDED!!!") but fails on spear phishing — messages written in professional language with a single malicious link buried inside. DistilBERT's bidirectional attention reads the full context and catches these. That is why I chose a transformer over simpler models.

The system uses three detection layers:

1. **URL hard override** — If the message contains a `.biz`, `.xyz`, `.tk` domain or IP-based URL, it is flagged as phishing regardless of ML score.
2. **BEC pattern override** — If BEC/aviation/enterprise keywords fire together with urgency and credential harvesting signals, it is flagged as phishing.
3. **Weighted blend** — `(DistilBERT confidence × 0.50) + (rule risk score × 0.50)` maps to three tiers: Legitimate (0–30%), Suspicious (30–70%), Phishing (70–100%).

---

## Results

Trained on 30,000 real emails from 2 Kaggle sources.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 89.49% | 99.22% | 79.28% | 88.14% |
| Logistic Regression | 97.48% | 96.99% | 97.92% | 97.45% |
| Random Forest | 97.04% | 96.93% | 97.06% | 96.99% |
| Linear SVM | 97.82% | 97.49% | 98.10% | 97.79% |
| **DistilBERT (selected)** | **98.30%** | **98.14%** | **98.47%** | **98.30%** |

ROC-AUC: **0.9980**

Recall is the priority metric — a missed phishing attack is worse than a false alarm. DistilBERT has the highest recall at 98.47%.

---

## Project structure

```
app.py                    Flask API — 6 endpoints, hybrid scoring
src/
  preprocess.py           Two NLP pipelines (DistilBERT + classical)
  features.py             25 hand-crafted features + TF-IDF
  keyword_detector.py     9-category regex scanner + URL signals
  train.py                DistilBERT fine-tuning with feedback ingestion
  evaluate.py             Metrics + diagnostic plots
  model_comparison.py     5-model comparison + SHAP explanations
data/
  dataset.csv             30k balanced emails
  download_dataset.py     Multi-source downloader with fallback
models/
  phishing_model.pkl      Best classical model (Linear SVM pipeline)
  metrics.json            All evaluation scores
  *.png                   Confusion matrix, ROC curve, comparison chart
notebook/
  phishing_detection_experimentation.ipynb
frontend/                 React dashboard
docker-compose.yml        One-command deployment
Makefile                  install / train / run targets
```

---

## NLP preprocessing

Two separate pipelines because DistilBERT and classical models need different input.

**DistilBERT pipeline:** Minimal cleaning — strip email headers, replace URLs with a `URL` token, remove HTML. Keep natural sentence structure intact. Over-processing hurts transformer performance because it destroys the contextual cues that attention relies on.

**Classical pipeline (TF-IDF):** Full normalization — lowercase, tokenize, remove stop words, lemmatize. One detail: NLTK's default stop word list includes `"urgent"`, `"verify"`, `"click"` — words that are strong phishing signals. I removed these from the exclusion list so they are preserved as features.

---

## Feature engineering

25 features across 5 categories:

- **Structural:** message length, uppercase ratio, exclamation marks, URL count, dollar signs, IP URLs
- **Keyword scores:** urgency, credential harvesting, threats, financial, lures — scored using a saturation scale (1 hit = 30, 2 = 65, 3 = 85, 4+ = 100) to prevent longer messages from scoring artificially high
- **Aviation/Enterprise:** crew portal mentions, DGCA/FAA impersonation, IT helpdesk, payroll — relevant to QMSMART's use case
- **URL signals:** suspicious TLD count, URL-to-text ratio
- **Contextual combinations:** 17 dangerous keyword pairs like `("verify", "immediately")` and `("crew portal", "verify")`

---

## Keyword detection

9 categories, regex-based, 100+ patterns. Runs alongside the ML model.

| Category | Examples |
|---|---|
| `urgency` | "urgent", "act now", "24 hours" |
| `credential_harvesting` | "verify your account", "reset password" |
| `threat_suspension` | "account suspended", "unauthorized access" |
| `bec_spear_phishing` | "mandatory training portal", "password rotation policy" |
| `prize_lure` | "you have won", "claim your prize" |
| `financial` | "wire transfer", "credit card", dollar amounts |
| `suspicious_links` | .biz/.xyz/.tk TLDs, IP URLs |
| `aviation_sector` | crew portal, DGCA/FAA compliance |
| `enterprise_sector` | IT helpdesk, payroll update, Office 365 |

Detected keywords are highlighted in the React dashboard so users can see exactly what triggered the classification.

---

## API

Backend runs on port 5001.

POST /api/predict      Hybrid prediction — returns label, confidence, risk level, keywords
GET  /api/health       Model load status + device info
GET  /api/keywords     Full keyword lexicon with category counts
GET  /api/demo         5 pre-built test messages
POST /api/parse-file   Upload .eml, .pdf, or .txt for analysis
POST /api/feedback     Submit correction — stored in data/feedback.csv for retraining. One sample entry is included in `data/feedback.csv` to show the schema.

---

## Dataset

**Source 1** — `naserabdullahalam/phishing-email-dataset`: 6 real corpora (Enron, Nazario, CEAS, etc.), ~82k emails, 2000–2008.  
**Source 2** — `subhajournal/phishingemails`: ~18k modern emails, 2019–2021. Added because phishing tactics changed significantly since 2008.

After merge, dedup, and balancing: **30,000 rows, 15,000 per class.**

The downloader has a 4-tier fallback (Kaggle → secondary → HuggingFace → synthetic) so it works without credentials.

---

## Additional work beyond requirements

These were not required by the assignment but I added them because they solve real problems:

- **Unicode normalization (NFKD):** Phishers swap characters (Greek 'ο' for Latin 'o') to bypass keyword filters. NFKD normalization in preprocessing strips this.
- **INT8 quantization:** Reduces DistilBERT memory by ~50% and speeds up CPU inference 2–4x. Applied automatically for CPU deployments.
- **SHAP explanations:** Shows which words contributed most to a prediction. Useful for auditing false positives.
- **Feedback endpoint:** Users can flag incorrect predictions. Feedback is saved to CSV and merged into training data on next retrain cycle.
- **Docker deployment:** One-command deployment with `docker-compose up --build`

---

## Running locally

```bash
git clone https://github.com/papiyamazumder/phishing_detection_project_aiml.git
cd phishing_detection_project_aiml
pip install -r requirements.txt

# Download dataset (skip if data/dataset.csv exists)
python data/download_dataset.py

# Train DistilBERT (~27 min on GPU/MPS)
python src/train.py

# Generate evaluation plots and metrics
python src/evaluate.py

# Compare models and save .pkl
python src/model_comparison.py

# Start API
python app.py

# Start dashboard (new terminal)
cd frontend && npm install && npm start
```

Or with Docker:
```bash
docker-compose up --build
```

Note: Run `src/train.py` first to generate DistilBERT weights (~27 min, requires GPU/MPS). The classical model (`phishing_model pkl`) and all evaluation plots are already in the repo and work immediately after clone.

---

## Stack

Python 3.11 · PyTorch · HuggingFace Transformers · scikit-learn · NLTK · Flask · React · Docker