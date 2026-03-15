# PhishGuard AI — Phishing Detection System

A text classification system that detects phishing attempts in emails and messages using a hybrid approach: rule-based heuristics combined with a fine-tuned DistilBERT model.

Built as part of an AI Engineer assignment, but extended beyond the base requirements.

---

## What it does

Given an email or message as input, the system:
- Classifies it as **Phishing** or **Legitimate**
- Returns a confidence score
- Highlights the specific keywords or phrases that triggered the flag
- Breaks down URL risk signals if a link is present

---

## Why DistilBERT instead of Logistic Regression / SVM

The assignment suggested simpler models, which are totally valid. I went with DistilBERT because keyword-based features alone miss a lot of context. A message like *"We need you to verify your identity to keep your account secure"* contains no single alarming word, but the intent is clearly suspicious. Transformers handle that better.

The tradeoff is inference latency (~200ms vs near-instant for LR), which is worth it here since this isn't a real-time stream processor.

---

## Architecture

The system runs three checks in sequence:

**1. Rules Layer**
Catches obvious signals immediately — malicious TLDs (`.xyz`, `.biz`), known phishing URL patterns, or flagged domains. If something hits here, we don't even bother with the model.

**2. Heuristics Layer**
Looks at linguistic signals: urgency density, credential-request patterns, uppercase ratio, special character frequency. Also includes some aviation-specific BEC patterns (crew portals, flight schedule spoofs) since I used a custom dataset slice for that domain.

**3. Model Layer**
Fine-tuned DistilBERT handles the cases that slip past rules and heuristics — obfuscated phrasing, novel attack patterns, or messages that read normally but carry phishing intent in context.

---

## Dataset

- Base: public phishing datasets from Kaggle (~80k samples, balanced)
- Extended with a small custom set of aviation-domain BEC examples I synthesized manually (~500 samples)
- Preprocessing: tokenization, stopword removal, lemmatization via NLTK

The model performance numbers below are on a held-out test split, not the full corpus.

---

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.15% |
| Precision | 98.49% |
| Recall | 97.80% |
| F1 | 98.14% |
| ROC-AUC | 0.998 |

These are solid numbers, though I'd note that phishing datasets can be relatively separable — real-world performance on novel attacks would likely be lower. The confusion matrix and full eval logs are in `src/evaluate.py`.

---

## Project Structure

```
phishguard/
├── src/
│   ├── preprocess.py        # Data cleaning and NLP pipeline
│   ├── features.py          # Feature extraction (urgency, credentials, etc.)
│   ├── train.py             # DistilBERT fine-tuning
│   ├── evaluate.py          # Metrics, confusion matrix
│   └── keyword_detector.py  # Keyword scan and highlighting
├── models/
│   └── best_model/          # Saved DistilBERT weights
├── frontend/
│   └── src/App.js           # React dashboard
├── app.py                   # Flask API
├── docker-compose.yml
└── requirements.txt
```

---

## Running it

### Docker (easier)

```bash
docker-compose up --build
```

- Dashboard: `http://localhost:3000`
- API: `http://localhost:5001`

### Local

```bash
# Set up environment
source venv_3.11/bin/activate
pip install -r requirements.txt

# Terminal 1
python app.py

# Terminal 2
cd frontend && npm start
```

---

## API

**POST** `/analyze`

```json
{
  "text": "Your account has been suspended. Verify your password immediately at secure-login.xyz"
}
```

Response:
```json
{
  "prediction": "Phishing",
  "confidence": 0.97,
  "keywords_detected": ["verify", "password", "immediately"],
  "url_signals": ["suspicious TLD: .xyz"]
}
```

---

## Stack

- **Model:** HuggingFace Transformers (DistilBERT)
- **Backend:** Python, Flask
- **Frontend:** React
- **NLP:** NLTK, Scikit-learn (TF-IDF features)
- **Infra:** Docker, Docker Compose

---

## What I'd improve with more time

- Add an active learning loop to flag uncertain predictions for human review
- Better calibration on the confidence scores (Platt scaling or temperature scaling)
- Swap Flask for FastAPI for async support
- The aviation dataset is small — more domain-specific data would help that slice meaningfully

---

*Built by Papiya Mazumder — open to feedback or questions about the approach.*