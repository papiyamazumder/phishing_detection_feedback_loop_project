# PhishGuard AI — Aviation & Enterprise Security Intelligence

> **Production-grade phishing detection system powered by DistilBERT transformers and rule-based heuristics.**

---

## 📋 Overview

PhishGuard AI is an end-to-end security solution designed for **aviation safety, compliance, and enterprise operational communication** (tailored for QMSmart-style environments). It identifies malicious intent in emails and messages by combining deep learning transformer models with a multi-layered rule engine.

### Key Capabilities
- **DistilBERT Intelligence:** Fine-tuned transformer model (98.15% accuracy) for deep semantic analysis.
- **Aviation Domain Focus:** Specific detection for crew portal phishing, flight schedule BEC, and DGCA/FAA/ICAO compliance alerts.
- **Tri-Level Verdicts:** Classifies messages as **Legitimate**, **Suspicious**, or **Phishing** with confidence-based risk tiers.
- **Hybrid Scoring:** 3-layer decision system (URL Signature → BEC Heuristics → Weighted ML Blend).
- **Enterprise Dashboard:** Professional React UI featuring real-time risk gauges, URL signal analysis, and annotated payload traces.

---

## 🏗️ System Architecture

### 1. Hybrid Detection Pipeline
The system uses a 3-layer "Defense in Depth" approach:
- **Layer 1 (Hard Override):** Detects high-confidence malicious indicators (e.g., `.biz` domains mimicking corporate portals). Signals here trigger an immediate "Phishing" verdict.
- **Layer 2 (BEC Pattern):** Checks for structural Business Email Compromise (BEC) patterns combining source spoofing, urgency, and credential requests.
- **Layer 3 (Weighted Blend):** Combines DistilBERT confidence (50%) with hand-crafted rule-based risk (50%) to produce a final probability score.

### 2. Project Structure
```
phishing-detector/
├── app.py                       ← Flask REST API (tri-level classification)
├── src/
│   ├── preprocess.py            ← Aviation-aware text cleaning (strips headers/signatures)
│   ├── keyword_detector.py      ← Rule-based scanner (9 categories, including Aviation)
│   ├── features.py              ← Hand-crafted feature extraction (Aviation & Enterprise signals)
│   ├── train.py                 ← DistilBERT fine-tuning pipeline
│   └── evaluate.py              ← Performance metrics & diagnostic plots
├── models/
│   └── best_model/              ← Fine-tuned DistilBERT weights
├── frontend/
│   ├── src/
│   │   ├── App.js               ← Enterprise Dashboard UI
│   │   └── App.css              ← Premium Security Aesthetic
│   └── public/index.html
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Deployment

### 1. Core Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Dataset Initialization (Optional)
If starting fresh, download the training dataset (+30,000 samples):
```bash
python data/download_dataset.py
```

### 3. Start Secure API
The backend serves on port 5000:
```bash
python app.py
```

### 4. Start Dashboard
```bash
cd frontend
npm install
npm start
```
*Accessible at http://localhost:3000*

---

## 📊 Performance & Accuracy

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.15% |
| **Precision (Phishing)** | 98.49% |
| **Recall (Phishing)** | 97.80% |
| **F1 Score** | 98.14% |
| **ROC-AUC** | 0.9983 |

---

## 🔬 Skills & Technologies Demonstrated

- **Deep Learning:** Fine-tuning DistilBERT (Transformers library).
- **Advanced NLP:** Custom preprocessing for noisy enterprise communication (NLTK, RegEx).
- **Feature Engineering:** Domain-specific vocabulary extraction (Aviation/Enterprise).
- **Backend Engineering:** Multi-layered decision systems in Flask.
- **Frontend Excellence:** Premium React dashboard development with real-time feedback.
- **UI/UX Design:** Dark-mode security aesthetic with data visualization components.

---

## 🛡️ Security Use Case: QMSmart Integration
Designed to protect aviation personnel (pilots, cabin crew, maintenance engineers) from targeted spear-phishing such as:
- **Crew Portal Phishing:** Fake login pages for roster/payroll.
- **Flight Ops BEC:** Malicious instructions disguised as maintenance or schedule updates.
- **Regulatory Impersonation:** Fake compliance alerts from DGCA, FAA, or EASA.

---
*Developed as a high-quality technical submission for Senior AI Engineering roles.*
