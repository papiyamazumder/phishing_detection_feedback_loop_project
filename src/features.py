"""
features.py
-----------
Extracts numerical features from text for the PhishGuard AI pipeline.

Designed for **aviation safety & compliance environments** (QMSmart-style).
Features supplement the DistilBERT model with transparent, explainable signals.

FEATURE CATEGORIES:
  1. Structural features      — message length, special chars, uppercase words
  2. Keyword features          — phishing vocabulary density per category
  3. URL / link features       — URL count, suspicious TLD detection
  4. Aviation-specific features — aviation / crew / compliance phishing vocab
  5. Enterprise features       — corporate IT / HR / finance phishing vocab
  6. Contextual combinations   — dangerous keyword co-occurrence patterns
  7. TF-IDF features           — statistical word importance (for training)

CONCEPT — TF-IDF:
  Term Frequency × Inverse Document Frequency.
  TF(t,d)  = count(t in d) / total_words(d)
  IDF(t,D) = log(total_docs / docs_containing_t)
  TFIDF    = TF × IDF
  → "verify" scores high in phishing emails; "the" scores low everywhere.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ── PHISHING KEYWORD LISTS ────────────────────────────────────────────────────
# Shorter than keyword_detector.py — these are for feature COUNTING, not
# comprehensive detection.

URGENCY_WORDS = [
    "urgent", "immediately", "now", "asap", "expires", "deadline",
    "final notice", "act now", "right now", "today only", "24 hours",
    "limited time", "time sensitive"
]

CREDENTIAL_WORDS = [
    "verify", "confirm", "update", "password", "credential",
    "login", "sign in", "authenticate", "validate", "reactivate"
]

THREAT_WORDS = [
    "suspended", "locked", "blocked", "compromised", "unauthorized",
    "suspicious", "unusual", "terminated", "deleted", "disabled"
]

FINANCIAL_WORDS = [
    "bank", "credit card", "debit card", "wire transfer", "account",
    "payment", "billing", "refund", "prize", "reward", "gift card"
]

LURE_WORDS = [
    "congratulations", "won", "winner", "selected", "free",
    "claim", "prize", "reward", "lucky", "exclusive"
]

# ── AVIATION-SPECIFIC VOCABULARY ──────────────────────────────────────────────
AVIATION_WORDS = [
    "crew portal", "pilot portal", "flight schedule", "flight roster",
    "crew management", "flight operations", "crew login", "airline portal",
    "dgca", "faa", "easa", "icao", "civil aviation", "airworthiness",
    "maintenance portal", "aircraft system", "crew verification",
    "flight assignment", "aviation compliance", "recurrent training",
    "air operator", "crew roster", "per diem", "duty hours",
]

# ── ENTERPRISE IT / HR / FINANCE VOCABULARY ───────────────────────────────────
ENTERPRISE_WORDS = [
    "it department", "helpdesk", "office 365", "sharepoint", "teams",
    "multi-factor", "mfa", "two-factor", "2fa",
    "hr department", "human resources", "payroll", "direct deposit",
    "benefits enrollment", "open enrollment", "w-2", "tax document",
    "corporate security", "compliance training", "data breach",
    "purchase order", "expense report", "invoice", "reimbursement",
]

# ── DANGEROUS KEYWORD COMBINATIONS ───────────────────────────────────────────
# Pairs that, when both present, strongly indicate phishing intent.
DANGEROUS_COMBOS = [
    ("verify", "immediately"),
    ("verify", "account"),
    ("portal", "credentials"),
    ("login", "immediately"),
    ("security", "alert"),
    ("reset", "password"),
    ("suspended", "verify"),
    ("update", "credentials"),
    ("confirm", "identity"),
    ("crew portal", "verify"),
    ("flight schedule", "verify"),
    ("compliance", "portal"),
    ("payroll", "update"),
    ("direct deposit", "verify"),
    ("invoice", "approval"),
    ("urgent", "click"),
    ("expire", "login"),
]

SUSPICIOUS_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq", ".biz",
                   ".info", ".work", ".click", ".top", ".xyz"]


def extract_structural_features(text: str) -> dict:
    """
    Extract observable structural properties of the message.

    CONCEPT — Why these matter:
      • Phishing emails tend to be SHORT (quick fear hook) or LONG (fake bank template)
      • UPPERCASE WORDS signal alarm ("URGENT", "WARNING", "ACT NOW")
      • Special chars like ! $ % are overused in phishing ("You WON $1000!!!")
      • URLs in a message shift prior probability toward phishing
    """
    if not text:
        return {k: 0 for k in ["msg_length", "word_count", "uppercase_count",
                                "special_char_count", "exclamation_count",
                                "url_count", "dollar_sign_count",
                                "has_ip_url", "avg_word_length"]}

    words           = text.split()
    uppercase_words = [w for w in words if w.isupper() and len(w) > 1]
    urls            = re.findall(r"https?://\S+", text, re.IGNORECASE)
    ip_urls         = re.findall(r"https?://\d+\.\d+\.\d+\.\d+", text)
    special_chars   = re.findall(r"[!$%@#&*()]", text)

    return {
        "msg_length":         len(text),
        "word_count":         len(words),
        "uppercase_count":    len(uppercase_words),
        "special_char_count": len(special_chars),
        "exclamation_count":  text.count("!"),
        "url_count":          len(urls),
        "dollar_sign_count":  text.count("$"),
        "has_ip_url":         int(len(ip_urls) > 0),
        "avg_word_length":    np.mean([len(w) for w in words]) if words else 0,
    }


def extract_keyword_features(text: str) -> dict:
    """
    Count occurrences of phishing vocabulary per category.
    Returns NORMALIZED counts (per 100 words) to avoid bias toward
    longer messages.
    """
    if not text:
        return {k: 0.0 for k in ["urgency_score", "credential_score",
                                  "threat_score", "financial_score", "lure_score"]}

    lower   = text.lower()
    n_words = max(len(text.split()), 1)

    def score(word_list):
        count = sum(1 for w in word_list if w in lower)
        return (count / n_words) * 100   # per-100-words normalization

    return {
        "urgency_score":    score(URGENCY_WORDS),
        "credential_score": score(CREDENTIAL_WORDS),
        "threat_score":     score(THREAT_WORDS),
        "financial_score":  score(FINANCIAL_WORDS),
        "lure_score":       score(LURE_WORDS),
    }


def extract_aviation_features(text: str) -> dict:
    """Score aviation-specific and enterprise phishing vocabulary."""
    if not text:
        return {"aviation_phish_score": 0.0, "enterprise_phish_score": 0.0,
                "contextual_combo_score": 0.0}

    lower   = text.lower()
    n_words = max(len(text.split()), 1)

    def score(word_list):
        count = sum(1 for w in word_list if w in lower)
        return (count / n_words) * 100

    # Contextual combination score — counts how many dangerous pairs co-occur
    combo_hits = sum(1 for a, b in DANGEROUS_COMBOS if a in lower and b in lower)
    combo_score = min(combo_hits * 2.0, 10.0)  # cap at 10

    return {
        "aviation_phish_score":    score(AVIATION_WORDS),
        "enterprise_phish_score":  score(ENTERPRISE_WORDS),
        "contextual_combo_score":  combo_score,
    }


def extract_url_features(text: str) -> dict:
    """Detect URL-based signals."""
    lower = text.lower()
    urls  = re.findall(r"https?://\S+", lower)
    suspicious = sum(1 for u in urls if any(tld in u for tld in SUSPICIOUS_TLDS))

    return {
        "suspicious_tld_count": suspicious,
        "has_url":              int(len(urls) > 0),
        "url_to_text_ratio":    len(urls) / max(len(text.split()), 1),
    }


def extract_all_features(text: str) -> dict:
    """Combine all feature extractors into one flat feature dict."""
    feats = {}
    feats.update(extract_structural_features(text))
    feats.update(extract_keyword_features(text))
    feats.update(extract_aviation_features(text))
    feats.update(extract_url_features(text))
    return feats


def build_feature_matrix(texts: list[str]) -> np.ndarray:
    """
    Build the full feature matrix for a list of texts.
    Returns shape (n_samples, n_features).
    """
    rows = [extract_all_features(t) for t in texts]
    df   = pd.DataFrame(rows)
    return df.values, df.columns.tolist()


# ── TF-IDF VECTORIZER ─────────────────────────────────────────────────────────

class TFIDFFeaturizer:
    """
    Wraps sklearn's TfidfVectorizer with save/load support.

    CONCEPT — Why TF-IDF instead of bag-of-words?
      Bag-of-words counts raw word frequencies. TF-IDF penalizes
      common words like "the" and "and" that appear everywhere —
      they carry no discriminative information for phishing detection.

    PARAMETERS:
      max_features=5000  — top 5000 most important vocab terms
      ngram_range=(1,2)  — unigrams ("click") + bigrams ("click here")
                           bigrams capture multi-word phishing phrases
      sublinear_tf=True  — use log(1+tf) instead of raw tf to dampen
                           effect of very frequent terms
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features  = max_features,
            ngram_range   = (1, 2),        # unigrams + bigrams
            sublinear_tf  = True,          # log-frequency scaling
            min_df        = 2,             # ignore terms that appear < 2 times
            strip_accents = "unicode",
        )
        self.fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit on training data and return TF-IDF matrix."""
        matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return matrix.toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform new texts using the already-fitted vocabulary."""
        if not self.fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        return self.vectorizer.transform(texts).toarray()

    def save(self, path: str):
        joblib.dump(self.vectorizer, path)
        print(f"TF-IDF vectorizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TFIDFFeaturizer":
        obj = cls()
        obj.vectorizer = joblib.load(path)
        obj.fitted = True
        return obj


# ── QUICK DEMO ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "URGENT: Your Chase account has been SUSPENDED! Verify your password at http://secure-phish.now.biz immediately!!!"

    print("=" * 60)
    print("STRUCTURAL FEATURES:")
    for k, v in extract_structural_features(sample).items():
        print(f"  {k:<25} {v}")

    print("\nKEYWORD FEATURES:")
    for k, v in extract_keyword_features(sample).items():
        print(f"  {k:<25} {v:.3f}")

    print("\nURL FEATURES:")
    for k, v in extract_url_features(sample).items():
        print(f"  {k:<25} {v}")
    print("=" * 60)
