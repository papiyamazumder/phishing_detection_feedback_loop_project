"""
preprocess.py
-------------
WHAT THIS MODULE DOES:
  Converts raw email/SMS text into a clean, normalized form
  before feature extraction and model training.

KEY NLP CONCEPTS EXPLAINED:
  1. Tokenization   - splitting text into individual words/tokens
  2. Lowercasing    - "URGENT" and "urgent" should be the same token
  3. Stop words     - common words ("the", "is", "at") that add noise
  4. Lemmatization  - reducing words to root form: "verifying" → "verify"
  5. Punctuation/URL cleaning - removing noise that doesn't carry meaning

WHY PREPROCESS FOR DISTILBERT?
  DistilBERT has its own internal tokenizer (WordPiece), so it handles
  sub-word splitting automatically. However, cleaning URLs, HTML artifacts,
  and excessive whitespace still improves quality significantly.

  For the hand-crafted features (keyword counts, special char counts),
  preprocessing is critical — these features are extracted BEFORE
  DistilBERT sees the text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (runs once, cached after)
nltk.download("punkt",          quiet=True)
nltk.download("punkt_tab",      quiet=True)
nltk.download("stopwords",      quiet=True)
nltk.download("wordnet",        quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# ── SINGLETON OBJECTS (expensive to create, reuse them) ──────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# We KEEP these stop words because they appear in phishing patterns:
# "now", "free", "click" are not informative in normal English but ARE
# in phishing context. Removing them would hurt our keyword features.
KEEP_WORDS = {"now", "free", "click", "limited", "urgent", "immediately",
              "account", "verify", "update", "confirm", "suspend", "expires"}
_stop_words -= KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Stage 1 — Raw cleaning.
    Removes URLs, emails, HTML tags, quoted replies, signatures,
    and normalizes whitespace.

    Handles real-world email formatting including:
    - Quoted replies (lines starting with '>')
    - Email signatures (text after '--' or 'Regards,')
    - Email headers (From:, Subject:, To:, Date:)

    CONCEPT — Why remove URLs?
      A phishing URL like http://secure-verify.now.biz carries no semantic
      meaning after cleaning. DistilBERT has never seen these scam domains
      in pre-training, so they'd just be [UNK] tokens. Better to replace
      them with the token 'URL' so the model learns: "URL presence = signal".
    """
    if not isinstance(text, str):
        return ""

    # Strip email headers (From:, To:, Subject:, Date:, CC:, BCC:)
    text = re.sub(r"^(From|To|Cc|Bcc|Date|Subject|Reply-To|Sent):.*$",
                  " ", text, flags=re.MULTILINE | re.IGNORECASE)

    # Strip quoted reply lines (lines starting with > or >>)
    text = re.sub(r"^>+.*$", " ", text, flags=re.MULTILINE)

    # Strip "On ... wrote:" forwarding markers
    text = re.sub(r"On .{10,80} wrote:\s*$", " ", text, flags=re.MULTILINE)

    # Strip email signatures (text after common sign-off patterns)
    text = re.sub(r"\n--\s*\n.*", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"\n(Best regards|Kind regards|Regards|Thanks|Sincerely|Cheers),?\s*\n.*",
        " ", text, flags=re.DOTALL | re.IGNORECASE
    )

    # Replace URLs with placeholder token (preserves the SIGNAL that a URL exists)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)

    # Replace email addresses
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # Strip HTML tags (some phishing emails contain raw HTML)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace (multiple spaces → single space)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_normalize(text: str) -> list[str]:
    """
    Stage 2 — Tokenization + normalization.
    Returns a list of clean, lowercased, lemmatized tokens.

    CONCEPT — WordNetLemmatizer:
      Unlike stemming (which just chops endings: "running" → "runn"),
      lemmatization uses a vocabulary dictionary to return valid words:
        "verifying" → "verify"
        "accounts"  → "account"
        "suspended" → "suspend"
      This groups semantically identical words, reducing vocab size.
    """
    # Lowercase everything
    text = text.lower()

    # Tokenize into words
    tokens = word_tokenize(text)

    # Remove pure punctuation tokens (keep alphanumeric)
    tokens = [t for t in tokens if t not in string.punctuation]

    # Remove stop words (KEEP_WORDS are excluded from _stop_words above)
    tokens = [t for t in tokens if t not in _stop_words]

    # Lemmatize each token
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    # Remove very short tokens (single characters add noise)
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


def preprocess_for_features(text: str) -> str:
    """
    Returns a clean string (joined tokens) used for keyword matching
    and TF-IDF vectorization.

    NOTE: This is DIFFERENT from what DistilBERT receives.
          DistilBERT gets the minimally-cleaned text (clean_text only),
          NOT the fully lemmatized version — its tokenizer is trained
          to handle natural language, not pre-processed tokens.
    """
    cleaned = clean_text(text)
    tokens  = tokenize_and_normalize(cleaned)
    return " ".join(tokens)


def preprocess_for_distilbert(text: str) -> str:
    """
    Returns text for DistilBERT — only URL/HTML cleaning, no lemmatization.
    DistilBERT's WordPiece tokenizer handles subwords better on natural text.
    """
    return clean_text(text)


def batch_preprocess(texts: list[str], for_distilbert: bool = False) -> list[str]:
    """
    Process a list of texts. Set for_distilbert=True when preparing
    inputs for the transformer model.
    """
    fn = preprocess_for_distilbert if for_distilbert else preprocess_for_features
    return [fn(t) for t in texts]


# ── QUICK DEMO ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "URGENT: Your Chase account has been SUSPENDED! Verify your password immediately at http://secure-phish.now.biz or lose access FOREVER!!!"

    print("=" * 60)
    print("ORIGINAL TEXT:")
    print(sample)
    print("\nCLEANED (for features):")
    print(preprocess_for_features(sample))
    print("\nCLEANED (for DistilBERT):")
    print(preprocess_for_distilbert(sample))
    print("=" * 60)
