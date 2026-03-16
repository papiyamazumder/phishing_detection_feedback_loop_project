"""
app.py  — Flask REST API for PhishGuard AI
-------------------------------------------
Serves the trained DistilBERT model as a REST API for aviation and
enterprise phishing detection (Aviation-Enterprise style security platform).

ENDPOINTS:
  POST /api/predict       — classify a message (main endpoint)
  GET  /api/health        — health check
  GET  /api/keywords      — return the full phishing keyword lexicon
  GET  /api/demo          — return example messages for UI demo buttons

TRI-LEVEL CLASSIFICATION:
  0–30 % confidence  → Legitimate   (LOW risk)
  30–70 % confidence → Suspicious   (MEDIUM risk) — mixed-content messages
  70–100 % confidence → Phishing    (HIGH risk)

HYBRID SCORING (3-layer decision system):
  Layer 1 — Hard override:   Suspicious URL evidence → always phishing
  Layer 2 — BEC pattern:     BEC + urgency + credential → always phishing
  Layer 3 — Weighted blend:  50 % ML confidence + 50 % rule risk score
"""

import os, sys, json, time, re, email, logging, csv
import numpy as np
import torch
import PyPDF2
from src.preprocess import parse_eml_content, clean_text
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess      import preprocess_for_distilbert, parse_eml_content
from keyword_detector import scan_text
from features        import extract_all_features

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_model")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("phishguard-api")

# ── GLOBAL MODEL (loaded once at startup) ─────────────────────────────────────
_tokenizer = None
_model     = None


def load_model():
    """Load DistilBERT model and tokenizer into memory with Dynamic Quantization."""
    global _tokenizer, _model
    if _tokenizer is None:
        logger.info(f"Loading model from {MODEL_DIR}...")
        t0 = time.time()
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        base_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        
        # PRODUCTION OPTIMIZATION: Dynamic Quantization (INT8)
        # Reduces model size (~260MB -> ~130MB) and speeds up CPU inference by 2-4x.
        if DEVICE.type == "cpu":
            try:
                logger.info("Applying Dynamic Quantization (INT8) for CPU inference...")
                # Ensure the quantization engine is set correctly for the environment
                if sys.platform == "darwin": # Mac
                    torch.backends.quantized.engine = 'qnnpack'
                else:
                    torch.backends.quantized.engine = 'fbgemm'

                _model = torch.quantization.quantize_dynamic(
                    base_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Dynamic Quantization applied successfully.")
            except Exception as qe:
                logger.warning(f"Dynamic Quantization failed, falling back to base model: {qe}")
                _model = base_model
        else:
            _model = base_model

        _model.to(DEVICE)
        _model.eval()
        logger.info(f"Model loaded in {time.time()-t0:.2f}s  (device: {DEVICE}, quantized: {DEVICE.type == 'cpu'})")


def model_predict(text: str) -> tuple[int, float]:
    """
    Run one inference pass through DistilBERT.

    Returns:
        label:      0 (legitimate) or 1 (phishing)
        confidence: P(phishing) — float 0.0–1.0
    """
    clean = preprocess_for_distilbert(text)

    encoding = _tokenizer(
        clean,
        max_length     = 128,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1).squeeze()

    phishing_prob = probs[1].item()     # P(phishing)
    label         = int(probs.argmax().item())

    return label, phishing_prob


# ── RISK CLASSIFICATION ───────────────────────────────────────────────────────

def classify_risk(confidence: float, kw_risk: float,
                  has_suspicious_url: bool) -> tuple[str, str]:
    """
    Determine prediction label and risk level from combined signals.

    Risk tiers:
      0–30 %  → Legitimate (LOW)
      30–70 % → Suspicious (MEDIUM) — mixed content / ambiguous
      70–100 % → Phishing (HIGH)
    """
    combined = (confidence * 0.60) + (kw_risk * 0.40)

    if combined >= 0.70 or has_suspicious_url:
        return "Phishing", "HIGH"
    elif combined >= 0.30:
        return "Suspicious", "MEDIUM"
    else:
        return "Legitimate", "LOW"


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check — used by React to verify API is running."""
    model_loaded = _model is not None
    return jsonify({
        "status":       "ok",
        "model_loaded": model_loaded,
        "device":       str(DEVICE),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint with tri-level classification.

    Request body (JSON):
      { "text": "Your message here..." }

    Response body (JSON):
      {
        "label":            1,
        "prediction":       "Phishing" | "Suspicious" | "Legitimate",
        "confidence":       0.94,
        "confidence_pct":   94,
        "risk_level":       "HIGH" | "MEDIUM" | "LOW",
        "override_reason":  "suspicious_url" | "bec_pattern" | "weighted_score",
        "keywords":         { ... DetectionResult ... },
        "features":         { ... structural + aviation features ... },
        "processing_ms":    47
      }
    """
    t0 = time.time()

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain 'text' field"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text exceeds 5000 character limit"}), 400

    # ── EML Detection & Parsing ──────────────────────────────────────────────
    # If the text starts with common EML headers, parse it to extract clean Body
    is_raw_eml = any(text.lstrip().upper().startswith(h) for h in ["DELIVERED-TO:", "RECEIVED:", "RETURN-PATH:", "FROM:"])
    if is_raw_eml:
        logger.info("Raw EML detected, parsing content...")
        text = parse_eml_content(text)

    # ── Model prediction ──────────────────────────────────────────────────────
    try:
        label, confidence = model_predict(text)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    # ── Keyword detection (rule-based layer) ──────────────────────────────────
    kw_result = scan_text(text)

    # ── Structural + aviation features ────────────────────────────────────────
    features = extract_all_features(text)

    # ── HYBRID SCORING (3-layer decision) ─────────────────────────────────────
    #
    # Layer 1 — Hard override (URL evidence):
    #   Suspicious URLs (.biz/.info/.xyz with phishing keywords in domain)
    #   → ALWAYS flag as phishing. Real IT departments never use .biz domains.
    #
    # Layer 2 — BEC pattern override:
    #   BEC category + urgency + credential request → override to phishing.
    #
    # Layer 3 — Weighted combination:
    #   50 % ML + 50 % rules (equal weight).
    #
    has_suspicious_url = len(kw_result.url_signals) > 0
    has_bec_pattern    = "bec_spear_phishing" in kw_result.categories
    has_aviation       = "aviation_sector"    in kw_result.categories
    has_enterprise     = "enterprise_sector"  in kw_result.categories
    has_threat         = "threat_suspension"   in kw_result.categories
    has_urgency        = "urgency"             in kw_result.categories
    has_credential     = "credential_harvesting" in kw_result.categories

    # Base dynamic blend (50% ML + 50% Rule Risk)
    combined = (confidence * 0.50) + (kw_result.risk_score * 0.50)

    # Layer 1: Suspicious URL hard override
    if has_suspicious_url:
        final_label      = 1
        # Add a dynamic 25% bump for severe URL threat, capped at 99%
        final_confidence = min(combined + 0.25, 0.99)
        override_reason  = "suspicious_url"

    # Layer 2: BEC / aviation spear phishing pattern
    elif (has_bec_pattern or has_aviation or has_enterprise) and \
         (has_urgency or has_threat) and has_credential:
        final_label      = 1
        # Add a dynamic 15% bump for BEC threats
        final_confidence = min(combined + 0.15, 0.99)
        override_reason  = "bec_pattern"

    # Layer 3: Weighted combination
    else:
        final_label      = 1 if combined >= 0.45 else label
        final_confidence = combined
        override_reason  = "weighted_score"

    # ── Tri-level prediction label ────────────────────────────────────────────
    prediction, risk_level = classify_risk(
        final_confidence, kw_result.risk_score, has_suspicious_url
    )

    # Override label integer to match prediction
    if prediction == "Phishing":
        final_label = 1
    elif prediction == "Legitimate":
        final_label = 0
    # "Suspicious" keeps final_label as-is (could be 0 or 1)

    processing_ms = int((time.time() - t0) * 1000)

    response = {
        "label":           final_label,
        "prediction":      prediction,
        "confidence":      round(final_confidence, 4),
        "confidence_pct":  round(final_confidence * 100, 1),
        "risk_level":      risk_level,
        "override_reason": override_reason,
        "keywords": {
            "found":            kw_result.found_keywords,
            "categories":       kw_result.categories,
            "risk_score":       round(kw_result.risk_score, 3),
            "highlighted_text": kw_result.highlighted_text,
            "is_suspicious":    kw_result.is_suspicious,
            "url_signals":      kw_result.url_signals,
        },
        "features": {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in features.items()
        },
        "processing_ms":   processing_ms,
        "text_length":     len(text),
    }

    return jsonify(response)


@app.route("/api/parse-file", methods=["POST"])
def parse_file():
    """
    Endpoint to parse .eml, .pdf, and .txt files.
    Returns the extracted plain text body.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename.lower()
    
    # Check file size (2MB limit)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > 2 * 1024 * 1024:
        return jsonify({"error": "File size exceeds 2MB limit"}), 400
    file.seek(0)

    try:
        content = ""
        if filename.endswith('.eml'):
            raw_bytes = file.read()
            # Try to decode the whole file as string first to use our preprocess helper
            try:
                raw_str = raw_bytes.decode("utf-8", errors="ignore")
            except:
                raw_str = str(raw_bytes)
            
            content = parse_eml_content(raw_str)
            # Remove any residual Quoted-Printable artifacts if helper missed them
            content = re.sub(r'=\r?\n', '', content)
            content = re.sub(r'=[0-9A-F]{2}', '', content)

        elif filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            content = "\n".join(p.extract_text() for p in reader.pages).strip()

        elif filename.endswith('.txt'):
            content = file.read().decode("utf-8", errors="ignore").strip()

        else:
            return jsonify({"error": "Unsupported file format. Please use .eml, .pdf, or .txt"}), 400

        return jsonify({
            "text": content,
            "filename": file.filename
        })

    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 500


@app.route("/api/keywords", methods=["GET"])
def get_keywords():
    """Return the full phishing keyword lexicon (for UI display)."""
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from keyword_detector import PHISHING_KEYWORDS
    return jsonify({
        "categories": list(PHISHING_KEYWORDS.keys()),
        "total_patterns": sum(len(v) for v in PHISHING_KEYWORDS.values()),
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """
    Collect user feedback for misclassified messages.
    Supports simple Thumbs Up/Down + Comments.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data or "is_correct" not in data:
        return jsonify({"error": "Missing required fields (text, is_correct)"}), 400

    text       = data["text"].strip()
    is_correct = data["is_correct"] # True (Up) or False (Down)
    comment    = data.get("comment", "").strip()
    prediction = data.get("prediction", "Unknown")
    confidence = data.get("confidence", 0.0)

    feedback_file = os.path.join(BASE_DIR, "data", "feedback.csv")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    file_exists = os.path.isfile(feedback_file)
    
    try:
        with open(feedback_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "text", "is_correct", "prediction", "confidence", "comment"])
            
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                text.replace("\n", " "),
                is_correct,
                prediction,
                confidence,
                comment.replace("\n", " ")
            ])
        
        status_msg = "Upvote" if is_correct else "Downvote"
        logger.info(f"Feedback received: {status_msg} | Comment: {comment[:30]}...")
        return jsonify({"status": "success", "message": "Feedback saved. Thank you for helping us improve!"})
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")
        return jsonify({"error": "Failed to save feedback"}), 500


@app.route("/api/demo", methods=["GET"])
def demo():
    """Return example messages for UI demo buttons — includes aviation scenarios."""
    examples = [
        {
            "label": "phishing",
            "text": "URGENT: Your Chase account has been suspended! Verify your password immediately at http://secure-verify.now.biz or your account will be permanently deleted within 24 hours!"
        },
        {
            "label": "phishing",
            "text": "Urgent: Verify your airline crew portal credentials immediately. Your flight assignment access will be revoked if not completed by end of day. Use the secure link: http://crew-portal-login.biz/verify"
        },
        {
            "label": "suspicious",
            "text": "Hi Team,\nAs part of our quarterly IT security review, we are rolling out several updates to improve internal data protection.\n• New password rotation policy (every 90 days)\n• Updated VPN access guidelines\nPlease confirm your account activity using this portal: http://account-security-review.biz/employee/login\nRegards, Michael Carter\nIT Security Operations"
        },
        {
            "label": "legitimate",
            "text": "Hi Sarah, just a reminder that our weekly team meeting is scheduled for Wednesday at 2:00 PM in Conference Room B. Please bring your Q3 progress report. Let me know if you can make it."
        },
        {
            "label": "legitimate",
            "text": "Flight OPS Notice: Aircraft B737-800 (VT-AXR) maintenance check completed successfully. All airworthiness directives complied with. Aircraft cleared for next scheduled departure at 0600 UTC."
        },
    ]
    return jsonify({"examples": examples})


# ── STARTUP ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    app.run(
        host    = "0.0.0.0",
        port    = 5001,
        debug   = False,
    )