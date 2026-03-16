import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_BASE = "http://localhost:5001/api";

const DEMO_MESSAGES = [
  {
    type: "staff",
    label: "✈️ Fake Staff Login",
    text: `Our system has detected multiple failed login attempts on your staff account over the past 24 hours. As a precautionary measure, all employees are required to re-verify their credentials through our secure helpdesk portal before end of business today to avoid temporary suspension of system and roster access. Please confirm your identity here: http://staff-account-verify.biz/employee/login`
  },

  {
    type: "schedule",
    label: "📅 Fake Schedule Alert",
    text: `Your upcoming duty schedule has been updated in our system. Due to a recent platform migration, you are required to log in and re-confirm your schedule details to ensure accurate records before the next rostering cycle. Failure to confirm your details by end of day may result in scheduling conflicts and temporary loss of access to your duty assignments. Verify here: http://schedule-update-portal.info/account/verify`
  },

  {
    type: "it",
    label: "🔒 Fake IT Warning",
    text: `As part of our quarterly IT security review, our monitoring system detected several unusual login attempts across employee accounts last night. While most were automatically blocked, all staff are required to review their recent account activity and reset their credentials before end of business today to avoid any interruption to email and system access. Complete your verification here: http://security-review-portal.net/staff/credentials`
  },

  {
    type: "notice",
    label: "✅ Fake Company Notice",
    text: `As part of our annual payroll system migration, all employees are required to re-verify their personal and banking details to ensure uninterrupted salary processing for the upcoming payment cycle. Please confirm your information before this Friday to avoid any delay in your salary disbursement. Update your details here: http://hr-payroll-staff-verify.biz/employee/banking`
  },

  {
    type: "general",
    label: "🌐 Miscellaneous Threats",
    text: `Congratulations! You have been selected as one of our loyalty reward winners for this quarter. As a valued customer, you are eligible to claim a gift voucher worth $500 as a token of our appreciation. This offer is strictly limited and expires within the next 24 hours, so act now to avoid missing out on your reward: http://loyalty-rewards-claim.biz/voucher/redeem`
  }
];

function HighlightedText({ text }) {
  if (!text) return null;
  const parts = text.split(/(<<[^>]+>>)/g);
  return (
    <div className="highlighted-text">
      {parts.map((part, i) => {
        if (part.startsWith("<<") && part.endsWith(">>")) {
          return (
            <mark key={i} className="phish-mark">
              {part.slice(2, -2)}
            </mark>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </div>
  );
}

function ConfidenceBar({ confidence, label }) {
  const pct = Math.round(confidence * 100);

  let background = "linear-gradient(90deg, #00C8FF, #00D68F)";
  if (pct >= 71) {
    background = "linear-gradient(90deg, #FFB800, #FF3B3B)";
  } else if (pct >= 41) {
    background = "#FFB800"; // Warning amber
  }

  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-labels">
        <span>Safe</span>
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
      <div className="conf-bar-container">
        <div className="conf-bar-bg" />
        <div className="conf-bar-fill" style={{ width: `${pct}%`, background }} />
        <div className="conf-bar-needle" style={{ left: `${pct}%`, backgroundColor: "#ffffff" }} />
      </div>
      <div className="conf-bar-value" style={{ color: label === 1 ? "#FF3B3B" : "#00D68F" }}>
        {pct}% Phishing Risk
      </div>
      <div className="conf-bar-description" style={{ color: label === 1 ? "#FF3B3B" : "#00D68F" }}>
        {label === 1 ? "⚠️ This message is likely phishing" : "✓ This message appears safe"}
      </div>
    </div>
  );
}

function RiskBadge({ level }) {
  return (
    <span className={`risk-badge risk-${level.toLowerCase()}`}>
      {level === "HIGH" ? "🚨" : level === "MEDIUM" ? "⚠️" : "🛡️"} {level} RISK
    </span>
  );
}

function UrlSignals({ signals }) {
  if (!signals || signals.length === 0) return null;
  return (
    <div className="url-signals">
      <h5>Suspicious Links Detected</h5>
      {signals.map((sig, i) => (
        <div key={i} className="url-signal-card">
          <div className="url-text">{sig.url}</div>
          <div className="url-reasons">
            {sig.reasons.map((r, j) => (
              <span key={j} className="url-reason-tag">{r}</span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function KeywordCategory({ name, keywords }) {
  const icons = {
    urgency: "⏱️",
    credential_harvesting: "🔑",
    threat_suspension: "🔒",
    bec_spear_phishing: "🎭",
    aviation_sector: "✈️",
    enterprise_sector: "🏢",
    prize_lure: "🎁",
    financial: "💳",
    suspicious_links: "🔗",
  };

  const labels = {
    urgency: "Urgency Indicators",
    credential_harvesting: "Credential Harvesting",
    threat_suspension: "Threat / Suspension",
    bec_spear_phishing: "BEC / Spear Phishing",
    aviation_sector: "Aviation Domain Signals",
    enterprise_sector: "Enterprise IT/HR Signals",
    prize_lure: "Prize / Lure",
    financial: "Financial Signals",
    suspicious_links: "Suspicious URLs",
  };

  return (
    <div className="kw-category">
      <span className="kw-cat-title">
        {icons[name] || "◉"} {labels[name] || name}
      </span>
      <div className="kw-chips">
        {keywords.map((kw, i) => (
          <span key={i} className="kw-chip">{kw}</span>
        ))}
      </div>
    </div>
  );
}

function FeatureMeter({ label, value, max = 100, format }) {
  const pct = Math.min((value / max) * 100, 100);
  const formatted = format === "pct" ? `${value.toFixed(1)}%` :
    format === "int" ? Math.round(value) :
      `${value.toFixed(0)}%`;

  return (
    <div className="feat-row">
      <span className="feat-label">{label}</span>
      <div className="feat-bar-wrap">
        <div className="feat-bar" style={{ width: `${pct}%`, opacity: value > 0 ? 1 : 0.3 }} />
      </div>
      <span className="feat-value">{formatted}</span>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("unknown");
  const [feedbackStatus, setFeedbackStatus] = useState(null); // null, 'sending', 'sent', 'error'
  const [feedbackComment, setFeedbackComment] = useState("");
  const resultRef = useRef(null);
  const fileInputRef = useRef(null);

  const [isDark, setIsDark] = useState(
    () => localStorage.getItem('theme') !== 'light'
  );

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Client-side size check (2MB)
    if (file.size > 2 * 1024 * 1024) {
      setError("File size exceeds 2MB limit.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/parse-file`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "File parsing failed");
      }

      const data = await res.json();
      setText(data.text.slice(0, 5000));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      e.target.value = null; // reset input
    }
  };

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setApiStatus(data.model_loaded ? "online" : "loading");
    } catch {
      setApiStatus("offline");
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const analyze = async (inputText) => {
    const msg = (inputText ?? text).trim();
    if (!msg) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: msg }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
      setApiStatus("online");
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (e) {
      setError(e.message.includes("fetch") ?
        "Security System Offline. Ensure the backend analyzer is running." :
        e.message
      );
    } finally {
      setLoading(false);
      setFeedbackStatus(null);
      setFeedbackComment("");
    }
  };

  const submitFeedback = async (isCorrect) => {
    if (!result || !text) return;
    setFeedbackStatus('sending');

    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          is_correct: isCorrect,
          comment: feedbackComment,
          prediction: result.prediction,
          confidence: result.confidence
        }),
      });

      if (!res.ok) throw new Error("Feedback submission failed");
      setFeedbackStatus('sent');
      setFeedbackComment("");
    } catch (e) {
      setFeedbackStatus('error');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && text.trim() && !loading) {
      e.preventDefault();
      analyze();
    }
  };

  const loadDemo = (demo) => {
    setText(demo.text);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app" data-theme={isDark ? 'dark' : 'light'}>
      {/* ── HEADER ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">📡</div>
            <div>
              <h1>PhishGuard AI</h1>
              <p>Aviation & Enterprise Security Intelligence</p>
            </div>
          </div>
          <div className="header-actions">
            <div className="system-indicator">
              <span className="label">SEC-LEVEL:</span>
              <span className="value">CLASS-A</span>
            </div>
            <button className="theme-toggle-btn" onClick={toggleTheme}>
              {isDark ? '☀️' : '🌙'}
            </button>
            <button
              className={`status-btn status-${apiStatus}`}
              onClick={checkHealth}
            >
              <span className="status-dot" />
              {apiStatus === "online" ? "ANALYZER ONLINE" :
                apiStatus === "offline" ? "SYSTEM OFFLINE" :
                  apiStatus === "loading" ? "MODEL SYNCING..." : "RETRY CONNECT"}
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── HERO ── */}
        <section className="hero">
          <div className="hero-badge">Aviation Compliance · Enterprise Intelligence</div>
          <h2>Threat Detection Hub</h2>
          <p className="hero-sub">
            Enterprise-grade phishing analysis for aviation communication.
            Identify credential harvesting, BEC, and malicious payloads across operations.
          </p>
        </section>

        {/* ── DEMO BUTTONS ── */}
        <section className="demo-section">
          <span className="demo-label">Test with Real-World Phishing Scenarios:</span>
          <div className="demo-btns">
            {DEMO_MESSAGES.map((d, i) => (
              <button
                key={i}
                className={`demo-btn demo-${d.type}`}
                onClick={() => loadDemo(d)}
              >
                {d.label}
              </button>
            ))}
          </div>
        </section>

        {/* ── INPUT ── */}
        <section className="input-section">
          <div className="input-card">
            <div className="input-header">
              <span className="input-title">MESSAGE PAYLOAD ANALYZER</span>
              <div className="input-header-utils">
                <span className="char-count">{text.length} / 5000 chars</span>
                <button
                  className="clear-btn"
                  onClick={() => { setText(""); setResult(null); setError(null); }}
                  disabled={!text && !result}
                  title="Clear all text and results"
                >
                  🗑️ CLEAR
                </button>
              </div>
            </div>
            <textarea
              className="msg-input"
              placeholder={`Paste any suspicious email, SMS, or message directly in this box to scan for phishing threats.
Include the subject line and any URLs for better accuracy.
Or upload a .eml, .txt, or .pdf file using the button below.`}
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, 5000))}
              onKeyDown={handleKeyDown}
              rows={8}
            />
            <div className="input-footer">
              <div className="input-actions-left">
                <input
                  type="file"
                  accept=".txt,.eml,.pdf"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  style={{ display: "none" }}
                />
                <div className="upload-wrapper">
                  <button
                    className="upload-btn"
                    onClick={() => fileInputRef.current?.click()}
                    title="Upload text-based files (.txt, .eml, etc.)"
                  >
                    📁 Upload File
                  </button>
                  <span className="upload-hint">Supported: .txt, .eml, .pdf</span>
                </div>
              </div>
              <button
                className={`analyze-btn ${loading ? 'loading' : ''}`}
                onClick={() => analyze()}
                disabled={!text.trim() || loading}
              >
                {loading ? (
                  <span className="btn-loading">
                    <span className="spinner" /> SCANNING PAYLOAD...
                  </span>
                ) : (
                  "INITIATE ANALYSIS >>"
                )}
              </button>
            </div>
          </div>
        </section>

        {/* ── ERROR ── */}
        {error && (
          <div className="error-card">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* ── RESULTS ── */}
        {result && (
          <section className="results" ref={resultRef}>
            {/* Verdict banner */}
            <div className={`verdict-banner verdict-${result.prediction.toLowerCase()}`}>
              <div className="verdict-icon">
                {result.prediction === "Phishing" ? "🚨" : result.prediction === "Suspicious" ? "⚠️" : "🛡️"}
              </div>
              <div className="verdict-text">
                <h3>DETECTION: {result.prediction.toUpperCase()}</h3>
                <p>
                  Phishing Probability: <strong>{Math.round(result.confidence * 100)}%</strong>
                  &nbsp;·&nbsp;
                  Analysis Latency: {result.processing_ms}ms
                </p>
              </div>
              <RiskBadge level={result.risk_level} />
            </div>

            <div className={`results-grid ${!result.keywords.is_suspicious ? 'single-col' : ''}`}>
              {/* Left Column */}
              <div className="results-col">
                {/* Risk Gauge */}
                <div className="result-card card-risk">
                  <h4>📈 Phishing Risk Level</h4>
                  <ConfidenceBar confidence={result.confidence} label={result.label} />
                </div>

                {/* Threat Summary */}
                <div className="result-card card-summary">
                  <h4>🧠 Threat Intelligence Summary</h4>
                  <div className="threat-summary">
                    <div className="threat-item">
                      <span className="t-label">Inference Logic:</span>
                      <span className="t-value">{result.override_reason === 'suspicious_url' ? 'URL Signature Match (Manual Override)' : result.override_reason === 'bec_pattern' ? 'Heuristic BEC Pattern Match' : 'Statistical ML Weighted Blend'}</span>
                    </div>
                    <div className="threat-item">
                      <span className="t-label">Vector Analysis:</span>
                      <span className="t-value">{result.label === 1 ? 'Malicious' : 'Safe/Ambiguous'}</span>
                    </div>
                  </div>
                </div>

                {/* URL Signals */}
                {result.keywords.url_signals && result.keywords.url_signals.length > 0 && (
                  <div className="result-card card-urls">
                    <h4>🔗 Malicious Link Signatures</h4>
                    <UrlSignals signals={result.keywords.url_signals} />
                  </div>
                )}

                {/* Feature breakdown */}
                <div className="result-card card-features">
                  <h4>⚡ Risk Signal Breakdown</h4>
                  <div className="features-grid">
                    <FeatureMeter label="Domain Risk Score" value={result.features.aviation_phish_score} />
                    <FeatureMeter label="Threat Score" value={result.features.enterprise_phish_score} />
                    <FeatureMeter label="Pattern Match" value={result.features.contextual_combo_score} />
                    <FeatureMeter label="Urgency Score" value={result.features.urgency_score} />
                    <FeatureMeter label="Password Theft" value={result.features.credential_score} />
                    <FeatureMeter label="Suspicious Links" value={result.features.suspicious_tld_count} max={5} format="int" />
                    <FeatureMeter label="Uppercase Words" value={result.features.uppercase_count} max={10} format="int" />
                    <FeatureMeter label="Special Characters" value={result.features.special_char_count} max={15} format="int" />
                  </div>
                </div>
              </div>

              {/* Right Column - Only rendered if suspicious content exists */}
              {result.keywords.is_suspicious && (
                <div className="results-col">
                  {/* Keyword detection */}
                  <div className="result-card card-patterns">
                    <h4>🎯 Phishing Pattern Identification</h4>
                    <div className="kw-categories">
                      {Object.entries(result.keywords.categories).map(([cat, kws]) => (
                        <KeywordCategory key={cat} name={cat} keywords={kws} />
                      ))}
                    </div>
                  </div>

                  {/* Highlighted text */}
                  {result.keywords.highlighted_text && (
                    <div className="result-card card-trace">
                      <h4>📝 Annotated Payload Trace</h4>
                      <HighlightedText text={result.keywords.highlighted_text} />
                      <p className="highlight-legend">
                        <span className="mark-indicator"></span> = Identified Threat Pattern
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Dynamic Result Summary Cards */}
            {result.prediction === "Phishing" && (
              <div className="threat-card phish-summary">
                <span className="threat-icon">🚨</span>
                <p>Critical threat signatures detected. High correlation with known credential harvesting or BEC patterns identified in this sector.</p>
              </div>
            )}

            {result.prediction === "Suspicious" && (
              <div className="threat-card suspicious-summary">
                <span className="threat-icon">⚠️</span>
                <p>Anomalous signals identified. Content contains secondary threat layers or unverified redirection paths requiring manual review.</p>
              </div>
            )}

            {result.prediction === "Legitimate" && (
              <div className="safe-card">
                <span className="safe-icon">🛡️</span>
                <p>No actionable threat patterns detected. Message aligns with standard enterprise and aviation communication profiles.</p>
              </div>
            )}

            {/* ── FEEDBACK LOOP ── */}
            <div className="feedback-section">
              <div className="feedback-inner">
                {feedbackStatus === 'sent' ? (
                  <div className="feedback-thanks">
                    <span className="check-icon">✓</span>
                    <p>Feedback received! Thank you for helping us improve.</p>
                  </div>
                ) : (
                  <>
                    <p className="feedback-prompt">Was this analysis accurate?</p>
                    <div className="feedback-actions-wrap">
                      <div className="feedback-thumbs">
                        <button
                          className="thumb-btn up"
                          onClick={() => submitFeedback(true)}
                          disabled={feedbackStatus === 'sending'}
                          title="This result is correct"
                        >
                          👍
                        </button>
                        <button
                          className="thumb-btn down"
                          onClick={() => submitFeedback(false)}
                          disabled={feedbackStatus === 'sending'}
                          title="This result is incorrect"
                        >
                          👎
                        </button>
                      </div>

                      <div className="feedback-comment-box">
                        <textarea
                          placeholder="What did you like or dislike? (optional)"
                          value={feedbackComment}
                          onChange={(e) => setFeedbackComment(e.target.value.slice(0, 500))}
                          disabled={feedbackStatus === 'sending'}
                        />
                      </div>

                      {feedbackStatus === 'error' && <span className="feedback-err">Submission failed.</span>}
                    </div>
                  </>
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>
            PhishGuard AI · Phishing Detection & Analysis Platform · © 2026 Aviation Enterprise Technologies
          </p>
          <p className="footer-sub">
            Powered by DistilBERT Transformer · Built with React, Flask & PyTorch
          </p>
        </div>
      </footer>
    </div>
  );
}
