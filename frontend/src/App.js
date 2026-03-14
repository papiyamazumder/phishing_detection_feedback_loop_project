import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_BASE = "http://localhost:5001/api";

const AVIATION_DEMO = [
  {
    type: "phishing",
    label: "✈️ Crew Portal",
    text: "Urgent: Verify your airline crew portal credentials immediately. Your flight assignment access will be revoked if not completed by end of day. Use the secure link: http://crew-portal-login.biz/verify",
  },
  {
    type: "phishing",
    label: "📅 Flight Schedule",
    text: "Flight schedule update requires immediate account verification. Log in to the crew management portal before your next assignment to avoid roster delays: http://crew-mgmt-portal.xyz/login",
  },
  {
    type: "suspicious",
    label: "🔒 IT Security",
    text: "Hi Team,\nAs part of our quarterly IT security review, we are rolling out several updates to improve internal data protection.\n• New password rotation policy (every 90 days)\n• Updated VPN access guidelines\nPlease confirm your account activity using this portal: http://account-security-review.biz/employee/login\nRegards, Michael Carter\nIT Security Operations",
  },
  {
    type: "legit",
    label: "✅ Ops Notice",
    text: "Flight OPS Notice: Aircraft B737-800 (VT-AXR) maintenance check completed successfully. All airworthiness directives (ADs) complied with. Aircraft cleared for next scheduled departure at 0600 UTC.",
  },
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

function RiskGauge({ confidence }) {
  const angle = (confidence * 180) - 90;
  const pct = Math.round(confidence * 100);

  let color = "#22c55e"; // Legit
  if (pct >= 70) color = "#ff4444"; // Phish
  else if (pct >= 30) color = "#f97316"; // Suspicious

  return (
    <div className="risk-gauge-wrap">
      <div className="gauge-container">
        <div className="gauge-bg"></div>
        <div className="gauge-fill" style={{
          background: `conic-gradient(from -90deg at 50% 100%, ${color} ${pct * 1.8}deg, var(--bg3) 0)`
        }}></div>
        <div className="gauge-needle" style={{ transform: `rotate(${angle}deg)` }}></div>
        <div className="gauge-center">
          <span className="gauge-value" style={{ color }}>{pct}%</span>
          <span className="gauge-label">RISK</span>
        </div>
      </div>
      <div className="gauge-scales">
        <span>LEGIT</span>
        <span>SUSPICIOUS</span>
        <span>PHISHING</span>
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

function FeatureMeter({ label, value, max, format }) {
  const pct = Math.min((value / max) * 100, 100);
  const formatted = format === "pct" ? `${value.toFixed(1)}%` :
    format === "int" ? Math.round(value) : value.toFixed(1);
  return (
    <div className="feat-row">
      <span className="feat-label">{label}</span>
      <div className="feat-bar-wrap">
        <div className="feat-bar" style={{ width: `${pct}%` }} />
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
  const resultRef = useRef(null);

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
    <div className="app">
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
          <div className="hero-badge">Aviation Compliance · QMSmart Intelligence</div>
          <h2>Threat Detection Hub</h2>
          <p className="hero-sub">
            Enterprise-grade phishing analysis for aviation communication.
            Identify credential harvesting, BEC, and malicious payloads across operations.
          </p>
        </section>

        {/* ── DEMO BUTTONS ── */}
        <section className="demo-section">
          <span className="demo-label">Threat Scenarios:</span>
          <div className="demo-btns">
            {AVIATION_DEMO.map((d, i) => (
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
              <span className="char-count">{text.length} / 5000 chars</span>
            </div>
            <textarea
              className="msg-input"
              placeholder="Paste suspicious aviation communication, crew alerts, or enterprise emails here..."
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, 5000))}
              onKeyDown={handleKeyDown}
              rows={8}
            />
            <div className="input-footer">
              <button
                className="clear-btn"
                onClick={() => { setText(""); setResult(null); setError(null); }}
                disabled={!text && !result}
              >
                Clear Terminal
              </button>
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
                  Confidence: <strong>{result.confidence_pct}%</strong>
                  &nbsp;·&nbsp;
                  Analysis Latency: {result.processing_ms}ms
                </p>
              </div>
              <RiskBadge level={result.risk_level} />
            </div>

            <div className="results-grid">
              {/* Left Column */}
              <div className="results-col">
                {/* Risk Gauge */}
                <div className="result-card">
                  <h4>Risk Probability Gauge</h4>
                  <RiskGauge confidence={result.confidence} />
                </div>

                {/* Threat Summary */}
                <div className="result-card">
                  <h4>Threat Intelligence Summary</h4>
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
                  <div className="result-card">
                    <h4>Malicious Link Signatures</h4>
                    <UrlSignals signals={result.keywords.url_signals} />
                  </div>
                )}
              </div>

              {/* Right Column */}
              <div className="results-col">
                {/* Keyword detection */}
                {result.keywords.is_suspicious && (
                  <div className="result-card">
                    <h4>Phishing Pattern Identification</h4>
                    <div className="kw-categories">
                      {Object.entries(result.keywords.categories).map(([cat, kws]) => (
                        <KeywordCategory key={cat} name={cat} keywords={kws} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Highlighted text */}
                {result.keywords.highlighted_text && result.keywords.is_suspicious && (
                  <div className="result-card">
                    <h4>Annotated Payload Trace</h4>
                    <HighlightedText text={result.keywords.highlighted_text} />
                    <p className="highlight-legend">
                      <span className="mark-indicator"></span> = Identified Threat Pattern
                    </p>
                  </div>
                )}

                {/* Feature breakdown */}
                <div className="result-card">
                  <h4>Signal Strength Breakdown</h4>
                  <div className="features-grid">
                    <FeatureMeter label="Aviation Domain Sig" value={result.features.aviation_phish_score} max={10} />
                    <FeatureMeter label="Enterprise Sig" value={result.features.enterprise_phish_score} max={10} />
                    <FeatureMeter label="Contextual Combo" value={result.features.contextual_combo_score} max={10} />
                    <FeatureMeter label="Urgency Pressure" value={result.features.urgency_score} max={10} />
                    <FeatureMeter label="Credential Signals" value={result.features.credential_score} max={10} />
                    <FeatureMeter label="Suspicious URLs" value={result.features.url_count} max={5} format="int" />
                    <FeatureMeter label="Uppercase Ratio" value={result.features.uppercase_count} max={10} format="int" />
                    <FeatureMeter label="Special Char Count" value={result.features.special_char_count} max={15} format="int" />
                  </div>
                </div>
              </div>
            </div>

            {/* No keywords legit message */}
            {!result.keywords.is_suspicious && result.prediction === "Legitimate" && (
              <div className="safe-card">
                <span className="safe-icon">🛡️</span>
                <p>No actionable threat patterns detected. Message aligns with standard enterprise communication profiles.</p>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>
            PHISHGUARD SECURE INTELLIGENCE · POWERED BY DISTILBERT-TRANSFORMER (L6-H768)
          </p>
          <p className="footer-sub">
            Deployed Cluster: QMS-SAFE-01 · Aviation Compliance Engine v2.1.0
          </p>
        </div>
      </footer>
    </div>
  );
}
