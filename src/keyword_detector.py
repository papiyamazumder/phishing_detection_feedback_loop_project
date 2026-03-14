"""
keyword_detector.py
-------------------
Rule-based phishing keyword scanner — 9 categories.

Designed for enterprise environments including **aviation safety, compliance,
and operational communication** (e.g., QMSmart-style platforms).

CATEGORIES:
  1. urgency              — time pressure tactics
  2. credential_harvesting — password / login / verify signals
  3. threat_suspension     — account lock / suspension threats
  4. bec_spear_phishing    — Business Email Compromise patterns
  5. prize_lure            — lottery / prize scams
  6. financial             — banking / payment signals
  7. suspicious_links      — URL / TLD anomalies
  8. aviation_sector       — aviation-specific phishing patterns   ← NEW
  9. enterprise_sector     — corporate IT/HR/finance phishing     ← NEW

WHY RULES + ML?
  DistilBERT votes on the whole email — when 85 % of tokens are clean
  business language, one malicious URL gets outvoted.  Rules catch
  SPECIFIC signals with 100 % recall regardless of surrounding text.

  Hybrid score = (ML confidence × 0.50) + (rule risk × 0.50)
"""

import re
from dataclasses import dataclass, field


PHISHING_KEYWORDS: dict[str, list[str]] = {

    # ── 1. URGENCY ────────────────────────────────────────────────────────────
    "urgency": [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bright now\b",
        r"\bact now\b",
        r"\baction required\b",
        r"\bfinal notice\b",
        r"\blast chance\b",
        r"\bexpires?\b",
        r"\b24 hours?\b",
        r"\bwithin \d+ (hour|minute|day)",
        r"\bdeadline\b",
        r"\btime.?sensitive\b",
        r"\bbefore end of day\b",
        r"\bby end of (day|business)\b",
        r"\btoday (only|to avoid|or)\b",
        r"\bno later than\b",
        r"\bavoid interruption\b",
        r"\bavoid (being )?locked out\b",
        r"\bbefore (tomorrow|monday|the deadline)\b",
    ],

    # ── 2. CREDENTIAL HARVESTING ──────────────────────────────────────────────
    "credential_harvesting": [
        r"\bverify (your )?(account|identity|password|email|credentials|details|access)\b",
        r"\bconfirm (your )?(account|identity|password|credentials|details|activity)\b",
        r"\bupdate (your )?(password|credentials|billing|payment|account|details)\b",
        r"\breset (your )?password\b",
        r"\bre-?authenticate\b",
        r"\bre-?verify\b",
        r"\benter (your )?(password|credentials|details|ssn)\b",
        r"\bprovide (your )?(bank|account|personal) (details|information|number)\b",
        r"\blogin (to|at|here|below|using|with)\b",
        r"\bsign in (to|at|here|below|using|with)\b",
        r"\bupdate (your )?credentials\b",
        r"\bconfirm (your )?account activity\b",
        r"\breview (your )?(login|account|credentials)\b",
    ],

    # ── 3. THREAT / SUSPENSION ────────────────────────────────────────────────
    "threat_suspension": [
        r"\baccount (will be |has been )?(suspended|locked|blocked|disabled|terminated|deleted)\b",
        r"\bpermanent(ly)? (suspend|block|lock|delete|disable)\b",
        r"\blose (access|your account)\b",
        r"\bservice (will be |has been )?terminated\b",
        r"\b(unusual|suspicious|unauthori[sz]ed) (activity|login|access|attempt)\b",
        r"\bcompromised\b",
        r"\bfailed login attempt",
        r"\bmultiple (failed|unsuccessful) (login|sign.?in) attempt",
        r"\bpotential lockout\b",
        r"\binterruption of (email |account )?access\b",
        r"\baccount (may be |will be )?restricted\b",
        r"\bsecurity (breach|incident|alert|warning|issue)\b",
        r"\bdetected.{0,30}(unusual|suspicious|unauthori[sz]ed)\b",
    ],

    # ── 4. BEC / SPEAR PHISHING ───────────────────────────────────────────────
    "bec_spear_phishing": [
        r"\btemporary (portal|link|url|page|site|access)\b",
        r"\bsecurity portal\b",
        r"\bemployee (portal|login|verification|access)\b",
        r"\bIT (security |support )?(portal|helpdesk|notification|department)\b",
        r"\bmandatory (security|compliance|verification|update|training)\b",
        r"\brolling out.{0,40}(update|change|policy|patch)\b",
        r"\bquarterly (security|IT|compliance|review)\b",
        r"\bpassword (rotation|expir|reset|update) (policy|required|reminder)\b",
        r"\bVPN (access|guideline|policy|update|change)\b",
        r"\bsecurity awareness (training|program)\b",
        r"\bclick (the |this )?(link|button|here|below) (to|and) (verify|confirm|reset|update|access)\b",
        r"\buse (the |this )?link below\b",
        r"\bportal below\b",
        r"\baccess (the |your )?(portal|link|dashboard) below\b",
        r"\bfollowing link\b",
        r"\bsecure (link|portal|page|access)\b",
        r"\bdata protection.{0,30}(update|review|policy)\b",
        r"\binternal (data|security|system) (protection|review|update)\b",
    ],

    # ── 5. PRIZE / LURE ───────────────────────────────────────────────────────
    "prize_lure": [
        r"\bcongratulations?\b",
        r"\byou (have |'ve )?(won|been selected)\b",
        r"\bclaim (your )?(prize|reward|gift|money|winnings)\b",
        r"\bfree (gift|reward|prize|money|offer)\b",
        r"\bspecial (offer|reward|prize|discount|selection)\b",
        r"\bunclaimed (reward|prize|money|funds)\b",
        r"\blottery\b",
        r"\bwinnings?\b",
    ],

    # ── 6. FINANCIAL ──────────────────────────────────────────────────────────
    "financial": [
        r"\bwire transfer\b",
        r"\bbank account\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bpayment (failed|declined|required)\b",
        r"\btax refund\b",
        r"\bpending (payment|transfer|funds)\b",
        r"\$\d+[\d,]*",
    ],

    # ── 7. SUSPICIOUS LINKS ───────────────────────────────────────────────────
    "suspicious_links": [
        r"\bclick here\b",
        r"\bclick (the )?link\b",
        r"\bfollow (this |the )?link\b",
        # Suspicious TLDs — commonly used in phishing domains
        r"https?://[^\s]*\.(biz|info|xyz|work|click|top|tk|ml|ga|cf|gq|zip|mov)\b",
        # Hyphenated domains mimicking corporate IT (e.g. account-security-review.biz)
        r"https?://[a-z0-9]+(-[a-z0-9]+){2,}\.(biz|info|net|org|xyz|com)\b",
        # IP-based URLs
        r"https?://\d+\.\d+\.\d+\.\d+",
        # Subdomain impersonation (e.g. security.company-support-helpdesk.net)
        r"https?://[^\s]*(security|account|login|verify|portal|helpdesk|support)[^\s]*\.(biz|info|net|xyz)\b",
        r"\bshort(ened)? (url|link)\b",
    ],

    # ── 8. AVIATION SECTOR (NEW) ──────────────────────────────────────────────
    # Phishing patterns targeting aviation personnel: pilots, crew, engineers,
    # dispatchers, and compliance officers.
    "aviation_sector": [
        # Crew portal / login phishing
        r"\b(crew|pilot|cabin crew|flight crew) (portal|login|access|verification)\b",
        r"\bcrew (management|scheduling|roster) (portal|system|update)\b",
        r"\bverify (your )?(crew|pilot|airline|flight) (portal|credentials|access|account)\b",
        # Flight schedule / operations phishing
        r"\bflight (schedule|operation|assignment|roster) (update|change|requires?|verification)\b",
        r"\bflight (ops|operations) (portal|access|system|login)\b",
        r"\b(pre.?flight|post.?flight) (check|report|verification) (portal|system|required)\b",
        # Maintenance portal phishing
        r"\b(aircraft|maintenance|MRO) (system|portal|access|login) (update|required|verification)\b",
        r"\bmaintenance (log|record|report) (portal|system|access)\b",
        # Aviation authority / compliance phishing
        r"\b(DGCA|FAA|EASA|ICAO|CAA|civil aviation) (compliance|authority|regulation|portal|update)\b",
        r"\baviation (authority|compliance|safety|regulation) (portal|update|alert|notice)\b",
        r"\bcivil aviation (authority|compliance|portal|regulation)\b",
        r"\bairworthiness (directive|certificate|update|portal)\b",
        # Aviation payroll / roster / HR
        r"\b(airline|aviation|crew) (payroll|roster|schedule|bid) (portal|update|login|verification)\b",
        r"\b(layover|per diem|flight pay|duty hours?) (portal|update|verification|login)\b",
        # AOC / license phishing
        r"\b(AOC|air operator|operating certificate|pilot license) (renewal|update|verification|portal)\b",
        r"\brecurrent (training|check|assessment) (portal|login|required|overdue)\b",
    ],

    # ── 9. ENTERPRISE SECTOR (NEW) ────────────────────────────────────────────
    # General corporate phishing patterns across IT, HR, and finance.
    "enterprise_sector": [
        # IT / Password resets
        r"\b(IT department|IT team|system administrator|helpdesk) (requires?|notification|alert)\b",
        r"\bpassword (expir|reset|change|update) (required|immediately|today|by)\b",
        r"\b(Office ?365|O365|SharePoint|Teams|Outlook) (access|login|update|verification)\b",
        r"\b(multi.?factor|MFA|two.?factor|2FA) (enrollment|setup|required|verification|update)\b",
        # HR / Payroll phishing
        r"\b(HR|human resources|payroll) (department|team|portal|update|verification|notice)\b",
        r"\b(direct deposit|salary|compensation|benefits|W-?2|tax document) (update|verification|portal|review)\b",
        r"\b(open enrollment|benefits enrollment|annual review) (portal|period|required|deadline)\b",
        # Corporate security
        r"\bcorporate (security|compliance|policy) (update|portal|review|alert|notification)\b",
        r"\b(data breach|security incident|compliance violation) (notification|alert|reported|detected)\b",
        r"\b(annual|quarterly) compliance (training|review|certification|assessment)\b",
        # Fake invoice / purchase order
        r"\b(invoice|purchase order|PO) (#?\d+)?.{0,20}(review|approval|pending|attached)\b",
        r"\b(expense|reimbursement|travel) (report|claim|portal|submission) (required|pending|overdue)\b",
    ],
}


@dataclass
class DetectionResult:
    found_keywords:   list       = field(default_factory=list)
    categories:       dict       = field(default_factory=dict)
    risk_score:       float      = 0.0
    highlighted_text: str        = ""
    url_signals:      list       = field(default_factory=list)   # extracted suspicious URLs

    @property
    def is_suspicious(self) -> bool:
        return len(self.found_keywords) > 0

    def to_dict(self) -> dict:
        return {
            "found_keywords":   self.found_keywords,
            "categories":       self.categories,
            "risk_score":       round(self.risk_score, 3),
            "highlighted_text": self.highlighted_text,
            "is_suspicious":    self.is_suspicious,
            "url_signals":      self.url_signals,
        }


def extract_suspicious_urls(text: str) -> list[str]:
    """
    Extract and score all URLs in the text.
    Returns list of suspicious URLs found.

    SUSPICIOUS URL SIGNALS:
      - Scam TLDs: .biz .info .xyz .tk .ml .ga .zip .mov
      - Hyphenated subdomains: account-security-review.biz
      - IP-based URLs
      - Keywords in domain: security, account, login, verify, helpdesk, portal
      - Non-HTTPS (http://) for sensitive operations
    """
    SCAM_TLDS    = {".biz", ".info", ".xyz", ".tk", ".ml", ".ga", ".cf",
                    ".gq", ".zip", ".mov", ".work", ".click", ".top"}
    PHISH_WORDS  = {"security", "account", "login", "verify", "portal",
                    "helpdesk", "support", "reset", "employee", "update",
                    "confirm", "auth", "secure", "credential"}

    all_urls = re.findall(r"https?://[^\s\)]+", text, re.IGNORECASE)
    suspicious = []

    for url in all_urls:
        url_lower = url.lower()
        reasons   = []

        # Check TLD
        for tld in SCAM_TLDS:
            if tld in url_lower:
                reasons.append(f"suspicious TLD ({tld})")
                break

        # Hyphenated domain (3+ parts before TLD = likely fake)
        domain_match = re.search(r"https?://([^/\s]+)", url_lower)
        if domain_match:
            domain = domain_match.group(1).split(":")[0]
            parts  = domain.replace("www.", "").split(".")
            host   = parts[0] if parts else ""
            if host.count("-") >= 2:
                reasons.append("hyphenated fake domain")
            # Phishing keywords in domain name
            for pw in PHISH_WORDS:
                if pw in domain:
                    reasons.append(f"'{pw}' in domain")
                    break

        # IP-based URL
        if re.search(r"https?://\d+\.\d+\.\d+\.\d+", url_lower):
            reasons.append("IP-based URL")

        if reasons:
            suspicious.append({"url": url, "reasons": reasons})

    return suspicious


def scan_text(text: str) -> DetectionResult:
    """
    Main entry point — scans text for all phishing signals.

    RISK SCORE FORMULA:
      Base: 1 - (0.65 ^ n_categories_hit)   [independent probability combination]
      URL boost: +0.25 per suspicious URL found (capped at 0.95)
      BEC boost: extra +0.15 if bec_spear_phishing category fires

    This ensures a single .biz URL in an otherwise clean email still
    gets a high risk score — addressing the spear phishing blind spot.
    """
    result     = DetectionResult()
    lower_text = text.lower()
    all_matches = []

    for category, patterns in PHISHING_KEYWORDS.items():
        cat_hits = []
        for pattern in patterns:
            for match in re.finditer(pattern, lower_text):
                keyword = match.group(0).strip()
                cat_hits.append(keyword)
                all_matches.append((match.start(), match.end(), keyword))

        if cat_hits:
            unique_hits = list(dict.fromkeys(cat_hits))
            result.categories[category]  = unique_hits
            result.found_keywords.extend(unique_hits)

    result.found_keywords = list(dict.fromkeys(result.found_keywords))

    # ── URL signal extraction ─────────────────────────────────────────────────
    result.url_signals = extract_suspicious_urls(text)

    # ── RISK SCORE ────────────────────────────────────────────────────────────
    c = len(result.categories)
    base_risk = 1.0 - (0.65 ** c) if c > 0 else 0.0

    # URL boost — each suspicious URL adds 0.25 (this catches spear phishing)
    url_boost = min(0.35, len(result.url_signals) * 0.25)

    # BEC category boost — spear phishing specific
    bec_boost = 0.15 if "bec_spear_phishing" in result.categories else 0.0

    result.risk_score = min(0.95, base_risk + url_boost + bec_boost)

    # ── HIGHLIGHTING ─────────────────────────────────────────────────────────
    # Also highlight suspicious URLs even if not caught by keyword regex
    for url_info in result.url_signals:
        url = url_info["url"]
        idx = text.find(url)
        if idx >= 0:
            all_matches.append((idx, idx + len(url), url))

    result.highlighted_text = _build_highlighted(text, all_matches)
    return result


def _build_highlighted(text: str, matches: list[tuple]) -> str:
    """Wrap detected keywords/URLs in << >> markers for React highlighting."""
    if not matches:
        return text

    matches = sorted(set(matches), key=lambda m: m[0])

    merged = [list(matches[0])]
    for start, end, kw in matches[1:]:
        if start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, kw])

    result = text
    for start, end, _ in reversed(merged):
        result = result[:start] + "<<" + result[start:end] + ">>" + result[end:]
    return result


if __name__ == "__main__":
    # Test with the exact failing email
    test = """Hello Team,
As part of our quarterly IT security review, we are rolling out several updates to improve internal data protection.
• New password rotation policy (every 90 days) • Updated VPN access guidelines • Mandatory security awareness training
Our monitoring system detected several unusual login attempts across multiple employee accounts late last night.
You can review your recent login history using the temporary security portal below:
http://account-security-review.biz/employee/login
Please ensure this is completed before end of day today to avoid interruption of email access tomorrow morning.
Regards, Michael Carter IT Security Operations"""

    r = scan_text(test)
    print(f"Suspicious:    {r.is_suspicious}")
    print(f"Risk Score:    {r.risk_score:.2f}")
    print(f"Categories:    {list(r.categories.keys())}")
    print(f"URL Signals:   {r.url_signals}")
    print(f"Keywords:      {r.found_keywords[:8]}")