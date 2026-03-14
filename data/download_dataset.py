"""
download_dataset.py
--------------------
Downloads phishing dataset from Kaggle using credentials from .env file.

PRIMARY DATASET:
  naserabdullahalam/phishing-email-dataset
  ~82,000 real emails from 6 corpora:
    - Enron       (real corporate legit emails)
    - Nazario     (real scraped phishing)
    - CEAS 2008   (spam/phishing competition)
    - Ling Spam   (classic benchmark)
    - Nigerian    (advance fee fraud)
    - SpamAssassin(diverse spam + ham)

  WHY THIS DATASET?
  These are REAL emails — including subtle spear phishing like:
  "Hi team, agenda for tomorrow... reset your password at http://evil.net"
  Our synthetic data could never replicate this nuance.

FALLBACK ORDER:
  1. subhajournal/phishingemails       (Kaggle — 18k emails)
  2. zefang-liu/phishing-email-dataset (HuggingFace — no key needed)
  3. Synthetic data                    (always works offline)

SETUP:
  1. Copy .env.example → .env in project root
  2. Add KAGGLE_USERNAME and KAGGLE_KEY
  3. pip install kaggle
  4. python data/download_dataset.py

USAGE:
  python data/download_dataset.py                # auto mode
  python data/download_dataset.py --synthetic    # force synthetic
  python data/download_dataset.py --huggingface  # force HuggingFace
"""

import os
import sys
import json
import argparse
import tempfile
import random

import pandas as pd

# ── LOAD .env BEFORE ANYTHING ELSE ────────────────────────────────────────────

def load_env_file() -> bool:
    """
    Parse .env from project root → inject into os.environ.
    Must run before `import kaggle` so the library sees the credentials.

    Priority: existing env vars > .env file > ~/.kaggle/kaggle.json
    """
    env_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".env")
    )

    if not os.path.exists(env_path):
        print(f"  ! .env not found at {env_path}")
        print(f"    Create it from .env.example and add KAGGLE_USERNAME + KAGGLE_KEY")
        return False

    loaded = []
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.environ.get(key):
                os.environ[key] = value
                loaded.append(key)

    if loaded:
        print(f"  ✓ .env loaded → set: {', '.join(loaded)}")
    else:
        print(f"  ✓ .env found (variables already in environment)")
    return True


print("\n[.env] Loading Kaggle credentials...")
_env_loaded = load_env_file()

# ── CONFIG ─────────────────────────────────────────────────────────────────────

OUTPUT_PATH      = os.path.join(os.path.dirname(__file__), "dataset.csv")
PRIMARY_DATASET  = "naserabdullahalam/phishing-email-dataset"
FALLBACK_DATASET = "subhajournal/phishingemails"
HF_DATASET       = "zefang-liu/phishing-email-dataset"

# CSV files inside the primary dataset zip and their column mappings
# The dataset contains multiple CSVs — we'll merge them all
PRIMARY_CSV_CONFIGS = [
    {"filename": "Nazario.csv",       "text": "Email Text", "label": "Email Type"},
    {"filename": "CEAS_08.csv",       "text": "Email Text", "label": "Email Type"},
    {"filename": "Ling.csv",          "text": "Email Text", "label": "Email Type"},
    {"filename": "Nigerian_Fraud.csv","text": "Email Text", "label": "Email Type"},
    {"filename": "SpamAssasin.csv",   "text": "Email Text", "label": "Email Type"},
    {"filename": "Enron.csv",         "text": "Email Text", "label": "Email Type"},
]


# ── CREDENTIALS CHECK ──────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key      = os.environ.get("KAGGLE_KEY", "").strip()

    if username and key:
        print(f"  ✓ Credentials ready (username: {username})")
        return True

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        try:
            creds = json.load(open(kaggle_json))
            if creds.get("username") and creds.get("key"):
                print(f"  ✓ Found ~/.kaggle/kaggle.json")
                return True
        except Exception:
            pass

    print("  ✗ No Kaggle credentials found.")
    print("    → Add KAGGLE_USERNAME and KAGGLE_KEY to your .env file")
    return False


# ── KAGGLE DOWNLOAD ────────────────────────────────────────────────────────────

def kaggle_download_all_csvs(slug: str) -> "pd.DataFrame | None":
    """
    Download Kaggle dataset, merge ALL CSVs found inside into one DataFrame.
    This is important for naserabdullahalam/phishing-email-dataset which
    ships 6 separate CSV files — we want ALL of them for maximum coverage.
    """
    try:
        import kaggle
        print(f"  Downloading: kaggle → {slug}")
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle.api.dataset_download_files(
                slug, path=tmpdir, unzip=True, quiet=False
            )
            frames = []
            for root, _, files in os.walk(tmpdir):
                for f in sorted(files):
                    if not f.endswith(".csv"):
                        continue
                    fpath = os.path.join(root, f)
                    try:
                        df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
                        cleaned = clean_dataframe(df, source_name=f)
                        if cleaned is not None and len(cleaned) > 0:
                            frames.append(cleaned)
                            print(f"    ✓ {f}: {len(cleaned)} rows")
                        else:
                            print(f"    ✗ {f}: could not parse columns")
                    except Exception as e:
                        print(f"    ✗ {f}: {e}")

            if not frames:
                print("  ✗ No usable CSVs found in archive")
                return None

            combined = pd.concat(frames, ignore_index=True)
            print(f"  Combined shape: {combined.shape}")
            return combined

    except Exception as e:
        print(f"  ✗ Kaggle download failed: {e}")
    return None


def clean_dataframe(df: pd.DataFrame, source_name: str = "") -> "pd.DataFrame | None":
    """
    Auto-detect text + label columns → normalise to (text str, label int 0/1).
    Handles all column naming conventions across the 6 sub-datasets.
    """
    text_col = label_col = None

    for c in df.columns:
        cl = c.lower().strip()
        if text_col  is None and any(x in cl for x in
            ["email text", "text", "body", "content", "message", "mail", "email body"]):
            text_col = c
        if label_col is None and any(x in cl for x in
            ["email type", "type", "label", "class", "category", "spam", "phishing"]):
            label_col = c

    if not text_col or not label_col:
        return None

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label_raw"]
    out = out.dropna()

    # Map all string label variants → 0 / 1
    uniq = out["label_raw"].unique()
    lmap = {}
    for v in uniq:
        s = str(v).lower().strip()
        lmap[v] = 1 if any(x in s for x in
            ["phish", "spam", "fraud", "1", "malicious", "nigerian"]) else 0

    out["label"] = out["label_raw"].map(lmap).astype(int)
    out["text"]  = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 20][["text", "label"]]
    return out.reset_index(drop=True)


# ── HUGGINGFACE FALLBACK ───────────────────────────────────────────────────────

def download_huggingface() -> "pd.DataFrame | None":
    """
    HuggingFace mirror of the Kaggle phishing dataset.
    No API key required.
    Dataset: zefang-liu/phishing-email-dataset
    """
    try:
        print(f"  Downloading: HuggingFace → {HF_DATASET}")
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split="train")
        df = ds.to_pandas()
        print(f"  Shape: {df.shape}  Columns: {list(df.columns)}")
        cleaned = clean_dataframe(df)
        if cleaned is not None:
            return cleaned
        # Fallback column names for this specific HF dataset
        df["text"]  = df.get("email_text", df.iloc[:, 0]).astype(str).str.strip()
        df["label"] = df.get("label", df.iloc[:, -1]).astype(int)
        return df[["text","label"]].dropna()
    except Exception as e:
        print(f"  ✗ HuggingFace failed: {e}")
    return None


# ── SYNTHETIC FALLBACK ─────────────────────────────────────────────────────────

def generate_synthetic() -> pd.DataFrame:
    """
    600-sample synthetic dataset covering 3 phishing categories:
      1. Obvious phishing    (urgent/threat/prize — 200 samples)
      2. SPEAR PHISHING      (legitimate-looking + 1 malicious element — 200 samples)
      3. Legitimate emails   (clean corporate/personal — 200 samples)

    SPEAR PHISHING is the critical addition — these are realistic mixed
    emails that look mostly legitimate but contain one phishing signal.
    This is exactly the pattern the model was missing.
    """
    random.seed(42)

    # ── Category 1: Obvious phishing ──────────────────────────────────────────
    OBVIOUS_PHISHING = [
        "URGENT: Your {bank} account has been SUSPENDED! Verify your password at {url} within 24 hours or lose access permanently!",
        "Congratulations! You have WON a ${amount} prize. Click here {url} to claim NOW. Expires today!",
        "Your PayPal account is LIMITED. Verify identity immediately to avoid permanent suspension: {url}",
        "ALERT: Your {bank} debit card has been BLOCKED. Confirm your details NOW at {url}!",
        "FINAL NOTICE: Your Netflix payment FAILED. Update credit card NOW or lose access: {url}",
        "IRS ALERT: Tax refund of ${amount} pending. Verify SSN and bank account immediately: {url}",
        "WARNING: Unauthorized access detected. Account will be LOCKED. Verify NOW: {url}",
        "You have ${amount} in UNCLAIMED rewards. Click to claim before midnight: {url}",
        "Your Apple ID was used to make a purchase. If NOT you, click here IMMEDIATELY: {url}",
        "SECURITY BREACH: Your password was exposed. Reset it NOW at: {url}",
    ]

    # ── Category 2: Spear Phishing / BEC (THE CRITICAL CATEGORY) ────────────
    # Business Email Compromise style — legitimate-looking body + 1 malicious signal.
    SPEAR_PHISHING = [
        "Subject: IT Security Update\nHi Team,\nOur IT department has rolled out a mandatory security patch. Please log in to verify your credentials before end of day:\n{url}\nIf you have already updated, please disregard.\nIT Support",

        "Hello Team,\nAs part of our quarterly IT security review, we are rolling out updates to improve internal data protection.\n• New password rotation policy (every 90 days)\n• Updated VPN access guidelines\n• Mandatory security awareness training\nOur monitoring system detected unusual login attempts across employee accounts. Please confirm your account activity to avoid potential lockouts:\n{url}\nPlease complete this before end of day to avoid interruption of email access.\nRegards,\nMichael Carter\nIT Security Operations",

        "Hi Team,\nWe are updating our VPN access policy. All remote employees must re-authenticate by Friday:\n{url}\nThis is mandatory. Failure to complete will result in loss of remote access.\nNetwork Security",

        "Dear {name},\nYour password will expire in 2 days. To avoid being locked out, please update it using the company portal:\n{url}\nThis is a routine security measure.\nIT Helpdesk",

        "Subject: Mandatory Security Compliance Update\nDear Staff,\nAs part of our annual compliance review, all employees are required to re-verify their access credentials. This is mandatory per our updated data protection policy.\nPlease complete verification before {day}:\n{url}\nFailure to complete may result in temporary account suspension.\nCompliance Team",

        "Hi {name},\nIT has implemented new two-factor authentication across all systems. You need to re-enrol your device before {day} to maintain access.\nEnrol here: {url}\nContact the helpdesk if you need assistance.\nIT Operations",

        "Hi {name},\nJust a reminder about tomorrow's all-hands meeting at 10 AM. Also, HR has asked all staff to re-verify their employee portal access:\n{url}\nPlease complete before the meeting.\nOperations Team",

        "Hi {name},\nThe HR team is collecting updated direct deposit information for the upcoming payroll cycle. Please verify your banking details by Thursday:\n{url}\nThis is required to ensure timely payment.\nHR Department",

        "Hello {name},\nYour benefits enrollment period ends this Friday. Please log in to confirm or update your selections:\n{url}\nIf you miss the deadline, your current selections will be auto-renewed.\nHR Benefits Team",

        "Dear {name},\nAs part of our annual salary review cycle, please confirm your personal and banking details are up to date:\n{url}\nThis must be completed by {day} to ensure correct processing.\nPayroll Department",

        "Subject: Invoice #INV-{order} Requires Your Review\nDear {name},\nWe have a new invoice pending your approval. Please log in to review and authorise:\n{url}\nThis requires action within 48 hours.\nAccounts Payable",

        "Subject: Shared Document - Q{quarter} Report\nHello,\nPlease find the Q{quarter} performance report ready for review. Sign in with your work credentials to access:\n{url}\nFinance Team",

        "Hi {name},\nThe project files you requested have been shared to an external portal. Use your company credentials to access:\n{url}\nProject Manager",

        "Dear {name},\nYour cloud storage is nearly full (98% used). Please review and manage your files:\n{url}\nFiles will stop syncing in 24 hours if no action is taken.\nIT Storage Team",

        "Hi,\nWe noticed a login attempt to your account from a new device in {city}. If this was not you, please secure your account immediately:\n{url}\nSecurity Team",

        "Subject: Unusual Account Activity Detected\nDear {name},\nOur security systems detected several failed login attempts on your account from an unrecognised location. Please verify your identity:\n{url}\nThis link expires in 6 hours.\nAccount Security",

        "Hi Team,\nWe have detected unusual activity across several employee accounts overnight. While most attempts were blocked automatically, we recommend all staff review their recent login history:\n{url}\nPlease complete this review before end of business today.\nIT Security",

        "Hi,\nWe have updated our Terms of Service. To continue using your account, please review and accept the new terms:\n{url}\nYour account access will be restricted until you complete this step.\nLegal & Compliance",

        "Dear {name},\nYour recent shipment #{order} requires customs verification before release. Please confirm your details:\n{url}\nCustomer Service",

        "Hi {name},\nYour IT support ticket #{order} has been updated. Please log in to the helpdesk portal to review and provide additional information:\n{url}\nThe ticket will auto-close in 24 hours without a response.\nIT Helpdesk",

        "Dear {name},\nYour W-2 / tax document for this year is ready. Please log in to the employee portal to download:\n{url}\nPayroll & Tax Team",

        "Hi {name},\nYour email mailbox has reached 95% capacity. Please log in and manage your storage:\n{url}\nEmail delivery will be paused once capacity is reached.\nEmail Services",

        "Subject: Q{quarter} Review + IT Action Required\nHi Team,\nTomorrow's agenda:\n1. Q{quarter} performance review\n2. Product roadmap\n3. Security policy updates\nAdditionally, IT flagged multiple failed login attempts. Please reset your password before the meeting:\n{url}\nMeeting at 10 AM, Conference Room B.\nThanks, Alex",
    ]

        # ── Category 3: Legitimate emails ─────────────────────────────────────────
    LEGIT = [
        "Hi {name}, team meeting scheduled for {day} at {time}. Please confirm your attendance. Looking forward to seeing everyone there.",
        "Your monthly bank statement for {month} is now available. You can log in to your account at yourbank.com to view your transactions.",
        "Thank you for your purchase! Your order #{order} has been shipped and will arrive by {day}. Track it on our website.",
        "Hi {name}, following up on the project we discussed last week. Are you free for a 30-minute call on {day}?",
        "Your subscription has been renewed successfully for another year. Your next billing date is {date}.",
        "Appointment with Dr. {name} confirmed for {day} at {time}. Please arrive 10 minutes early and bring your insurance card.",
        "Hi team, the Q3 performance report is ready for review. I've shared it in the team folder — please take a look before our meeting.",
        "Your password was recently changed from a recognised device. If this was you, no action is needed.",
        "We have updated our privacy policy. The changes take effect on {date}. You can read the full document at our website.",
        "Your weekly digest: 3 new followers, 12 likes, and 5 comments on your recent posts.",
        "Reminder: your annual subscription renews in 30 days. No action needed if you wish to continue.",
        "Hi {name}, the document you shared earlier has been reviewed. I've left comments directly in the file.",
        "Your flight confirmation: New York → London on {date}. Check-in opens 24 hours before departure on the airline's website.",
        "Please submit your timesheet for the week ending {date}. The deadline is Friday at 5 PM.",
        "Hi {name}, great meeting you at the conference last week. Would love to stay in touch — connecting with you on LinkedIn.",
        "Your package was delivered to your front door at {time} today. A photo confirmation has been saved to your account.",
        "Hi team, just a reminder that the office will be closed on {day} for the public holiday. Enjoy the long weekend!",
        "Your gym membership is confirmed. Your first class is booked for {day} at {time}. See you there!",
        "Hi {name}, the budget proposal you submitted has been approved. Finance will process it in the next cycle.",
        "Meeting notes from {day}'s call are in the shared folder. Let me know if you need any clarification on the action items.",
    ]

    BANKS   = ["Chase", "Wells Fargo", "Bank of America", "Citibank", "HSBC"]
    BAD_URLS= ["http://security-reset-helpdesk.net/password",
               "http://it-portal-login.support-desk.net",
               "http://secure-verify.employee-portal.biz",
               "http://accounts-update.corp-helpdesk.net",
               "http://staff-auth.internal-portal.xyz",
               "http://hr-verification.company-access.info"]
    AMTS    = ["500", "1,000", "250", "750", "5,000"]
    NAMES   = ["John", "Sarah", "Michael", "Emily", "David", "Priya", "Alex"]
    DAYS    = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    TIMES   = ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM"]
    MONTHS  = ["January", "February", "March", "April"]
    DATES   = ["March 15", "April 1", "May 10", "June 5"]
    ORDERS  = [str(random.randint(10000, 99999)) for _ in range(20)]
    CITIES  = ["Chicago", "London", "Singapore", "Mumbai", "Berlin"]
    QUARTERS= ["1", "2", "3", "4"]

    def fill(t, lbl):
        return (t.replace("{bank}",    random.choice(BANKS))
                 .replace("{url}",     random.choice(BAD_URLS) if lbl == 1 else "https://company.com/portal")
                 .replace("{amount}",  random.choice(AMTS))
                 .replace("{name}",    random.choice(NAMES))
                 .replace("{day}",     random.choice(DAYS))
                 .replace("{time}",    random.choice(TIMES))
                 .replace("{month}",   random.choice(MONTHS))
                 .replace("{date}",    random.choice(DATES))
                 .replace("{order}",   random.choice(ORDERS))
                 .replace("{city}",    random.choice(CITIES))
                 .replace("{quarter}", random.choice(QUARTERS)))

    rows = []
    # 200 obvious phishing
    for _ in range(200):
        rows.append({"text": fill(random.choice(OBVIOUS_PHISHING), 1), "label": 1})
    # 200 spear phishing  ← THE KEY ADDITION
    for _ in range(200):
        rows.append({"text": fill(random.choice(SPEAR_PHISHING), 1), "label": 1})
    # 200 legitimate
    for _ in range(200):
        rows.append({"text": fill(random.choice(LEGIT), 0), "label": 0})

    random.shuffle(rows)
    print(f"  Generated {len(rows)} synthetic samples "
          f"(200 obvious phishing + 200 spear phishing + 200 legit)")
    return pd.DataFrame(rows)


# ── FINALIZE ───────────────────────────────────────────────────────────────────

def finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Deduplicate + balance classes + print summary.
    Cap at 10,000 per class to keep training fast on CPU.
    """
    print(f"\n  Source : {source}")
    print(f"  Raw    : {len(df)} rows")
    print(f"  Phishing (1) : {df['label'].sum()}")
    print(f"  Legit    (0) : {(df['label']==0).sum()}")

    df = df.drop_duplicates(subset=["text"]).dropna()
    n  = min(df[df.label==1].shape[0], df[df.label==0].shape[0], 10000)

    df = pd.concat([
        df[df.label==1].sample(n=n, random_state=42),
        df[df.label==0].sample(n=n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  After balancing : {len(df)} rows ({n} per class)")
    return df[["text", "label"]]


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download phishing dataset")
    parser.add_argument("--synthetic",   action="store_true", help="Force synthetic dataset")
    parser.add_argument("--huggingface", action="store_true", help="Force HuggingFace dataset")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PHISHING DATASET DOWNLOADER")
    print("="*60)

    df = source = None

    if args.synthetic:
        df, source = generate_synthetic(), "Synthetic (forced)"

    elif args.huggingface:
        df     = download_huggingface()
        source = f"HuggingFace: {HF_DATASET}"
        if df is None:
            df, source = generate_synthetic(), "Synthetic (HF fallback)"

    else:
        # ── Auto: Kaggle primary (6 real corpora) ────────────────────────────
        print("\n[1] Checking Kaggle credentials...")
        has_creds = check_kaggle_credentials()

        if has_creds:
            print(f"\n[2] Downloading primary dataset ({PRIMARY_DATASET})...")
            print("    This contains Nazario + CEAS + Ling + Enron + Nigerian + SpamAssassin")
            df = kaggle_download_all_csvs(PRIMARY_DATASET)
            if df is not None:
                source = f"Kaggle: {PRIMARY_DATASET} (6 real corpora)"

        # ── Kaggle fallback ───────────────────────────────────────────────────
        if df is None and has_creds:
            print(f"\n[2b] Trying Kaggle fallback ({FALLBACK_DATASET})...")
            df = kaggle_download_all_csvs(FALLBACK_DATASET)
            if df is not None:
                source = f"Kaggle: {FALLBACK_DATASET}"

        # ── HuggingFace ───────────────────────────────────────────────────────
        if df is None:
            if not has_creds:
                print("\n    No Kaggle credentials — falling back to HuggingFace...")
            else:
                print("\n[3] All Kaggle sources failed — trying HuggingFace...")
            df     = download_huggingface()
            source = f"HuggingFace: {HF_DATASET}"

        # ── Synthetic last resort ─────────────────────────────────────────────
        if df is None:
            print("\n[4] All remote sources failed — generating synthetic data...")
            df, source = generate_synthetic(), "Synthetic (auto-fallback)"

    # ── Save ──────────────────────────────────────────────────────────────────
    df = finalize(df, source)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"  DONE ✓")
    print(f"  Saved  : {args.output}")
    print(f"  Shape  : {df.shape}")
    print(f"  Labels : {dict(df['label'].value_counts().sort_index())}")
    print(f"{'='*60}")
    print(f"\n  Phishing sample → {df[df.label==1].iloc[0]['text'][:100]}...")
    print(f"  Legit    sample → {df[df.label==0].iloc[0]['text'][:100]}...")


if __name__ == "__main__":
    main()