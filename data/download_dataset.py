"""
download_dataset.py
--------------------
Downloads and merges phishing datasets from Kaggle.

DATASET STRATEGY - WHY 3 SOURCES:

  Gap analysis of email-only corpora revealed two critical blind spots:
    1. CHANNEL GAP: All 6 original corpora are email only.
       Real attackers also use SMS (SMiShing). A model trained only on
       email patterns may miss mobile phishing attacks.
    2. ERA GAP: All 6 original corpora are from 2000-2008.
       Modern phishing (2019+) uses different vocabulary, templates,
       and social engineering techniques (COVID lures, crypto scams, etc.)

  Three datasets were selected to close both gaps:

  SOURCE 1 - naserabdullahalam/phishing-email-dataset  (PRIMARY)
    ~82,000 real emails from 6 corpora:
      - Enron         (real corporate legitimate emails, 2000-2002)
      - Nazario       (real scraped phishing, 2005-2007)
      - CEAS 2008     (spam/phishing competition, 2008)
      - Ling Spam     (classic academic benchmark, 2000-2005)
      - Nigerian Fraud(advance fee fraud, 2000s)
      - SpamAssassin  (diverse spam + ham, 2002-2006)
    Covers: Classic phishing patterns, BEC, legitimate corporate language

  SOURCE 2 - subhajournal/phishingemails  (ERA GAP FIX)
    ~18,000 emails, 2019-2021
    Covers: Modern phishing vocabulary, updated BEC tactics
    Why needed: Phishing evolves - a 2008 model misses 2021 attacks

  SOURCE 3 - uciml/sms-spam-collection  (CHANNEL GAP FIX)
    5,574 SMS messages, 2011-2012
    Covers: SMS phishing (SMiShing), short-form social engineering
    Why needed: Aviation personnel receive SMS alerts - attackers exploit this
    Labels: ham=0 (legitimate), spam=1 (phishing/spam)

  COMBINED: ~105,000 raw samples across 2 channels and 2 decades
  AFTER BALANCING: 15,000 per class (30,000 total) for fast, fair training

FALLBACK ORDER (if Kaggle credentials unavailable):
  1. HuggingFace mirror of Source 1 (no API key needed)
  2. Synthetic dataset (always works offline, 600 samples)

SETUP:
  1. Copy .env.example to .env in project root
  2. Add KAGGLE_USERNAME and KAGGLE_KEY
  3. pip install kaggle
  4. python data/download_dataset.py

USAGE:
  python data/download_dataset.py                 # auto mode (recommended)
  python data/download_dataset.py --synthetic     # force synthetic (offline)
  python data/download_dataset.py --huggingface   # force HuggingFace
  python data/download_dataset.py --source1-only  # only original 6 corpora
"""

import os
import sys
import json
import argparse
import tempfile
import random

import pandas as pd


# -- LOAD .env BEFORE ANYTHING ELSE -------------------------------------------

def load_env_file() -> bool:
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
        print(f"  .env loaded -> set: {', '.join(loaded)}")
    else:
        print(f"  .env found (variables already in environment)")
    return True


print("\n[.env] Loading Kaggle credentials...")
_env_loaded = load_env_file()


# -- DATASET CONFIGS ----------------------------------------------------------

OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "dataset.csv")
SOURCE1_SLUG  = "naserabdullahalam/phishing-email-dataset"
SOURCE2_SLUG  = "subhajournal/phishingemails"
SOURCE3_SLUG  = "uciml/sms-spam-collection"
HF_DATASET    = "zefang-liu/phishing-email-dataset"
PER_CLASS_CAP = 15000


# -- CREDENTIALS CHECK --------------------------------------------------------

def check_kaggle_credentials() -> bool:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key      = os.environ.get("KAGGLE_KEY", "").strip()

    if username and key:
        print(f"  Credentials ready (username: {username})")
        return True

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        try:
            creds = json.load(open(kaggle_json))
            if creds.get("username") and creds.get("key"):
                print(f"  Found ~/.kaggle/kaggle.json")
                return True
        except Exception:
            pass

    print("  No Kaggle credentials found.")
    print("    -> Add KAGGLE_USERNAME and KAGGLE_KEY to your .env file")
    return False


# -- SHARED: COLUMN AUTO-DETECTION --------------------------------------------

def clean_dataframe(df: pd.DataFrame, source_name: str = "") -> "pd.DataFrame | None":
    """
    Auto-detect text + label columns -> normalise to (text str, label int 0/1).

    Handles all column naming conventions:
      Source 1: 'Email Text', 'Email Type'
      Source 2: 'Email Text', 'Email Type'
      Source 3: 'v2' (text), 'v1' (ham/spam label)

    Label mapping:
      1 = Phishing / Spam / Fraud / Malicious
      0 = Legitimate / Ham / Safe
    """
    text_col = label_col = None

    for c in df.columns:
        cl = c.lower().strip()
        if text_col is None and any(x in cl for x in
            ["email text", "text", "body", "content", "message",
             "mail", "email body", "sms", "v2"]):
            text_col = c
        if label_col is None and any(x in cl for x in
            ["email type", "type", "label", "class", "category",
             "spam", "phishing", "v1"]):
            label_col = c

    if not text_col or not label_col:
        return None

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label_raw"]
    out = out.dropna()

    uniq = out["label_raw"].unique()
    lmap = {}
    for v in uniq:
        s = str(v).lower().strip()
        lmap[v] = 1 if any(x in s for x in
            ["phish", "spam", "fraud", "1", "malicious",
             "nigerian", "smishing"]) else 0

    out["label"] = out["label_raw"].map(lmap).astype(int)
    out["text"]  = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 10][["text", "label"]]
    return out.reset_index(drop=True)


# -- KAGGLE DOWNLOADER (GENERIC) ----------------------------------------------

def kaggle_download_all_csvs(slug: str,
                              source_label: str = "") -> "pd.DataFrame | None":
    """
    Download any Kaggle dataset and merge all CSVs inside into one DataFrame.
    Works for all 3 sources despite their different internal structures.
    """
    try:
        import kaggle
        label = source_label or slug
        print(f"  Downloading: {label}")
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
                        df = pd.read_csv(fpath, encoding="utf-8",
                                         on_bad_lines="skip")
                        cleaned = clean_dataframe(df, source_name=f)
                        if cleaned is not None and len(cleaned) > 0:
                            frames.append(cleaned)
                            print(f"    + {f}: {len(cleaned):,} rows")
                        else:
                            print(f"    - {f}: could not parse columns")
                    except Exception as e:
                        print(f"    - {f}: {e}")

            if not frames:
                print(f"  No usable CSVs found in {slug}")
                return None

            combined = pd.concat(frames, ignore_index=True)
            print(f"  Combined: {combined.shape[0]:,} rows "
                  f"(phishing={combined['label'].sum():,} "
                  f"legit={(combined['label']==0).sum():,})")
            return combined

    except Exception as e:
        print(f"  Kaggle download failed for {slug}: {e}")
    return None


# -- SOURCE-SPECIFIC DOWNLOADERS ----------------------------------------------

def download_source1() -> "pd.DataFrame | None":
    """
    Source 1: naserabdullahalam/phishing-email-dataset
    6 real-world email corpora (Enron, Nazario, CEAS, Ling, Nigerian, SpamAssassin)
    Era: 2000-2008 | Channel: Email
    """
    print("\n  [Source 1] Classic email corpora (6 datasets, 2000-2008)")
    return kaggle_download_all_csvs(
        SOURCE1_SLUG,
        "Enron + Nazario + CEAS + Ling + Nigerian + SpamAssassin"
    )


def download_source2() -> "pd.DataFrame | None":
    """
    Source 2: subhajournal/phishingemails
    Modern phishing emails 2019-2021 -- closes the ERA GAP.
    Era: 2019-2021 | Channel: Email
    """
    print("\n  [Source 2] Modern phishing emails (2019-2021) -- ERA GAP FIX")
    return kaggle_download_all_csvs(
        SOURCE2_SLUG,
        "Modern Phishing Emails 2019-2021"
    )


def download_source3() -> "pd.DataFrame | None":
    """
    Source 3: uciml/sms-spam-collection
    5,574 SMS messages -- closes the CHANNEL GAP.
    Adds mobile/SMS phishing (SMiShing) patterns.
    Columns: v1 (ham/spam label), v2 (message text)
    Era: 2011-2012 | Channel: SMS/Mobile
    """
    print("\n  [Source 3] SMS Spam Collection -- CHANNEL GAP FIX")
    df = kaggle_download_all_csvs(
        SOURCE3_SLUG,
        "UCI SMS Spam Collection (5,574 messages)"
    )
    if df is not None:
        print(f"  SMS channel added: {len(df):,} messages")
    return df


# -- HUGGINGFACE FALLBACK -----------------------------------------------------

def download_huggingface() -> "pd.DataFrame | None":
    """HuggingFace mirror of Source 1 -- no API key required."""
    try:
        print(f"  Downloading HuggingFace -> {HF_DATASET}")
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split="train")
        df = ds.to_pandas()
        print(f"  Shape: {df.shape}  Columns: {list(df.columns)}")
        cleaned = clean_dataframe(df)
        if cleaned is not None:
            return cleaned
        df["text"]  = df.get("email_text", df.iloc[:, 0]).astype(str).str.strip()
        df["label"] = df.get("label", df.iloc[:, -1]).astype(int)
        return df[["text", "label"]].dropna()
    except Exception as e:
        print(f"  HuggingFace failed: {e}")
    return None


# -- SYNTHETIC FALLBACK -------------------------------------------------------

def generate_synthetic() -> pd.DataFrame:
    """
    600-sample synthetic dataset for offline use.
    Covers: obvious phishing (200) + spear phishing/BEC (200) + legitimate (200)
    """
    random.seed(42)

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

    SPEAR_PHISHING = [
        "Subject: IT Security Update\nHi Team,\nOur IT department has rolled out a mandatory security patch. Please log in to verify your credentials before end of day:\n{url}\nIT Support",
        "Hello Team,\nAs part of our quarterly IT security review, our monitoring system detected unusual login attempts. Please confirm your account activity:\n{url}\nMichael Carter, IT Security",
        "Hi Team,\nWe are updating our VPN access policy. All remote employees must re-authenticate by Friday:\n{url}\nNetwork Security",
        "Dear {name},\nYour password will expire in 2 days. Update it using the company portal:\n{url}\nIT Helpdesk",
        "Dear Staff,\nAll employees are required to re-verify their access credentials before {day}:\n{url}\nCompliance Team",
        "Hi {name},\nNew two-factor authentication implemented. Re-enrol your device before {day}:\n{url}\nIT Operations",
        "Hi {name},\nHR requests all staff to re-verify their employee portal access:\n{url}\nOperations Team",
        "Hi {name},\nPlease verify your banking details for the upcoming payroll cycle by Thursday:\n{url}\nHR Department",
        "Dear {name},\nInvoice #INV-{order} requires your approval. Log in to review:\n{url}\nAccounts Payable",
        "Subject: URGENT Crew Portal Update\nDear Crew Member,\nYour airline crew portal credentials are expiring. Verify immediately to maintain flight assignment access:\n{url}\nAirline Operations",
        "Dear Pilot,\nYour DGCA compliance portal requires immediate re-verification. Failure will suspend your flight ops access:\n{url}\nAviation Authority",
        "Hi {name},\nHR payroll migration requires all staff to verify direct deposit details by {day}:\n{url}\nPayroll Department",
    ]

    LEGIT = [
        "Hi {name}, team meeting scheduled for {day} at {time}. Please confirm attendance.",
        "Your monthly bank statement for {month} is available at yourbank.com.",
        "Thank you for your purchase! Order #{order} shipped. Arriving by {day}.",
        "Hi {name}, are you free for a 30-minute call on {day}?",
        "Your subscription has been renewed. Next billing date is {date}.",
        "Appointment confirmed for {day} at {time}. Please arrive 10 minutes early.",
        "Hi team, Q3 report ready for review. Shared in the team folder.",
        "Your password was changed from a recognised device. No action needed.",
        "Privacy policy updated. Changes take effect on {date}.",
        "Flight OPS: Aircraft B737-800 maintenance check complete. Cleared for 0600 UTC.",
        "Crew briefing for flight AI-204 at 0430 at the crew lounge. Bring documents.",
        "Your recurrent training certificate issued. Next due in 12 months.",
        "Hi {name}, the budget proposal has been approved. Finance will process next cycle.",
        "Meeting notes from {day} call are in the shared folder.",
        "Your order #{order} delivered at {time}. Photo saved to your account.",
        "Reminder: office closed on {day} for public holiday.",
        "Please submit your timesheet for week ending {date} by Friday 5 PM.",
        "NOTAM: Runway 27L at VABB closed for maintenance until further notice.",
        "Hi {name}, great meeting you at the conference. Connecting on LinkedIn.",
        "Your package is out for delivery and will arrive by {time} today.",
    ]

    BANKS    = ["Chase", "Wells Fargo", "Bank of America", "Citibank", "HSBC"]
    BAD_URLS = [
        "http://security-reset-helpdesk.net/password",
        "http://it-portal-login.support-desk.net",
        "http://secure-verify.employee-portal.biz",
        "http://accounts-update.corp-helpdesk.net",
        "http://staff-auth.internal-portal.xyz",
        "http://hr-verification.company-access.info",
    ]
    AMTS     = ["500", "1,000", "250", "750", "5,000"]
    NAMES    = ["John", "Sarah", "Michael", "Emily", "David", "Priya", "Alex"]
    DAYS     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    TIMES    = ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM"]
    MONTHS   = ["January", "February", "March", "April"]
    DATES    = ["March 15", "April 1", "May 10", "June 5"]
    ORDERS   = [str(random.randint(10000, 99999)) for _ in range(20)]
    CITIES   = ["Chicago", "London", "Singapore", "Mumbai", "Berlin"]
    QUARTERS = ["1", "2", "3", "4"]

    def fill(t, lbl):
        return (
            t.replace("{bank}",    random.choice(BANKS))
             .replace("{url}",     random.choice(BAD_URLS) if lbl == 1
                                   else "https://company.com/portal")
             .replace("{amount}",  random.choice(AMTS))
             .replace("{name}",    random.choice(NAMES))
             .replace("{day}",     random.choice(DAYS))
             .replace("{time}",    random.choice(TIMES))
             .replace("{month}",   random.choice(MONTHS))
             .replace("{date}",    random.choice(DATES))
             .replace("{order}",   random.choice(ORDERS))
             .replace("{city}",    random.choice(CITIES))
             .replace("{quarter}", random.choice(QUARTERS))
        )

    rows = []
    for _ in range(200):
        rows.append({"text": fill(random.choice(OBVIOUS_PHISHING), 1), "label": 1})
    for _ in range(200):
        rows.append({"text": fill(random.choice(SPEAR_PHISHING),   1), "label": 1})
    for _ in range(200):
        rows.append({"text": fill(random.choice(LEGIT),             0), "label": 0})

    random.shuffle(rows)
    print(f"  Generated {len(rows)} synthetic samples "
          f"(200 obvious + 200 spear phishing + 200 legit)")
    return pd.DataFrame(rows)


# -- MERGE + BALANCE ----------------------------------------------------------

def merge_and_finalize(frames: list,
                       sources: list,
                       per_class_cap: int = PER_CLASS_CAP) -> pd.DataFrame:
    """
    Merge multiple DataFrames, deduplicate, balance classes.

    WHY BALANCE?
      Imbalanced classes cause the model to predict the majority class for
      high accuracy without learning phishing patterns. Equal class sizes
      force it to learn both distributions properly.

    WHY CAP AT 15k PER CLASS?
      30k total samples is the sweet spot: enough for strong generalisation,
      fast enough for CPU training, and representative of all 3 sources.
    """
    valid = [(df, src) for df, src in zip(frames, sources) if df is not None]

    if not valid:
        print("  All sources failed. Using synthetic fallback.")
        df = generate_synthetic()
        return df

    print(f"\n  Merging {len(valid)} dataset(s)...")
    combined = pd.concat([df for df, _ in valid], ignore_index=True)

    print(f"\n  MERGE SUMMARY")
    print(f"  {'='*50}")
    for df, src in valid:
        p = df["label"].sum()
        l = (df["label"] == 0).sum()
        print(f"  {src:<40} {len(df):>7,} rows  "
              f"(phishing={p:,}  legit={l:,})")
    print(f"  {'='*50}")
    print(f"  Combined raw total : {len(combined):,} rows")

    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).dropna()
    print(f"  After dedup        : {len(combined):,} rows "
          f"(removed {before - len(combined):,} duplicates)")

    n = min(
        combined[combined.label == 1].shape[0],
        combined[combined.label == 0].shape[0],
        per_class_cap,
    )
    combined = pd.concat([
        combined[combined.label == 1].sample(n=n, random_state=42),
        combined[combined.label == 0].sample(n=n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  After balancing    : {len(combined):,} rows "
          f"({n:,} per class)")
    return combined[["text", "label"]]


# -- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and merge phishing datasets from Kaggle"
    )
    parser.add_argument("--synthetic",    action="store_true",
                        help="Force offline synthetic dataset (600 samples)")
    parser.add_argument("--huggingface",  action="store_true",
                        help="Force HuggingFace source only")
    parser.add_argument("--source1-only", action="store_true",
                        help="Download Source 1 only (original 6 corpora)")
    parser.add_argument("--output",       default=OUTPUT_PATH,
                        help="Output CSV path")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PHISHGUARD AI -- MULTI-SOURCE DATASET DOWNLOADER")
    print("  3 Kaggle datasets | 2 channels | 2 eras | ~30k balanced")
    print("=" * 60)

    if args.synthetic:
        df = generate_synthetic()
        df = merge_and_finalize([df], ["Synthetic (forced)"])

    elif args.huggingface:
        df_hf  = download_huggingface()
        frames = [df_hf] if df_hf is not None else [generate_synthetic()]
        src    = [f"HuggingFace: {HF_DATASET}" if df_hf else "Synthetic"]
        df     = merge_and_finalize(frames, src)

    elif args.source1_only:
        df1 = download_source1()
        df  = merge_and_finalize([df1], [SOURCE1_SLUG])

    else:
        print("\n[1] Checking Kaggle credentials...")
        has_creds = check_kaggle_credentials()

        frames  = []
        sources = []

        if has_creds:
            print("\n[2] Source 1 -- Classic email corpora (6 datasets, 2000-2008)")
            df1 = download_source1()
            if df1 is not None:
                frames.append(df1)
                sources.append("Source1: 6 classic corpora (2000-2008)")

            print("\n[3] Source 2 -- Modern phishing emails (ERA GAP FIX)")
            df2 = download_source2()
            if df2 is not None:
                frames.append(df2)
                sources.append("Source2: Modern emails (2019-2021)")
            else:
                print("  Source 2 unavailable -- era gap partially unfilled")

            print("\n[4] Source 3 -- SMS Spam Collection (CHANNEL GAP FIX)")
            df3 = download_source3()
            if df3 is not None:
                frames.append(df3)
                sources.append("Source3: SMS spam (mobile channel)")
            else:
                print("  Source 3 unavailable -- channel gap partially unfilled")

        if not frames:
            print("\n  No Kaggle sources available. Trying HuggingFace...")
            df_hf = download_huggingface()
            if df_hf is not None:
                frames.append(df_hf)
                sources.append(f"HuggingFace: {HF_DATASET}")

        if not frames:
            print("\n  All remote sources failed. Generating synthetic data...")
            frames.append(generate_synthetic())
            sources.append("Synthetic (auto-fallback)")

        df = merge_and_finalize(frames, sources)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Saved  : {args.output}")
    print(f"  Shape  : {df.shape}")
    print(f"  Labels : {dict(df['label'].value_counts().sort_index())}")
    print(f"{'='*60}")
    print(f"\n  Phishing sample: {df[df.label==1].iloc[0]['text'][:100]}...")
    print(f"  Legit    sample: {df[df.label==0].iloc[0]['text'][:100]}...")


if __name__ == "__main__":
    main()