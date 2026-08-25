"""
Part B — SEC EDGAR 10-K retrieval (50-firm pilot).

Takes the firm universe from clean_firm_bridge.csv, resolves one CIK per firm at
the reference date, queries the SEC submissions API for each firm's filing
history, selects the most recent 10-K filed strictly before the reference date
(within a staleness window), downloads and caches the primary HTML document, and
logs the per-firm outcome (or the specific reason no filing was retrieved).

Scoped to a configurable subset (default: first 50 firms by PERMNO). No text
parsing, tokenization, or scoring is performed here (that is Step 2).

    python edgar_pull.py [--n-firms 50] [--reference-date 2025-04-02]

IMPORTANT: set a real name/email via the EDGAR_USER_AGENT environment variable (or a
gitignored .env file) before any run - the SEC fair-access policy requires a genuine
contact. If unset, a non-functional placeholder is used and a warning is printed.
"""

import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE .env loader (no python-dotenv dependency).

    Loads variables from a local .env file (gitignored) into the environment
    without overriding variables already set in the real environment. Blank
    lines and '#' comments are ignored.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
BRIDGE_CSV = BASE / "clean_firm_bridge.csv"
FILINGS_DIR = BASE / "filings_raw"
LOG_CSV = BASE / "edgar_pull_log.csv"

# SEC fair access requires a descriptive User-Agent "AppName ContactEmail".
# Supply a real, monitored contact via the EDGAR_USER_AGENT environment variable
# (set it in your shell or a gitignored .env file); never hard-code it. The
# default below is a non-functional placeholder that triggers a warning.
_load_dotenv(BASE / ".env")
_PLACEHOLDER_UA = "TariffFactorResearch your-email@example.com"
USER_AGENT = os.environ.get("EDGAR_USER_AGENT", _PLACEHOLDER_UA)
USER_AGENT_IS_PLACEHOLDER = USER_AGENT == _PLACEHOLDER_UA

REFERENCE_DATE = pd.Timestamp("2025-04-02")  # "Liberation Day" tariffs
STALENESS_DAYS = 364                          # filing_date >= ref - 364 days
N_FIRMS = 50                                  # pilot size
REQUEST_SLEEP = 0.15                          # seconds between requests (<10 req/s)
TIMEOUT = 60

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def _headers() -> dict:
    return {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


# --------------------------------------------------------------------------- #
# Firm-subset resolution                                                      #
# --------------------------------------------------------------------------- #
def _resolve_cik(group: pd.DataFrame, ref: pd.Timestamp) -> pd.Series:
    """Pick one link row for a PERMNO: the interval covering the reference date,
    else the most recent valid interval. Ties prefer LINKPRIM 'P', latest linkdt.
    """
    linkdt = pd.to_datetime(group["linkdt"])
    linkend = pd.to_datetime(group["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    covers = (linkdt <= ref) & (linkend >= ref)
    prim_rank = (group["linkprim"] == "P").astype(int)  # P before C
    g = group.assign(_linkdt=linkdt, _linkend=linkend, _prim=prim_rank)
    pool = g[covers] if covers.any() else g
    pool = pool.sort_values(["_prim", "_linkend", "_linkdt"], ascending=False)
    return pool.iloc[0]


def load_firm_subset(bridge_csv: Path, n: int, ref: pd.Timestamp) -> list[dict]:
    """Return the first ``n`` firms by PERMNO with a single resolved CIK each.

    Firms whose resolved link has no CIK are still returned (cik='') so the pull
    logs them as ``no_cik`` rather than silently omitting them.
    """
    bridge = pd.read_csv(bridge_csv, dtype=str)
    permnos = sorted(bridge["permno"].astype(int).unique())[:n]
    firms = []
    for p in permnos:
        row = _resolve_cik(bridge[bridge["permno"].astype(int) == p], ref)
        firms.append({
            "permno": int(p),
            "gvkey": row["gvkey"],
            "cik": (row["cik"] if isinstance(row["cik"], str) else "").strip(),
        })
    return firms


# --------------------------------------------------------------------------- #
# SEC access                                                                   #
# --------------------------------------------------------------------------- #
def get_submissions(cik: str) -> dict:
    """Fetch a firm's submission history from the SEC submissions API."""
    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=_headers(),
                     timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def select_10k(submissions: dict, ref: pd.Timestamp) -> tuple[dict | None, str]:
    """Select the most recent 10-K filed within [ref-STALENESS_DAYS, ref-1 day].

    Uses ``form == '10-K'`` exactly (no variants). Records ``reportDate`` as the
    fiscal period of report. Returns (selection|None, reason).
    """
    recent = submissions.get("filings", {}).get("recent", {})
    keys = ["form", "filingDate", "reportDate", "accessionNumber", "primaryDocument"]
    if not recent.get("accessionNumber"):
        return None, "no_10k_on_file"
    df = pd.DataFrame({k: recent.get(k, []) for k in keys})

    tenk = df[df["form"] == "10-K"].copy()
    if tenk.empty:
        return None, "no_10k_on_file"

    tenk["filing_dt"] = pd.to_datetime(tenk["filingDate"], errors="coerce")
    lo = ref - pd.Timedelta(days=STALENESS_DAYS)
    hi = ref - pd.Timedelta(days=1)
    win = tenk[(tenk["filing_dt"] >= lo) & (tenk["filing_dt"] <= hi)]
    if win.empty:
        return None, "no_10k_in_window"

    sel = win.loc[win["filing_dt"].idxmax()]
    return {
        "accession": sel["accessionNumber"],
        "filing_date": sel["filingDate"],
        "period_of_report": sel["reportDate"],
        "primary_document": sel["primaryDocument"],
    }, "ok"


def download_primary(permno: int, cik: str, accession: str, doc: str) -> Path:
    """Download and cache the primary HTML document; skip if already cached."""
    acc = accession.replace("-", "")
    path = FILINGS_DIR / f"{permno}_{int(cik):010d}_{acc}.html"
    if path.exists():
        return path
    url = ARCHIVE_URL.format(cik=int(cik), acc=acc, doc=doc)
    r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


# --------------------------------------------------------------------------- #
# Pilot driver                                                                 #
# --------------------------------------------------------------------------- #
def pull_pilot(n: int, ref: pd.Timestamp) -> pd.DataFrame:
    """Run the pull over the first ``n`` firms and write the per-firm log."""
    FILINGS_DIR.mkdir(exist_ok=True)
    if USER_AGENT_IS_PLACEHOLDER:
        print("!! WARNING: USER_AGENT is a placeholder — set a real name/email "
              "before any non-pilot run (SEC fair-access requirement).")
    firms = load_firm_subset(BRIDGE_CSV, n, ref)
    print(f"Pulling 10-Ks for {len(firms)} firms "
          f"(reference date {ref.date()}, window "
          f"[{(ref - pd.Timedelta(days=STALENESS_DAYS)).date()} .. {(ref - pd.Timedelta(days=1)).date()}])\n")

    rows = []
    for i, f in enumerate(firms, 1):
        rec = {"permno": f["permno"], "gvkey": f["gvkey"], "cik": f["cik"],
               "found_10k": False, "accession": "", "filing_date": "",
               "period_of_report": "", "local_path": "", "fail_reason": ""}

        if not f["cik"]:
            rec["fail_reason"] = "no_cik"
            rows.append(rec)
            print(f"[{i:>2}/{len(firms)}] permno={f['permno']:<7} no_cik")
            continue

        try:
            subs = get_submissions(f["cik"])
            time.sleep(REQUEST_SLEEP)
        except Exception as e:  # network / HTTP / JSON
            rec["fail_reason"] = f"api_error: {type(e).__name__}"
            rows.append(rec)
            print(f"[{i:>2}/{len(firms)}] permno={f['permno']:<7} cik={f['cik']} api_error")
            continue

        sel, reason = select_10k(subs, ref)
        if sel is None:
            rec["fail_reason"] = reason
            rows.append(rec)
            print(f"[{i:>2}/{len(firms)}] permno={f['permno']:<7} cik={f['cik']} {reason}")
            continue

        try:
            path = download_primary(f["permno"], f["cik"], sel["accession"],
                                    sel["primary_document"])
            time.sleep(REQUEST_SLEEP)
        except Exception as e:
            rec.update(accession=sel["accession"], filing_date=sel["filing_date"],
                       period_of_report=sel["period_of_report"],
                       fail_reason=f"download_error: {type(e).__name__}")
            rows.append(rec)
            print(f"[{i:>2}/{len(firms)}] permno={f['permno']:<7} cik={f['cik']} download_error")
            continue

        rec.update(found_10k=True, accession=sel["accession"],
                   filing_date=sel["filing_date"],
                   period_of_report=sel["period_of_report"],
                   local_path=str(path.relative_to(BASE)))
        rows.append(rec)
        print(f"[{i:>2}/{len(firms)}] permno={f['permno']:<7} cik={f['cik']} "
              f"OK  filed={sel['filing_date']} period={sel['period_of_report']}")

    log = pd.DataFrame(rows)
    log.to_csv(LOG_CSV, index=False)
    print(f"\nWrote {LOG_CSV.name} ({len(log)} rows).")
    return log


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(log: pd.DataFrame) -> None:
    """Print pilot summary stats and spot-check 3 downloaded filings."""
    n = len(log)
    ok = log[log["found_10k"]]
    print("\n" + "=" * 70)
    print("PILOT VALIDATION")
    print("=" * 70)
    print(f"Usable 10-K downloaded : {len(ok)} / {n}")
    print(f"Failed                 : {n - len(ok)} / {n}")
    fails = log.loc[~log["found_10k"], "fail_reason"]
    if len(fails):
        print("Failure reasons:")
        for reason, cnt in fails.value_counts().items():
            print(f"  {reason:24s} {cnt}")

    paths = [BASE / p for p in ok["local_path"] if p]
    if paths:
        random.seed(0)
        sample = random.sample(paths, min(3, len(paths)))
        print("\nSpot-check - first 500 chars of 3 downloaded filings:")
        for p in sample:
            text = p.read_text(encoding="utf-8", errors="replace")[:500]
            print("\n" + "-" * 70)
            print(f"FILE: {p.name}")
            print("-" * 70)
            print(text)
    else:
        print("\nNo successful downloads to spot-check.")


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC EDGAR 10-K pilot pull.")
    ap.add_argument("--n-firms", type=int, default=N_FIRMS)
    ap.add_argument("--reference-date", type=str, default=str(REFERENCE_DATE.date()))
    args = ap.parse_args()
    ref = pd.Timestamp(args.reference_date)
    log = pull_pilot(args.n_firms, ref)
    validate(log)


if __name__ == "__main__":
    main()
