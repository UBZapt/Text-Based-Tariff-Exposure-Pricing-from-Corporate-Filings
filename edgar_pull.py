"""
Part B — SEC EDGAR 10-K retrieval (gap-only pull).

Takes the firm universe from clean_firm_bridge.csv, resolves one CIK per firm at
the reference date, queries the SEC submissions API for each firm's filing
history, selects the most recent 10-K filed strictly before the reference date
(within a staleness window), downloads and caches the primary HTML document, and
logs the per-firm outcome (or the specific reason no filing was retrieved).

The pull is incremental: the work list is the first --n-ciks distinct CIKs of the
universe MINUS every CIK already marked found_10k=True in edgar_pull_log.csv, so
cached filings are never re-downloaded while previously-failed firms are retried.
Outcomes are appended to the log, which is the accumulated history of all runs.
Cached filings whose firm has left the universe are pruned. No text parsing,
tokenization, or scoring is performed here (that is Step 2).

    python edgar_pull.py [--n-ciks 50] [--reference-date 2025-04-02]

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
N_CIKS = 50                                   # unique CIKs to cover this run
REQUEST_SLEEP = 0.15                          # seconds between requests (<10 req/s)
TIMEOUT = 60

LOG_COLUMNS = ["permno", "gvkey", "cik", "found_10k", "accession", "filing_date",
               "period_of_report", "local_path", "fail_reason"]

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


def resolve_firms(bridge_csv: Path, ref: pd.Timestamp) -> list[dict]:
    """Resolve one CIK per PERMNO across the whole bridge, in PERMNO order.

    Firms whose resolved link has no CIK are still returned (cik='') so the pull
    logs them as ``no_cik`` rather than silently omitting them.
    """
    bridge = pd.read_csv(bridge_csv, dtype=str)
    bridge["_permno"] = bridge["permno"].astype(int)
    firms = []
    for permno, group in bridge.groupby("_permno", sort=True):
        row = _resolve_cik(group, ref)
        firms.append({
            "permno": int(permno),
            "gvkey": row["gvkey"],
            "cik": (row["cik"] if isinstance(row["cik"], str) else "").strip(),
        })
    return firms


def select_gap_firms(firms: list[dict], done: set[str],
                     n_ciks: int) -> tuple[list[dict], int]:
    """Take the first ``n_ciks`` distinct CIKs in PERMNO order, minus ``done``.

    De-duplicating on CIK means a CIK shared by several PERMNOs is fetched once.
    Firms with no CIK do not consume a slot but are still returned so they log as
    ``no_cik``. Returns (firms to pull, count skipped as already pulled).
    """
    seen, target = set(), []
    for f in firms:
        if len(seen) >= n_ciks:
            break
        if f["cik"]:
            if f["cik"] in seen:
                continue
            seen.add(f["cik"])
        target.append(f)
    todo = [f for f in target if f["cik"] not in done]
    return todo, len(target) - len(todo)


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
# Accumulated pull log                                                         #
# --------------------------------------------------------------------------- #
def load_log() -> pd.DataFrame:
    """Read the accumulated pull log as text.

    Read as str deliberately: the log's CIKs are zero-padded to 10 digits and
    gvkeys to 6, and pandas' default parse would silently turn '0000043350' into
    the integer 43350 — breaking the CIK diff and corrupting the file on rewrite.
    """
    if not LOG_CSV.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_CSV, dtype=str).fillna("")


def _succeeded(log: pd.DataFrame) -> pd.Series:
    """Boolean mask of successful pulls, tolerant of a bool or text found_10k.

    Rows read back from disk are text ('True'), rows built in this run are bool,
    and a concatenated log holds both.
    """
    return log["found_10k"].astype(str).str.strip().str.lower() == "true"


def successful_ciks(log: pd.DataFrame) -> set[str]:
    """CIKs already pulled successfully. Failures are NOT counted as done, so a
    firm that previously failed is retried rather than treated as complete."""
    if log.empty:
        return set()
    return {c.strip().zfill(10) for c in log.loc[_succeeded(log), "cik"] if c.strip()}


def append_log(new: pd.DataFrame) -> pd.DataFrame:
    """Append this run's outcomes to the log; never overwrite earlier entries.

    Rows are not de-duplicated: a firm that failed earlier and succeeds now keeps
    both rows, so the log stays a faithful record of every attempt. "Already
    pulled" is defined as any row with found_10k True, which remains correct.
    """
    old = load_log()
    out = pd.concat([old, new], ignore_index=True)[LOG_COLUMNS] if len(old) else new
    out.to_csv(LOG_CSV, index=False)
    print(f"\nWrote {LOG_CSV.name} ({len(old)} existing + {len(new)} new = {len(out)} rows).")
    return out


def prune_stale(log: pd.DataFrame, universe_ciks: set[str]) -> list[str]:
    """Delete cached filings whose CIK has left the firm universe.

    filings_raw/ is a cache of the current universe; the log is the permanent
    pull history, so its rows are left in place and only the files are removed.
    """
    if log.empty:
        return []
    removed = []
    for _, r in log[_succeeded(log)].iterrows():
        cik, rel = r["cik"].strip().zfill(10), r["local_path"].strip()
        if not rel or cik in universe_ciks:
            continue
        path = BASE / rel.replace("\\", "/")
        if path.exists():
            path.unlink()
            removed.append(path.name)
            print(f"  removed {path.name} (permno={r['permno']} cik={cik}, out of universe)")
    return removed


# --------------------------------------------------------------------------- #
# Pull driver                                                                  #
# --------------------------------------------------------------------------- #
def _pull_firms(firms: list[dict], ref: pd.Timestamp) -> pd.DataFrame:
    """Run the per-firm pull over ``firms`` and return this run's log rows."""
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

    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def pull_gap(n_ciks: int, ref: pd.Timestamp) -> pd.DataFrame:
    """Pull only the universe CIKs not already pulled successfully.

    Scope is the first ``n_ciks`` distinct CIKs of the rebuilt bridge universe in
    PERMNO order, less every CIK already marked successful in the accumulated
    log. Cached filings for firms that have left the universe are pruned first.
    """
    FILINGS_DIR.mkdir(exist_ok=True)
    if USER_AGENT_IS_PLACEHOLDER:
        print("!! WARNING: USER_AGENT is a placeholder — set a real name/email "
              "before any non-pilot run (SEC fair-access requirement).")

    firms = resolve_firms(BRIDGE_CSV, ref)
    universe_ciks = {f["cik"] for f in firms if f["cik"]}
    log = load_log()
    done = successful_ciks(log)
    todo, n_skipped = select_gap_firms(firms, done, n_ciks)
    attempted = {c.strip().zfill(10) for c in log["cik"]} if not log.empty else set()
    n_retry = len([f for f in todo if f["cik"] and f["cik"] in attempted])

    print(f"Universe ({BRIDGE_CSV.name}): {len(firms):,} PERMNOs, "
          f"{len(universe_ciks):,} unique CIKs.")
    print(f"Already pulled successfully: {len(done)} CIKs "
          f"({len(done & universe_ciks)} still in universe, "
          f"{len(done - universe_ciks)} no longer).")
    print(f"Scope: first {n_ciks} unique CIKs by PERMNO -> {n_skipped} already done "
          f"(not re-pulled), {len(todo)} to pull "
          f"({n_retry} previously-failed retries, {len(todo) - n_retry} new).\n")

    removed = prune_stale(log, universe_ciks)
    print(f"Pruned {len(removed)} cached filing(s) no longer in the universe.\n")

    new = _pull_firms(todo, ref)
    return append_log(new)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(log: pd.DataFrame) -> None:
    """Print accumulated-log summary stats and spot-check 3 downloaded filings."""
    n = len(log)
    hit = _succeeded(log)
    ok = log[hit]
    print("\n" + "=" * 70)
    print("VALIDATION (accumulated log)")
    print("=" * 70)
    print(f"Usable 10-K downloaded : {len(ok)} / {n}")
    print(f"Failed                 : {n - len(ok)} / {n}")
    fails = log.loc[~hit, "fail_reason"]
    if len(fails):
        print("Failure reasons:")
        for reason, cnt in fails.value_counts().items():
            print(f"  {reason:24s} {cnt}")

    paths = [BASE / str(p).replace("\\", "/") for p in ok["local_path"] if str(p).strip()]
    paths = [p for p in paths if p.exists()]
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


def print_assumptions(n_ciks: int, ref: pd.Timestamp) -> None:
    """Print the assumptions that governed this run."""
    print("\n" + "=" * 70)
    print("ASSUMPTIONS (this run)")
    print("=" * 70)
    for line in [
        f"Firm universe = {BRIDGE_CSV.name}, rebuilt after the NYSE micro-cap drop.",
        f"Scope = the first {n_ciks} distinct CIKs in ascending PERMNO order "
        "(same ordering as the original pilot); raise --n-ciks to extend coverage.",
        "Work list = scope CIKs MINUS CIKs already marked found_10k=True in the log. "
        "Previously-failed firms are retried; successful ones are never re-pulled.",
        "De-duplication is on CIK, so a CIK shared by several PERMNOs is fetched once "
        "under the lowest PERMNO.",
        "The log is append-only and not de-duplicated: it is the accumulated history of "
        "every attempt, and 'already pulled' means any row with found_10k True.",
        "The log is read as text so zero-padded CIK (10-digit) and gvkey (6-digit) "
        "identifiers survive the round-trip.",
        "Cached filings whose CIK has left the universe are deleted; their log rows "
        "are kept as history.",
        f"Reference date = {ref.date()}; 10-K must be filed in "
        f"[{(ref - pd.Timedelta(days=STALENESS_DAYS)).date()} .. {(ref - pd.Timedelta(days=1)).date()}] "
        "(unchanged from the pilot).",
    ]:
        print(f"  - {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC EDGAR 10-K gap-only pull.")
    ap.add_argument("--n-ciks", type=int, default=N_CIKS,
                    help="cover the first N distinct CIKs by PERMNO (default 50)")
    ap.add_argument("--reference-date", type=str, default=str(REFERENCE_DATE.date()))
    args = ap.parse_args()
    ref = pd.Timestamp(args.reference_date)
    log = pull_gap(args.n_ciks, ref)
    validate(log)
    print_assumptions(args.n_ciks, ref)


if __name__ == "__main__":
    main()
