"""
Part B — SEC EDGAR 10-K retrieval (gap-only pull).

Takes the firm universe from clean_firm_bridge.csv, resolves one CIK per firm at
the reference date, queries the SEC submissions API for each firm's filing
history, selects the most recent 10-K filed strictly before the reference date
(within a staleness window), downloads and caches the primary HTML document, and
logs the per-firm outcome (or the specific reason no filing was retrieved).

The pull is incremental and resumable: the work list is the scope CIKs MINUS every
CIK already complete at this reference date (pulled successfully, or failed for a
structural reason that a retry cannot change). Transient api_error/download_error
firms are always retried. Each outcome is flushed to edgar_pull_log.csv as the firm
finishes, so an interrupted run keeps everything it recorded and re-running the same
command carries on from where it stopped. Cached filings whose firm has left the
universe are pruned. No text parsing, tokenization, or scoring is performed here
(that is Step 2).

    python edgar_pull.py --all                  # full universe (hours; run in background)
    python edgar_pull.py --status               # coverage report; no network calls
    python edgar_pull.py [--n-ciks 50] [--reference-date 2025-04-02] [--retry-all]

IMPORTANT: set a real name/email via the EDGAR_USER_AGENT environment variable (or a
gitignored .env file) before any run - the SEC fair-access policy requires a genuine
contact. If unset, a non-functional placeholder is used and a warning is printed.
"""

import argparse
import csv
import ctypes
import os
import random
import sys
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

MAX_RETRIES = 4               # HTTP attempts per request before the firm is failed
BACKOFF_BASE = 2.0            # backoff seconds = BACKOFF_BASE ** attempt + jitter
RETRY_STATUS = {429, 500, 502, 503, 504}
BLOCK_SLEEP = 600             # SEC fair-access throttles clear in ~10 minutes
MAX_BLOCK_EVENTS = 3          # 403 blocks tolerated in one run
MAX_CONSECUTIVE_ERRORS = 20   # circuit breaker on runaway transient failures
MIN_FILING_BYTES = 5_000      # smaller responses are throttle stubs, not filings
THROTTLE_MARKER = b"undeclared automated tool"   # SEC's rate-limit stub page
WRITE_RETRIES = 5             # OneDrive briefly locks files it is uploading
PROGRESS_EVERY = 25           # firms between progress/ETA lines

# Structural failures are deterministic given a fixed reference date: the firm has no
# CIK, no 10-K on file, or none inside the staleness window. Retrying them only burns
# API calls, so they count as complete. Transient errors are always retried.
STRUCTURAL_FAILURES = {"no_cik", "no_10k_on_file", "no_10k_in_window"}
DEFAULT_REF_DATE = "2025-04-02"   # reference date assumed for pre-column log rows

LOG_COLUMNS = ["permno", "gvkey", "cik", "reference_date", "found_10k", "accession",
               "filing_date", "period_of_report", "local_path", "fail_reason"]
LOG_BACKUP = LOG_CSV.with_name("edgar_pull_log.pre-migration.csv")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_SESSION: requests.Session | None = None


class SecBlocked(Exception):
    """SEC returned 403 - fair-access throttle or a rejected User-Agent."""


class PullAborted(Exception):
    """Run stopped early. The log is already flushed, so re-running resumes."""


def _session() -> requests.Session:
    """Module-level session so TCP/TLS connections are reused across the run.

    A full run makes ~7,000 requests; without reuse each pays a fresh TLS handshake.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": USER_AGENT,
                                 "Accept-Encoding": "gzip, deflate"})
    return _SESSION


def _get(url: str) -> requests.Response:
    """GET with exponential backoff on transient errors.

    Retries RETRY_STATUS responses and connection/timeout errors; a 404 (or any other
    non-retryable status) fails immediately. A 403 is raised as SecBlocked so the
    caller applies the long fair-access cool-off instead of hammering a throttled
    endpoint - the failure mode that would otherwise mark thousands of firms as errors
    in minutes.
    """
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = _session().get(url, timeout=TIMEOUT)
            if r.status_code == 403:
                raise SecBlocked(url)
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()          # 404 etc. - terminal, no retry
                return r
            last = requests.HTTPError(f"HTTP {r.status_code} for {url}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE ** attempt + random.uniform(0, 1))
    raise last


def _prevent_sleep() -> None:
    """Stop Windows sleeping mid-run (the likeliest cause of a multi-hour stoppage).

    Released automatically when the process exits. Does not override a lid-close
    sleep policy.
    """
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def _cached_path(local_path: str) -> Path:
    """Resolve a log local_path to a file under FILINGS_DIR, by name.

    Log rows carry OS-specific separators from earlier runs; resolving by filename
    keeps the lookup correct regardless of separator.
    """
    return FILINGS_DIR / str(local_path).replace("\\", "/").rsplit("/", 1)[-1]


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
                     n_ciks: int | None) -> tuple[list[dict], int]:
    """Take the first ``n_ciks`` distinct CIKs in PERMNO order, minus ``done``.

    ``n_ciks=None`` covers the whole universe. De-duplicating on CIK means a CIK
    shared by several PERMNOs is fetched once. Firms with no CIK do not consume a
    slot but are still returned so they log as ``no_cik``. Returns (firms to pull,
    count skipped as already complete).
    """
    seen, target = set(), []
    for f in firms:
        if n_ciks is not None and len(seen) >= n_ciks:
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
    return _get(SUBMISSIONS_URL.format(cik=cik)).json()


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


def _replace_with_retry(tmp: Path, path: Path) -> None:
    """Atomically move tmp -> path, retrying briefly on Windows/OneDrive file locks."""
    for attempt in range(WRITE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == WRITE_RETRIES - 1:
                raise
            time.sleep(BACKOFF_BASE ** attempt)


def clear_partials() -> int:
    """Delete .part files left behind by a previous interrupted run."""
    stale = list(FILINGS_DIR.glob("*.part"))
    for p in stale:
        p.unlink()
    return len(stale)


def download_primary(permno: int, cik: str, accession: str, doc: str) -> Path:
    """Download and cache the primary HTML document; skip if already cached.

    Written to a .part file and atomically renamed, so a process killed mid-write
    never leaves a truncated .html that a later resume would accept as complete.
    Short responses and SEC's rate-limit stub page are rejected rather than cached,
    which would otherwise poison the corpus silently during a throttle.
    """
    if not doc:
        raise ValueError("no primaryDocument in the submissions record")
    acc = accession.replace("-", "")
    path = FILINGS_DIR / f"{permno}_{int(cik):010d}_{acc}.html"
    if path.exists():
        return path

    r = _get(ARCHIVE_URL.format(cik=int(cik), acc=acc, doc=doc))
    if len(r.content) < MIN_FILING_BYTES:
        raise ValueError(f"response too small to be a filing ({len(r.content)} bytes)")
    if THROTTLE_MARKER in r.content[:4000].lower():
        raise SecBlocked(f"rate-limit stub returned for {path.name}")

    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    _replace_with_retry(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# Accumulated pull log                                                         #
# --------------------------------------------------------------------------- #
def migrate_log() -> bool:
    """Add the reference_date column to a pre-existing log, once.

    Rows written before the column existed were all pulled at DEFAULT_REF_DATE, so
    they are stamped with it. Without the column, "already complete" cannot be scoped
    to a reference date, and the later 2018-vintage pull would wrongly inherit the
    2025 outcomes. The original file is kept as a backup before the rewrite.
    """
    if not LOG_CSV.exists():
        return False
    log = pd.read_csv(LOG_CSV, dtype=str).fillna("")
    if "reference_date" in log.columns:
        return False
    if not LOG_BACKUP.exists():
        LOG_BACKUP.write_bytes(LOG_CSV.read_bytes())
    log["reference_date"] = DEFAULT_REF_DATE
    tmp = LOG_CSV.with_suffix(".csv.tmp")
    log[LOG_COLUMNS].to_csv(tmp, index=False)
    _replace_with_retry(tmp, LOG_CSV)
    print(f"[migrate] {LOG_CSV.name}: added reference_date={DEFAULT_REF_DATE} to "
          f"{len(log)} existing row(s); backup at {LOG_BACKUP.name}.")
    return True


def load_log() -> pd.DataFrame:
    """Read the accumulated pull log as text.

    Read as str deliberately: the log's CIKs are zero-padded to 10 digits and
    gvkeys to 6, and pandas' default parse would silently turn '0000043350' into
    the integer 43350 — breaking the CIK diff and corrupting the file on rewrite.
    Rows predating the reference_date column are backfilled with DEFAULT_REF_DATE.
    """
    if not LOG_CSV.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(LOG_CSV, dtype=str).fillna("")
    if "reference_date" not in log.columns:
        log["reference_date"] = DEFAULT_REF_DATE
    log["reference_date"] = log["reference_date"].replace("", DEFAULT_REF_DATE)
    return log


def _succeeded(log: pd.DataFrame) -> pd.Series:
    """Boolean mask of successful pulls, tolerant of a bool or text found_10k.

    Rows read back from disk are text ('True'), rows built in this run are bool,
    and a concatenated log holds both.
    """
    return log["found_10k"].astype(str).str.strip().str.lower() == "true"


def successful_ciks(log: pd.DataFrame) -> set[str]:
    """CIKs pulled successfully at any reference date, i.e. those holding a file.

    Used only to decide which cached files to keep; completion is decided by
    completed_ciks, which is reference-date aware.
    """
    if log.empty:
        return set()
    return {c.strip().zfill(10) for c in log.loc[_succeeded(log), "cik"] if c.strip()}


def completed_ciks(log: pd.DataFrame, ref: pd.Timestamp,
                   retry_all: bool = False) -> set[str]:
    """CIKs needing no further work at this reference date.

    Complete = pulled successfully, or failed for a structural reason, which for a
    fixed reference date cannot change on a retry (Step 1c confirmed this: all 11
    retries failed identically). Transient api_error/download_error rows are always
    retried. Scoped by reference_date so a later 2018-vintage pull starts clean.
    ``retry_all`` restores the old success-only rule.
    """
    if log.empty:
        return set()
    same_ref = log[log["reference_date"] == str(ref.date())]
    if same_ref.empty:
        return set()
    terminal = _succeeded(same_ref)
    if not retry_all:
        terminal = terminal | same_ref["fail_reason"].isin(STRUCTURAL_FAILURES)
    return {c.strip().zfill(10) for c in same_ref.loc[terminal, "cik"] if c.strip()}


def open_log():
    """Open the accumulated log for append, writing the header if it is new.

    Returns (file handle, DictWriter). Rows are flushed as each firm completes so an
    interrupted run keeps every outcome it recorded; the end-of-run write this
    replaced discarded the entire run. Append-only and non-deduplicated semantics are
    unchanged: a firm that failed earlier and succeeds now keeps both rows.
    """
    is_new = not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0
    fh = LOG_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


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
        path = _cached_path(rel)
        if path.exists():
            path.unlink()
            removed.append(path.name)
            print(f"  removed {path.name} (permno={r['permno']} cik={cik}, out of universe)")
    return removed


# --------------------------------------------------------------------------- #
# Pull driver                                                                  #
# --------------------------------------------------------------------------- #
def _pull_one(f: dict, ref: pd.Timestamp) -> dict:
    """Resolve, download and describe one firm's 10-K as a log row.

    SecBlocked propagates to the caller, which applies the fair-access cool-off; every
    other failure is captured in fail_reason so the firm is retried on the next run.
    """
    rec = {"permno": f["permno"], "gvkey": f["gvkey"], "cik": f["cik"],
           "reference_date": str(ref.date()), "found_10k": False, "accession": "",
           "filing_date": "", "period_of_report": "", "local_path": "",
           "fail_reason": ""}

    if not f["cik"]:
        rec["fail_reason"] = "no_cik"
        return rec

    try:
        subs = get_submissions(f["cik"])
        time.sleep(REQUEST_SLEEP)
    except SecBlocked:
        raise
    except Exception as e:  # network / HTTP / JSON
        rec["fail_reason"] = f"api_error: {type(e).__name__}"
        return rec

    sel, reason = select_10k(subs, ref)
    if sel is None:
        rec["fail_reason"] = reason
        return rec

    rec.update(accession=sel["accession"], filing_date=sel["filing_date"],
               period_of_report=sel["period_of_report"])
    try:
        path = download_primary(f["permno"], f["cik"], sel["accession"],
                                sel["primary_document"])
        time.sleep(REQUEST_SLEEP)
    except SecBlocked:
        raise
    except Exception as e:
        rec["fail_reason"] = f"download_error: {type(e).__name__}"
        return rec

    rec.update(found_10k=True, local_path=str(path.relative_to(BASE)))
    return rec


def _pull_firms(firms: list[dict], ref: pd.Timestamp, writer, fh) -> dict:
    """Run the per-firm pull, flushing each outcome to the log as it completes.

    Two guards bound the damage a bad run can do. A 403 pauses for BLOCK_SLEEP and
    retries the firm once, aborting past MAX_BLOCK_EVENTS rather than continuing into
    a throttle. MAX_CONSECUTIVE_ERRORS transient failures in a row abort as well,
    since that signals a broken network rather than firm-specific problems. Both
    raise PullAborted, and every row already written stays on disk, so re-running the
    same command resumes. Returns this run's outcome counts.
    """
    total = len(firms)
    lo = (ref - pd.Timedelta(days=STALENESS_DAYS)).date()
    print(f"Pulling 10-Ks for {total:,} firms (reference date {ref.date()}, window "
          f"[{lo} .. {(ref - pd.Timedelta(days=1)).date()}])\n", flush=True)

    counts = {"ok": 0, "failed": 0}
    consecutive, blocks, bytes_got, t0 = 0, 0, 0, time.monotonic()

    for i, f in enumerate(firms, 1):
        for block_attempt in range(2):
            try:
                rec = _pull_one(f, ref)
                break
            except SecBlocked as exc:
                blocks += 1
                if block_attempt or blocks > MAX_BLOCK_EVENTS:
                    raise PullAborted(
                        f"SEC 403 persisted ({blocks} block event(s)): {exc}. "
                        f"{i - 1} firm(s) logged this run.") from exc
                print(f"!! SEC 403 (block {blocks}/{MAX_BLOCK_EVENTS}); sleeping "
                      f"{BLOCK_SLEEP}s before retrying permno={f['permno']}", flush=True)
                time.sleep(BLOCK_SLEEP)

        writer.writerow(rec)
        fh.flush()

        if rec["found_10k"]:
            counts["ok"] += 1
            consecutive = 0
            path = _cached_path(rec["local_path"])
            bytes_got += path.stat().st_size if path.exists() else 0
            print(f"[{i}/{total}] permno={f['permno']:<7} cik={f['cik']} "
                  f"OK  filed={rec['filing_date']} period={rec['period_of_report']}",
                  flush=True)
        else:
            counts["failed"] += 1
            reason = rec["fail_reason"]
            if reason.startswith(("api_error", "download_error")):
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_ERRORS:
                    raise PullAborted(
                        f"{consecutive} consecutive transient failures at "
                        f"permno={f['permno']}; network or SEC access is broken.")
            else:
                consecutive = 0
            print(f"[{i}/{total}] permno={f['permno']:<7} cik={f['cik']} {reason}",
                  flush=True)

        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = time.monotonic() - t0
            eta = (total - i) * elapsed / i / 60
            print(f"[progress] {i:,}/{total:,} ({i / total:.1%}) | ok={counts['ok']:,} "
                  f"failed={counts['failed']:,} | {elapsed / 60:.1f} min elapsed | "
                  f"ETA {eta:.0f} min | {bytes_got / 1024 ** 3:.2f} GB", flush=True)

    return counts


def pull_gap(n_ciks: int | None, ref: pd.Timestamp,
             retry_all: bool = False) -> pd.DataFrame:
    """Pull only the universe CIKs not already complete at this reference date.

    Scope is the first ``n_ciks`` distinct CIKs of the rebuilt bridge universe in
    PERMNO order (``None`` = the whole universe), less every CIK already complete
    (see completed_ciks). Cached
    filings for firms that have left the universe are pruned and .part files from an
    interrupted run are cleared first. The log is written row-by-row, so an
    interruption at any point leaves a resumable state.
    """
    FILINGS_DIR.mkdir(exist_ok=True)
    _prevent_sleep()
    if USER_AGENT_IS_PLACEHOLDER:
        print("!! WARNING: USER_AGENT is a placeholder — set a real name/email "
              "before any non-pilot run (SEC fair-access requirement).")
    migrate_log()

    firms = resolve_firms(BRIDGE_CSV, ref)
    universe_ciks = {f["cik"] for f in firms if f["cik"]}
    log = load_log()
    done = completed_ciks(log, ref, retry_all)
    todo, n_skipped = select_gap_firms(firms, done, n_ciks)
    attempted = {c.strip().zfill(10) for c in log["cik"]} if not log.empty else set()
    n_retry = len([f for f in todo if f["cik"] and f["cik"] in attempted])

    print(f"Universe ({BRIDGE_CSV.name}): {len(firms):,} PERMNOs, "
          f"{len(universe_ciks):,} unique CIKs.")
    print(f"Already complete at {ref.date()}: {len(done):,} CIKs "
          f"({len(done & universe_ciks):,} still in universe, "
          f"{len(done - universe_ciks):,} no longer)"
          f"{' [--retry-all: successes only]' if retry_all else ''}.")
    scope = "whole universe" if n_ciks is None else f"first {n_ciks:,} unique CIKs by PERMNO"
    print(f"Scope: {scope} -> {n_skipped:,} already done "
          f"(not re-pulled), {len(todo):,} to pull "
          f"({n_retry:,} previously-failed retries, {len(todo) - n_retry:,} new).")

    n_part = clear_partials()
    removed = prune_stale(log, universe_ciks)
    print(f"Cleared {n_part} partial download(s); pruned {len(removed)} cached "
          f"filing(s) no longer in the universe.\n")

    fh, writer = open_log()
    try:
        counts = _pull_firms(todo, ref, writer, fh)
        print(f"\nRun complete: {counts['ok']:,} pulled, {counts['failed']:,} failed.")
    finally:
        fh.close()
    return load_log()


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

    paths = [_cached_path(p) for p in ok["local_path"] if str(p).strip()]
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


def print_status(ref: pd.Timestamp) -> None:
    """Report pull coverage from the log and disk only — no network calls.

    Safe to run while a pull is in flight: the log is appended row-by-row, so the
    counts are current as of the last completed firm.
    """
    migrate_log()
    firms = resolve_firms(BRIDGE_CSV, ref)
    universe = {f["cik"] for f in firms if f["cik"]}
    log = load_log()
    done = completed_ciks(log, ref) & universe
    same_ref = log[log["reference_date"] == str(ref.date())] if not log.empty else log
    files = list(FILINGS_DIR.glob("*.html"))
    size = sum(p.stat().st_size for p in files)

    print("=" * 70)
    print(f"PULL STATUS  (reference date {ref.date()})")
    print("=" * 70)
    print(f"Universe CIKs          : {len(universe):,}")
    print(f"Complete               : {len(done):,} ({len(done) / len(universe):.1%})")
    print(f"Remaining              : {len(universe - done):,}")
    print(f"Log rows (this ref)    : {len(same_ref):,} of {len(log):,} total")
    if len(same_ref):
        outcomes = same_ref["fail_reason"].replace("", "ok").value_counts()
        for reason, n in outcomes.items():
            print(f"  {reason:24s} {n:,}")
    print(f"Cached files           : {len(files):,} ({size / 1024 ** 3:.2f} GB)")

    if len(log):
        ok = log[_succeeded(log)]
        missing = [r.local_path for r in ok.itertuples(index=False)
                   if str(r.local_path).strip() and not _cached_path(r.local_path).exists()]
        print(f"Successes missing file : {len(missing):,} "
              f"(expected for pruned out-of-universe firms)")
    parts = list(FILINGS_DIR.glob("*.part"))
    if parts:
        print(f"Orphan .part files     : {len(parts)} (cleared on the next run)")
    truncated = [p.name for p in files if p.stat().st_size < MIN_FILING_BYTES]
    if truncated:
        print(f"!! Suspiciously small files: {len(truncated)} {truncated[:5]}")


def print_assumptions(n_ciks: int | None, ref: pd.Timestamp) -> None:
    """Print the assumptions that governed this run."""
    print("\n" + "=" * 70)
    print("ASSUMPTIONS (this run)")
    print("=" * 70)
    for line in [
        f"Firm universe = {BRIDGE_CSV.name}, rebuilt after the NYSE micro-cap drop.",
        ("Scope = the whole universe." if n_ciks is None else
         f"Scope = the first {n_ciks:,} distinct CIKs in ascending PERMNO order "
         "(same ordering as the original pilot); --all covers the whole universe."),
        "Work list = scope CIKs MINUS CIKs already complete at this reference date. "
        "Complete = pulled successfully, or failed structurally (no_cik / no_10k_on_file "
        "/ no_10k_in_window), which a retry cannot change while the reference date is "
        "fixed. Transient api_error / download_error firms are always retried; "
        "--retry-all falls back to the old success-only rule.",
        "De-duplication is on CIK, so a CIK shared by several PERMNOs is fetched once "
        "under the lowest PERMNO.",
        "The log is append-only and not de-duplicated: it is the accumulated history of "
        "every attempt. Each row is flushed as its firm finishes, so an interrupted run "
        "keeps every outcome it recorded and re-running the command resumes.",
        "Every log row carries the reference_date it was pulled at, so outcomes from one "
        "policy cycle are never treated as complete for another. Rows predating the "
        f"column are stamped {DEFAULT_REF_DATE}.",
        "The log is read as text so zero-padded CIK (10-digit) and gvkey (6-digit) "
        "identifiers survive the round-trip.",
        "Filings are written to a .part file and atomically renamed, so a killed process "
        "cannot leave a truncated .html that a later run would accept as complete. "
        f"Responses under {MIN_FILING_BYTES:,} bytes or carrying SEC's rate-limit stub "
        "are rejected, not cached.",
        f"Transient HTTP errors are retried {MAX_RETRIES}x with exponential backoff; a 403 "
        f"pauses {BLOCK_SLEEP}s (SEC fair-access throttle) for up to {MAX_BLOCK_EVENTS} "
        f"block events; {MAX_CONSECUTIVE_ERRORS} consecutive transient failures abort the "
        "run rather than marking the rest of the universe as errors.",
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
    ap.add_argument("--all", action="store_true",
                    help="cover the entire universe (overrides --n-ciks)")
    ap.add_argument("--status", action="store_true",
                    help="report pull coverage from the log and disk; no network calls")
    ap.add_argument("--retry-all", action="store_true",
                    help="also retry structural failures, not just transient ones")
    ap.add_argument("--reference-date", type=str, default=str(REFERENCE_DATE.date()))
    args = ap.parse_args()
    ref = pd.Timestamp(args.reference_date)

    if args.status:
        print_status(ref)
        return

    n_ciks = None if args.all else args.n_ciks   # None = no limit
    try:
        log = pull_gap(n_ciks, ref, args.retry_all)
    except PullAborted as exc:
        print(f"\n!! ABORTED: {exc}\n   The log is flushed up to this point - re-run the "
              f"same command to resume.", file=sys.stderr)
        sys.exit(1)
    validate(log)
    print_assumptions(n_ciks, ref)
    print_status(ref)


if __name__ == "__main__":
    main()
