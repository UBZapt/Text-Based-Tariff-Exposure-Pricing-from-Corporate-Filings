"""
Part B — SEC EDGAR 10-K retrieval (gap-only pull).

Takes the firm list for the requested scope, resolves one CIK per firm at the
reference date, queries the SEC submissions API for each firm's filing history,
selects the most recent 10-K filed strictly before the reference date (within a
staleness window), downloads and caches the primary HTML document, and logs the
per-firm outcome (or the specific reason no filing was retrieved).

Two scopes and two named multi-date batches (Step 1e):

    --scope full       clean_firm_bridge.csv, the default and unchanged
    --scope subsample  output/full_panel_firm_sample.csv, the fixed 1,000-firm draw
    --batch cross_cycle  9 Section 301 event dates, 2018-2020, at scope full   (7.6)
    --batch full_panel   9 annual dates, 2017-2025, at scope subsample         (7.4/7.5)

Whatever the scope, the CIK for each PERMNO is still resolved against the bridge per
(PERMNO, reference_date): the persisted lists supply which firms to pull, never their
identifiers. Cache retention is likewise always decided against the bridge.

The pull is incremental and resumable: the work list is the scope CIKs MINUS every
CIK already complete at this reference date (pulled successfully, or failed for a
structural reason that a retry cannot change). Transient api_error/download_error
firms are always retried. Each outcome is flushed to edgar_pull_log.csv as the firm
finishes, so an interrupted run keeps everything it recorded and re-running the same
command carries on from where it stopped. Cached filings whose firm has left the
universe are pruned. No text parsing, tokenization, or scoring is performed here
(that is Step 2).

    python src/edgar_pull.py --all                  # full universe (hours; run in background)
    python src/edgar_pull.py --status               # coverage report; no network calls
    python src/edgar_pull.py --status --batch cross_cycle      # per-date coverage matrix
    python src/edgar_pull.py --batch cross_cycle               # download-bound; run in background
    python src/edgar_pull.py --batch full_panel                # run cross_cycle first: it caches
                                                          # nearly all the 1,001 CIKs it needs
    python src/edgar_pull.py [--n-ciks 50] [--reference-date 2025-04-02] [--scope full] [--retry-all]

IMPORTANT: set a real name/email via the EDGAR_USER_AGENT environment variable (or a
gitignored .env file) before any run - the SEC fair-access policy requires a genuine
contact. If unset, a non-functional placeholder is used and a warning is printed.
"""

import argparse
import csv
import ctypes
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from config import (BASE, CLEAN_DIR, EDGAR_LOG as LOG_CSV, ENV_FILE, FILINGS_DIR, OUTPUT_DIR,
                    SUBMISSIONS_DIR)


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
BRIDGE_CSV = CLEAN_DIR / "clean_firm_bridge.csv"
# SUBMISSIONS_DIR holds gzipped submissions JSON, one file per CIK. The submissions history is
# identical for every reference date, so one fetch serves a whole multi-date batch (see
# get_submissions). LOG_CSV is resumable pull state paired with FILINGS_DIR, not an analysis
# output: if it cannot be found, every CIK reads as incomplete and re-pulls.

# SEC fair access requires a descriptive User-Agent "AppName ContactEmail".
# Supply a real, monitored contact via the EDGAR_USER_AGENT environment variable
# (set it in your shell or a gitignored .env file); never hard-code it. The
# default below is a non-functional placeholder that triggers a warning.
_load_dotenv(ENV_FILE)
_PLACEHOLDER_UA = "TariffFactorResearch your-email@example.com"
USER_AGENT = os.environ.get("EDGAR_USER_AGENT", _PLACEHOLDER_UA)
USER_AGENT_IS_PLACEHOLDER = USER_AGENT == _PLACEHOLDER_UA

REFERENCE_DATE = pd.Timestamp("2025-04-02")  # "Liberation Day" tariffs
STALENESS_DAYS = 364                          # filing_date >= ref - 364 days
N_CIKS = 50                                   # unique CIKs to cover this run
REQUEST_SLEEP = 0.15                          # min seconds between requests; 6.7/s < SEC's 10/s
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

# --- Firm-list sourcing and multi-date batches (Step 1e) ------------------- #
# Written by persist_2025_universe.py and sample_full_panel_firms.py. Read here directly rather
# than by import: those modules import this one, so importing back would be circular.
UNIVERSE_CSV = OUTPUT_DIR / "event_study_firm_universe.csv"   # 2025 event-study universe (audit)
SAMPLE_CSV = OUTPUT_DIR / "full_panel_firm_sample.csv"        # fixed 1,000-firm draw, section 7.0

# Pull scopes. 'full' is the bridge, unchanged from every previous run: the bridge is
# ever-qualifying across 2017-2026 while the persisted universe is point-in-time at end-March
# 2025, and filtering a 2018 event pull through a 2025 screen would impose a survivorship
# filter the design does not ask for (section 7.2 screens point-in-time at each event).
# 'subsample' restricts to the persisted draw, for the annual full-panel refresh only.
SCOPES = {"full": None, "subsample": SAMPLE_CSV}

# Annual full-panel refresh, section 7.0. April 2 each year, so 2025-04-02 is a member and is
# already complete - completed_ciks() skips it with no special-casing. With the 364-day window a
# December-FY firm gets its FY(y-1) 10-K filed in Feb/Mar of year y.
FULL_PANEL_DATES = [f"{year}-04-02" for year in range(2017, 2026)]

# Cross-cycle event pull, section 7.6. Seven trade-war escalations from Bruno, Goltz & Luyten
# (2024) Table 3 (sourced from Amiti, Kong & Weinstein 2020), used exactly as published so the
# event-date selection carries no discretion, plus the two reversal dates: 2019-10-11 (primary)
# and 2020-01-15 (Phase One, robustness).
CROSS_CYCLE_DATES = ["2018-03-01", "2018-03-22", "2018-04-02", "2018-06-15", "2018-09-17",
                     "2019-05-10", "2019-08-23", "2019-10-11", "2020-01-15"]

# 2018-04-02 is in both lists. Run cross_cycle first: it covers that date at bridge scope, which
# makes it free for full_panel, since completion is keyed on (CIK, reference_date) not on scope.
BATCHES = {"full_panel": (FULL_PANEL_DATES, "subsample"),
           "cross_cycle": (CROSS_CYCLE_DATES, "full")}

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


_last_request = 0.0


def _throttle() -> None:
    """Block until at least REQUEST_SLEEP has passed since the previous request.

    Enforced here rather than by the caller so the floor holds per *request*: retries
    inside _get are spaced too, and firms answered from a cache without issuing a
    request cost nothing. Caps the run at 1/REQUEST_SLEEP req/s against SEC's 10/s
    fair-access limit, for every code path, present and future.
    """
    global _last_request
    wait = REQUEST_SLEEP - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


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
            _throttle()
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

    # 97% of PERMNOs hold exactly one bridge row, where _resolve_cik has no choice to make and
    # returns that row whatever the reference date. Splitting them out cuts a groupby-apply over
    # 4,291 groups to one over ~120 (5.8 s -> 0.2 s), which matters because --status resolves
    # once per batch date. _resolve_cik itself is unchanged and still decides every real tie.
    n_rows = bridge["_permno"].map(bridge["_permno"].value_counts())
    chosen = {int(r["_permno"]): r
              for _, r in bridge[n_rows.eq(1)].iterrows()}
    for permno, group in bridge[n_rows.gt(1)].groupby("_permno"):
        chosen[int(permno)] = _resolve_cik(group, ref)

    return [{"permno": permno,
             "gvkey": chosen[permno]["gvkey"],
             "cik": (chosen[permno]["cik"] if isinstance(chosen[permno]["cik"], str)
                     else "").strip()}
            for permno in sorted(chosen)]


def bridge_ciks(bridge_csv: Path = BRIDGE_CSV) -> set[str]:
    """Every CIK appearing anywhere in the bridge, independent of any reference date.

    This is the cache-retention set, and it must NOT be the date-resolved universe that
    resolve_firms returns: _resolve_cik picks one link row per PERMNO according to which
    interval covers the reference date, so the resolved CIK set shifts with the date (4,230 at
    2025-04-02, 4,190 at 2019-10-11). Pruning against a date-resolved set therefore deletes
    filings pulled at a different date - harmless while 2025-04-02 was the only date in use,
    but destructive as soon as the cache spans several. data/filings/raw/ spans every reference
    date, so retention has to as well.
    """
    cik = pd.read_csv(bridge_csv, dtype=str, usecols=["cik"])["cik"].fillna("")
    return {c.strip().zfill(10) for c in cik if c.strip()}


def read_firm_list(path: Path) -> list[int]:
    """Read a persisted PERMNO list (universe or subsample), skipping provenance comments.

    ``comment='#'`` is required for the sample file, whose header carries the recorded seed
    and draw date. Raises rather than falling back to the bridge: a silently wider pull would
    cost hours of SEC requests before anyone noticed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run persist_2025_universe.py, then "
            f"sample_full_panel_firms.py, before pulling at this scope.")
    frame = pd.read_csv(path, dtype=str, comment="#")
    return sorted(int(p) for p in frame["permno"])


def scope_firms(scope: str, bridge_firms: list[dict]) -> list[dict]:
    """Restrict an already-resolved bridge firm list to the requested scope.

    'full' returns it unchanged, so the default scope is byte-identical to every previous run.
    'subsample' keeps only the persisted 1,000-firm draw. The CIK is never read from the
    persisted file - it is resolved by _resolve_cik per (PERMNO, reference_date) in
    resolve_firms, so a link that changed between reference dates resolves correctly at each.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {sorted(SCOPES)}.")
    path = SCOPES[scope]
    if path is None:
        return bridge_firms
    keep = set(read_firm_list(path))
    firms = [f for f in bridge_firms if f["permno"] in keep]
    missing = keep - {f["permno"] for f in firms}
    if missing:
        print(f"!! {len(missing)} PERMNO(s) in {path.name} are absent from "
              f"{BRIDGE_CSV.name} and cannot be pulled: {sorted(missing)[:10]}")
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
def get_submissions(cik: str, ref: pd.Timestamp) -> dict:
    """Return a firm's submission history, reusing a cached copy when valid for ref.

    The submissions file is an append-only filing history and does not depend on the
    reference date, so one fetch serves an entire multi-date batch. A cached copy is
    valid for ref only if it was fetched at or after ref, since everything filed up to
    ref-1 is then already present; otherwise it is refetched and the newer copy - valid
    for that ref and every earlier one - replaces it. Which 10-K gets selected is
    unaffected: select_10k computes that locally from whichever copy is used.
    """
    path = SUBMISSIONS_DIR / f"CIK{int(cik):010d}.json.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            cached = json.load(fh)
        if pd.Timestamp(cached["fetched_at"]) >= ref:
            return cached["submissions"]
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError):
        path.unlink(missing_ok=True)      # truncated, corrupt or written by an older format

    subs = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump({"fetched_at": pd.Timestamp.now().isoformat(), "submissions": subs}, fh)
    _replace_with_retry(tmp, path)
    return subs


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

    data/filings/raw/ is a cache of the current universe; the log is the permanent
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
        subs = get_submissions(f["cik"], ref)
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
             retry_all: bool = False, scope: str = "full") -> pd.DataFrame:
    """Pull only the scope CIKs not already complete at this reference date.

    Scope is the first ``n_ciks`` distinct CIKs of the requested firm list in PERMNO order
    (``None`` = the whole list), less every CIK already complete (see completed_ciks).
    ``scope='full'`` is the rebuilt bridge, unchanged from every previous run; 'subsample' is
    the persisted 1,000-firm draw. Cached filings for firms that have left the *bridge* are
    pruned and .part files from an interrupted run are cleared first. The log is written
    row-by-row, so an interruption at any point leaves a resumable state.
    """
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    _prevent_sleep()
    if USER_AGENT_IS_PLACEHOLDER:
        print("!! WARNING: USER_AGENT is a placeholder — set a real name/email "
              "before any non-pilot run (SEC fair-access requirement).")
    migrate_log()

    bridge_firms = resolve_firms(BRIDGE_CSV, ref)
    firms = scope_firms(scope, bridge_firms)
    # Retention is decided against the whole bridge and independently of this reference date -
    # never the active scope, and never the date-resolved universe (see bridge_ciks).
    # data/filings/raw/ is a cache of the bridge across every reference date; pruning against a
    # narrower or date-specific set would delete filings the full-universe scoring and the
    # section 8 unscreened cross-section depend on.
    retain_ciks = bridge_ciks()
    universe_ciks = {f["cik"] for f in firms if f["cik"]}
    log = load_log()
    done = completed_ciks(log, ref, retry_all)
    todo, n_skipped = select_gap_firms(firms, done, n_ciks)
    attempted = {c.strip().zfill(10) for c in log["cik"]} if not log.empty else set()
    n_retry = len([f for f in todo if f["cik"] and f["cik"] in attempted])

    source = BRIDGE_CSV.name if SCOPES[scope] is None else SCOPES[scope].name
    print(f"Scope '{scope}' ({source}): {len(firms):,} PERMNOs, "
          f"{len(universe_ciks):,} unique CIKs.")
    print(f"Already complete at {ref.date()}: {len(done):,} CIKs "
          f"({len(done & universe_ciks):,} still in scope, "
          f"{len(done - universe_ciks):,} not)"
          f"{' [--retry-all: successes only]' if retry_all else ''}.")
    limit = "whole scope" if n_ciks is None else f"first {n_ciks:,} unique CIKs by PERMNO"
    print(f"Work list: {limit} -> {n_skipped:,} already done "
          f"(not re-pulled), {len(todo):,} to pull "
          f"({n_retry:,} previously-failed retries, {len(todo) - n_retry:,} new).")

    n_part = clear_partials()
    removed = prune_stale(log, retain_ciks)
    print(f"Cleared {n_part} partial download(s); pruned {len(removed)} cached "
          f"filing(s) no longer in the bridge.\n")

    fh, writer = open_log()
    try:
        counts = _pull_firms(todo, ref, writer, fh)
        print(f"\nRun complete: {counts['ok']:,} pulled, {counts['failed']:,} failed.")
    finally:
        fh.close()
    return load_log()


def missing_cached_files(log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Successful log rows whose cached file is gone, split by whether it should come back.

    Returns (repairable, expected). A row is *expected* to be missing when its CIK has left
    the bridge: prune_stale deleted the file on purpose and re-downloading it would only have
    the next pull delete it again. Only in-bridge rows are repairable. print_status has always
    reported the combined count; the split is what makes it actionable.
    """
    empty = log.iloc[0:0]
    if log.empty:
        return empty, empty
    ok = log[_succeeded(log)]
    gone = [bool(str(r.local_path).strip()) and not _cached_path(r.local_path).exists()
            for r in ok.itertuples(index=False)]
    missing = ok[pd.Series(gone, index=ok.index)]
    if missing.empty:
        return empty, empty
    retain = bridge_ciks()
    in_bridge = missing["cik"].str.strip().str.zfill(10).isin(retain)
    return missing[in_bridge], missing[~in_bridge]


def restore_missing(dry_run: bool = False) -> pd.DataFrame:
    """Re-download cached filings that a successful log row points at but disk no longer holds.

    Runs the ordinary per-firm path at each row's own reference date, so select_10k is
    deterministic and resolves to the same accession and the same cached filename. Firms are
    de-duplicated on (cik, reference_date). The log stays append-only: the repair adds a fresh
    row rather than editing the original, matching how a retried failure is already recorded.
    """
    migrate_log()
    log = load_log()
    missing, expected = missing_cached_files(log)
    print("=" * 70)
    print("RESTORE MISSING CACHED FILINGS")
    print("=" * 70)
    if not expected.empty:
        print(f"Skipping {len(expected):,} row(s) whose CIK has left the bridge "
              f"({expected['cik'].nunique():,} firm(s)): prune_stale deleted those files "
              f"deliberately,\n  and re-downloading them would only have the next pull delete "
              f"them again.")
    if missing.empty:
        print("Nothing to restore: every repairable success has its file on disk.")
        return log

    by_ref: dict[str, list[dict]] = {}
    for ref_date, group in missing.groupby("reference_date"):
        seen, firms = set(), []
        for r in group.itertuples(index=False):
            key = r.cik.strip().zfill(10)
            if not key or key in seen:
                continue
            seen.add(key)
            firms.append({"permno": int(r.permno), "gvkey": r.gvkey, "cik": r.cik.strip()})
        by_ref[ref_date] = firms
        print(f"  {ref_date}: {len(group):,} missing row(s) -> {len(firms):,} firm(s) to refetch")

    if dry_run:
        print("\n--dry-run: nothing fetched. Re-run without it to restore.")
        return log

    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    _prevent_sleep()
    clear_partials()
    fh, writer = open_log()
    try:
        for ref_date, firms in by_ref.items():
            print(f"\nRestoring {len(firms):,} firm(s) at {ref_date} ...")
            counts = _pull_firms(firms, pd.Timestamp(ref_date), writer, fh)
            print(f"  {counts['ok']:,} restored, {counts['failed']:,} failed.")
    finally:
        fh.close()
    still, still_expected = missing_cached_files(load_log())
    print(f"\nStill missing after restore: {len(still):,} repairable, "
          f"{len(still_expected):,} expected (out of bridge).")
    return load_log()


def run_batch(name: str, retry_all: bool = False) -> pd.DataFrame:
    """Run one named batch's reference dates in sequence, in this process.

    Each date is an ordinary pull_gap call, so completion, caching and resumption behave
    exactly as for a single-date run: nothing already complete at a (CIK, reference_date) pair
    is re-pulled and nothing already cached under an accession is re-downloaded. PullAborted
    propagates rather than being swallowed - a 403 storm should stop the batch, not push on
    into SEC's throttle - and re-running the identical command resumes from the log.
    """
    dates, scope = BATCHES[name]
    log = load_log()
    for i, date in enumerate(dates, start=1):
        print(f"\n{'=' * 70}\nBATCH {name}  [{i}/{len(dates)}]  "
              f"reference_date={date}  scope={scope}\n{'=' * 70}")
        log = pull_gap(None, pd.Timestamp(date), retry_all, scope)
    print(f"\n{'=' * 70}\nBATCH {name} COMPLETE: {len(dates)} reference date(s) at "
          f"scope '{scope}'.\n{'=' * 70}")
    return log


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


def _firm_list_sizes() -> dict[str, int | None]:
    """PERMNO counts of the two persisted firm lists; None where a file is absent."""
    sizes = {}
    for path in (UNIVERSE_CSV, SAMPLE_CSV):
        try:
            sizes[path.name] = len(read_firm_list(path))
        except FileNotFoundError:
            sizes[path.name] = None
    return sizes


def print_coverage_matrix(dates: list[str], scope: str, log: pd.DataFrame) -> None:
    """Print complete/remaining per reference date at one scope, from the log alone.

    Scope membership is derived from the persisted lists, so the log schema is unchanged - it
    carries no scope column and does not need one. A CIK is counted complete if it is complete
    at that reference date under ANY scope, since completion is keyed on (CIK, reference_date):
    a date already covered by a wider scope is genuinely done for a narrower one.
    """
    print(f"\n  scope '{scope}'")
    print(f"  {'reference_date':<16}{'scope CIKs':>12}{'complete':>11}{'remaining':>11}"
          f"{'pct':>8}   outcomes at this date")
    print("  " + "-" * 96)
    for date in dates:
        ref = pd.Timestamp(date)
        firms = scope_firms(scope, resolve_firms(BRIDGE_CSV, ref))
        ciks = {f["cik"] for f in firms if f["cik"]}
        done = completed_ciks(log, ref) & ciks
        same_ref = log[log["reference_date"] == date] if not log.empty else log
        outcomes = ""
        if len(same_ref):
            counts = same_ref["fail_reason"].replace("", "ok").value_counts()
            outcomes = "  ".join(f"{r}={n:,}" for r, n in counts.head(4).items())
        print(f"  {date:<16}{len(ciks):>12,}{len(done):>11,}{len(ciks - done):>11,}"
              f"{len(done) / max(len(ciks), 1):>8.1%}   {outcomes}")


def print_status(ref: pd.Timestamp, batch: str | None = None) -> None:
    """Report pull coverage from the log and disk only — no network calls.

    Safe to run while a pull is in flight: the log is appended row-by-row, so the
    counts are current as of the last completed firm. ``batch`` adds the per-date
    coverage matrix for that batch's reference dates.
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

    # Persisted firm lists, reported beside the coverage so a mismatch between the two is
    # visible here rather than discovered downstream.
    print("\nPersisted firm lists:")
    print(f"  {BRIDGE_CSV.name:<32} {len(firms):>7,} PERMNOs  {len(universe):>6,} CIKs  "
          f"(scope 'full')")
    for name, n in _firm_list_sizes().items():
        scope_of = "scope 'subsample'" if name == SAMPLE_CSV.name else "draw pool / audit"
        print(f"  {name:<32} {'absent' if n is None else format(n, ',') + ' PERMNOs':>7}"
              f"{'':>17}  {scope_of}")
    sizes = _firm_list_sizes()
    n_uni, n_sam = sizes[UNIVERSE_CSV.name], sizes[SAMPLE_CSV.name]
    if n_uni is not None and n_sam is not None:
        try:
            nested = set(read_firm_list(SAMPLE_CSV)) <= set(read_firm_list(UNIVERSE_CSV))
        except FileNotFoundError:
            nested = False
        print(f"  sample subset of universe: {nested}"
              f"{'' if nested else '   !! MISMATCH - redraw from the current universe'}")

    if batch:
        dates, scope = BATCHES[batch]
        print(f"\n{'=' * 70}\nBATCH COVERAGE: {batch}\n{'=' * 70}")
        print_coverage_matrix(dates, scope, log)

    if len(log):
        repairable, expected = missing_cached_files(log)
        print(f"Successes missing file : {len(repairable) + len(expected):,}")
        print(f"  out of bridge        : {len(expected):,} "
              f"({expected['cik'].nunique() if len(expected) else 0:,} firm(s)) - expected, "
              f"pruned on purpose")
        print(f"  repairable           : {len(repairable):,}"
              f"{'  -> --restore-missing' if len(repairable) else ''}")
    parts = list(FILINGS_DIR.glob("*.part"))
    if parts:
        print(f"Orphan .part files     : {len(parts)} (cleared on the next run)")
    truncated = [p.name for p in files if p.stat().st_size < MIN_FILING_BYTES]
    if truncated:
        print(f"!! Suspiciously small files: {len(truncated)} {truncated[:5]}")


def print_assumptions(n_ciks: int | None, ref: pd.Timestamp, scope: str = "full",
                      batch: str | None = None) -> None:
    """Print the assumptions that governed this run."""
    print("\n" + "=" * 70)
    print("ASSUMPTIONS (this run)")
    print("=" * 70)
    for line in [
        (f"Scope '{scope}' = {BRIDGE_CSV.name}, rebuilt after the NYSE micro-cap drop. This is "
         "the firm list every previous run used and it is unchanged."
         if SCOPES[scope] is None else
         f"Scope '{scope}' = {SCOPES[scope].name}, the fixed random 1,000-firm draw "
         "(Research Design v6 section 7.0, seed recorded in that file's header). The draw is "
         "made once and reused unchanged across every annual reference date."),
        "Firm-list sourcing: the persisted list supplies WHICH PERMNOs to pull; the CIK for "
        "each is still resolved by _resolve_cik against the bridge once per "
        "(PERMNO, reference_date), so a link that changed between dates resolves correctly at "
        "each date rather than being frozen from the file.",
        (f"Batch = {batch}: {len(BATCHES[batch][0])} reference date(s) run in sequence in one "
         f"process, each an ordinary gap-only pull." if batch else
         f"Single reference date {ref.date()}."),
        ("Cache retention (prune_stale) is decided against the BRIDGE, never the active scope. "
         "data/filings/raw/ is a cache of the bridge; pruning against a narrower scope would delete "
         "filings the full-universe scoring and the section 8 unscreened cross-section need."),
        ("Scope = the whole list." if n_ciks is None else
         f"Work list capped at the first {n_ciks:,} distinct CIKs in ascending PERMNO order "
         "(same ordering as the original pilot); --all covers the whole list."),
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
    ap.add_argument("--scope", choices=sorted(SCOPES), default="full",
                    help="firm list: full = the bridge (default, unchanged); "
                         "subsample = the persisted 1,000-firm draw")
    ap.add_argument("--batch", choices=sorted(BATCHES), default=None,
                    help="run a named batch's reference dates in sequence at its own scope")
    ap.add_argument("--restore-missing", action="store_true",
                    help="re-download cached filings a successful log row points at but disk "
                         "no longer holds (the 'Successes missing file' count in --status)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --restore-missing: report what would be fetched, fetch nothing")
    args = ap.parse_args()
    ref = pd.Timestamp(args.reference_date)

    if args.status:
        print_status(ref, args.batch)
        return

    if args.restore_missing:
        try:
            restore_missing(args.dry_run)
        except PullAborted as exc:
            print(f"\n!! ABORTED: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.batch:
        dates, scope = BATCHES[args.batch]   # a batch always covers its whole scope
        try:
            log = run_batch(args.batch, args.retry_all)
        except PullAborted as exc:
            print(f"\n!! ABORTED: {exc}\n   The log is flushed up to this point - re-run the "
                  f"same command to resume.", file=sys.stderr)
            sys.exit(1)
        validate(log)
        print_assumptions(None, pd.Timestamp(dates[-1]), scope, args.batch)
        print_status(pd.Timestamp(dates[-1]), args.batch)
        return

    n_ciks = None if args.all else args.n_ciks   # None = no limit
    try:
        log = pull_gap(n_ciks, ref, args.retry_all, args.scope)
    except PullAborted as exc:
        print(f"\n!! ABORTED: {exc}\n   The log is flushed up to this point - re-run the "
              f"same command to resume.", file=sys.stderr)
        sys.exit(1)
    validate(log)
    print_assumptions(n_ciks, ref, args.scope)
    print_status(ref)


if __name__ == "__main__":
    main()
