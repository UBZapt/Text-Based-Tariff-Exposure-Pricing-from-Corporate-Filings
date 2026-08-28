"""
Step 4a - Script 1 of 3: cleaned, merged control panel for the section 7.2 event study.

Merges CRSP daily returns, Compustat fundamentals (book equity, leverage), the CCM
PERMNO-GVKEY-CIK bridge and the Ken French FF12 industry definitions into one daily panel
covering the estimation and event windows of both 2025 tariff events.

    impose  = 2025-04-02  (Liberation Day)
    reverse = 2025-08-29  (Federal Circuit, V.O.S. Selections v. Trump)

Cleaning only. No abnormal returns, no market model, no regression - those are Scripts 2 and 3.

Fundamentals are gated point-in-time on the date they actually became public: the real 10-K
filing date from edgar_pull_log.csv where known, otherwise datadate + FUNDAMENTAL_LAG_DAYS.

    python clean_controls_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
CLEAN_DIR = BASE / "clean_data"
OUTPUT_DIR = BASE / "output"

DAILY_FILE = BASE / "Daily returns.csv"
MONTHLY_RAW_FILE = BASE / "Monthly Returns.csv"   # unscreened history, for the momentum control
FUNDA_FILE = BASE / "BM and Lev.csv"
SICCODES_FILE = BASE / "Siccodes12.txt"
BRIDGE_CSV = CLEAN_DIR / "clean_firm_bridge.csv"
MONTHLY_CSV = CLEAN_DIR / "clean_returns.csv"
EDGAR_LOG = BASE / "edgar_pull_log.csv"

PANEL_OUT = OUTPUT_DIR / "controls_panel"        # extension added from OUTPUT_FORMAT
REPORT_OUT = OUTPUT_DIR / "cleaning_validation_report.txt"

# Event dates (research design section 7.2). Both must be trading days.
EVENT_DATES = {"impose": "2025-04-02", "reverse": "2025-08-29"}

EST_WINDOW = (-250, -46)          # market-model estimation window, trading days
MAX_EVENT_WINDOW = (-10, 10)      # widest event window used in the section 8 robustness table
BUFFER_TRADING_DAYS = 10          # slack either side; also supplies the t-1 lag at the left edge

# Availability lag applied when no actual 10-K filing date is known. 90 days is the SEC's outer
# 10-K deadline (non-accelerated filers) and matches the p95 of observed EDGAR filing gaps.
FUNDAMENTAL_LAG_DAYS = 90
FUNDAMENTAL_STALE_DAYS = 730      # fundamentals older than this at an event are flagged, not dropped

MOM12_LOOKBACK = 12               # months in the prior-return window
MOM12_MIN_MONTHS = 10             # of 12; below this mom12 is NaN, never dropped

RESTRICT_TO_EVENT_UNIVERSE = True   # keep only firms with a return on at least one event date
OUTPUT_FORMAT = "csv"               # "parquet" once the pyarrow OS block is lifted

GVKEY_WIDTH = 6                   # the bridge zero-pads gvkey
CIK_WIDTH = 10
OPEN_END = pd.Timestamp("2099-12-31")     # still-active CCM link sentinel
CHUNK_ROWS = 1_000_000

# CRSP ShrOut is in thousands and DlyPrc in dollars, so me is in $thousands; Compustat is in
# $millions. Verified: DlyPrc * ShrOut reproduces the monthly panel's me exactly.
ME_TO_MILLIONS = 1_000.0

DAILY_REQUIRED = ["PERMNO", "PERMCO", "Ticker", "DlyCalDt", "DlyPrc", "DlyRet", "ShrOut"]
MONTHLY_RAW_REQUIRED = ["PERMNO", "MthCalDt", "MthRet"]
FUNDA_REQUIRED = ["gvkey", "cik", "conm", "datadate", "fyear",
                  "at", "ceq", "dlc", "dltt", "pstk", "pstkl", "pstkrv", "txditc"]
FUNDA_NUMERIC = ["at", "ceq", "dlc", "dltt", "pstk", "pstkl", "pstkr", "pstkrv", "txditc"]

SPOT_CHECK_PERMNOS = [10107, 93436, 12490, 10145, 14593]   # MSFT, TSLA, IBM, Honeywell, Apple

OUTPUT_COLUMNS = [
    "permno", "permco", "ticker", "date", "gvkey", "cik",
    "ret", "prc", "shrout", "me", "me_lag", "ln_me_lag",
    "siccd", "ff12_num", "ff12", "sic_invalid", "in_screened_universe",
    "datadate", "fyear", "available_date", "pit_source", "fundamentals_stale",
    "at", "ceq", "txditc", "txditc_imputed", "ps", "be", "debt",
    "bm", "lev", "be_nonpositive", "ceq_nonpositive",
    "mom12", "mom12_n_months",
    "alive_at_impose", "alive_at_reverse",
]

_REPORT: list[str] = []


def _say(line: str = "") -> None:
    """Append one line to the single consolidated validation report."""
    _REPORT.append(line)


def _section(title: str) -> None:
    """Start a titled report section."""
    _say()
    _say(title)
    _say("-" * len(title))


def _events() -> dict[str, pd.Timestamp]:
    return {k: pd.Timestamp(v) for k, v in EVENT_DATES.items()}


# --------------------------------------------------------------------------- #
# CRSP daily: schema, calendar, load                                          #
# --------------------------------------------------------------------------- #
def validate_daily_schema(path: Path) -> None:
    """Halt before any processing if the CRSP daily export is not the expected shape.

    Reads the header only. A silently renamed or reordered WRDS export would otherwise surface
    as a wrong number many steps downstream.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; it is a required raw input.")
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in DAILY_REQUIRED if c not in header]
    if missing:
        raise ValueError(
            f"{path.name} is missing required field(s) {missing}. "
            f"Found columns: {header}. Re-pull the CRSP daily extract with these fields."
        )


def build_calendar(path: Path) -> np.ndarray:
    """Collect the distinct trading days from the daily file (first pass, dates only)."""
    dates: set = set()
    for chunk in pd.read_csv(path, usecols=["DlyCalDt"], chunksize=CHUNK_ROWS):
        dates.update(chunk["DlyCalDt"].unique())
    return pd.DatetimeIndex(sorted(pd.to_datetime(list(dates)))).to_numpy()


def derive_bounds(calendar: np.ndarray) -> tuple[pd.Timestamp, pd.Timestamp, dict]:
    """Derive the panel span from the event dates and the window constants, never hardcoded.

    Span = earliest (event - 250 trading days) - buffer .. latest (event + 10) + buffer. The left
    buffer also supplies the t-1 market-equity lag for the first day of the estimation window.
    """
    cal = pd.DatetimeIndex(calendar)
    idx = {}
    for name, day in _events().items():
        pos = cal.searchsorted(day)
        if pos >= len(cal) or cal[pos] != day:
            raise ValueError(f"event '{name}' {day:%Y-%m-%d} is not a trading day in {DAILY_FILE.name}")
        idx[name] = int(pos)

    start_i = min(i + EST_WINDOW[0] for i in idx.values())
    end_i = max(i + MAX_EVENT_WINDOW[1] for i in idx.values())
    want_start, want_end = start_i - BUFFER_TRADING_DAYS, end_i + BUFFER_TRADING_DAYS
    clipped_start, clipped_end = max(want_start, 0), min(want_end, len(cal) - 1)

    # Clipping means the file does not reach far enough for the full window; Script 2 would then
    # estimate the market model on a short sample, so it is surfaced rather than absorbed.
    stats = {
        "n_trading_days": clipped_end - clipped_start + 1,
        "short_at_start": max(0, clipped_start - want_start),
        "short_at_end": max(0, want_end - clipped_end),
    }
    return cal[clipped_start], cal[clipped_end], stats


def load_daily(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Chunked read of the daily file, filtered to the derived span inside each chunk."""
    dtypes = {"PERMNO": "int32", "PERMCO": "int32", "Ticker": "object",
              "DlyPrc": "float64", "DlyRet": "float64", "ShrOut": "float64"}
    kept = []
    for chunk in pd.read_csv(path, usecols=DAILY_REQUIRED, dtype=dtypes, chunksize=CHUNK_ROWS):
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
        kept.append(chunk[(chunk["DlyCalDt"] >= start) & (chunk["DlyCalDt"] <= end)])
    daily = pd.concat(kept, ignore_index=True)
    return daily.rename(columns={"PERMNO": "permno", "PERMCO": "permco", "Ticker": "ticker",
                                 "DlyCalDt": "date", "DlyPrc": "prc", "DlyRet": "ret",
                                 "ShrOut": "shrout"})


def dedupe_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows, having first proved they carry no conflicting values.

    The CRSP export repeats whole rows (the monthly file has the same defect). A PERMNO-date pair
    duplicated with *differing* values would be a different problem entirely and must not be
    silently collapsed, so that case raises instead.
    """
    key = ["permno", "date"]
    full_dup = daily.duplicated(keep=False)
    key_dup = daily.duplicated(subset=key, keep=False)
    conflicting = daily[key_dup & ~full_dup]
    if len(conflicting):
        raise ValueError(
            f"{len(conflicting):,} PERMNO-date rows are duplicated with differing values "
            f"(first: permno={conflicting.iloc[0]['permno']}, "
            f"date={conflicting.iloc[0]['date']:%Y-%m-%d}); these are not export artifacts."
        )
    out = daily.drop_duplicates().reset_index(drop=True)
    if out.duplicated(subset=key).any():
        raise ValueError("duplicate PERMNO-date rows survived de-duplication")
    return out, {"rows_in": len(daily), "rows_out": len(out), "removed": len(daily) - len(out),
                 "permnos_affected": int(daily.loc[key_dup, "permno"].nunique())}


# --------------------------------------------------------------------------- #
# Market equity                                                               #
# --------------------------------------------------------------------------- #
def add_market_equity(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build me = |prc| * shrout ($thousands) and its one-trading-day lag.

    CRSP stores a bid/ask midpoint as a negative price, so the absolute value is taken; this file
    happens to contain none, and the count is reported rather than assumed. The lag is the firm's
    previous *available* observation, which is why the span carries a left buffer.
    """
    d = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    neg = int((d["prc"] < 0).sum())
    d["me"] = d["prc"].abs() * d["shrout"]
    nonpos = int((d["me"] <= 0).sum())
    d.loc[d["me"] <= 0, "me"] = np.nan
    d["me_lag"] = d.groupby("permno")["me"].shift(1)
    d["ln_me_lag"] = np.where(d["me_lag"] > 0, np.log(d["me_lag"]), np.nan)
    return d, {"negative_prices": neg, "nonpositive_me": nonpos,
               "me_null": int(d["me"].isna().sum()),
               "me_lag_null": int(d["me_lag"].isna().sum()),
               "ln_me_lag_null": int(d["ln_me_lag"].isna().sum())}


# --------------------------------------------------------------------------- #
# Prior 12-month return                                                       #
# --------------------------------------------------------------------------- #
def load_monthly_returns(path: Path, permnos: set) -> tuple[pd.DataFrame, dict]:
    """Raw CRSP monthly return history for the panel firms, for the prior-return control.

    Read from the unscreened monthly file rather than clean_returns.csv on purpose. The cleaned
    panel drops firm-months failing the price and NYSE micro-cap screens, which punches holes in
    the return history of exactly the smaller firms - 58% of panel firms have a complete history
    there against 92% here. Momentum is a realised-return characteristic, so those months must be
    compounded, not skipped; using the screened panel would make the control go missing
    non-randomly with size.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; it is a required raw input.")
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in MONTHLY_RAW_REQUIRED if c not in header]
    if missing:
        raise ValueError(f"{path.name} is missing required field(s) {missing}. Found: {header}")

    kept = []
    for chunk in pd.read_csv(path, usecols=MONTHLY_RAW_REQUIRED, chunksize=CHUNK_ROWS):
        kept.append(chunk[chunk["PERMNO"].isin(permnos)])
    m = pd.concat(kept, ignore_index=True).drop_duplicates()
    m["date"] = pd.to_datetime(m["MthCalDt"])
    m = m.rename(columns={"PERMNO": "permno", "MthRet": "ret"})[["permno", "date", "ret"]]
    m["ym"] = m["date"].dt.to_period("M")

    dups = int(m.duplicated(["permno", "ym"]).sum())
    if dups:
        raise ValueError(f"{dups:,} duplicate PERMNO-month rows survive exact de-duplication in "
                         f"{path.name}; the momentum window would double-count them.")
    return m, {"rows": len(m), "permnos": int(m["permno"].nunique()),
               "ret_null": int(m["ret"].isna().sum())}


def add_momentum(daily: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach r_(t-12:t-1): the compounded return over the twelve months ending one month back.

    Computed on a complete monthly grid so the twelve-month window is twelve *calendar* months
    even when a firm has a gap, then matched to daily rows on calendar month. Months with no
    return compound at 1.0 and are excluded from the observation count; a firm-month with fewer
    than MOM12_MIN_MONTHS observations gets NaN rather than a short-window figure.
    """
    m = monthly[["permno", "ym", "ret"]].copy()
    wide = m.pivot(index="permno", columns="ym", values="ret")
    grid = pd.period_range(wide.columns.min(), wide.columns.max(), freq="M", name="ym")
    wide = wide.reindex(columns=grid)   # reindex drops the axis name; the grid carries it

    ret = wide.to_numpy(dtype="float64")
    n_firms, n_months = ret.shape
    if n_months <= MOM12_LOOKBACK:
        raise ValueError(f"monthly panel spans {n_months} months; need more than {MOM12_LOOKBACK}")
    observed = ~np.isnan(ret)
    gross = np.where(observed, 1.0 + ret, 1.0)

    # Window j covers months [j, j+11]; it is the prior-return window for month j + 12.
    prod = sliding_window_view(gross, MOM12_LOOKBACK, axis=1).prod(axis=2)
    count = sliding_window_view(observed, MOM12_LOOKBACK, axis=1).sum(axis=2)

    mom = np.full(ret.shape, np.nan)
    n_obs = np.zeros(ret.shape, dtype="int16")
    mom[:, MOM12_LOOKBACK:] = prod[:, : n_months - MOM12_LOOKBACK] - 1.0
    n_obs[:, MOM12_LOOKBACK:] = count[:, : n_months - MOM12_LOOKBACK]
    mom = np.where(n_obs >= MOM12_MIN_MONTHS, mom, np.nan)

    long = (pd.DataFrame(mom, index=wide.index, columns=wide.columns)
            .stack(future_stack=True).rename("mom12").reset_index())
    long["mom12_n_months"] = (pd.DataFrame(n_obs, index=wide.index, columns=wide.columns)
                              .stack(future_stack=True).to_numpy())
    long = long[long["mom12_n_months"] > 0]

    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M")
    before = len(d)
    d = d.merge(long, on=["permno", "ym"], how="left").drop(columns="ym")
    if len(d) != before or d.duplicated(subset=["permno", "date"]).any():
        raise ValueError("momentum merge changed the daily grain")
    d["mom12_n_months"] = d["mom12_n_months"].astype("Int16")
    return d, {"firm_months_computed": len(long),
               "daily_rows_with_mom12": int(d["mom12"].notna().sum()),
               "daily_rows_missing_mom12": int(d["mom12"].isna().sum())}


# --------------------------------------------------------------------------- #
# CCM link: PERMNO -> GVKEY                                                   #
# --------------------------------------------------------------------------- #
def link_gvkey(daily: pd.DataFrame, bridge: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach gvkey/cik per firm-date, requiring the date inside the CCM link interval.

    Same resolution rule as foreign_sales.link_permno and edgar_pull._resolve_cik, applied in the
    opposite direction: LINKPRIM 'P' first, then the latest interval. Rows whose date falls in no
    valid interval keep a null gvkey rather than being dropped.
    """
    b = bridge.copy()
    b["permno"] = b["permno"].astype("int32")
    b["gvkey"] = b["gvkey"].str.strip().str.zfill(GVKEY_WIDTH)
    b["cik"] = b["cik"].fillna("").str.strip().str.zfill(CIK_WIDTH).replace("0" * CIK_WIDTH, "")
    b["linkdt"] = pd.to_datetime(b["linkdt"])
    b["linkenddt"] = pd.to_datetime(b["linkenddt"]).fillna(OPEN_END)
    b["_prim"] = (b["linkprim"] == "P").astype("int8")

    pairs = daily[["permno", "date"]].drop_duplicates()
    cand = pairs.merge(b[["permno", "gvkey", "cik", "linkdt", "linkenddt", "_prim"]],
                       on="permno", how="left")
    valid = cand["gvkey"].notna() & (cand["linkdt"] <= cand["date"]) & (cand["linkenddt"] >= cand["date"])
    resolved = (cand[valid].sort_values(["_prim", "linkenddt", "linkdt"])
                .drop_duplicates(["permno", "date"], keep="last")[["permno", "date", "gvkey", "cik"]])

    before = len(daily)
    d = daily.merge(resolved, on=["permno", "date"], how="left")
    if len(d) != before or d.duplicated(subset=["permno", "date"]).any():
        raise ValueError("CCM link merge changed the daily grain")

    unlinked = d[d["gvkey"].isna()]
    return d, {"rows_unlinked": len(unlinked),
               "permnos_unlinked": int(unlinked["permno"].nunique()),
               "permnos_linked": int(d.loc[d["gvkey"].notna(), "permno"].nunique()),
               "gvkeys_linked": int(d["gvkey"].nunique())}


# --------------------------------------------------------------------------- #
# Compustat fundamentals and the point-in-time gate                           #
# --------------------------------------------------------------------------- #
def load_fundamentals(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the funda extract with identifiers padded and one row per gvkey-fiscal year.

    gvkey is zero-padded at read time regardless of how the export arrives: an earlier Compustat
    file in this project shipped unpadded gvkeys and merged silently against a third of its rows.
    Where a fiscal year carries two datadates (a fiscal year-end change) the later is kept, the
    rule foreign_sales.build_firm_year uses.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; it is a required raw input.")
    f = pd.read_csv(path, dtype=str)
    missing = [c for c in FUNDA_REQUIRED if c not in f.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required field(s) {missing}. Found: {list(f.columns)}")

    f["gvkey"] = f["gvkey"].str.strip().str.zfill(GVKEY_WIDTH)
    f["cik"] = f["cik"].fillna("").str.strip().str.zfill(CIK_WIDTH)
    f["datadate"] = pd.to_datetime(f["datadate"])
    f["fyear"] = pd.to_numeric(f["fyear"], errors="coerce").astype("Int16")
    for col in FUNDA_NUMERIC:
        f[col] = pd.to_numeric(f[col], errors="coerce")

    if f.duplicated(["gvkey", "datadate"]).any():
        raise ValueError("Compustat funda holds duplicate gvkey-datadate rows")
    dup_fy = f[f.duplicated(["gvkey", "fyear"], keep=False)].copy()
    f = f.sort_values("datadate").drop_duplicates(["gvkey", "fyear"], keep="last")
    return f.reset_index(drop=True), dup_fy


def load_edgar_filing_dates(path: Path) -> tuple[pd.DataFrame, dict]:
    """Actual 10-K filing dates per (gvkey, fiscal period), from the EDGAR pull log.

    Two PERMNOs can share a gvkey (and co-registrants share an accession), so the log is collapsed
    to the earliest filing date per gvkey-period - the date the disclosure first became public.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; it records the real filing dates.")
    log = pd.read_csv(path, dtype=str)
    log = log[log["found_10k"].astype(str).str.lower() == "true"].copy()
    log["gvkey"] = log["gvkey"].str.strip().str.zfill(GVKEY_WIDTH)
    log["filing_date"] = pd.to_datetime(log["filing_date"])
    log["period_of_report"] = pd.to_datetime(log["period_of_report"])

    backwards = int((log["filing_date"] < log["period_of_report"]).sum())
    log = log[log["filing_date"] >= log["period_of_report"]]
    grouped = (log.groupby(["gvkey", "period_of_report"], as_index=False)["filing_date"].min())
    stats = {"log_filings": len(log), "filing_dates_before_period": backwards,
             "gvkey_periods": len(grouped)}
    return grouped, stats


def add_availability_date(fund: pd.DataFrame, filings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stamp each fundamentals row with the date it became public knowledge.

    Primary source is the real 10-K filing date; where the cache holds no filing for that
    gvkey-period the SEC's outer 10-K deadline (FUNDAMENTAL_LAG_DAYS) is applied to datadate.
    pit_source records which rule was used so the gate is auditable per row.
    """
    f = fund.merge(filings.rename(columns={"period_of_report": "datadate"}),
                   on=["gvkey", "datadate"], how="left")
    if len(f) != len(fund):
        raise ValueError("filing-date merge changed the fundamentals grain")

    from_edgar = f["filing_date"].notna()
    f["available_date"] = f["filing_date"].where(
        from_edgar, f["datadate"] + pd.Timedelta(days=FUNDAMENTAL_LAG_DAYS))
    f["pit_source"] = np.where(from_edgar, "edgar_filing_date", "fallback_lag")
    return f.drop(columns="filing_date"), {
        "rows": len(f),
        "from_edgar_filing_date": int(from_edgar.sum()),
        "from_fallback_lag": int((~from_edgar).sum()),
    }


def build_book_equity(fund: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fama-French book equity and total debt, with every imputation flagged.

    BE = ceq + txditc - preferred stock, preferred taken as redemption -> liquidating -> par value.
    Missing txditc is treated as zero, the standard Fama-French/Davis-Fama-French convention;
    the flag makes the affected rows recoverable downstream. Nothing is dropped here.
    """
    f = fund.copy()
    f["ps"] = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"])
    f["txditc_imputed"] = f["txditc"].isna()
    f["be"] = f["ceq"] + f["txditc"].fillna(0.0) - f["ps"]
    f["debt"] = f["dlc"].fillna(0.0) + f["dltt"].fillna(0.0)
    f.loc[f["dlc"].isna() & f["dltt"].isna(), "debt"] = np.nan
    f["be_nonpositive"] = f["be"].notna() & (f["be"] <= 0)
    f["ceq_nonpositive"] = f["ceq"].notna() & (f["ceq"] <= 0)
    return f, {"rows": len(f), "be_null": int(f["be"].isna().sum()),
               "be_nonpositive": int(f["be_nonpositive"].sum()),
               "ceq_nonpositive": int(f["ceq_nonpositive"].sum()),
               "txditc_imputed": int(f["txditc_imputed"].sum()),
               "ps_unresolvable": int(f["ps"].isna().sum()),
               "debt_null": int(f["debt"].isna().sum())}


def merge_fundamentals(daily: pd.DataFrame, fund: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Backward as-of merge on available_date: only fundamentals already public on that date.

    allow_exact_matches=False excludes a filing dated on the day itself, so nothing published on
    an event date can enter that event's regressors.
    """
    carry = ["gvkey", "datadate", "fyear", "available_date", "pit_source", "conm",
             "at", "ceq", "txditc", "txditc_imputed", "ps", "be", "debt",
             "be_nonpositive", "ceq_nonpositive"]
    right = fund[carry].sort_values("available_date")
    # merge_asof cannot match on a null 'by' key; a sentinel keeps unlinked rows in the frame.
    left = daily.copy()
    left["_gvkey"] = left["gvkey"].fillna("")
    right = right.copy()
    right["_gvkey"] = right["gvkey"]

    before = len(left)
    out = pd.merge_asof(left.sort_values("date"),
                        right.drop(columns="gvkey").sort_values("available_date"),
                        left_on="date", right_on="available_date", by="_gvkey",
                        direction="backward", allow_exact_matches=False)
    out = out.drop(columns="_gvkey").sort_values(["permno", "date"]).reset_index(drop=True)
    if len(out) != before or out.duplicated(subset=["permno", "date"]).any():
        raise ValueError("fundamentals as-of merge changed the daily grain")

    out["fundamentals_stale"] = (out["datadate"].notna() &
                                 ((out["date"] - out["datadate"]).dt.days > FUNDAMENTAL_STALE_DAYS))
    matched = out["datadate"].notna()
    return out, {"rows_with_fundamentals": int(matched.sum()),
                 "rows_without_fundamentals": int((~matched).sum()),
                 "permnos_with_fundamentals": int(out.loc[matched, "permno"].nunique()),
                 "rows_stale": int(out["fundamentals_stale"].sum())}


def build_ratios(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Book-to-market and debt-to-equity, both undefined rather than misleading when they must be.

    Compustat is in $millions and CRSP market equity in $thousands, so me is rescaled before the
    ratio is formed. BM is left null where book equity is non-positive and leverage where common
    equity is non-positive: a negative denominator flips the sign and would silently pollute the
    cross-section. Both cases are flagged in the panel, not dropped.
    """
    d = daily.copy()
    me_millions = d["me_lag"] / ME_TO_MILLIONS
    d["bm"] = np.where((d["be"] > 0) & (me_millions > 0), d["be"] / me_millions, np.nan)
    d["lev"] = np.where(d["ceq"] > 0, d["debt"] / d["ceq"], np.nan)
    return d, {"bm_null": int(d["bm"].isna().sum()), "lev_null": int(d["lev"].isna().sum()),
               "bm_present": int(d["bm"].notna().sum()), "lev_present": int(d["lev"].notna().sum())}


# --------------------------------------------------------------------------- #
# FF12 industry                                                               #
# --------------------------------------------------------------------------- #
def parse_ff12(path: Path) -> pd.DataFrame:
    """Parse the Ken French FF12 definition file into (industry number, label, SIC range) rows.

    Industry 12 'Other' is defined in the file with no ranges - it is the residual - so anything
    matching no range is assigned to it. Overlapping ranges would make first-match arbitrary, so
    that is checked rather than assumed.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; it is a required raw input.")
    import re
    head_re = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z]+)\s+")
    range_re = re.compile(r"^\s+(\d{4})-(\d{4})")

    rows, current = [], None
    for line in path.read_text().splitlines():
        rng = range_re.match(line)
        if rng and current:
            rows.append((current[0], current[1], int(rng.group(1)), int(rng.group(2))))
            continue
        head = head_re.match(line)
        if head:
            current = (int(head.group(1)), head.group(2))
    ff = pd.DataFrame(rows, columns=["ff12_num", "ff12", "lo", "hi"])
    if ff.empty:
        raise ValueError(f"no SIC ranges parsed from {path.name}")

    lo, hi = ff["lo"].to_numpy(), ff["hi"].to_numpy()
    overlaps = [(i, j) for i in range(len(ff)) for j in range(i + 1, len(ff))
                if lo[i] <= hi[j] and lo[j] <= hi[i]]
    if overlaps:
        raise ValueError(f"{len(overlaps)} overlapping SIC ranges in {path.name}; "
                         "first-match assignment would be arbitrary")
    return ff


def assign_ff12(daily: pd.DataFrame, monthly: pd.DataFrame,
                ff: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Give every firm exactly one FF12 label, from its SIC just before the first event.

    CRSP SIC is time-varying, so a per-date mapping would give some firms two labels and make the
    industry fixed effects in section 7.2 differ between the imposition and reversal legs of the
    same test. A single pre-event snapshot keeps the label fixed across both legs. SICCD 0 is
    invalid rather than 'Other' and is flagged separately, kept, and reported.
    """
    snapshot_date = min(_events().values())
    ordered = monthly[["permno", "date", "siccd"]].sort_values("date")
    snap = (ordered[ordered["date"] < snapshot_date]
            .groupby("permno", as_index=False).last()[["permno", "siccd"]])

    # Firms first listed (or first clearing the screens) after the imposition date have no prior
    # SIC. Their earliest available one is used instead: an industry label is a classification,
    # not a financial quantity, so this introduces no look-ahead into any regressor.
    fallback = (ordered[~ordered["permno"].isin(snap["permno"])]
                .groupby("permno", as_index=False).first()[["permno", "siccd"]])
    n_fallback = len(fallback)
    snap = pd.concat([snap, fallback], ignore_index=True)
    snap["siccd"] = snap["siccd"].astype("int32")

    lo, hi, num, lab = (ff["lo"].to_numpy(), ff["hi"].to_numpy(),
                        ff["ff12_num"].to_numpy(), ff["ff12"].to_numpy())
    codes = np.sort(snap["siccd"].unique())
    mapping, unmapped = {}, []
    for code in codes:
        hit = np.flatnonzero((lo <= code) & (hi >= code))
        if len(hit):
            mapping[int(code)] = (int(num[hit[0]]), str(lab[hit[0]]))
        else:
            mapping[int(code)] = (12, "Other")
            unmapped.append(int(code))
    snap["ff12_num"] = snap["siccd"].map(lambda c: mapping[int(c)][0]).astype("int8")
    snap["ff12"] = snap["siccd"].map(lambda c: mapping[int(c)][1])
    snap["sic_invalid"] = snap["siccd"] <= 0

    before = len(daily)
    d = daily.merge(snap, on="permno", how="left")
    if len(d) != before or d.duplicated(subset=["permno", "date"]).any():
        raise ValueError("FF12 merge changed the daily grain")

    per_firm = d.groupby("permno")["ff12"].nunique(dropna=False)
    if (per_firm > 1).any():
        raise ValueError(f"{int((per_firm > 1).sum())} firms carry more than one FF12 label")

    d["siccd"] = d["siccd"].astype("Int32")
    changed = monthly[monthly["permno"].isin(d["permno"].unique())].groupby("permno")["siccd"].nunique()
    return d, {"firms_labelled": int(d.loc[d["ff12"].notna(), "permno"].nunique()),
               "firms_without_label": int(d.loc[d["ff12"].isna(), "permno"].nunique()),
               "firms_from_fallback_sic": n_fallback,
               "distinct_sic": len(codes), "unmapped_sic": unmapped,
               "sic_invalid_firms": int(snap["sic_invalid"].sum()),
               "sic_changed_in_sample": int((changed > 1).sum()),
               "snapshot_date": snapshot_date,
               "distribution": d.drop_duplicates("permno")["ff12"].value_counts()}


def add_screened_universe_flag(daily: pd.DataFrame,
                               monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mark firm-months that clear the design's price and NYSE micro-cap screens.

    clean_data.py applies price > $1 and the NYSE 10th-percentile size cut monthly, so presence in
    clean_returns.csv *is* the screen. Carrying it as a flag lets Script 3 impose the section 6
    sample definition without re-deriving breakpoints, and keeps the decision visible rather than
    silently baked into this panel.
    """
    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M")
    screened = monthly[["permno", "date"]].copy()
    screened["ym"] = screened["date"].dt.to_period("M")
    screened = screened[["permno", "ym"]].drop_duplicates()
    screened["in_screened_universe"] = True

    before = len(d)
    d = d.merge(screened, on=["permno", "ym"], how="left").drop(columns="ym")
    if len(d) != before or d.duplicated(subset=["permno", "date"]).any():
        raise ValueError("screened-universe merge changed the daily grain")
    d["in_screened_universe"] = d["in_screened_universe"].eq(True)   # unmatched -> False

    at_event = {name: int(d.loc[(d["date"] == day) & d["in_screened_universe"], "permno"].nunique())
                for name, day in _events().items()}
    return d, {"rows_in_screen": int(d["in_screened_universe"].sum()),
               "rows_out_of_screen": int((~d["in_screened_universe"]).sum()),
               "at_event": at_event}


# --------------------------------------------------------------------------- #
# Event universe                                                              #
# --------------------------------------------------------------------------- #
def restrict_universe(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Flag firms trading on each event date and, by default, keep only those.

    A firm with no return on either event date cannot contribute a CAR to section 7.2, so keeping
    it only inflates the panel. Every excluded firm is returned with its reason rather than
    vanishing.
    """
    d = daily.copy()
    alive = {}
    for name, day in _events().items():
        on_day = d.loc[(d["date"] == day) & d["ret"].notna(), "permno"].unique()
        col = f"alive_at_{name}"
        d[col] = d["permno"].isin(on_day)
        alive[name] = len(on_day)

    flags = [f"alive_at_{name}" for name in _events()]
    keep_mask = d[flags].any(axis=1)
    dropped = (d.loc[~keep_mask, ["permno", "ticker"]].drop_duplicates("permno")
               .assign(reason="no_return_on_any_event_date"))
    stats = {"alive": alive, "permnos_in": int(d["permno"].nunique()),
             "permnos_dropped": int(dropped["permno"].nunique()),
             "rows_in": len(d), "applied": RESTRICT_TO_EVENT_UNIVERSE}
    if RESTRICT_TO_EVENT_UNIVERSE:
        d = d[keep_mask].reset_index(drop=True)
    stats["permnos_out"] = int(d["permno"].nunique())
    stats["rows_out"] = len(d)
    return d, dropped, stats


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def write_panel(daily: pd.DataFrame) -> Path:
    """Write the panel in the configured format."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = daily.reindex(columns=OUTPUT_COLUMNS)
    path = PANEL_OUT.with_suffix(f".{OUTPUT_FORMAT}")
    if OUTPUT_FORMAT == "parquet":
        out.to_parquet(path, index=False)
    elif OUTPUT_FORMAT == "csv":
        out.to_csv(path, index=False)
    else:
        raise ValueError(f"OUTPUT_FORMAT must be 'csv' or 'parquet', got {OUTPUT_FORMAT!r}")
    return path


def _describe(frame: pd.DataFrame, cols: list[str]) -> None:
    """Report distribution and missingness for the constructed variables."""
    _say(f"  {'variable':<14}{'n':>11}{'null':>9}{'min':>17}{'p25':>17}"
         f"{'median':>17}{'p75':>17}{'max':>17}")
    for col in cols:
        s = pd.to_numeric(frame[col], errors="coerce")
        v = s.dropna()
        if v.empty:
            _say(f"  {col:<14}{0:>11,}{len(s):>9,}" + f"{'-':>17}" * 5)
            continue
        # Large-magnitude series (market equity, in $thousands) lose nothing to fewer decimals.
        fmt = ",.0f" if v.abs().max() >= 1e6 else ",.4f"
        cells = "".join(format(x, f">17{fmt}") for x in
                        (v.min(), v.quantile(.25), v.median(), v.quantile(.75), v.max()))
        _say(f"  {col:<14}{len(v):>11,}{int(s.isna().sum()):>9,}{cells}")


def validate(panel: pd.DataFrame, fund: pd.DataFrame, bridge: pd.DataFrame,
             stats: dict, dropped: pd.DataFrame, dup_fy: pd.DataFrame,
             bounds: tuple, panel_path: Path) -> None:
    """Assemble the consolidated validation report and write it to disk."""
    start, end = bounds
    events = _events()

    _say("=" * 78)
    _say("SECTION 7.2 CONTROLS PANEL - CLEANING VALIDATION REPORT")
    _say("=" * 78)
    _say("Events        : " + ", ".join(f"{k}={v:%Y-%m-%d}" for k, v in events.items()))
    _say(f"Derived span  : {start:%Y-%m-%d} .. {end:%Y-%m-%d} "
         f"({stats['bounds']['n_trading_days']} trading days)")
    if stats["bounds"]["short_at_start"] or stats["bounds"]["short_at_end"]:
        _say(f"  WARNING: the daily file does not reach the full requested span - short by "
             f"{stats['bounds']['short_at_start']} trading days at the start and "
             f"{stats['bounds']['short_at_end']} at the end. Estimation windows will be truncated.")
    _say(f"               from estimation window {EST_WINDOW}, event window {MAX_EVENT_WINDOW}, "
         f"buffer {BUFFER_TRADING_DAYS}")
    _say(f"Output        : {panel_path.name}  ({len(panel):,} rows x {len(OUTPUT_COLUMNS)} cols)")

    _section("1. Funnel (every row and firm accounted for)")
    d, m, lk, fm, uni = (stats["dedupe"], stats["mom"], stats["link"],
                         stats["fund_merge"], stats["universe"])
    _say(f"  daily rows in span                         {d['rows_in']:>10,}")
    _say(f"    - exact duplicate rows removed           {d['removed']:>10,}"
         f"   ({d['permnos_affected']} PERMNOs affected)")
    _say(f"  = de-duplicated daily rows                 {d['rows_out']:>10,}")
    if d["rows_out"] + d["removed"] != d["rows_in"]:
        raise ValueError("de-duplication funnel does not reconcile")
    _say(f"    reconciles: {d['rows_out']:,} + {d['removed']:,} = {d['rows_in']:,}")
    _say(f"  PERMNOs before event-universe filter       {uni['permnos_in']:>10,}")
    _say(f"    - no return on either event date         {uni['permnos_dropped']:>10,}"
         f"   ({'applied' if uni['applied'] else 'flagged only'})")
    _say(f"  = PERMNOs in final panel                   {uni['permnos_out']:>10,}")
    if uni["applied"] and uni["permnos_out"] + uni["permnos_dropped"] != uni["permnos_in"]:
        raise ValueError("universe funnel does not reconcile")
    _say(f"  = rows in final panel                      {uni['rows_out']:>10,}")
    for name, n in uni["alive"].items():
        _say(f"    firms trading on {name:<8s} ({events[name]:%Y-%m-%d})  {n:>10,}")

    _section("2. Grain integrity")
    _say("  de-duplication, the momentum merge, the CCM link merge, the fundamentals as-of merge")
    _say("  and the FF12 merge each raise on a row-count change or a duplicated (permno, date);")
    _say("  reaching this report means all five passed.")
    _say(f"  final panel duplicate (permno, date)       "
         f"{int(panel.duplicated(['permno', 'date']).sum()):>10,}   (re-measured here)")
    _say(f"  final panel distinct (permno, date)        "
         f"{len(panel.drop_duplicates(['permno', 'date'])):>10,}")

    _section("3. Market equity and the t-1 lag")
    me = stats["me"]
    _say(f"  negative prices (CRSP bid/ask midpoint)    {me['negative_prices']:>10,}")
    _say(f"  non-positive market equity -> null         {me['nonpositive_me']:>10,}")
    _say(f"  rows with null me                          {me['me_null']:>10,}")
    _say(f"  rows with null me_lag (incl. span left edge){me['me_lag_null']:>10,}")
    _say(f"  rows with null ln_me_lag                   {me['ln_me_lag_null']:>10,}")
    _say(f"  units: me = |prc| x shrout is $thousands; divided by {ME_TO_MILLIONS:,.0f} "
         f"before forming BM against Compustat $millions")

    _section("4. Prior 12-month return")
    src = stats["mom_source"]
    _say(f"  window: months t-{MOM12_LOOKBACK} .. t-1, minimum {MOM12_MIN_MONTHS} observations")
    _say(f"  source: {MONTHLY_RAW_FILE.name} (unscreened history), NOT {MONTHLY_CSV.name}.")
    _say("    The cleaned monthly panel drops firm-months failing the price and micro-cap screens,")
    _say("    which would make the momentum control go missing non-randomly with firm size.")
    _say(f"  monthly rows read for panel firms          {src['rows']:>10,}"
         f"   ({src['permnos']:,} PERMNOs)")
    _say(f"  firm-months computed                       {m['firm_months_computed']:>10,}")
    _say(f"  daily rows carrying mom12                  {m['daily_rows_with_mom12']:>10,}")
    _say(f"  daily rows without mom12                   {m['daily_rows_missing_mom12']:>10,}")

    _section("5. CCM link (PERMNO -> GVKEY)")
    linked_rows = int(panel["gvkey"].notna().sum())
    _say(f"  before the event-universe filter:")
    _say(f"    rows with no date-valid link (kept)      {lk['rows_unlinked']:>10,}"
         f"   ({lk['permnos_unlinked']} PERMNOs)")
    _say(f"  in the final panel:")
    _say(f"    rows with a date-valid link              {linked_rows:>10,}")
    _say(f"    rows with no valid link (null gvkey)     {len(panel) - linked_rows:>10,}"
         f"   ({int(panel.loc[panel['gvkey'].isna(), 'permno'].nunique())} PERMNOs)")
    _say(f"    distinct gvkeys                          {int(panel['gvkey'].nunique()):>10,}")

    _section("6. Point-in-time gate on fundamentals")
    av, be = stats["avail"], stats["be"]
    _say(f"  rule: actual 10-K filing date where known, else datadate + {FUNDAMENTAL_LAG_DAYS}d; "
         f"same-day filings excluded")
    fl = stats["filings"]
    _say(f"  EDGAR filings supplying a real date        {fl['log_filings']:>10,}"
         f"   ({fl['gvkey_periods']:,} distinct gvkey-periods)")
    _say(f"    dropped: filing_date before the period   {fl['filing_dates_before_period']:>10,}")
    _say(f"  fundamentals rows                          {av['rows']:>10,}")
    _say(f"    from a real EDGAR filing date            {av['from_edgar_filing_date']:>10,}")
    _say(f"    from the {FUNDAMENTAL_LAG_DAYS}-day fallback lag             "
         f"{av['from_fallback_lag']:>10,}")
    if dup_fy.empty:
        _say("  gvkey-fiscal year duplicates                     0")
    else:
        n_dup = dup_fy[["gvkey", "fyear"]].drop_duplicates().shape[0]
        _say(f"  gvkey-fiscal year duplicates (later datadate kept) {n_dup:>6,}")
        for r in dup_fy.head(5).itertuples(index=False):
            _say(f"      gvkey={r.gvkey} fyear={r.fyear} datadate={r.datadate:%Y-%m-%d}")
    for name, day in events.items():
        at_event = panel[(panel["date"] == day) & panel["datadate"].notna()]
        if at_event.empty:
            _say(f"  {name}: no firm carries fundamentals at the event date")
            continue
        latest = at_event["available_date"].max()
        if latest >= day:
            raise ValueError(f"point-in-time gate failed at {name}: an available_date "
                             f"{latest:%Y-%m-%d} is not strictly before {day:%Y-%m-%d}")
        src = at_event["pit_source"].value_counts().to_dict()
        fy = at_event["datadate"].dt.year.value_counts().sort_index().to_dict()
        _say(f"  {name} ({day:%Y-%m-%d}): {len(at_event):,} firms with fundamentals")
        _say(f"    latest available_date {latest:%Y-%m-%d} < event date  (gate holds)")
        _say(f"    source: {src}")
        _say(f"    fiscal year-end mix: {fy}")
        _say(f"    stale > {FUNDAMENTAL_STALE_DAYS}d (flagged, kept): "
             f"{int(at_event['fundamentals_stale'].sum()):,}")

    _section("7. Constructed fundamentals")
    _say(f"  BE = ceq + txditc - PS (PS: pstkrv -> pstkl -> pstk); missing txditc treated as 0")
    _say(f"  rows with txditc imputed to 0              {be['txditc_imputed']:>10,}"
         f"   ({100 * be['txditc_imputed'] / be['rows']:.1f}%)")
    _say(f"  rows with unresolvable preferred stock     {be['ps_unresolvable']:>10,}")
    _say(f"  rows with null BE                          {be['be_null']:>10,}")
    _say(f"  rows with BE <= 0 (BM left null, flagged)  {be['be_nonpositive']:>10,}"
         f"   ({100 * be['be_nonpositive'] / be['rows']:.1f}%)")
    _say(f"  rows with ceq <= 0 (Lev left null, flagged){be['ceq_nonpositive']:>10,}"
         f"   ({100 * be['ceq_nonpositive'] / be['rows']:.1f}%)")
    _say(f"  panel rows with fundamentals attached      {fm['rows_with_fundamentals']:>10,}")
    _say(f"  panel rows without fundamentals            {fm['rows_without_fundamentals']:>10,}")
    _say(f"  panel rows with a usable BM                {stats['ratios']['bm_present']:>10,}"
         f"   (null: {stats['ratios']['bm_null']:,})")
    _say(f"  panel rows with a usable Lev               {stats['ratios']['lev_present']:>10,}"
         f"   (null: {stats['ratios']['lev_null']:,})")

    _section("8. FF12 industry")
    ind = stats["ff12"]
    _say(f"  label taken from each firm's SIC at its last month before "
         f"{ind['snapshot_date']:%Y-%m-%d}")
    _say(f"  firms with exactly one label               {ind['firms_labelled']:>10,}  (asserted)")
    _say(f"  firms without a label                      {ind['firms_without_label']:>10,}")
    _say(f"    of which labelled from earliest SIC      {ind['firms_from_fallback_sic']:>10,}"
         f"   (first listed after the imposition date)")
    _say(f"  firms whose SIC changed inside the sample  {ind['sic_changed_in_sample']:>10,}")
    _say(f"  firms with SICCD 0 (invalid, flagged)      {ind['sic_invalid_firms']:>10,}")
    _say(f"  distinct SIC codes                         {ind['distinct_sic']:>10,}")
    _say(f"  SIC codes matching no range -> 'Other'     {len(ind['unmapped_sic']):>10,}")
    if ind["unmapped_sic"]:
        codes = ind["unmapped_sic"]
        for i in range(0, len(codes), 16):
            _say("      " + " ".join(f"{c:>4d}" for c in codes[i:i + 16]))
    _say("  firm counts by industry:")
    for label, n in ind["distribution"].items():
        _say(f"      {label:<8s}{n:>6,}")

    _section("9. Section 6 sample screen (carried as a flag, not applied)")
    sc = stats["screen"]
    _say("  in_screened_universe marks firm-months clearing price > $1 and the NYSE 10th-percentile")
    _say("  size cut, i.e. presence in clean_returns.csv. Script 3 applies it; this panel does not.")
    _say(f"  rows inside the screen                     {sc['rows_in_screen']:>10,}")
    _say(f"  rows outside the screen                    {sc['rows_out_of_screen']:>10,}")
    for name, n in sc["at_event"].items():
        _say(f"    firms inside the screen on {name:<8s}     {n:>10,}"
             f"   (of {uni['alive'][name]:,} trading)")

    _section("10. Distributions of constructed variables")
    _describe(panel, ["ret", "me", "ln_me_lag", "bm", "lev", "mom12"])
    _say("  no winsorising is applied here: BM and Lev have long right tails driven by small")
    _say("  book or common equity, and trimming is an estimation decision for Script 3.")

    _section("11. Universe reconciliation")
    _say(f"  bridge gvkeys                              {bridge['gvkey'].nunique():>10,}")
    _say(f"  fundamentals gvkeys                        {fund['gvkey'].nunique():>10,}")
    no_funda = sorted(set(bridge["gvkey"].dropna()) - set(fund["gvkey"]))
    _say(f"  bridge gvkeys with no fundamentals         {len(no_funda):>10,}")
    if no_funda:
        for i in range(0, min(len(no_funda), 64), 8):
            _say("      " + " ".join(no_funda[i:i + 8]))
        if len(no_funda) > 64:
            _say(f"      ... and {len(no_funda) - 64} more")
    _say(f"  PERMNOs in the final panel                 {panel['permno'].nunique():>10,}")
    _say(f"  PERMNOs dropped, with reason               {len(dropped):>10,}")
    if len(dropped):
        for r in dropped.head(10).itertuples(index=False):
            _say(f"      permno={r.permno} ticker={r.ticker} reason={r.reason}")
        if len(dropped) > 10:
            _say(f"      ... and {len(dropped) - 10} more (all: {dropped['reason'].unique().tolist()})")

    _section("12. Spot check (eyeball against known filings)")

    def _num(value, fmt: str) -> str:
        return "NA" if pd.isna(value) else format(value, fmt)

    for permno in SPOT_CHECK_PERMNOS:
        rows = panel[panel["permno"] == permno]
        if rows.empty:
            _say(f"  permno {permno}: not in the final panel")
            continue
        name = rows["conm"].dropna()
        _say(f"  permno {permno}  {name.iloc[0] if len(name) else ''}")
        for ev_name, day in events.items():
            r = rows[rows["date"] == day]
            if r.empty:
                _say(f"    {ev_name:<8s} {day:%Y-%m-%d}  no observation on this date")
                continue
            r = r.iloc[0]
            _say(f"    {ev_name:<8s} {day:%Y-%m-%d}  ticker={r['ticker']} "
                 f"gvkey={r['gvkey']} ff12={r['ff12']} (siccd {r['siccd']})")
            _say(f"      prc={_num(r['prc'], ',.2f')}  "
                 f"me_lag={_num(r['me_lag'] / ME_TO_MILLIONS, ',.0f')}m  "
                 f"ln_me_lag={_num(r['ln_me_lag'], '.3f')}  "
                 f"mom12={_num(r['mom12'], '.4f')} ({r['mom12_n_months']} months)")
            if pd.isna(r["datadate"]):
                _say("      fundamentals: none available point-in-time")
                continue
            _say(f"      fiscal year-end {r['datadate']:%Y-%m-%d}, available "
                 f"{r['available_date']:%Y-%m-%d} via {r['pit_source']}"
                 f"{'  [STALE]' if r['fundamentals_stale'] else ''}")
            _say(f"      be={_num(r['be'], ',.0f')}m  debt={_num(r['debt'], ',.0f')}m  "
                 f"ceq={_num(r['ceq'], ',.0f')}m  bm={_num(r['bm'], '.4f')}  "
                 f"lev={_num(r['lev'], '.4f')}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_OUT.write_text("\n".join(_REPORT), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def main() -> pd.DataFrame:
    for path in (BRIDGE_CSV, MONTHLY_CSV):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; run clean_data.py first.")

    validate_daily_schema(DAILY_FILE)
    calendar = build_calendar(DAILY_FILE)
    start, end, bound_stats = derive_bounds(calendar)

    daily = load_daily(DAILY_FILE, start, end)
    daily, dedupe_stats = dedupe_daily(daily)
    daily, me_stats = add_market_equity(daily)

    monthly = pd.read_csv(MONTHLY_CSV, usecols=["permno", "date", "siccd"], parse_dates=["date"])
    monthly_raw, raw_stats = load_monthly_returns(MONTHLY_RAW_FILE, set(daily["permno"].unique()))
    daily, mom_stats = add_momentum(daily, monthly_raw)

    bridge = pd.read_csv(BRIDGE_CSV, dtype=str)
    daily, link_stats = link_gvkey(daily, bridge)

    fund, dup_fy = load_fundamentals(FUNDA_FILE)
    filings, filing_stats = load_edgar_filing_dates(EDGAR_LOG)
    fund, avail_stats = add_availability_date(fund, filings)
    fund, be_stats = build_book_equity(fund)
    daily, fund_stats = merge_fundamentals(daily, fund)
    daily, ratio_stats = build_ratios(daily)

    daily, ff12_stats = assign_ff12(daily, monthly, parse_ff12(SICCODES_FILE))
    daily, screen_stats = add_screened_universe_flag(daily, monthly)
    daily, dropped, univ_stats = restrict_universe(daily)

    panel_path = write_panel(daily)
    validate(daily, fund, bridge,
             {"bounds": bound_stats, "dedupe": dedupe_stats, "me": me_stats, "mom": mom_stats,
              "mom_source": raw_stats, "link": link_stats, "filings": filing_stats,
              "avail": avail_stats, "be": be_stats,
              "fund_merge": fund_stats, "ratios": ratio_stats, "ff12": ff12_stats,
              "screen": screen_stats, "universe": univ_stats},
             dropped, dup_fy, (start, end), panel_path)
    print("\n".join(_REPORT))
    return daily


if __name__ == "__main__":
    main()
