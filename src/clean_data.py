"""
Part A — Clean raw data and build the firm-level PERMNO-GVKEY-CIK bridge.

Reads the raw CRSP monthly file (CIZ format) and the CCM linking table, applies
the common-equity and price filters, flags NYSE micro-caps, and constructs a
long-format bridge table (one row per valid PERMNO-GVKEY-CIK link interval,
respecting linkdt/linkenddt validity against each firm's CRSP observation range).

Downstream modules should import these functions and pass the returned DataFrames
directly rather than re-reading the raw files. Running this file as a script
executes the full pipeline and writes clean_firm_bridge.csv.

No text parsing, tokenization, or scoring is performed here (that is Step 2).
"""

import io
import re
import argparse

import pandas as pd

import run_report
from config import CLEAN_DIR, OUTPUT_DIR, RAW_DIR

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
REPORT_OUT = OUTPUT_DIR / "source_cleaning_validation_report.txt"
CRSP_FILE = RAW_DIR / "Monthly Returns.csv"
LINK_FILE = RAW_DIR / "PERMNO - GVKEY - CIK.csv"
GEO_FILE = RAW_DIR / "Compustat Geographic segment data.csv"  # date-checked here; read by foreign_sales
FF_FILE = RAW_DIR / "FF5_MOM_Factors.csv"
EPU_FILE = RAW_DIR / "US_Policy_Uncertainty_Data.xlsx"
ME_BP_FILE = RAW_DIR / "ME_Breakpoints.csv"      # Ken French NYSE ME breakpoints
BRIDGE_OUT = CLEAN_DIR / "clean_firm_bridge.csv"
RETURNS_OUT = CLEAN_DIR / "clean_returns.csv"
FF_OUT = CLEAN_DIR / "clean_ff5_mom.csv"
EPU_OUT = CLEAN_DIR / "clean_epu.csv"

EPU_SHEET = "Main News Index"                 # sheet holding the chosen EPU series
EPU_VALUE_COL = "News_Based_Policy_Uncert_Index"  # chosen EPU variant

REFERENCE_DATE = pd.Timestamp("2025-04-02")  # "Liberation Day" tariffs
PRICE_MIN = 1.0                              # keep price > $1
NYSE_PCTILE = 10                             # NYSE micro-cap breakpoint (10th pct)
DROP_MICROCAPS = True                        # drop firm-months below the NYSE p10 breakpoint
ME_BP_P10_FIELD = 3                          # French layout: [0]=YYYYMM [1]=count [2]=p5 [3]=p10
ME_BP_SCALE = 1000.0                         # French ME is $millions; CRSP MthCap is $thousands
OPEN_END = pd.Timestamp("2099-12-31")        # sentinel for still-active links ('E')

COMMON_EQUITY_FILTER = {                      # CIZ-to-SIZ SHRCD 10/11 replication
    "ShareType": {"NS"},
    "SecurityType": {"EQTY"},
    "SecuritySubType": {"COM"},
    "USIncFlg": {"Y"},
    "IssuerType": {"ACOR", "CORP"},
}

_QUALITY_LOG: list[str] = []  # data-quality notes accumulated across the run


def _note(msg: str) -> None:
    """Record a data-quality observation for the end-of-run summary."""
    _QUALITY_LOG.append(msg)
    print(msg)


def to_month_end(dates) -> pd.Series:
    """Normalise a datetime series to the calendar month-end Timestamp.

    Gives one uniform monthly join key across the CRSP, factor, and EPU panels:
    CRSP MthCalDt and FF dateff carry last-trading-day dates while EPU has only
    Year+Month, so all are collapsed to the last calendar day of the month.
    """
    return pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp("M")


def _me_bp_data_rows() -> list[str]:
    """Return only the data rows of the French breakpoints file.

    The file has a one-line text header, blank lines and a copyright footer, and
    no column header row, so data rows are selected by pattern (leading YYYYMM)
    rather than by offset. A plain read_csv fails on this file: pandas infers the
    field count from the one-field text header and then rejects the 22-field rows.
    """
    rows = [ln for ln in ME_BP_FILE.read_text(encoding="utf-8").splitlines()
            if re.match(r"^\s*\d{6}\s*,", ln)]
    if not rows:
        raise ValueError(f"{ME_BP_FILE.name}: no rows matched the leading-YYYYMM data "
                         f"pattern; the file layout has changed.")
    return rows


# --------------------------------------------------------------------------- #
# Part A.1 — Inspect raw date fields BEFORE any parsing                        #
# --------------------------------------------------------------------------- #
def inspect_dates() -> None:
    """Print the raw dtype and a sample of every date-like field, unparsed.

    Confirms format assumptions (yyyymm, datadate, linkdt, linkenddt, dateff,
    Year/Month) against the actual files instead of assuming consistency.
    """
    print("=" * 70)
    print("DATE-FIELD INSPECTION (raw, pre-parse)")
    print("=" * 70)
    checks = [
        (CRSP_FILE, ["YYYYMM", "MthCalDt"]),
        (LINK_FILE, ["LINKDT", "LINKENDDT"]),
        (GEO_FILE, ["datadate"]),
        (FF_FILE, ["dateff"]),
    ]
    for path, fields in checks:
        head = pd.read_csv(path, usecols=fields, dtype=str, nrows=5)
        for f in fields:
            print(f"  {path.name:45s} {f:12s} dtype=str  sample={head[f].tolist()}")
    epu = pd.read_excel(EPU_FILE, sheet_name=EPU_SHEET, nrows=3)[["Year", "Month"]]
    print(f"  {EPU_FILE.name:45s} {'Year/Month':12s} "
          f"sample={list(zip(epu['Year'], epu['Month']))}")
    bp_rows = _me_bp_data_rows()[:3]
    print(f"  {ME_BP_FILE.name:45s} {'YYYYMM':12s} dtype=str  "
          f"sample={[r.split(',')[0].strip() for r in bp_rows]}  (no column header row)")
    print()


# --------------------------------------------------------------------------- #
# Part A.2-A.5 — CRSP cleaning                                                 #
# --------------------------------------------------------------------------- #
def load_crsp() -> pd.DataFrame:
    """Load the CRSP monthly file, parse dates, and coerce numeric fields.

    Deduplicates fully identical rows (an export artifact in the raw file) and
    returns one row per PERMNO-month for the retained columns.
    """
    cols = [
        "PERMNO", "PrimaryExch", "USIncFlg", "IssuerType", "SecurityType",
        "SecuritySubType", "ShareType", "SICCD", "Ticker", "YYYYMM",
        "MthCalDt", "MthPrc", "MthCap", "ShrOut", "MthRet",
    ]
    df = pd.read_csv(CRSP_FILE, usecols=cols, dtype=str)

    n_raw = len(df)
    df = df.drop_duplicates()
    if n_raw - len(df):
        _note(f"[dq] CRSP: dropped {n_raw - len(df):,} fully-identical duplicate "
              f"rows ({n_raw:,} -> {len(df):,}).")

    df["PERMNO"] = pd.to_numeric(df["PERMNO"], errors="coerce").astype("Int64")
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"], format="%Y-%m-%d")
    for c in ["MthPrc", "MthCap", "ShrOut", "MthRet"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dup_key = int(df.duplicated(["PERMNO", "YYYYMM"], keep=False).sum())
    if dup_key:
        _note(f"[dq] CRSP: {dup_key:,} PERMNO-YYYYMM rows remain duplicated on the "
              f"key after exact-dedup (differ only in delisting/last-known fields); "
              f"resolved within the common-equity subset below.")
    return df


def filter_common_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to US common equity, replicating legacy CRSP SHRCD 10/11.

    Rule (CIZ->SIZ): ShareType=='NS' AND SecurityType=='EQTY' AND
    SecuritySubType=='COM' AND USIncFlg=='Y' AND IssuerType in {ACOR,CORP}.
    """
    mask = pd.Series(True, index=df.index)
    for col, allowed in COMMON_EQUITY_FILTER.items():
        mask &= df[col].isin(allowed)
    out = df[mask].copy()
    _note(f"[info] Common-equity filter: {len(df):,} -> {len(out):,} rows, "
          f"{out['PERMNO'].nunique():,} unique PERMNOs.")

    key_dups = int(out.duplicated(["PERMNO", "YYYYMM"], keep=False).sum())
    if key_dups:
        before = len(out)
        out = out.drop_duplicates(["PERMNO", "YYYYMM"], keep="first")
        _note(f"[dq] Common-equity: {key_dups:,} residual PERMNO-YYYYMM duplicate "
              f"rows collapsed to one ({before:,} -> {len(out):,}).")
    return out


def apply_price_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep firm-months with price > PRICE_MIN.

    CRSP encodes bid/ask-average prices as negatives; take absolute value before
    filtering rather than dropping those rows. Null prices cannot clear the
    threshold and are dropped (logged).
    """
    df = df.copy()
    df["price"] = df["MthPrc"].abs()
    n_neg = (df["MthPrc"] < 0).sum()
    n_null = df["price"].isna().sum()
    kept = df[df["price"] > PRICE_MIN].copy()
    _note(f"[info] Price filter (>{PRICE_MIN}): {len(df):,} -> {len(kept):,} rows "
          f"(neg-price rows recovered via abs: {n_neg:,}; "
          f"null-price rows dropped: {n_null:,}).")
    return kept


def load_nyse_breakpoints() -> pd.DataFrame:
    """Load the official Ken French NYSE ME breakpoints (date + nyse_p10).

    Fields are [YYYYMM, NYSE firm count, p5, p10, ..., p100] with no header row;
    the 10th percentile is field ME_BP_P10_FIELD. French quotes ME in $millions
    while CRSP MthCap is in $thousands, so the level is scaled by ME_BP_SCALE.
    Returned dates use the shared calendar month-end join key.
    """
    raw = pd.read_csv(io.StringIO("\n".join(_me_bp_data_rows())),
                      header=None, skipinitialspace=True)
    bp = pd.DataFrame({
        "date": to_month_end(pd.to_datetime(raw[0].astype(int).astype(str), format="%Y%m")),
        "nyse_p10": pd.to_numeric(raw[ME_BP_P10_FIELD], errors="coerce") * ME_BP_SCALE,
    })
    if bp["date"].duplicated().any() or bp["nyse_p10"].isna().any():
        raise ValueError(f"{ME_BP_FILE.name}: duplicate months or unparsable values in "
                         f"field {ME_BP_P10_FIELD}; check the file layout.")
    _note(f"[info] NYSE breakpoints: {len(bp):,} months "
          f"{bp['date'].min():%Y-%m} -> {bp['date'].max():%Y-%m} from {ME_BP_FILE.name} "
          f"(field {ME_BP_P10_FIELD} = p{NYSE_PCTILE}; $M x{ME_BP_SCALE:,.0f} -> $thousands).")
    return bp


def flag_microcaps(df: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    """Drop firm-months below the official NYSE 10th-percentile market cap.

    The breakpoint is the Ken French NYSE ME series (``bp``), merged on the
    canonical month-end date. The previously internally-computed per-month
    breakpoint is retained as a diagnostic only: both classifications are
    reported so the change of source can be sanity-checked, but French governs
    the ``below_nyse_p10`` flag and the drop (unless DROP_MICROCAPS is False).
    """
    df = df.copy()
    df["date"] = to_month_end(df["MthCalDt"])

    q = NYSE_PCTILE / 100.0
    internal = (df[df["PrimaryExch"] == "N"]
                .groupby("date")["MthCap"].quantile(q).rename("nyse_p10_internal"))
    df = df.merge(internal, on="date", how="left").merge(bp, on="date", how="left")

    if df["nyse_p10"].isna().any():
        gaps = sorted(df.loc[df["nyse_p10"].isna(), "date"].dt.strftime("%Y-%m").unique())
        raise ValueError(f"{ME_BP_FILE.name} has no p{NYSE_PCTILE} breakpoint for "
                         f"{len(gaps)} CRSP month(s): {gaps[:12]}")

    df["below_nyse_p10"] = df["MthCap"] < df["nyse_p10"]
    below_internal = df["MthCap"] < df["nyse_p10_internal"]
    n_fr, n_int, n = int(df["below_nyse_p10"].sum()), int(below_internal.sum()), len(df)
    _note(f"[info] Micro-cap breakpoint sources over {n:,} firm-months: internal "
          f"p{NYSE_PCTILE} flags {n_int:,} ({n_int / n:.1%}), French p{NYSE_PCTILE} flags "
          f"{n_fr:,} ({n_fr / n:.1%}); agreement "
          f"{(df['below_nyse_p10'] == below_internal).mean():.2%}; mean threshold ratio "
          f"internal/French {(df['nyse_p10_internal'] / df['nyse_p10']).mean():.3f}.")

    if not DROP_MICROCAPS:
        _note(f"[info] Micro-caps flagged only (DROP_MICROCAPS=False).")
        return df
    n_permno = df["PERMNO"].nunique()
    df = df[~df["below_nyse_p10"]].copy()
    _note(f"[info] Micro-caps dropped on French p{NYSE_PCTILE}: {n:,} -> {len(df):,} "
          f"firm-months, {n_permno:,} -> {df['PERMNO'].nunique():,} PERMNOs.")
    return df


# --------------------------------------------------------------------------- #
# Linking table + bridge construction                                         #
# --------------------------------------------------------------------------- #
def load_link() -> pd.DataFrame:
    """Load the CCM linking table, filter to primary links, parse link dates.

    Keeps LINKPRIM in {'P','C'} (drops joint 'J' and non-primary 'N'). Maps the
    still-active LINKENDDT sentinel 'E' to OPEN_END and zero-pads CIK to 10 digits.
    """
    link = pd.read_csv(LINK_FILE, dtype=str)
    n_raw = len(link)
    link = link[link["LINKPRIM"].isin(["P", "C"])].copy()
    _note(f"[info] Link table: LINKPRIM in (P,C) filter {n_raw:,} -> {len(link):,} rows.")

    link["LPERMNO"] = pd.to_numeric(link["LPERMNO"], errors="coerce").astype("Int64")
    link["linkdt"] = pd.to_datetime(link["LINKDT"], format="%Y-%m-%d")
    link["link_open"] = link["LINKENDDT"] == "E"
    end = link["LINKENDDT"].where(~link["link_open"])
    link["linkenddt"] = pd.to_datetime(end, format="%Y-%m-%d").fillna(OPEN_END)

    cik = link["cik"].str.strip()
    link["cik"] = cik.where(cik.notna() & (cik != ""), "").apply(
        lambda x: x.zfill(10) if x else "")
    link["cik_missing"] = link["cik"] == ""
    n_miss = int(link["cik_missing"].sum())
    _note(f"[dq] Link table: {n_miss:,} of {len(link):,} primary links have no CIK.")
    return link


def build_bridge(crsp_universe: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Build the long-format PERMNO-GVKEY-CIK bridge with date validity.

    For each PERMNO, its CRSP observation range [obs_start, obs_end] is matched
    against every candidate link interval [linkdt, linkenddt]; a link is retained
    only if the intervals overlap (date-valid join, not a static merge). One row
    is emitted per valid PERMNO-link interval.
    """
    obs = (crsp_universe.groupby("PERMNO")["MthCalDt"]
           .agg(obs_start="min", obs_end="max").reset_index())

    m = obs.merge(link, left_on="PERMNO", right_on="LPERMNO", how="left")
    matched = m["LPERMNO"].notna()
    overlap = matched & (m["linkdt"] <= m["obs_end"]) & (m["linkenddt"] >= m["obs_start"])

    no_link = sorted(obs.loc[~obs["PERMNO"].isin(m.loc[overlap, "PERMNO"]), "PERMNO"]
                     .astype(int).tolist())
    _note(f"[dq] Bridge: {len(no_link):,} universe PERMNOs have no valid CCM "
          f"primary link and are excluded from the bridge.")

    bridge = m[overlap].copy()  # overlap implies matched -> these flags are non-null
    bridge["link_open"] = bridge["link_open"].astype(bool)
    bridge["cik_missing"] = bridge["cik_missing"].astype(bool)
    bridge["linkenddt_out"] = bridge["linkenddt"].where(~bridge["link_open"])
    out = bridge[[
        "PERMNO", "gvkey", "cik", "cik_missing", "LINKPRIM", "LINKTYPE",
        "linkdt", "linkenddt_out", "link_open", "obs_start", "obs_end",
    ]].rename(columns={
        "PERMNO": "permno", "LINKPRIM": "linkprim", "LINKTYPE": "linktype",
        "linkenddt_out": "linkenddt",
    }).sort_values(["permno", "linkdt"]).reset_index(drop=True)

    for c in ["linkdt", "linkenddt", "obs_start", "obs_end"]:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    _note(f"[info] Bridge: {len(out):,} rows, {out['permno'].nunique():,} PERMNOs, "
          f"{out['gvkey'].nunique():,} GVKEYs; "
          f"{int((out['cik'] == '').sum()):,} rows without CIK.")
    return out


def log_disagreements(bridge: pd.DataFrame) -> None:
    """Report PERMNOs that map to more than one GVKEY over their valid history.

    These are not silently resolved: the pull resolves a single CIK per firm at
    pull time using the reference-date interval; here we simply record them.
    """
    per = bridge.groupby("permno")["gvkey"].nunique()
    multi = sorted(per[per > 1].index.astype(int).tolist())
    _note(f"[dq] Bridge: {len(multi):,} PERMNOs map to >1 GVKEY over time "
          f"(logged, not resolved). First 15: {multi[:15]}")


# --------------------------------------------------------------------------- #
# Clean panel exports (returns, FF5+MOM, EPU)                                  #
# --------------------------------------------------------------------------- #
def export_returns(ce: pd.DataFrame) -> pd.DataFrame:
    """Export the cleaned CRSP monthly returns panel to clean_returns.csv.

    Standardises the date to calendar month-end and drops columns not needed for
    the pricing tests, logging each drop and its reason.
    """
    dropped = {
        "USIncFlg": "constant after common-equity filter (all 'Y')",
        "IssuerType": "filter field (CORP/ACOR); no pricing use downstream",
        "SecurityType": "constant after filter (all 'EQTY')",
        "SecuritySubType": "constant after filter (all 'COM')",
        "ShareType": "constant after filter (all 'NS')",
        "YYYYMM": "superseded by canonical month-end 'date'",
        "MthCalDt": "superseded by canonical month-end 'date'",
        "MthPrc": "replaced by absolute-valued 'prc'",
        "nyse_p10": "intermediate breakpoint threshold; not needed downstream",
        "nyse_p10_internal": "diagnostic only (internal-vs-French breakpoint comparison)",
        "ShrOut": "redundant (me = prc x shrout)",
        "Ticker": "not needed; permno is the join key",
    }
    cols = {
        "permno": ce["PERMNO"],
        "date": to_month_end(ce["MthCalDt"]),
        "ret": ce["MthRet"],
        "me": ce["MthCap"],
        "prc": ce["price"],
        "siccd": ce["SICCD"],
        "primaryexch": ce["PrimaryExch"],
    }
    if DROP_MICROCAPS:
        dropped["below_nyse_p10"] = "constant False after the micro-cap drop"
    else:
        cols["below_nyse_p10"] = ce["below_nyse_p10"]
    panel = pd.DataFrame(cols).sort_values(["permno", "date"]).reset_index(drop=True)

    for col, reason in dropped.items():
        _note(f"[drop] returns.{col}: {reason}")
    panel.to_csv(RETURNS_OUT, index=False)
    _note(f"[info] Wrote {RETURNS_OUT.name} ({len(panel):,} rows; "
          f"kept={list(panel.columns)}).")
    return panel


def clean_ff5_mom() -> pd.DataFrame:
    """Clean the FF5+Momentum factor file to clean_ff5_mom.csv.

    Standardises the date, verifies the percent-vs-decimal convention (converting
    only if needed), and maps any missing-value sentinels to NaN. rf is retained
    because the abnormal-return pipeline needs it: estimate_car forms excess returns as
    r - rf before fitting FF5+MOM loadings. The v5 design also cited Sharpe-ratio
    computation, which v6 retires along with the long-short portfolio; the excess-return
    use is what keeps rf here.
    """
    factors = ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
    ff = pd.read_csv(FF_FILE)
    ff[factors] = ff[factors].apply(pd.to_numeric, errors="coerce")

    max_abs = float(ff[factors].abs().max().max())
    scale = "decimal" if max_abs < 1 else "percent"
    _note(f"[info] FF units: max|value|={max_abs:.4f} -> {scale}; "
          f"{'no conversion applied' if scale == 'decimal' else 'divided by 100'}.")
    if scale == "percent":
        ff[factors] = ff[factors] / 100.0

    sentinels = [-99.99, -999, -0.9999, -9.99]
    n_hits = int(ff[factors].isin(sentinels).sum().sum())
    if n_hits:
        ff[factors] = ff[factors].mask(ff[factors].isin(sentinels))
    _note(f"[info] FF sentinel scan {sentinels}: {n_hits} value(s) -> NaN.")

    ff["date"] = to_month_end(pd.to_datetime(ff["dateff"], format="%d/%m/%Y"))
    out = ff[["date"] + factors].sort_values("date").reset_index(drop=True)
    out.to_csv(FF_OUT, index=False)
    _note(f"[info] Wrote {FF_OUT.name} ({len(out):,} rows; cols={list(out.columns)}; "
          f"rf retained for excess returns).")
    return out


def clean_epu() -> pd.DataFrame:
    """Clean the EPU index to clean_epu.csv.

    Reports every EPU variant present (none silently dropped), keeps the chosen
    News-Based series, and standardises the monthly date to calendar month-end.
    """
    xl = pd.ExcelFile(EPU_FILE)
    variants = {s: [c for c in xl.parse(s, nrows=0).columns
                    if c not in ("Year", "Month")] for s in xl.sheet_names}
    _note(f"[info] EPU variants present (not silently picked): {variants}. "
          f"Chosen = '{EPU_VALUE_COL}' from '{EPU_SHEET}'; "
          f"no trade-policy sub-index exists in this file.")

    df = xl.parse(EPU_SHEET)
    for c in ["Year", "Month", EPU_VALUE_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_raw = len(df)
    df = df.dropna(subset=["Year", "Month", EPU_VALUE_COL])
    if n_raw - len(df):
        _note(f"[dq] EPU: dropped {n_raw - len(df)} non-data row(s) "
              f"(trailing source-attribution text).")

    ym = pd.to_datetime(dict(year=df["Year"].astype(int),
                             month=df["Month"].astype(int), day=1))
    out = (pd.DataFrame({"date": to_month_end(ym).values,
                         "epu_news": df[EPU_VALUE_COL].values})
           .sort_values("date").reset_index(drop=True))
    _note(f"[info] EPU frequency = monthly; {len(out):,} rows, "
          f"{out['date'].min().date()} -> {out['date'].max().date()}.")
    out.to_csv(EPU_OUT, index=False)
    _note(f"[info] Wrote {EPU_OUT.name} ({len(out):,} rows; cols={list(out.columns)}).")
    return out


# --------------------------------------------------------------------------- #
# Assumptions summary                                                         #
# --------------------------------------------------------------------------- #
def print_assumptions() -> None:
    """Print the assumptions that governed this run (spec Part A requirement)."""
    print("\n" + "=" * 70)
    print("ASSUMPTIONS (this run)")
    print("=" * 70)
    for line in [
        f"Reference date (Liberation Day) = {REFERENCE_DATE.date()}.",
        "Common equity = CIZ->SIZ SHRCD 10/11 rule "
        "(NS & EQTY & COM & USInc=Y & IssuerType in {ACOR,CORP}); excludes REITs/funds/ADRs.",
        "Observation date = MthCalDt (ISO month-end); YYYYMM used only as a cross-check key.",
        f"Price filter uses abs(MthPrc) > {PRICE_MIN} (CRSP negative = bid/ask average, not missing).",
        f"NYSE micro-cap breakpoint = official Ken French NYSE ME series ({ME_BP_FILE.name}, "
        f"field {ME_BP_P10_FIELD} = p{NYSE_PCTILE}), merged on the month-end date; French "
        f"quotes $millions and CRSP MthCap is $thousands, so the level is scaled "
        f"x{ME_BP_SCALE:,.0f}.",
        f"DROP_MICROCAPS={DROP_MICROCAPS}: firm-months below the breakpoint are dropped. "
        "The screen is applied per firm-month, so firms crossing the breakpoint have gaps "
        "rather than being excluded wholesale.",
        "The internally-computed p10 is retained as a diagnostic only (both classifications "
        "reported for comparison); the French series governs the drop.",
        "Link table filtered to LINKPRIM in {P,C}; LINKTYPE already LC/LU at query time.",
        "LINKENDDT 'E' (still active) mapped to open-ended validity.",
        "Bridge is long: one row per valid PERMNO-GVKEY-CIK link interval; "
        "date validity requires link interval to overlap the firm's CRSP observation range.",
        "Multi-GVKEY PERMNOs are logged, not resolved, at the cleaning stage.",
        "All monthly panels (returns/FF/EPU) share a calendar month-end Timestamp 'date' as the "
        "uniform join key; CRSP MthCalDt and FF dateff (last trading day) are normalised to it.",
        "FF5+MOM values are decimals (not percent) - no /100 conversion; no missing sentinels present.",
        "clean_ff5_mom retains rf (risk-free) for excess-return computation in estimate_car; "
        "v6 retires the Sharpe ratio the v5 design also cited it for.",
        "EPU = News-Based index (Main News Index sheet); "
        "other variants reported, not silently dropped.",
    ]:
        print(f"  - {line}")
    print()


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def _up_to_date(outputs, label: str, force: bool) -> bool:
    """True when every output already exists and the caller has not passed --force.

    Cleaning is deterministic in its inputs, so re-deriving an output that is already on disk
    costs I/O and produces the same bytes. The stages that stream per document (clean_filings,
    score_filings) have always resumed; this gives the whole-file stages the same courtesy, with
    an explicit override rather than an implicit one.
    """
    missing = [p for p in outputs if not p.exists()]
    if force or missing:
        if missing and not force:
            print(f"{label}: rebuilding - missing "
                  f"{', '.join(p.name for p in missing)}")
        return False
    print(f"{label}: already built, not regenerated. Outputs:")
    for p in outputs:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:,.1f} MB)")
    print("  Pass --force to rebuild.")
    return True


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    ap = argparse.ArgumentParser(description="Clean the CRSP, CCM, factor and EPU sources.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the cleaned files even if they already exist")
    args = ap.parse_args(argv)
    outputs = [RETURNS_OUT, BRIDGE_OUT, FF_OUT, EPU_OUT]
    if _up_to_date(outputs, "clean_data", args.force):
        return None
    with run_report.capture(REPORT_OUT, title="STEP 1 - SOURCE DATA CLEANING VALIDATION"):
        return _run()


def _run() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    inspect_dates()
    crsp = load_crsp()
    ce = filter_common_equity(crsp)
    ce = apply_price_filter(ce)
    ce = flag_microcaps(ce, load_nyse_breakpoints())
    export_returns(ce)
    link = load_link()
    bridge = build_bridge(ce, link)
    log_disagreements(bridge)
    bridge.to_csv(BRIDGE_OUT, index=False)
    _note(f"[info] Wrote {BRIDGE_OUT.name} ({len(bridge):,} rows).")
    clean_ff5_mom()
    clean_epu()
    print_assumptions()
    return bridge


if __name__ == "__main__":
    main()
