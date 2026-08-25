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

from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
CRSP_FILE = BASE / "Monthly Returns.csv"
LINK_FILE = BASE / "PERMNO - GVKEY - CIK.csv"
GEO_FILE = BASE / "Compustat Geographic segment data.csv"
BRIDGE_OUT = BASE / "clean_firm_bridge.csv"

REFERENCE_DATE = pd.Timestamp("2025-04-02")  # "Liberation Day" tariffs
PRICE_MIN = 1.0                              # keep price > $1
NYSE_PCTILE = 10                             # NYSE micro-cap breakpoint (10th pct)
DROP_MICROCAPS = False                       # flag only by default; toggle to drop
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


# --------------------------------------------------------------------------- #
# Part A.1 — Inspect raw date fields BEFORE any parsing                        #
# --------------------------------------------------------------------------- #
def inspect_dates() -> None:
    """Print the raw dtype and a sample of every date-like field, unparsed.

    Confirms format assumptions (yyyymm, datadate, linkdt, linkenddt) against the
    actual files instead of assuming consistency across sources.
    """
    print("=" * 70)
    print("DATE-FIELD INSPECTION (raw, pre-parse)")
    print("=" * 70)
    checks = [
        (CRSP_FILE, ["YYYYMM", "MthCalDt"]),
        (LINK_FILE, ["LINKDT", "LINKENDDT"]),
        (GEO_FILE, ["datadate"]),
    ]
    for path, fields in checks:
        head = pd.read_csv(path, usecols=fields, dtype=str, nrows=5)
        for f in fields:
            vals = head[f].tolist()
            print(f"  {path.name:45s} {f:12s} dtype=str  sample={vals}")
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


def flag_microcaps(df: pd.DataFrame) -> pd.DataFrame:
    """Flag firm-months below the per-month NYSE 10th-percentile market cap.

    Breakpoints are computed internally from NYSE-listed (PrimaryExch=='N')
    common-equity firms each month — standard Fama-French practice. Adds the
    boolean column ``below_nyse_p10``; rows are dropped only if DROP_MICROCAPS.
    """
    df = df.copy()
    q = NYSE_PCTILE / 100.0
    bp = (df[df["PrimaryExch"] == "N"]
          .groupby("YYYYMM")["MthCap"].quantile(q).rename("nyse_p10"))
    df = df.merge(bp, on="YYYYMM", how="left")
    df["below_nyse_p10"] = df["MthCap"] < df["nyse_p10"]
    n_flag = int(df["below_nyse_p10"].sum())
    _note(f"[info] Micro-cap flag: {n_flag:,} firm-months below NYSE p{NYSE_PCTILE} "
          f"(DROP_MICROCAPS={DROP_MICROCAPS}).")
    if DROP_MICROCAPS:
        df = df[~df["below_nyse_p10"]].copy()
        _note(f"[info] Micro-caps dropped -> {len(df):,} rows.")
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
        f"NYSE micro-cap breakpoint = per-month {NYSE_PCTILE}th pct of MthCap over "
        "PrimaryExch=='N' common-equity firms (computed internally, Fama-French style); "
        f"flag only unless DROP_MICROCAPS=True.",
        "Link table filtered to LINKPRIM in {P,C}; LINKTYPE already LC/LU at query time.",
        "LINKENDDT 'E' (still active) mapped to open-ended validity.",
        "Bridge is long: one row per valid PERMNO-GVKEY-CIK link interval; "
        "date validity requires link interval to overlap the firm's CRSP observation range.",
        "Multi-GVKEY PERMNOs are logged, not resolved, at the cleaning stage.",
    ]:
        print(f"  - {line}")
    print()


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def main() -> pd.DataFrame:
    inspect_dates()
    crsp = load_crsp()
    ce = filter_common_equity(crsp)
    ce = apply_price_filter(ce)
    ce = flag_microcaps(ce)
    link = load_link()
    bridge = build_bridge(ce, link)
    log_disagreements(bridge)
    bridge.to_csv(BRIDGE_OUT, index=False)
    _note(f"[info] Wrote {BRIDGE_OUT.name} ({len(bridge):,} rows).")
    print_assumptions()
    return bridge


if __name__ == "__main__":
    main()
