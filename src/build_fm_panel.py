"""Step 7a of 2 - assemble the monthly firm panel for the section 7.4 Fama-MacBeth pricing test.

One row per (subsample firm, return month). Every regressor is resolved at ``pit_date``, the last
calendar day of month t-1; the only month-t quantity is the dependent variable. Firms and vintages
come from the two fixed lists section 7.0 defines - the 1,000-firm random draw and the nine
April-2 reference dates of edgar_pull's full_panel batch - so the exposure a month carries is the
10-K that was actually on file when the month opened, and nothing else.

Every candidate firm-month is written, kept or not, carrying the reason it was excluded. The panel
is therefore self-describing: Script 7b derives the funnel, the per-vintage table and the monthly
cross-section counts from it without a second source.

    python src/build_fm_panel.py
    python src/build_fm_panel.py --status     # funnel and monthly counts, writes nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_report

import clean_controls_data as ccd
import clean_data as cd
import run_car_regression as rcr
from config import BASE, CLEAN_DIR, INTERMEDIATE_DIR, OUTPUT_DIR

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
REPORT_OUT = OUTPUT_DIR / "fm_panel_validation_report.txt"

MONTHLY_RAW_FILE = ccd.MONTHLY_RAW_FILE      # unscreened CRSP monthly: return, cap, SIC
SCREENED_CSV = ccd.MONTHLY_CSV               # clean_returns.csv - presence *is* the section 6 screen
BRIDGE_CSV = ccd.BRIDGE_CSV
SUBSAMPLE_CSV = OUTPUT_DIR / "full_panel_firm_sample.csv"
TEXP_PANEL_CSV = CLEAN_DIR / "texp_panel.csv"
EPU_CSV = CLEAN_DIR / "clean_epu.csv"
PANEL_OUT = INTERMEDIATE_DIR / "fm_panel.csv"

# Sample window. The left edge is set by the momentum control, not by choice: Monthly Returns.csv
# begins 2017-01-31 and no earlier return history exists anywhere in the project, so the twelve
# months ending t-1 are first complete at 2018-01. TExp is independently unavailable before
# 2017-05 (first vintage 2017-04-02), so the eight months this costs are 2017-05..2017-12.
SAMPLE_START = pd.Period("2018-01", freq="M")
SAMPLE_END = pd.Period("2025-12", freq="M")

# The full_panel batch: April 2 each year 2017-2025, pulled at scope subsample (edgar_pull.BATCHES). The
# other eight reference dates in texp_panel.csv belong to the cross-cycle event study and are not
# this test's vintages.
VINTAGE_SUFFIX = "-04-02"
N_VINTAGES = 9

# A vintage dated y-04-02 holds 10-Ks filed in [y-04-02 minus STALENESS_DAYS, y-04-01], so its
# newest document is public before 1 May y. It takes effect in May y and is carried forward twelve
# months - to April y+1 - and no further: a firm with no vintage in year y+1 has no exposure for
# that block rather than a score whose fiscal period is two years old. Unlimited carry-forward
# would add 120 firm-vintage-years (+1.7%), measured on this sample, so the bounded rule is nearly
# free.
VINTAGE_FIRST_MONTH = 5
CARRY_MONTHS = 12

TEXP_RAW = "TExp_item1a"          # Item 1A measure, as in sections 7.2 and 7.3
TEXP_COLUMN = "texp_z"            # standardised within (vintage x subsample); see load_texp
FS_COLUMN = rcr.FS_COLUMN
CONTROLS = rcr.CONTROLS           # ln_me_lag, bm, lev, mom12
INDUSTRY_COLUMN = rcr.INDUSTRY_COLUMN

# build_texp_panel.report's tolerances, applied to the recomputed z.
Z_MEAN_TOL = 1e-9
Z_SD_TOL = 1e-6

RAW_VALUE_FIELDS = ["PERMNO", "MthCalDt", "MthRet", "MthCap", "SICCD"]
RAW_REQUIRED = RAW_VALUE_FIELDS + list(cd.COMMON_EQUITY_FILTER)

OUTPUT_COLUMNS = [
    "permno", "ym", "date_t", "pit_date", "ret",
    "vintage_year", "reference_date", "accession", "filing_date", TEXP_RAW, TEXP_COLUMN,
    FS_COLUMN, "me_lag", "ln_me_lag", "bm", "lev", "mom12", "mom12_n_months",
    "siccd_lag", "ff12_num", INDUSTRY_COLUMN, "sic_invalid",
    "gvkey", "datadate", "fyear", "available_date", "pit_source", "fundamentals_stale",
    "in_screen_lag", "epu_lag", "exclusion_reason",
]

RULE = "=" * 78


def _section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def _month_end(periods: pd.Series) -> pd.Series:
    """Last calendar day of each period, as a normalised timestamp."""
    return periods.dt.to_timestamp(how="end").dt.normalize()


def _assert_grain(frame: pd.DataFrame, before: int, step: str) -> None:
    """Every merge in this module must preserve one row per (permno, month)."""
    if len(frame) != before or frame.duplicated(["permno", "ym"]).any():
        raise ValueError(f"{step} changed the (permno, month) grain: {before:,} -> {len(frame):,}")


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_subsample(path: Path = SUBSAMPLE_CSV) -> set[int]:
    """The section 7.0 random draw, read from the file sample_full_panel_firms.py fixed once."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run sample_full_panel_firms.py first.")
    frame = pd.read_csv(path, comment="#", usecols=["permno"])
    if frame["permno"].duplicated().any():
        raise ValueError(f"{path.name} holds duplicate PERMNOs; the draw is without replacement")
    return set(frame["permno"].astype(int))


def load_raw_monthly(permnos: set[int],
                     path: Path = MONTHLY_RAW_FILE) -> tuple[pd.DataFrame, dict]:
    """Unscreened CRSP monthly history for the subsample: return, market cap and SIC.

    Read from the raw file rather than clean_returns.csv for the reason ccd.load_monthly_returns
    gives: the cleaned panel drops firm-months failing the price and micro-cap screens, so taking
    returns or market equity from it would make both go missing non-randomly with size. The
    screen enters separately, and only at t-1.

    clean_data's common-equity filter is applied so the universe definition matches
    clean_returns.csv's, and because without it ten firm-months carry a second CRSP row -
    PrimaryExch 'X', SICCD 0, null security metadata, the stub written the month a listing moves
    or ends. Those rows repeat the return and market cap exactly, so they cannot change a value,
    but they would fan out every downstream merge.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; it is a required raw input.")
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in RAW_REQUIRED if c not in header]
    if missing:
        raise ValueError(f"{path.name} is missing required field(s) {missing}.")

    kept = [chunk[chunk["PERMNO"].isin(permnos)] for chunk in
            pd.read_csv(path, usecols=RAW_REQUIRED, chunksize=ccd.CHUNK_ROWS)]
    raw = pd.concat(kept, ignore_index=True)
    rows_read = len(raw)

    mask = pd.Series(True, index=raw.index)
    for col, allowed in cd.COMMON_EQUITY_FILTER.items():
        mask &= raw[col].isin(allowed)
    raw = raw.loc[mask, RAW_VALUE_FIELDS].drop_duplicates()

    raw = raw.rename(columns={"PERMNO": "permno", "MthRet": "ret",
                              "MthCap": "me", "SICCD": "siccd"})
    raw["permno"] = raw["permno"].astype("int64")
    raw["ym"] = pd.to_datetime(raw["MthCalDt"]).dt.to_period("M")
    for col in ("ret", "me", "siccd"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw[["permno", "ym", "ret", "me", "siccd"]]

    # A surviving duplicate that disagreed on a value would make the choice of row material, so
    # it fails rather than resolving by read order - ccd.dedupe_daily's rule.
    disagreeing = raw.drop(columns="siccd").drop_duplicates()
    if disagreeing.duplicated(["permno", "ym"]).any():
        clash = disagreeing.loc[disagreeing.duplicated(["permno", "ym"], keep=False)]
        raise ValueError(f"{path.name} holds {len(clash):,} common-equity PERMNO-month rows that "
                         f"disagree on return or market cap:\n{clash.head().to_string(index=False)}")
    raw = raw.drop_duplicates(["permno", "ym"], keep="first")

    return raw, {"rows_read": rows_read, "rows": len(raw),
                 "permnos": int(raw["permno"].nunique()),
                 "span": (str(raw["ym"].min()), str(raw["ym"].max())),
                 "firms_absent_from_crsp": len(permnos) - int(raw["permno"].nunique())}


def load_texp(permnos: set[int], path: Path = TEXP_PANEL_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The nine full_panel vintages, sliced to the subsample and re-standardised within vintage.

    texp_panel.csv pools all seventeen edgar_pull reference dates, and two of the nine this test
    uses - 2018-04-02 and 2025-04-02 - were also pulled at scope full for the cross-cycle event
    study. Their TExp_item1a_z is therefore standardised over 2,289 and 2,868 firms against the
    560 and 958 subsample firms present (slice mean 0.045/0.073, sd 1.078/1.041, |z| moving by up
    to 0.70), so the panel's own z column is not comparable across vintages here. It is recomputed
    on exactly the population that enters this test.

    Standardising within vintage at all is not cosmetic. Raw TExp's cross-sectional sd rises 2.7x
    and its zero share falls from 72% to 16% between the 2017 and 2025 vintages, so one raw unit
    is a 2.7x larger move in exposure in 2017 than in 2025 and a time-series mean of raw slopes
    would silently mix the two scales. Section 4 defines TExp as standardised for the same reason.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run build_texp_panel.py first.")
    panel = pd.read_csv(path, usecols=["permno", "reference_date", "accession", "filing_date",
                                       TEXP_RAW],
                        parse_dates=["filing_date"], dtype={"accession": str})
    vintages = panel[panel["reference_date"].str.endswith(VINTAGE_SUFFIX)].copy()
    vintages = vintages[vintages["permno"].isin(permnos)].copy()

    dates = sorted(vintages["reference_date"].unique())
    if len(dates) != N_VINTAGES:
        raise ValueError(f"expected {N_VINTAGES} full_panel vintages ending {VINTAGE_SUFFIX}, "
                         f"found {len(dates)}: {', '.join(dates)}")
    if vintages.duplicated(["permno", "reference_date"]).any():
        raise ValueError("a vintage cross-section is not unique on permno; the merge would fan out")
    if not set(vintages["permno"]).issubset(permnos):
        raise ValueError("the vintage slice carries PERMNOs outside the section 7.0 draw")

    grouped = vintages.groupby("reference_date")[TEXP_RAW]
    sd = grouped.transform("std")
    vintages[TEXP_COLUMN] = (vintages[TEXP_RAW] - grouped.transform("mean")) / sd.where(sd > 0)
    moments = vintages.groupby("reference_date").agg(
        firms=(TEXP_RAW, "size"), raw_mean=(TEXP_RAW, "mean"), raw_sd=(TEXP_RAW, "std"),
        zero_share=(TEXP_RAW, lambda s: (s == 0).mean()),
        z_mean=(TEXP_COLUMN, "mean"), z_sd=(TEXP_COLUMN, "std"))
    if (moments["z_mean"].abs().max() > Z_MEAN_TOL
            or (moments["z_sd"] - 1).abs().max() > Z_SD_TOL):
        raise ValueError(f"{TEXP_COLUMN} is not standardised within vintage: worst mean "
                         f"{moments['z_mean'].abs().max():.2e}, worst sd {moments['z_sd'].max():.8f}")

    vintages["vintage_year"] = vintages["reference_date"].str[:4].astype(int)
    return vintages, moments


# --------------------------------------------------------------------------- #
# Panel assembly                                                              #
# --------------------------------------------------------------------------- #
def build_candidate_grid(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Candidate firm-months - a CRSP row in the window - with t-1 values attached by explicit lag.

    The lag is a merge on the previous calendar month, not a groupby shift: a shift takes the
    firm's previous *available* row, which across a listing gap would import market equity from
    several months earlier and call it t-1.
    """
    lag = raw[["permno", "ym", "me", "siccd"]].rename(
        columns={"me": "me_lag", "siccd": "siccd_lag"})
    lag["ym"] = lag["ym"] + 1                       # this month's values are next month's lag

    grid = raw.loc[raw["ym"].between(SAMPLE_START, SAMPLE_END),
                   ["permno", "ym", "ret"]].copy()
    before = len(grid)
    grid = grid.merge(lag, on=["permno", "ym"], how="left")
    _assert_grain(grid, before, "t-1 lag merge")

    # CRSP carries a zero or missing cap where the firm did not trade; a non-positive market
    # equity is not a small firm, it is an absent observation (ccd.add_market_equity).
    grid.loc[grid["me_lag"] <= 0, "me_lag"] = np.nan
    grid["ln_me_lag"] = np.where(grid["me_lag"] > 0, np.log(grid["me_lag"]), np.nan)
    grid["date_t"] = _month_end(grid["ym"])
    grid["pit_date"] = _month_end(grid["ym"] - 1)
    return grid.sort_values(["permno", "ym"]).reset_index(drop=True), {
        "candidate_rows": len(grid), "permnos": int(grid["permno"].nunique()),
        "months": int(grid["ym"].nunique()), "ret_missing": int(grid["ret"].isna().sum()),
        "me_lag_missing": int(grid["me_lag"].isna().sum())}


def add_screen_flag(grid: pd.DataFrame, path: Path = SCREENED_CSV) -> tuple[pd.DataFrame, dict]:
    """Was the firm in the section 6 screened universe in month t-1.

    Presence in clean_returns.csv *is* the screen, as ccd.add_screened_universe_flag argues, and
    it is read one month back for the reason estimate_car.pit_screen gives: same-month membership
    is decided partly by month-end market equity, which is not known when the month opens.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_data.py first.")
    screened = pd.read_csv(path, usecols=["permno", "date"], parse_dates=["date"])
    screened["ym"] = screened["date"].dt.to_period("M") + 1     # screened at t-1 -> flag month t
    screened = screened[["permno", "ym"]].drop_duplicates()
    screened["in_screen_lag"] = True

    before = len(grid)
    out = grid.merge(screened, on=["permno", "ym"], how="left")
    _assert_grain(out, before, "screened-universe merge")
    out["in_screen_lag"] = out["in_screen_lag"].eq(True)        # unmatched -> False
    return out, {"in_screen": int(out["in_screen_lag"].sum()),
                 "out_of_screen": int((~out["in_screen_lag"]).sum())}


def attach_texp(grid: pd.DataFrame, vintages: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map each month to the vintage in effect and attach that filing's exposure.

    Months May y .. April y+1 take the y-04-02 vintage, so a month is never scored by a filing
    that was not yet public when it opened. The PIT property is asserted rather than assumed: the
    document's own filing_date must not fall after pit_date.
    """
    year, month = grid["ym"].dt.year, grid["ym"].dt.month
    grid["vintage_year"] = np.where(month >= VINTAGE_FIRST_MONTH, year, year - 1)

    carry = ["permno", "vintage_year", "reference_date", "accession", "filing_date",
             TEXP_RAW, TEXP_COLUMN]
    before = len(grid)
    out = grid.merge(vintages[carry], on=["permno", "vintage_year"], how="left")
    _assert_grain(out, before, "TExp vintage merge")

    scored = out["filing_date"].notna()
    late = scored & (out["filing_date"] > out["pit_date"])
    if late.any():
        worst = out.loc[late, ["permno", "ym", "filing_date", "pit_date"]].head()
        raise ValueError(f"{int(late.sum()):,} firm-months carry a filing dated after their "
                         f"point-in-time date - look-ahead:\n{worst.to_string(index=False)}")

    # The carry bound is enforced, not merely documented: a firm-vintage spanning more than
    # CARRY_MONTHS months would mean the effective-month mapping had let a score run on past the
    # date its successor should have taken over.
    spanned = out.loc[scored].groupby(["permno", "vintage_year"])["ym"].nunique()
    if len(spanned) and spanned.max() > CARRY_MONTHS:
        raise ValueError(f"a firm-vintage covers {spanned.max()} months against a "
                         f"{CARRY_MONTHS}-month carry bound")
    return out, {"texp_matched": int(out[TEXP_COLUMN].notna().sum()),
                 "texp_missing": int(out[TEXP_COLUMN].isna().sum()),
                 "months_per_vintage": out.loc[scored].groupby("vintage_year")["ym"].nunique()}


def attach_momentum(grid: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prior 12-month return, evaluated at month t rather than at pit_date.

    ccd.add_momentum's window for month j is [j-12, j-1], which is already the prior-return
    characteristic known at the end of t-1. Called on the pit-dated frame it would close the
    window at t-2 and quietly drop the most informative month. The full unscreened history is
    passed, not the windowed grid, so the lookback has data before the sample opens.
    """
    before = len(grid)
    mom, stats = ccd.add_momentum(grid[["permno", "date_t"]].rename(columns={"date_t": "date"}),
                                  raw)
    out = grid.merge(mom.rename(columns={"date": "date_t"}), on=["permno", "date_t"], how="left")
    _assert_grain(out, before, "momentum merge")
    return out, stats


def attach_pit_controls(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """gvkey, Compustat fundamentals and the ratios, every one of them resolved at pit_date.

    ccd's functions are called unchanged on a frame whose ``date`` column is pit_date, so the
    backward as-of gate admits only fundamentals already public before month t opens, and bm and
    lev inherit Script 1's validated available_date rule rather than a second parallel one.
    """
    work = grid[["permno", "pit_date", "me_lag"]].rename(columns={"pit_date": "date"})
    bridge = pd.read_csv(BRIDGE_CSV, dtype=str)
    work, link = ccd.link_gvkey(work, bridge)

    fund, dup_fy = ccd.load_fundamentals(ccd.FUNDA_FILE)
    filings, filing_stats = ccd.load_edgar_filing_dates(ccd.EDGAR_LOG)
    fund, avail = ccd.add_availability_date(fund, filings)
    fund, be = ccd.build_book_equity(fund)
    work, merged = ccd.merge_fundamentals(work, fund)
    work, ratios = ccd.build_ratios(work)

    carry = ["permno", "date", "gvkey", "datadate", "fyear", "available_date", "pit_source",
             "fundamentals_stale", "bm", "lev", "be_nonpositive", "ceq_nonpositive"]
    before = len(grid)
    out = grid.merge(work[carry].rename(columns={"date": "pit_date"}),
                     on=["permno", "pit_date"], how="left")
    _assert_grain(out, before, "point-in-time controls merge")
    return out, {"link": link, "filings": filing_stats, "avail": avail, "be": be,
                 "fundamentals": merged, "ratios": ratios, "dup_fiscal_years": len(dup_fy)}


def assign_ff12_lagged(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FF12 industry from the firm's own SIC at t-1, remapped every month.

    ccd.assign_ff12 fixes one label per firm from a snapshot taken before the active cycle's first
    event, which for cycle 2025 would apply a 2025 classification to a 2018 cross-section. That
    convention exists to hold the industry effects identical across the two legs of one event
    study; section 7.4 estimates ninety-six independent cross-sections and has no such pair to
    protect, so the point-in-time label is used instead. SICCD 0 is invalid rather than 'Other'
    and is flagged, kept and reported, as in ccd.
    """
    ff = ccd.parse_ff12(ccd.SICCODES_FILE)
    lo, hi, num, lab = (ff["lo"].to_numpy(), ff["hi"].to_numpy(),
                        ff["ff12_num"].to_numpy(), ff["ff12"].to_numpy())

    codes = np.sort(grid.loc[grid["siccd_lag"].notna(), "siccd_lag"].unique()).astype(int)
    mapping, unmapped = {}, []
    for code in codes:
        hit = np.flatnonzero((lo <= code) & (hi >= code))
        if len(hit):
            mapping[int(code)] = (int(num[hit[0]]), str(lab[hit[0]]))
        else:                                     # industry 12 is the residual, by definition
            mapping[int(code)] = (12, "Other")
            unmapped.append(int(code))

    out = grid.copy()
    out["ff12_num"] = out["siccd_lag"].map(lambda c: mapping[int(c)][0] if pd.notna(c) else None)
    out[INDUSTRY_COLUMN] = out["siccd_lag"].map(
        lambda c: mapping[int(c)][1] if pd.notna(c) else None)
    out["sic_invalid"] = out["siccd_lag"].notna() & (out["siccd_lag"] <= 0)
    out["ff12_num"] = out["ff12_num"].astype("Int8")
    out["siccd_lag"] = out["siccd_lag"].astype("Int32")

    switched = out.loc[out[INDUSTRY_COLUMN].notna()].groupby("permno")[INDUSTRY_COLUMN].nunique()
    return out, {"distinct_sic": len(codes), "unmapped_sic": unmapped,
                 "rows_without_label": int(out[INDUSTRY_COLUMN].isna().sum()),
                 "sic_invalid_rows": int(out["sic_invalid"].sum()),
                 "firms_switching_ff12": int((switched > 1).sum()),
                 "distribution": out[INDUSTRY_COLUMN].value_counts()}


def attach_fs_and_epu(grid: pd.DataFrame) -> tuple[pd.DataFrame, set[str], dict]:
    """Foreign-sales share on the panel's own PIT (gvkey, datadate), and lagged EPU.

    FS joins exactly as run_car_regression.build_sample joins it, so the foreign-sales share, the
    book-to-market ratio and leverage all describe the same fiscal year under one availability
    gate. EPU is attached at t-1 and carried only - section 7.5 consumes it, this script does not.
    """
    fs = rcr.load_fs()
    before = len(grid)
    out = grid.merge(fs, on=["gvkey", "datadate"], how="left")
    _assert_grain(out, before, "FS merge")

    epu = pd.read_csv(EPU_CSV, parse_dates=["date"])
    epu["ym"] = epu["date"].dt.to_period("M") + 1               # EPU of t-1 -> month t
    out = out.merge(epu[["ym", "epu_news"]].rename(columns={"epu_news": "epu_lag"}),
                    on="ym", how="left")
    _assert_grain(out, before, "EPU merge")
    if out["epu_lag"].isna().any():
        raise ValueError(f"{EPU_CSV.name} does not cover every sample month lagged one period")
    return out, set(fs["gvkey"]), {"fs_matched": int(out[FS_COLUMN].notna().sum()),
                                   "fs_missing": int(out[FS_COLUMN].isna().sum())}


# --------------------------------------------------------------------------- #
# Exclusion ladder                                                            #
# --------------------------------------------------------------------------- #
def texp_reason_map(vintage_years: list[int]) -> dict[tuple[int, int], str]:
    """Why a firm carries no exposure at one vintage, from the module that made each decision.

    run_car_regression.load_texp_reasons is called once per vintage and keyed on
    (vintage_year, permno), because a firm can have a usable filing at one vintage and none at
    another and a pooled read would attribute the wrong reason.
    """
    reasons = {}
    for year in vintage_years:
        for permno, why in rcr.load_texp_reasons(f"{year}{VINTAGE_SUFFIX}").items():
            reasons[(year, permno)] = why
    return reasons


def exclusion_reasons(panel: pd.DataFrame, texp_reasons: dict[tuple[int, int], str],
                      fs_gvkeys: set[str]) -> pd.Series:
    """One reason per excluded firm-month, in fixed precedence so the counts cannot double.

    An empty string marks a row that survives into the regressions, so the funnel reconciles by
    construction. Same ladder and same order as run_car_regression.exclusion_reasons, with the
    event-window rung replaced by a missing monthly return.
    """
    reason = pd.Series("", index=panel.index, dtype=object)

    def mark(mask: pd.Series, value: str) -> None:
        reason.loc[mask & reason.eq("")] = value

    mark(~panel["in_screen_lag"], "outside_pit_screen")
    mark(panel["ret"].isna(), "ret_missing")

    pending = panel[TEXP_COLUMN].isna() & reason.eq("")
    for idx in panel.index[pending]:
        key = (int(panel.at[idx, "vintage_year"]), int(panel.at[idx, "permno"]))
        reason.at[idx] = "no_texp:" + texp_reasons.get(key, "not_in_pull_universe")

    pending = panel[FS_COLUMN].isna() & reason.eq("")
    for idx in panel.index[pending]:
        reason.at[idx] = "no_fs:" + rcr._fs_reason(panel.at[idx, "gvkey"],
                                                   panel.at[idx, "datadate"], fs_gvkeys)

    pending = reason.eq("")
    absent = panel.loc[pending, CONTROLS + [INDUSTRY_COLUMN]].isna()
    for idx in absent.index[absent.any(axis=1)]:
        reason.at[idx] = "no_control:" + "+".join(absent.columns[absent.loc[idx]])

    kept = int(reason.eq("").sum())
    if kept + int(reason.ne("").sum()) != len(panel):
        raise ValueError("exclusion reasons do not reconcile against the candidate row count")
    return reason


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def report(panel: pd.DataFrame, moments: pd.DataFrame, stats: dict) -> None:
    """Print the build's own summary; Script 7b re-derives all of it from the written panel."""
    kept = panel[panel["exclusion_reason"].eq("")]

    _section("FAMA-MACBETH MONTHLY PANEL (section 7.4)")
    print(f"Window          {SAMPLE_START} .. {SAMPLE_END}  "
          f"({(SAMPLE_END - SAMPLE_START).n + 1} months)")
    print(f"Subsample       {stats['subsample']:,} PERMNOs (section 7.0 draw), "
          f"{stats['raw']['permnos']:,} with CRSP monthly history")
    print(f"Candidates      {stats['grid']['candidate_rows']:,} firm-months over "
          f"{stats['grid']['months']} months")
    print(f"Estimable       {len(kept):,} firm-months over {kept['ym'].nunique()} months, "
          f"{kept['permno'].nunique():,} firms")

    print("\nExposure vintages - nine April-2 reference dates, subsample slice, z recomputed")
    print(f"{'vintage':<13}{'firms':>7}{'months':>8}{'raw mean':>11}{'raw sd':>10}"
          f"{'zero':>8}{'z mean':>10}{'z sd':>7}")
    months = stats["texp"]["months_per_vintage"]
    for date, row in moments.iterrows():
        year = int(date[:4])
        print(f"{date:<13}{int(row['firms']):>7,}{months.get(year, 0):>8}"
              f"{row['raw_mean']:>11.5f}{row['raw_sd']:>10.5f}{row['zero_share']:>7.1%}"
              f"{row['z_mean']:>10.1e}{row['z_sd']:>7.3f}")
    print("  Verified: every vintage's z has mean 0 and sd 1 on the population that enters.")

    print("\nMerge diagnostics - each step asserts the (permno, month) grain; these are what")
    print("the merges matched, so a silent coverage loss shows here rather than in the funnel")
    print(f"  raw monthly rows read      {stats['raw']['rows_read']:,} -> "
          f"{stats['raw']['rows']:,} after the common-equity filter")
    print(f"  screened at t-1            {stats['screen']['in_screen']:,} of "
          f"{stats['grid']['candidate_rows']:,} candidate firm-months")
    print(f"  TExp attached              {stats['texp']['texp_matched']:,}")
    print(f"  mom12 computed             {stats['mom']['daily_rows_with_mom12']:,}")
    print(f"  gvkey resolved at t-1      {stats['controls']['link']['permnos_linked']:,} firms, "
          f"{stats['controls']['link']['rows_unlinked']:,} rows unlinked")
    print(f"  fundamentals as-of t-1     "
          f"{stats['controls']['fundamentals']['rows_with_fundamentals']:,} rows, "
          f"{stats['controls']['fundamentals']['rows_stale']:,} stale")
    print(f"  FF12 assigned              {stats['ff12']['distinct_sic']} distinct SICs, "
          f"{stats['ff12']['firms_switching_ff12']} firms switch industry in sample")
    print(f"  FS attached                {stats['fs']['fs_matched']:,}")

    print("\nExclusion ladder")
    counts = panel.loc[panel["exclusion_reason"].ne(""), "exclusion_reason"].value_counts()
    for why, n in counts.items():
        print(f"  - {why:<52}{n:>9,}")
    print(f"  = reconciles: {len(kept):,} kept + {int(counts.sum()):,} dropped "
          f"= {len(panel):,} candidates")

    print("\nMonthly cross-sections entering the regressions")
    sizes = kept.groupby("ym")["permno"].size()
    print(f"{'year':<8}{'months':>8}{'min':>8}{'mean':>8}{'max':>8}")
    for year, group in sizes.groupby(sizes.index.year):
        print(f"{year:<8}{len(group):>8}{group.min():>8,}{group.mean():>8,.0f}{group.max():>8,}")
    print(f"{'all':<8}{len(sizes):>8}{sizes.min():>8,}{sizes.mean():>8,.0f}{sizes.max():>8,}")


def _up_to_date(outputs, label: str, force: bool) -> bool:
    """True when every output already exists and the caller has not passed --force."""
    missing = [p for p in outputs if not p.exists()]
    if force or missing:
        if missing and not force:
            print(f"{label}: rebuilding - missing {', '.join(p.name for p in missing)}")
        return False
    print(f"{label}: already built, not regenerated. Outputs:")
    for p in outputs:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:,.1f} MB)")
    print("  Pass --force to rebuild.")
    return True


def main() -> pd.DataFrame | None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the outputs already exist")
    ap.add_argument("--status", action="store_true",
                    help="report the funnel and monthly counts, then exit without writing")
    args = ap.parse_args()
    if not args.status and _up_to_date([PANEL_OUT, REPORT_OUT], 'build_fm_panel', args.force):
        return None

    permnos = load_subsample()
    raw, raw_stats = load_raw_monthly(permnos)
    vintages, moments = load_texp(permnos)

    grid, grid_stats = build_candidate_grid(raw)
    grid, screen_stats = add_screen_flag(grid)
    grid, texp_stats = attach_texp(grid, vintages)
    grid, mom_stats = attach_momentum(grid, raw)
    grid, control_stats = attach_pit_controls(grid)
    grid, ff12_stats = assign_ff12_lagged(grid)
    grid, fs_gvkeys, fs_stats = attach_fs_and_epu(grid)

    years = sorted(vintages["vintage_year"].unique())
    grid["exclusion_reason"] = exclusion_reasons(grid, texp_reason_map(years), fs_gvkeys)
    panel = grid.reindex(columns=OUTPUT_COLUMNS).sort_values(["ym", "permno"])

    with run_report.capture(REPORT_OUT, title="STEP 7A - FAMA-MACBETH PANEL VALIDATION"):
        report(panel, moments, {"subsample": len(permnos), "raw": raw_stats, "grid": grid_stats,
                                "screen": screen_stats, "texp": texp_stats, "mom": mom_stats,
                                "controls": control_stats, "ff12": ff12_stats, "fs": fs_stats})
    if args.status:
        return None

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_OUT, index=False)
    print(f"\nWrote {PANEL_OUT.relative_to(BASE)} "
          f"({len(panel):,} rows x {len(panel.columns)} cols).")
    return panel


if __name__ == "__main__":
    main()
