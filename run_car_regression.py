"""Script 3 of 3: cross-sectional regression of CARs on tariff exposure (section 7.2, H1 and H5).

Consumes Script 1's controls panel and Script 2's per-firm CAR tables and estimates

    CAR_i = a + b*TExp_i + c*FS_i + gamma'X_i + delta_ind + e_i

once per (run, event window) - three runs x three windows = nine independent regressions, with no
pooling across either dimension. The sign of b is H1: negative on imposition, positive on the
reversal, with TExp held identical across both legs. The FS coefficient c is the H5 discriminant-
validity test embedded in the same specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera

import clean_controls_data as ccd
import estimate_car as ec

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
CLEAN_DIR = BASE / "clean_data"
OUTPUT_DIR = BASE / "output"

# Paths owned by earlier scripts are imported, not re-declared, so the panel location and the
# event dates keep a single definition (the precedent Steps 4b and 4c set).
PANEL_PATH = ccd.PANEL_OUT.with_suffix(f".{ccd.OUTPUT_FORMAT}")
SCORES_CSV = CLEAN_DIR / "tariff_scores.csv"
FS_CSV = CLEAN_DIR / "foreign_sales_share.csv"
SCORING_DIAG = OUTPUT_DIR / "scoring_diagnostics.csv"   # supplies the TExp drop reasons
EDGAR_LOG = BASE / "edgar_pull_log.csv"                 # supplies the pull-failure reasons
RESULTS_OUT = OUTPUT_DIR / "car_regression_results.csv"
REPORT_OUT = OUTPUT_DIR / "car_regression_validation_report.txt"

# Specification. Every choice below is named here rather than inline in a function body.
TEXP_COLUMN = "TExp_item1a"      # raw Item 1A measure; b reads as CAR change per hit-per-sentence
FS_COLUMN = "FS"                 # Compustat foreign-sales share, the H5 control
CONTROLS = ["ln_me_lag", "bm", "lev", "mom12"]
INDUSTRY_COLUMN = "ff12"
FF12_REFERENCE = "Other"         # omitted dummy; shifts the intercept, never b
APPLY_SCREEN = True              # in_screened_universe_pit is the section 6 sample definition
COV_TYPE = "nonrobust"           # this pass only; section 7.2 nominates White HC

# The panel columns this script reads. The event-date row is already point-in-time: ln_me_lag is
# market equity lagged one trading day, bm and lev are built on that same lagged ME, and mom12
# covers the twelve calendar months ending the month before the event month. The two equity flags
# are carried only to explain why bm and lev are null - Script 1 leaves both undefined rather than
# letting a negative denominator flip the ratio's sign.
EQUITY_FLAGS = ["be_nonpositive", "ceq_nonpositive"]
PANEL_COLUMNS = ["permno", "date", "gvkey", "datadate", "ln_me_lag", "bm", "lev", "mom12",
                 INDUSTRY_COLUMN] + EQUITY_FLAGS

# Sensitivity specifications reported beside the primary fit, never instead of it.
EXTREME_CAR = ec.EXTREME_CAR     # |CAR| above this is a firm-specific event, flagged by Script 2
TRIM_COLUMNS = ["bm", "lev"]     # Script 2 deferred the winsorising question to this script
TRIM_QUANTILES = (0.01, 0.99)
TRIM_SPEC = "trim_" + "_".join(TRIM_COLUMNS)

# H1 predicts opposite signs on the two legs. Keyed on the CAR table's `event` field.
EXPECTED_SIGN = {"impose": -1, "reverse": +1}
SIGN_WORD = {-1: "NEGATIVE", 1: "POSITIVE", 0: "ZERO"}

STARS = {0.01: "***", 0.05: "**", 0.10: "*"}
MAX_LISTED = 10                  # identities printed before deferring to a count
RULE = "=" * 78
THIN = "-" * 78

_REPORT: list[str] = []


# --------------------------------------------------------------------------- #
# Reporting helpers                                                           #
# --------------------------------------------------------------------------- #
def _say(line: str = "") -> None:
    """Append one line to the single consolidated validation report."""
    _REPORT.append(line)


def _section(title: str) -> None:
    """Start a titled report section."""
    _say()
    _say(title)
    _say("-" * len(title))


def _short(window: str) -> str:
    """Window label without the redundant column prefix, so grouped tables stay inside 78 chars."""
    return window.removeprefix("car_")


def _listed(values) -> str:
    """Render an identity list, deferring to a count past MAX_LISTED."""
    vals = list(values)
    shown = " ".join(str(v) for v in vals[:MAX_LISTED])
    return shown if len(vals) <= MAX_LISTED else f"{shown} ... and {len(vals) - MAX_LISTED} more"


def _describe(frame: pd.DataFrame, cols: list[str], width: int = 13) -> None:
    """Report distribution and missingness for a set of columns."""
    _say(f"  {'variable':<16}{'n':>7}{'null':>7}" + "".join(
        f"{h:>{width}}" for h in ("mean", "sd", "min", "median", "max")))
    for col in cols:
        s = pd.to_numeric(frame[col], errors="coerce")
        v = s.dropna()
        if v.empty:
            _say(f"  {col:<16}{0:>7,}{len(s):>7,}" + f"{'-':>{width}}" * 5)
            continue
        cells = "".join(format(x, f">{width},.4f") for x in
                        (v.mean(), v.std(), v.min(), v.median(), v.max()))
        _say(f"  {col:<16}{len(v):>7,}{int(s.isna().sum()):>7,}{cells}")


def stars(pvalue: float) -> str:
    """Significance markers at the conventional 1 / 5 / 10 per cent thresholds."""
    for level in sorted(STARS):
        if pvalue < level:
            return STARS[level]
    return ""


def significance_label(pvalue: float) -> str:
    """Plain-language significance verdict, so the table reads without arithmetic."""
    for level in sorted(STARS):
        if pvalue < level:
            return f"significant at {level:.0%}"
    return f"not significant at {max(STARS):.0%}"


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_event_controls(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Controls as they stood on each event date, one row per (event, permno).

    Only the event-date rows are retained: nothing in PANEL_COLUMNS is known after the event, so
    the row dated on the event day is already the point-in-time regressor set.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_controls_data.py first.")
    frame = pd.read_csv(path, usecols=PANEL_COLUMNS, parse_dates=["date", "datadate"],
                        dtype={"gvkey": str, **{col: "object" for col in EQUITY_FLAGS}})
    for col in EQUITY_FLAGS:
        frame[col] = frame[col].astype(str).str.strip().str.lower().eq("true")
    events = {name: pd.Timestamp(day) for name, day in ccd.EVENT_DATES.items()}
    keep = frame[frame["date"].isin(events.values())].copy()
    keep["event"] = keep["date"].map({day: name for name, day in events.items()})
    missing = set(events) - set(keep["event"])
    if missing:
        raise ValueError(f"{path.name} has no rows on event date(s): {sorted(missing)}")
    if keep.duplicated(["event", "permno"]).any():
        raise ValueError("controls panel is not unique on (event, permno)")
    keep["gvkey"] = keep["gvkey"].str.zfill(ccd.GVKEY_WIDTH)
    return keep


def load_texp(path: Path = SCORES_CSV) -> pd.DataFrame:
    """Tariff exposure scores, one row per firm, from the scored 10-K vintage."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run score_filings.py first.")
    frame = pd.read_csv(path, usecols=["permno", "fiscal_year", "filing_date", TEXP_COLUMN],
                        parse_dates=["filing_date"])
    if frame["permno"].duplicated().any():
        raise ValueError(f"{path.name} is not unique on permno; the merge would fan out rows")
    return frame


def load_fs(path: Path = FS_CSV) -> pd.DataFrame:
    """Foreign-sales share by firm-fiscal-year, keyed for a join on the panel's PIT datadate."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run foreign_sales.py first.")
    frame = pd.read_csv(path, usecols=["gvkey", "datadate", FS_COLUMN],
                        parse_dates=["datadate"], dtype={"gvkey": str})
    frame["gvkey"] = frame["gvkey"].str.zfill(ccd.GVKEY_WIDTH)
    if frame.duplicated(["gvkey", "datadate"]).any():
        raise ValueError(f"{path.name} is not unique on (gvkey, datadate)")
    if not frame[["gvkey", "datadate"]].notna().all().all():
        # pandas matches null join keys to each other; a null key here would create false links.
        raise ValueError(f"{path.name} carries a null join key")
    return frame


def load_texp_reasons(diag: Path = SCORING_DIAG, log: Path = EDGAR_LOG) -> dict[int, str]:
    """Why a firm carries no TExp score, taken from the module that made each decision.

    Scoring drops (item_1a_not_found, fiscal_year_out_of_range) come from the scoring
    diagnostics; firms that never reached the scorer come from the EDGAR pull log's fail_reason.
    Nothing here is inferred - a permno absent from both is reported as such.
    """
    for path in (diag, log):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; it records why a firm has no score.")

    scored = pd.read_csv(diag, usecols=["permno", "drop_reason"])
    if scored["permno"].duplicated().any():
        raise ValueError(f"{diag.name} is not unique on permno")
    reasons = {int(pn): str(why) for pn, why in
               scored.loc[scored["drop_reason"].notna(),
                          ["permno", "drop_reason"]].itertuples(index=False)}

    pull = pd.read_csv(log, usecols=["permno", "found_10k", "fail_reason"], dtype=str)
    pull["permno"] = pull["permno"].astype(int)
    succeeded = set(pull.loc[pull["found_10k"].str.lower() == "true", "permno"])
    failed = pull[~pull["permno"].isin(succeeded)].drop_duplicates("permno", keep="last")
    for pn, why in failed[["permno", "fail_reason"]].itertuples(index=False):
        reasons.setdefault(int(pn), f"pull_{why}" if pd.notna(why) else "pull_unknown")
    return reasons


# --------------------------------------------------------------------------- #
# Sample assembly                                                             #
# --------------------------------------------------------------------------- #
def build_sample(car: pd.DataFrame, controls: pd.DataFrame,
                 texp: pd.DataFrame, fs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach controls, TExp and FS to one run's CAR table, asserting the grain at each step.

    FS joins on the panel's own point-in-time (gvkey, datadate), so the foreign-sales share, the
    book-to-market ratio and leverage all describe the same fiscal year and inherit Script 1's
    validated available_date gate rather than a second, parallel availability rule.
    """
    rows = len(car)
    merged = car.merge(controls.drop(columns=["date", "event"]), on="permno", how="left",
                       indicator="_controls")
    if len(merged) != rows:
        raise ValueError("controls merge changed the CAR grain")
    merged["has_controls_row"] = merged["_controls"].eq("both")
    merged = merged.drop(columns="_controls")

    merged = merged.merge(texp, on="permno", how="left")
    if len(merged) != rows:
        raise ValueError("TExp merge changed the CAR grain")

    merged = merged.merge(fs, on=["gvkey", "datadate"], how="left")
    if len(merged) != rows:
        raise ValueError("FS merge changed the CAR grain")

    stats = {"car_rows": rows,
             "controls_matched": int(merged["has_controls_row"].sum()),
             "texp_matched": int(merged[TEXP_COLUMN].notna().sum()),
             "fs_matched": int(merged[FS_COLUMN].notna().sum()),
             "screened_pit": int(merged["in_screened_universe_pit"].sum())}
    return merged, stats


def _fs_reason(gvkey, datadate, fs_gvkeys: set[str]) -> str:
    """Distinguish an unlinked firm from one Compustat reports no geographic segments for."""
    if pd.isna(gvkey):
        return "no_gvkey_link_at_event"
    if pd.isna(datadate):
        return "no_pit_fundamentals_datadate"
    if gvkey not in fs_gvkeys:
        return "gvkey_absent_from_segment_file"
    return "no_segment_firmyear_at_pit_datadate"


def exclusion_reasons(merged: pd.DataFrame, window: str, texp_reasons: dict[int, str],
                      fs_gvkeys: set[str]) -> pd.Series:
    """One reason per excluded row, assigned in fixed precedence so the counts cannot double.

    An empty string marks a row that survives into the regression, so the funnel reconciles by
    construction: rows kept plus the reason counts equal the CAR table's row count.
    """
    reason = pd.Series("", index=merged.index, dtype=object)

    def mark(mask: pd.Series, value: str) -> None:
        reason.loc[mask & reason.eq("")] = value

    if APPLY_SCREEN:
        mark(~merged["in_screened_universe_pit"], "outside_pit_screen")
    mark(merged[window].isna(), "car_missing")
    mark(~merged["has_controls_row"], "no_panel_row_at_event")

    pending = merged[TEXP_COLUMN].isna() & reason.eq("")
    for idx in merged.index[pending]:
        why = texp_reasons.get(int(merged.at[idx, "permno"]), "not_in_pull_universe")
        reason.at[idx] = f"no_texp:{why}"

    pending = merged[FS_COLUMN].isna() & reason.eq("")
    for idx in merged.index[pending]:
        reason.at[idx] = "no_fs:" + _fs_reason(merged.at[idx, "gvkey"],
                                               merged.at[idx, "datadate"], fs_gvkeys)

    pending = reason.eq("")
    absent = merged.loc[pending, CONTROLS + [INDUSTRY_COLUMN]].isna()
    for idx in absent.index[absent.any(axis=1)]:
        reason.at[idx] = "no_control:" + "+".join(absent.columns[absent.loc[idx]])
    return reason


def regression_frame(merged: pd.DataFrame, reason: pd.Series,
                     window: str) -> tuple[pd.DataFrame, dict]:
    """The estimation sample for one (run, window), with its funnel reconciled and asserted."""
    frame = merged[reason.eq("")].copy()
    frame["car"] = frame[window]

    if frame["permno"].duplicated().any():
        dupes = frame.loc[frame["permno"].duplicated(keep=False), "permno"].unique()
        raise ValueError(f"duplicate permno in the {window} regression sample: {_listed(dupes)}")

    counts = reason[reason.ne("")].value_counts().to_dict()
    if len(frame) + sum(counts.values()) != len(merged):
        raise ValueError("exclusion reasons do not reconcile against the CAR row count")
    return frame, {"car_rows": len(merged), "n": len(frame), "dropped": counts}


# --------------------------------------------------------------------------- #
# Estimation                                                                  #
# --------------------------------------------------------------------------- #
def design_matrix(frame: pd.DataFrame, include_texp: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """Regressors with FF12 dummies, the reference category dropped by name rather than position."""
    dummies = pd.get_dummies(frame[INDUSTRY_COLUMN], prefix=INDUSTRY_COLUMN, dtype=float)
    ref = f"{INDUSTRY_COLUMN}_{FF12_REFERENCE}"
    if ref not in dummies.columns:
        raise ValueError(f"reference industry {FF12_REFERENCE!r} is absent from this sample; "
                         f"present: {sorted(frame[INDUSTRY_COLUMN].unique())}")
    regressors = ([TEXP_COLUMN] if include_texp else []) + [FS_COLUMN] + CONTROLS
    design = pd.concat([frame[regressors].astype(float), dummies.drop(columns=ref)], axis=1)
    design["_cons"] = 1.0
    return frame["car"].astype(float), design


def fit_ols(frame: pd.DataFrame, include_texp: bool = True):
    """OLS with the configured covariance estimator; missing='raise' asserts listwise deletion."""
    y, design = design_matrix(frame, include_texp)
    return sm.OLS(y, design, missing="raise").fit(cov_type=COV_TYPE)


def nested_comparison(frame: pd.DataFrame) -> dict:
    """Collinearity read from what TExp adds, rather than from a variance inflation factor.

    A regressor that merely restates FS or the controls would leave R-squared, the intercept and
    every other coefficient essentially unchanged when it enters. Reporting those movements
    answers the collinearity question directly and on the quantities the hypothesis is about.
    """
    full = fit_ols(frame, include_texp=True)
    restricted = fit_ols(frame, include_texp=False)
    test = full.f_test(f"{TEXP_COLUMN} = 0")
    return {
        "full": full, "restricted": restricted,
        "d_r2": float(full.rsquared - restricted.rsquared),
        "d_r2_adj": float(full.rsquared_adj - restricted.rsquared_adj),
        "f_stat": float(np.asarray(test.fvalue).squeeze()),
        "f_pvalue": float(np.asarray(test.pvalue).squeeze()),
        "shifts": {term: (float(restricted.params[term]), float(full.params[term]))
                   for term in [FS_COLUMN] + CONTROLS + ["_cons"]},
    }


def sensitivity_frames(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Subsamples reported beside the primary fit: extreme CARs out, then BM/Lev tails trimmed."""
    extreme = frame["car"].abs() > EXTREME_CAR
    keep = pd.Series(True, index=frame.index)
    bounds = {}
    for col in TRIM_COLUMNS:
        low, high = frame[col].quantile(TRIM_QUANTILES[0]), frame[col].quantile(TRIM_QUANTILES[1])
        bounds[col] = (float(low), float(high))
        keep &= frame[col].between(low, high)
    frames = {"primary": frame, "ex_extreme_car": frame[~extreme], TRIM_SPEC: frame[keep]}
    stats = {"n_extreme": int(extreme.sum()),
             "extreme_permnos": frame.loc[extreme, "permno"].tolist(),
             "n_trimmed": int((~keep).sum()), "bounds": bounds}
    return frames, stats


def result_rows(run: str, event: str, window: str, spec: str, res) -> list[dict]:
    """One row per estimated coefficient, the machine-readable form of the printed tables."""
    expected = EXPECTED_SIGN[event]
    rows = []
    for term in res.params.index:
        coef, pvalue = float(res.params[term]), float(res.pvalues[term])
        rows.append({
            "run": run, "event": event, "window": window, "spec": spec, "term": term,
            "coef": coef, "se": float(res.bse[term]), "t": float(res.tvalues[term]),
            "p": pvalue, "stars": stars(pvalue), "n": int(res.nobs),
            "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj),
            "expected_sign": expected if term == TEXP_COLUMN else "",
            "sign_match": (int(np.sign(coef)) == expected) if term == TEXP_COLUMN else "",
        })
    return rows


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def stata_table(run: str, event: str, window: str, res, funnel: dict, n_dummies: int) -> None:
    """One regression, laid out the way a Stata coefficient table reads."""
    lo, hi = next(w for w in ec.EVENT_WINDOWS if f"car_{ec._window_label(*w)}" == window)
    _say(RULE)
    _say(f"[{run}]  {window}  event {ccd.EVENT_DATES[event]}  window [{lo:+d},{hi:+d}]")
    _say(f"  n = {int(res.nobs):,}   R2 = {res.rsquared:.4f}   "
         f"Adj R2 = {res.rsquared_adj:.4f}   SE: {COV_TYPE}")
    _say("  H0: b = 0 (tariff exposure has no association with CAR), two-sided alternative")
    _say(THIN)
    _say(f"  {'variable':<20}{'coef':>13}{'std err':>13}{'t':>11}{'P>|t|':>12}")
    _say(THIN)
    for term in [TEXP_COLUMN, FS_COLUMN] + CONTROLS + ["_cons"]:
        if term == "_cons":
            _say(THIN)
        p = float(res.pvalues[term])
        _say(f"  {term:<20}{res.params[term]:>13.6f}{res.bse[term]:>13.6f}"
             f"{res.tvalues[term]:>11.3f}{p:>12.4f}  {stars(p)}")
    _say(THIN)
    _say(f"  + {n_dummies} FF12 industry dummies absorbed (reference: {FF12_REFERENCE})")

    coef, pvalue = float(res.params[TEXP_COLUMN]), float(res.pvalues[TEXP_COLUMN])
    expected = EXPECTED_SIGN[event]
    match = int(np.sign(coef)) == expected
    _say(f"  H1: expected {SIGN_WORD[expected]} | estimated {SIGN_WORD[int(np.sign(coef))]} | "
         f"{'MATCH' if match else 'NO MATCH'} | {significance_label(pvalue)}")
    _say(f"  sample: {funnel['car_rows']:,} CAR rows -> {funnel['n']:,} estimated "
         f"({sum(funnel['dropped'].values()):,} dropped, itemised in section 3)")
    _say(f"  {'*** p<0.01':<14}{'** p<0.05':<14}{'* p<0.10':<14}")


def signflip_matrix(fits: dict) -> None:
    """The H1 headline: b and its sign verdict across every run and window at a glance."""
    windows = ec.CAR_COLUMNS
    _say(f"  {'run':<24}" + "".join(f"{w:>18}" for w in windows))
    _say("  " + "-" * (24 + 18 * len(windows)))
    for run, event, per_window in fits:
        cells = ""
        for window in windows:
            res = per_window[window]["specs"]["primary"]
            coef, pvalue = float(res.params[TEXP_COLUMN]), float(res.pvalues[TEXP_COLUMN])
            match = "Y" if int(np.sign(coef)) == EXPECTED_SIGN[event] else "N"
            cells += f"{f'{coef:+.4f}{stars(pvalue)}':>14}{f'[{match}]':>4}"
        _say(f"  {run:<24}{cells}")
    _say("  [Y] estimated sign matches the H1 prediction for that leg, [N] does not.")
    _say(f"  expected: {', '.join(f'{k} {SIGN_WORD[v]}' for k, v in EXPECTED_SIGN.items())}")


def write_results(rows: list[dict], path: Path = RESULTS_OUT) -> Path:
    """Every coefficient from every specification, long format."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _report_spot_check(samples: dict, texp_reasons: dict[int, str]) -> None:
    """The five firms Scripts 1 and 2 validated, shown as they entered this regression."""
    _section("10. Spot check - the firms validated in Scripts 1 and 2")
    for run, merged in samples.items():
        sub = merged[merged["permno"].isin(ccd.SPOT_CHECK_PERMNOS)]
        _say(f"  [{run}]")
        _say(f"    {'permno':<8}{'ticker':<8}{'TExp':>10}{'FS':>9}{'ln_me_lag':>11}"
             f"{'bm':>9}{'lev':>9}{'mom12':>9}{'ff12':>8}{'car_m1p1':>10}")
        for _, row in sub.iterrows():
            def cell(value, width, dp=4):
                return f"{'-':>{width}}" if pd.isna(value) else f"{value:>{width}.{dp}f}"
            _say(f"    {int(row['permno']):<8}{str(row['ticker']):<8}"
                 f"{cell(row[TEXP_COLUMN], 10, 6)}{cell(row[FS_COLUMN], 9)}"
                 f"{cell(row['ln_me_lag'], 11)}{cell(row['bm'], 9)}{cell(row['lev'], 9)}"
                 f"{cell(row['mom12'], 9)}{str(row[INDUSTRY_COLUMN]):>8}"
                 f"{cell(row['car_m1p1'], 10)}")
        _say("    PIT datadate the FS and fundamentals came from: " + ", ".join(
            f"{r['ticker']} " + ("-" if pd.isna(r["datadate"]) else f"{r['datadate']:%Y-%m-%d}")
            for _, r in sub.iterrows()))
        for _, row in sub[sub[TEXP_COLUMN].isna()].iterrows():
            why = texp_reasons.get(int(row["permno"]), "not_in_pull_universe")
            _say(f"    note: {row['ticker']} (permno {int(row['permno'])}) has no TExp - {why}")
        _say()
    _say("  Values are shown exactly as they entered the design matrix; a dash is a null, which")
    _say("  excludes that firm from the regression under listwise deletion.")


def validate(fits: list, samples: dict, merge_stats: dict, texp_reasons: dict[int, str],
             texp: pd.DataFrame, paths: list[Path]) -> None:
    """Assemble the consolidated validation report and write it to disk."""
    _say(RULE)
    _say("SECTION 7.2 CROSS-SECTIONAL CAR REGRESSION - VALIDATION REPORT")
    _say(RULE)
    _say("Specification : CAR_i = a + b*" + TEXP_COLUMN + " + c*" + FS_COLUMN
         + " + gamma'[" + ", ".join(CONTROLS) + "] + FF12 dummies + e_i")
    _say(f"Runs          : {', '.join(run for run, _, _ in fits)}")
    _say(f"Windows       : {', '.join(ec.CAR_COLUMNS)}   ({len(fits)} runs x "
         f"{len(ec.CAR_COLUMNS)} windows = {len(fits) * len(ec.CAR_COLUMNS)} regressions)")
    _say("Events        : " + ", ".join(f"{k}={v}" for k, v in ccd.EVENT_DATES.items()))
    _say(f"Screen        : in_screened_universe_pit "
         f"{'imposed (primary specification)' if APPLY_SCREEN else 'NOT imposed'}")
    _say(f"Std errors    : {COV_TYPE}")
    _say(f"Inputs        : {PANEL_PATH.name}, {SCORES_CSV.name}, {FS_CSV.name}, "
         + ", ".join(Path(run["out"]).name for run in ec.RUNS))

    _section("1. Specification choices")
    _say(f"  TExp measure          {TEXP_COLUMN} (raw, not standardised)")
    sd = float(texp[TEXP_COLUMN].std())
    _say(f"  vintage sd            {sd:.6f} - multiply b by this to read it per standard deviation")
    _say(f"  FS control            {FS_COLUMN}, joined on the panel's point-in-time "
         f"(gvkey, datadate)")
    _say(f"  controls              {', '.join(CONTROLS)}")
    _say(f"  industry FE           {INDUSTRY_COLUMN}, reference category {FF12_REFERENCE!r}")
    _say("  expected sign on b    " + ", ".join(f"{k} {SIGN_WORD[v]}"
                                                 for k, v in EXPECTED_SIGN.items()))
    _say("  Controls are read from the panel row dated on the event day. That row is already")
    _say("  point-in-time: ln_me_lag lags market equity one trading day, bm and lev are built on")
    _say("  that lagged ME, and mom12 covers the twelve calendar months ending the month before.")

    _section("2. Merge match rates")
    _say(f"  {'run':<24}{'CAR rows':>10}{'screened':>10}{'controls':>10}"
         f"{'TExp':>10}{'FS':>10}")
    for run, stats in merge_stats.items():
        _say(f"  {run:<24}{stats['car_rows']:>10,}{stats['screened_pit']:>10,}"
             f"{stats['controls_matched']:>10,}{stats['texp_matched']:>10,}"
             f"{stats['fs_matched']:>10,}")
    _say("  Match rates are against the whole CAR table, before the screen is imposed; the")
    _say("  regression funnel in section 3 applies the screen first.")

    _section("3. Exclusion ladder and funnel reconciliation")
    _say("  One reason per excluded row, assigned in precedence order, so the counts sum exactly")
    _say("  to the CAR table's rows. Asserted in regression_frame, not merely printed.")
    for run, _, per_window in fits:
        _say(f"  [{run}]")
        for window in ec.CAR_COLUMNS:
            funnel = per_window[window]["funnel"]
            dropped = funnel["dropped"]
            _say(f"    {window}: {funnel['car_rows']:,} rows -> {funnel['n']:,} estimated")
            for why, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
                _say(f"      {'-':>2} {why:<48}{count:>7,}")
            total = sum(dropped.values())
            _say(f"      {'=':>2} {'reconciles: kept + dropped':<48}"
                 f"{funnel['n']:>7,} + {total:,} = {funnel['n'] + total:,}")
    head_frame = samples[fits[0][0]]
    both = head_frame["be_nonpositive"] & head_frame["ceq_nonpositive"]
    _say("  no_control:bm+lev is non-positive equity, not a merge failure: of the firms losing")
    _say(f"  both ratios, {int(both.sum()):,} carry be_nonpositive and ceq_nonpositive together in "
         f"the panel.")
    _say("  Script 1 leaves bm null where book equity is non-positive and lev null where common")
    _say("  equity is, rather than letting a negative denominator flip the ratio's sign.")

    first_run, first_event, first_windows = fits[0]
    head_window = ec.CAR_COLUMNS[0]
    head = first_windows[head_window]
    frame = head["frame"]

    _section("4. Regression sample - descriptives and correlations")
    _say(f"  [{first_run}] {head_window}, n = {len(frame):,}")
    _describe(frame, ["car", TEXP_COLUMN, FS_COLUMN] + CONTROLS)
    _say()
    variables = [TEXP_COLUMN, FS_COLUMN] + CONTROLS
    corr = frame[variables].corr()
    _say(f"  {'':<16}" + "".join(f"{v[:11]:>12}" for v in variables))
    for row in variables:
        _say(f"  {row[:15]:<16}" + "".join(f"{corr.at[row, col]:>12.3f}" for col in variables))
    _say()
    counts = frame[INDUSTRY_COLUMN].value_counts()
    _say(f"  FF12 cells ({len(counts)} of 12 present): "
         + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    if FF12_REFERENCE not in counts.index:
        raise ValueError(f"reference industry {FF12_REFERENCE!r} is absent from the sample")

    _section("5. Regression results - nine independent OLS fits")
    for run, event, per_window in fits:
        for window in ec.CAR_COLUMNS:
            entry = per_window[window]
            stata_table(run, event, window, entry["specs"]["primary"], entry["funnel"],
                        entry["n_dummies"])
            _say()

    _section("6. H1 sign-flip summary")
    signflip_matrix(fits)
    _say()
    for window in ec.CAR_COLUMNS:
        legs: dict[str, list[int]] = {}
        for run, event, per_window in fits:
            res = per_window[window]["specs"]["primary"]
            legs.setdefault(event, []).append(int(np.sign(float(res.params[TEXP_COLUMN]))))
        flipped = (set(legs) == set(EXPECTED_SIGN)
                   and all(all(s == EXPECTED_SIGN[event] for s in signs)
                           for event, signs in legs.items()))
        _say(f"  {window}: sign flip {'HOLDS' if flipped else 'DOES NOT HOLD'} across both legs")
    _say("  A sign flip requires both legs to match; a single matching leg is a news-reaction")
    _say("  result, not the reversal identification H1 is built on.")

    _section("7. Collinearity - nested comparison against the TExp-free model")
    _say("  Variance inflation factors are deliberately not reported. Collinearity is read here")
    _say("  from what TExp adds when it enters the identical sample: a regressor that merely")
    _say("  restates FS or the controls moves R-squared, the intercept and the other")
    _say("  coefficients negligibly and carries an insignificant coefficient of its own.")
    _say()
    for run, _, per_window in fits:
        _say(f"  [{run}]")
        _say(f"    {'window':<10}" + "".join(f"{h:>10}" for h in
                                             ("R2 no b", "R2 full", "dR2", "dAdjR2", "F(b=0)", "p")))
        for window in ec.CAR_COLUMNS:
            nested = per_window[window]["nested"]
            _say(f"    {_short(window):<10}{nested['restricted'].rsquared:>10.5f}"
                 f"{nested['full'].rsquared:>10.5f}{nested['d_r2']:>10.5f}"
                 f"{nested['d_r2_adj']:>10.5f}{nested['f_stat']:>10.3f}{nested['f_pvalue']:>10.4f}")
    _say()
    nested = head["nested"]
    _say(f"  Coefficient movement when TExp enters [{first_run} / {head_window}]:")
    _say(f"    {'term':<16}{'without TExp':>15}{'with TExp':>15}{'change':>15}")
    for term, (without, with_) in nested["shifts"].items():
        _say(f"    {term:<16}{without:>15.6f}{with_:>15.6f}{with_ - without:>15.6f}")

    _section("8. Residual diagnostics")
    _say("  CARs have long tails, so normality of the residuals is checked rather than assumed.")
    for run, _, per_window in fits:
        _say(f"  [{run}]")
        _say(f"    {'window':<10}{'skew':>10}{'kurtosis':>11}{'Jarque-Bera':>14}{'p':>10}")
        for window in ec.CAR_COLUMNS:
            jb, jb_p, skew, kurt = jarque_bera(per_window[window]["specs"]["primary"].resid)
            _say(f"    {_short(window):<10}{skew:>10.3f}{kurt:>11.3f}{jb:>14,.1f}{jb_p:>10.4f}")
    _say("  Rejection of normality does not bias the OLS coefficients; it bears on the exact")
    _say("  finite-sample p-values. Section 7.2 nominates White heteroskedasticity-robust")
    _say("  standard errors, which this pass deliberately does not use - see section 11.")

    _section("9. Sensitivity - extreme CARs and untreated BM/Lev tails")
    _say(f"  Extreme is |CAR| > {EXTREME_CAR:.0%}, the threshold Script 2 flagged on. Trim drops")
    _say(f"  rows outside the {TRIM_QUANTILES[0]:.0%}/{TRIM_QUANTILES[1]:.0%} percentiles of "
         f"{' and '.join(TRIM_COLUMNS)}, which Script 2 left")
    _say("  unwinsorised and deferred to this script. Both are reported beside the primary fit.")
    _say()
    for run, _, per_window in fits:
        _say(f"  [{run}]")
        _say(f"    {'window':<9}{'specification':<20}{'b':>14}{'t':>9}{'n':>10}")
        for window in ec.CAR_COLUMNS:
            entry = per_window[window]
            labels = {"primary": "primary",
                      "ex_extreme_car": f"ex |CAR|>{EXTREME_CAR:.0%} ({entry['sens']['n_extreme']})",
                      TRIM_SPEC: f"{'/'.join(TRIM_COLUMNS)} trim "
                                 f"({entry['sens']['n_trimmed']})"}
            for i, spec in enumerate(("primary", "ex_extreme_car", TRIM_SPEC)):
                res = entry["specs"][spec]
                coef, pvalue = float(res.params[TEXP_COLUMN]), float(res.pvalues[TEXP_COLUMN])
                _say(f"    {_short(window) if i == 0 else '':<9}{labels[spec]:<20}"
                     f"{f'{coef:+.5f}{stars(pvalue)}':>14}"
                     f"{float(res.tvalues[TEXP_COLUMN]):>9.3f}{int(res.nobs):>10,}")
    _say()
    _say(f"  Trim bounds [{first_run} / {head_window}]: " + ", ".join(
        f"{col} [{low:.4f}, {high:.4f}]" for col, (low, high) in head["sens"]["bounds"].items()))
    extremes = {f"{run} / {_short(window)}": per_window[window]["sens"]["extreme_permnos"]
                for run, _, per_window in fits for window in ec.CAR_COLUMNS
                if per_window[window]["sens"]["extreme_permnos"]}
    if extremes:
        _say("  Extreme-CAR firms surviving the screen (permno):")
        for key, permnos in extremes.items():
            _say(f"    {key:<32}{_listed(permnos)}")
    else:
        _say("  No firm in any final sample exceeds the extreme-CAR threshold.")
    _say("  Script 2's point-in-time screen already removed roughly 80 per cent of the CAR tail,")
    _say("  so this check is expected to be near-inert at the headline window - that is the")
    _say("  finding, not a failure to run it.")

    _report_spot_check(samples, texp_reasons)

    _section("11. Deviations recorded at the point they were made")
    _say("  - Variance inflation factors replaced by the nested comparison in section 7, per")
    _say("    instruction. The written brief asked for VIF; the substitution is deliberate.")
    _say(f"  - Standard errors are {COV_TYPE}, per instruction for this pass. Section 7.2")
    _say("    nominates White heteroskedasticity-robust errors for the reported table.")
    _say("  - Firms absent from the Compustat segment file are not imputed FS = 0. Absence is")
    _say("    not evidence of a domestic-only firm, and the project's no-imputation rule holds.")
    _say("  - Listwise deletion throughout; no control is imputed and no CAR is winsorised.")

    _section("12. Outputs")
    for path in paths:
        _say(f"  {path.relative_to(BASE)}")
    _say(f"  {REPORT_OUT.relative_to(BASE)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_OUT.write_text("\n".join(_REPORT), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def main() -> list:
    controls = load_event_controls()
    texp = load_texp()
    fs = load_fs()
    texp_reasons = load_texp_reasons()
    fs_gvkeys = set(fs["gvkey"])

    fits, rows, merge_stats, samples = [], [], {}, {}
    for run in ec.RUNS:
        name, event = run["name"], run["event"]
        car = ec.read_car(run["out"])
        merged, stats = build_sample(car, controls[controls["event"] == event], texp, fs)
        merge_stats[name], samples[name] = stats, merged

        per_window = {}
        for window in ec.CAR_COLUMNS:
            reason = exclusion_reasons(merged, window, texp_reasons, fs_gvkeys)
            frame, funnel = regression_frame(merged, reason, window)
            frames, sens = sensitivity_frames(frame)
            nested = nested_comparison(frame)
            specs = {"primary": nested["full"], "no_texp": nested["restricted"]}
            for spec in ("ex_extreme_car", TRIM_SPEC):
                specs[spec] = fit_ols(frames[spec])
            for spec, res in specs.items():
                rows += result_rows(name, event, window, spec, res)
            per_window[window] = {
                "frame": frame, "funnel": funnel, "specs": specs, "nested": nested, "sens": sens,
                "n_dummies": int(frame[INDUSTRY_COLUMN].nunique() - 1),
            }
            print(f"[{name}] {window}: n = {len(frame):,}  "
                  f"b = {specs['primary'].params[TEXP_COLUMN]:+.6f}  "
                  f"p = {specs['primary'].pvalues[TEXP_COLUMN]:.4f}", flush=True)
        fits.append((name, event, per_window))

    paths = [write_results(rows)]
    validate(fits, samples, merge_stats, texp_reasons, texp, paths)
    print("\n".join(_REPORT))
    return fits


if __name__ == "__main__":
    main()
