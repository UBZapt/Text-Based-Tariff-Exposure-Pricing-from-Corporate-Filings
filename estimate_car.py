"""
Step 4c - Script 2 of 3: daily abnormal returns and CARs for one policy cycle's events.

Estimates FF5+MOM market-model loadings per firm over an event-specific estimation window and
cumulates daily abnormal returns into CARs over [-1,+1], [-5,+5] and [-10,+10].

    --cycle 2025          3 runs: 2 imposition windows (2025-04-02) + the reversal (2025-08-29)
    --cycle cross_cycle   9 runs, one per Section 301 event date, 2018-03-01 .. 2020-01-15

Runs are estimated independently, with no betas reused across events or across the two 2025
imposition windows. No cross-sectional regression on TExp - that is Script 3.

    python estimate_car.py                    # 2025, the default
    python estimate_car.py --cycle cross_cycle
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import clean_controls_data as ccd

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
# Derived analytical panels and per-document diagnostics live here, not in output/.
# output/ is for final results, validation reports and figures; these are inputs to a later
# stage or per-row audit tables, and two of them are the largest files in the repo.
INTERMEDIATE_DIR = BASE / "intermediate"

# The panel path, the event dates and the cycle registry are owned by Script 1, which writes them;
# importing rather than re-declaring keeps one definition (the precedent score_filings.py set for
# clean_filings.py in Step 4b). These three are rebound by select_cycle().
PANEL_PATH = ccd.PANEL_OUT.with_suffix(f".{ccd.OUTPUT_FORMAT}")
FF_DAILY_FILES = ccd.CYCLES[ccd.DEFAULT_CYCLE]["ff_daily"]
REPORT_OUT = OUTPUT_DIR / "car_estimation_validation_report.txt"

PANEL_COLUMNS = ["permno", "ticker", "date", "ret", "in_screened_universe"]
FACTORS = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]

# Estimation window opens 252 trading days before the event - one trading year, the standard
# market-model length. It closes on an event-specific anchor, not a fixed offset (see RUNS).
EST_START_OFFSET = -252

# A 7-parameter OLS (intercept + 6 factors) needs enough degrees of freedom for stable loadings.
# 100 is ~50% of the shortest window (187 days) and leaves ~13 observations per parameter; it is
# also not binding on the sample - only 25-29 firms per run fall below it, and those are recent
# listings with genuinely short histories rather than firms with sparse trading.
MIN_EST_OBS = 100

EVENT_WINDOWS = [(-1, 1), (-5, 5), (-10, 10)]

# |beta_MKT| above this is flagged for inspection, never dropped: at this magnitude a loading is
# usually a return-data defect rather than a firm characteristic.
EXTREME_BETA_MKT = 3.0

# |CAR| above this is a firm-specific corporate event rather than a tariff response. Reported with
# its screen status, never dropped or winsorised - both are Script 3's decisions.
EXTREME_CAR = 1.0

MAX_LISTED = 10          # identities printed before deferring to a count
AR_PER_LINE = 7          # daily ARs per line in the spot check
IN_SAMPLE_AR_TOL = 1e-10  # OLS residuals sum to zero; this is a machine-precision allowance

# Each run pairs an event with the date its estimation window must close before. The anchors are
# policy dates supplied by the research design and are not necessarily trading days, so the window
# ends on the last trading day strictly before the anchor.
RUNS_2025 = [
    {
        "name": "imposition_primary",
        "event": "impose",
        "anchor": "2025-01-20",
        "out": INTERMEDIATE_DIR / "car_imposition_primary.csv",
        # Change of administration. Tariff-policy expectations re-price from this date, so the
        # cleanest pre-regime loadings come from a window that closes before it. Not a trading
        # day (MLK Day), which is why the anchor is resolved against the calendar.
        "anchor_note": "second Trump inauguration; window closes before the change of "
                       "administration re-prices tariff expectations (not a trading day)",
    },
    {
        "name": "imposition_robustness",
        "event": "impose",
        "anchor": "2025-02-13",
        "out": INTERMEDIATE_DIR / "car_imposition_robustness.csv",
        # Later anchor, so a longer window (219 vs 202 days): it buys estimation data at the cost
        # of including the post-inauguration period in the loadings.
        "anchor_note": "reciprocal-trade memorandum; longer window, accepts post-inauguration "
                       "drift in the loadings in exchange for 17 more trading days",
    },
    {
        "name": "reversal_primary",
        "event": "reverse",
        "anchor": "2025-05-28",
        "out": INTERMEDIATE_DIR / "car_reversal_primary.csv",
        # First judicial reversal signal; closing before it keeps the reversal loadings free of
        # litigation-outcome repricing. This window deliberately CONTAINS the imposition event and
        # its aftermath - no dates are excised (see the contamination checks in validate).
        "anchor_note": "Court of International Trade ruling in V.O.S. Selections v. Trump, the "
                       "first judicial reversal signal; imposition event deliberately retained",
    },
]

RUNS = RUNS_2025


def build_runs(cycle_name: str) -> list[dict]:
    """The runs to estimate for one cycle: the 2025 anchor-dated three, or one per event.

    Where the cycle carries an ``anchor_offset`` the estimation window closes a fixed number of
    trading days before each event rather than before a hand-picked policy date. Nine Section 301
    events admit no such date - the escalation process ran continuously from January 2018, so any
    anchor sits inside the repricing it is meant to exclude - and inventing nine would reintroduce
    exactly the date discretion section 7.6 rules out.
    """
    cycle = ccd.CYCLES[cycle_name]
    if cycle["anchor_offset"] is None:
        return RUNS_2025
    offset = cycle["anchor_offset"]
    widest = min(lo for lo, _ in EVENT_WINDOWS)
    if offset > widest:
        raise ValueError(f"anchor_offset {offset} closes the estimation window inside the widest "
                         f"event window [{widest}, ...]; the event's own abnormal returns would "
                         f"be in-sample OLS residuals.")
    return [{
        "name": name,
        "event": name,
        "anchor_offset": offset,
        "out": INTERMEDIATE_DIR / f"car_cc_{name}.csv",
        "anchor_note": f"uniform rule: window closes {abs(offset)} trading days before the event, "
                       f"the last day before the widest event window [{widest},+{-widest}] opens",
    } for name in cycle["events"]]


def select_cycle(name: str) -> dict:
    """Rebind this module's cycle-dependent paths and runs, and Script 1's alongside them."""
    global PANEL_PATH, FF_DAILY_FILES, REPORT_OUT, RUNS
    cycle = ccd.select_cycle(name)
    PANEL_PATH = ccd.PANEL_OUT.with_suffix(f".{ccd.OUTPUT_FORMAT}")
    FF_DAILY_FILES = cycle["ff_daily"]
    REPORT_OUT = OUTPUT_DIR / f"car_estimation_validation_report{cycle['suffix']}.txt"
    RUNS = build_runs(name)
    return cycle


def sign_flip_pair(results: dict) -> tuple[tuple[str, pd.DataFrame], ...] | None:
    """The cycle's named (tightening, loosening) runs, as (name, frame) pairs.

    Named in ccd.CYCLES rather than inferred from run names, so which two legs carry the sign-flip
    test is a recorded decision and not an artefact of a string suffix.
    """
    names = ccd.CYCLES[ccd.CYCLE]["sign_flip_pair"]
    if not set(names) <= set(results):
        return None
    return tuple((name, results[name][0]) for name in names)


def _window_label(lo: int, hi: int) -> str:
    return f"m{abs(lo)}p{hi}"


CAR_COLUMNS = [f"car_{_window_label(*w)}" for w in EVENT_WINDOWS]
NAR_COLUMNS = [f"n_ar_{_window_label(*w)}" for w in EVENT_WINDOWS]
BETA_COLUMNS = [f"beta_{f}" for f in FACTORS]
OUTPUT_COLUMNS = (
    ["permno", "ticker", "run", "event", "event_date", "anchor_date", "est_start", "est_end",
     "n_est_obs", "alpha"]
    + BETA_COLUMNS
    + ["r2", "beta_mktrf_extreme"]
    + CAR_COLUMNS + NAR_COLUMNS
    + ["in_screened_universe_pit", "pit_screen_month", "exclusion_reason"]
)

_REPORT: list[str] = []


def _say(line: str = "") -> None:
    """Append one line to the single consolidated validation report."""
    _REPORT.append(line)


def _section(title: str) -> None:
    """Start a titled report section."""
    _say()
    _say(title)
    _say("-" * len(title))


def _listed(values) -> str:
    """Render an identity list, deferring to a count past MAX_LISTED."""
    vals = list(values)
    shown = " ".join(str(v) for v in vals[:MAX_LISTED])
    return shown if len(vals) <= MAX_LISTED else f"{shown} ... and {len(vals) - MAX_LISTED} more"


def _describe(frame: pd.DataFrame, cols: list[str], width: int = 18) -> None:
    """Report distribution and missingness for a set of columns."""
    _say(f"  {'variable':<22}{'n':>8}{'null':>8}" + "".join(
        f"{h:>{width}}" for h in ("min", "p25", "median", "p75", "max")))
    for col in cols:
        s = pd.to_numeric(frame[col], errors="coerce")
        v = s.dropna()
        if v.empty:
            _say(f"  {col:<22}{0:>8,}{len(s):>8,}" + f"{'-':>{width}}" * 5)
            continue
        cells = "".join(format(x, f">{width},.4f") for x in
                        (v.min(), v.quantile(.25), v.median(), v.quantile(.75), v.max()))
        _say(f"  {col:<22}{len(v):>8,}{int(s.isna().sum()):>8,}{cells}")


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_panel(path: Path | None = None) -> pd.DataFrame:
    """Read the five columns of Script 1's panel this script needs.

    in_screened_universe is coerced explicitly rather than trusted to pandas' inference: CSV does
    not carry dtypes, and a bool column silently arriving as object would make the point-in-time
    screen flag meaningless. Same class of hazard as the cik zero-padding closed in Step 4b.

    The path defaults to None and is resolved from the module global at call time, never bound as
    a default argument: Python evaluates defaults once at definition, so a default of PANEL_PATH
    would keep pointing at the cycle that was active on import and silently read the wrong panel.
    """
    path = PANEL_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_controls_data.py first.")
    if ccd.OUTPUT_FORMAT != "csv":
        raise ValueError(f"only csv panels are readable here (OUTPUT_FORMAT="
                         f"{ccd.OUTPUT_FORMAT!r}); Parquet is blocked by Smart App Control.")
    panel = pd.read_csv(path, usecols=PANEL_COLUMNS, parse_dates=["date"])
    panel["in_screened_universe"] = (panel["in_screened_universe"].astype(str)
                                     .str.strip().str.lower().eq("true"))
    if panel.duplicated(subset=["permno", "date"]).any():
        raise ValueError(f"{path.name} has duplicate (permno, date) rows")
    return panel


def load_factors(paths: list[Path] | None = None) -> pd.DataFrame:
    """Read the daily FF5+MOM factor files and the risk-free rate.

    Where a cycle supplies more than one export they are concatenated, exact duplicate rows
    dropped, and the date index then asserted unique - so an overlap that agrees is merged
    silently while one that disagrees raises rather than being resolved by read order. Same
    contract as clean_controls_data.dedupe_daily applies to the returns files.
    """
    paths = FF_DAILY_FILES if paths is None else paths
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; it is a required input.")
        ff = pd.read_csv(path, parse_dates=["date"])
        missing = [c for c in ["date", "rf"] + FACTORS if c not in ff.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required field(s): {missing}")
        frames.append(ff[["date", "rf"] + FACTORS])

    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    conflicting = out.loc[out["date"].duplicated(keep=False), "date"].unique()
    if len(conflicting):
        raise ValueError(f"{len(conflicting)} date(s) carry differing factor values across "
                         f"{', '.join(p.name for p in paths)}: "
                         f"{_listed(pd.DatetimeIndex(conflicting).strftime('%Y-%m-%d'))}")
    return out.sort_values("date").reset_index(drop=True)


def build_calendar(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """The reference trading-day sequence: the panel's own distinct sorted dates."""
    return pd.DatetimeIndex(np.sort(panel["date"].unique()))


def reconcile_calendar(cal: pd.DatetimeIndex,
                       factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """Inner-join the factors onto the panel calendar instead of assuming a shared one.

    Both directions of mismatch are counted: panel dates with no factor row (which would shorten
    the working calendar and shift every trading-day offset) and factor rows inside the panel span
    with no panel date. The working calendar returned is the join result, so offsets are always
    resolved against dates that actually carry factors.
    """
    aligned = pd.DataFrame({"date": cal}).merge(factors, on="date", how="inner")
    working = pd.DatetimeIndex(aligned["date"])
    ff_dates = pd.DatetimeIndex(factors["date"])
    panel_only = cal.difference(ff_dates)
    ff_only = ff_dates[(ff_dates >= cal[0]) & (ff_dates <= cal[-1])].difference(cal)
    nulls = int(aligned[["rf"] + FACTORS].isna().sum().sum())
    if nulls:
        raise ValueError(f"{nulls} null factor values inside the panel span; the market model "
                         f"cannot be estimated on a gapped factor series.")
    return aligned, working, {
        "panel_days": len(cal), "factor_rows": len(factors), "working_days": len(working),
        "panel_only": panel_only, "ff_only": ff_only,
        "factor_span": (ff_dates.min(), ff_dates.max()),
    }


# --------------------------------------------------------------------------- #
# Window resolution                                                           #
# --------------------------------------------------------------------------- #
def resolve_run(run: dict, cal: pd.DatetimeIndex) -> dict:
    """Resolve one run's event date, anchor and offsets onto the working trading calendar.

    Two ways to close the estimation window, and both end on the last trading day strictly before
    an anchor - only the anchor's provenance differs:

    ``anchor``        a policy date, which need not be a trading day (2025-01-20 is MLK Day), so a
                      calendar ``anchor - 1 day`` would land on a Sunday. Used by the 2025 runs.
    ``anchor_offset`` a fixed count of trading days before the event, used where a cycle has too
                      many events to anchor each on a defensible policy date. The equivalent
                      anchor date is derived back out for the report, so both paths document the
                      window the same way.

    Returns integer indices alongside the calendar dates they resolve to so the report can show
    both.
    """
    event_date = pd.Timestamp(ccd.EVENT_DATES[run["event"]])
    if event_date not in cal:
        raise ValueError(f"{run['name']}: event date {event_date:%Y-%m-%d} is not a trading day "
                         f"in the working calendar.")

    event_idx = int(cal.get_loc(event_date))
    est_lo = event_idx + EST_START_OFFSET
    if "anchor_offset" in run:
        est_hi = event_idx + run["anchor_offset"]
        if not 0 <= est_hi < len(cal) - 1:
            raise ValueError(f"{run['name']}: anchor offset {run['anchor_offset']} resolves off "
                             f"the calendar (index {est_hi} of {len(cal)}).")
        anchor = cal[est_hi + 1]        # the window closes strictly before this day, as above
    else:
        anchor = pd.Timestamp(run["anchor"])
        est_hi = int(cal.searchsorted(anchor)) - 1  # last trading day strictly before the anchor

    if est_lo < 0:
        raise ValueError(f"{run['name']}: estimation window opens {-est_lo} trading days before "
                         f"the calendar starts ({cal[0]:%Y-%m-%d}); the panel span is too short.")
    if est_hi <= est_lo:
        raise ValueError(f"{run['name']}: anchor {anchor:%Y-%m-%d} closes the estimation window "
                         f"at or before it opens.")
    if cal[est_hi] >= anchor:
        raise ValueError(f"{run['name']}: resolved window end {cal[est_hi]:%Y-%m-%d} is not "
                         f"strictly before the anchor {anchor:%Y-%m-%d}.")

    widest_lo = min(lo for lo, _ in EVENT_WINDOWS)
    widest_hi = max(hi for _, hi in EVENT_WINDOWS)
    if event_idx + widest_lo < 0 or event_idx + widest_hi >= len(cal):
        raise ValueError(f"{run['name']}: event window [{widest_lo}, +{widest_hi}] runs off the "
                         f"calendar ({cal[0]:%Y-%m-%d} .. {cal[-1]:%Y-%m-%d}).")

    return {**run, "event_date": event_date, "anchor_date": anchor, "event_idx": event_idx,
            "est_lo": est_lo, "est_hi": est_hi, "est_start": cal[est_lo], "est_end": cal[est_hi],
            "ar_lo": event_idx + widest_lo, "ar_hi": event_idx + widest_hi,
            "n_est_days": est_hi - est_lo + 1}


def pit_screen(panel: pd.DataFrame, event_date: pd.Timestamp) -> tuple[pd.Series, pd.Period, dict]:
    """Screen membership at the last completed calendar month strictly before the event.

    The panel's ``in_screened_universe`` is *same-month* membership, so read at 2025-04-02 it
    reflects 30-April market equity - information not available on the event date. Evaluating on
    the prior month and holding it fixed across the whole event window removes that look-ahead,
    the same point-in-time rule already applied to TExp filing dates and Compustat availability.
    Firms with no row in that month cannot be screened and are returned False, counted separately.
    """
    month = event_date.to_period("M") - 1
    rows = panel[panel["date"].dt.to_period("M") == month]
    if rows.empty:
        raise ValueError(f"no panel rows in {month}, the point-in-time screen month for "
                         f"{event_date:%Y-%m-%d}; the panel span does not cover it.")

    # Membership came from a monthly merge, so it must be constant within a firm-month; if it is
    # not, the flag does not mean what this function assumes.
    varying = rows.groupby("permno")["in_screened_universe"].nunique()
    if (varying > 1).any():
        raise ValueError(f"{int((varying > 1).sum())} firms have non-constant "
                         f"in_screened_universe within {month}")

    flag = rows.groupby("permno")["in_screened_universe"].max()

    # The same-month flag the panel carries, for contrast: the difference is the look-ahead this
    # function removes, and it is reported rather than merely asserted away.
    at_event = panel[panel["date"] == event_date]
    same_month = at_event.set_index("permno")["in_screened_universe"]
    pit_at_event = same_month.index.map(flag).to_series(index=same_month.index).eq(True)
    return flag, month, {
        "month": month, "firms_with_rows": len(flag), "in_screen": int(flag.sum()),
        "out_of_screen": int((~flag).sum()),
        "trading_no_pit_row": sorted(set(at_event["permno"]) - set(flag.index)),
        "same_month_in_screen": int(same_month.sum()),
        "pit_in_screen_at_event": int(pit_at_event.sum()),
        "pit_only": int((pit_at_event & ~same_month).sum()),
        "same_month_only": int((~pit_at_event & same_month).sum()),
    }


# --------------------------------------------------------------------------- #
# Estimation                                                                  #
# --------------------------------------------------------------------------- #
def estimate_betas(excess: pd.DataFrame, fmat: np.ndarray,
                   resolved: dict) -> tuple[pd.DataFrame, dict]:
    """Per-firm OLS of excess return on FF5+MOM over this run's estimation window.

    Coefficients only: no standard errors are needed here because inference on the CARs happens in
    Script 3, so np.linalg.lstsq is used rather than statsmodels. Firms are looped over as
    specified; each design is at most 219 x 7. Firms below MIN_EST_OBS get a row with null
    loadings and a populated exclusion_reason rather than disappearing.
    """
    lo, hi = resolved["est_lo"], resolved["est_hi"]
    window = excess.iloc[lo:hi + 1]
    design = np.column_stack([np.ones(len(window)), fmat[lo:hi + 1]])

    rows, excluded = [], []
    for permno in window.columns:
        y = window[permno].to_numpy(dtype=float)
        mask = np.isfinite(y)
        n_obs = int(mask.sum())
        row = {"permno": permno, "n_est_obs": n_obs}
        if n_obs < MIN_EST_OBS:
            # A firm with no observations at all did not trade in the window (listed after it or
            # delisted before it); one with 1-99 traded but too thinly. Different data situations,
            # so they carry different reasons rather than one blended count.
            row["exclusion_reason"] = (
                "no_estimation_obs" if n_obs == 0 else
                f"insufficient_estimation_obs ({n_obs} < {MIN_EST_OBS})")
            excluded.append(permno)
            rows.append(row)
            continue
        y_used, x_used = y[mask], design[mask]
        coef = np.linalg.lstsq(x_used, y_used, rcond=None)[0]
        resid = y_used - x_used @ coef
        tss = float(((y_used - y_used.mean()) ** 2).sum())
        row["alpha"] = coef[0]
        row.update(dict(zip(BETA_COLUMNS, coef[1:])))
        row["r2"] = 1.0 - float((resid ** 2).sum()) / tss if tss > 0 else np.nan
        row["exclusion_reason"] = ""
        rows.append(row)

    # Reindexed so the loading columns exist even if every firm were excluded.
    betas = pd.DataFrame(rows).reindex(
        columns=["permno", "n_est_obs", "alpha"] + BETA_COLUMNS + ["r2", "exclusion_reason"])
    betas["exclusion_reason"] = betas["exclusion_reason"].fillna("")
    betas["beta_mktrf_extreme"] = betas["beta_mktrf"].abs() > EXTREME_BETA_MKT
    valid = betas[betas["exclusion_reason"] == ""]
    obs = betas["n_est_obs"]
    used_dates = window.index[window.notna().any(axis=1)]
    stats = {
        "n_firms": len(betas), "n_valid": len(valid), "n_excluded": len(excluded),
        "excluded": excluded, "extreme": valid.loc[valid["beta_mktrf_extreme"], "permno"].tolist(),
        "obs_min": int(obs.min()), "obs_median": float(obs.median()), "obs_max": int(obs.max()),
        "valid_obs_min": int(valid["n_est_obs"].min()) if len(valid) else 0,
        "max_date_used": used_dates.max() if len(used_dates) else None,
        "no_obs": int((obs == 0).sum()), "thin_obs": int(((obs > 0) & (obs < MIN_EST_OBS)).sum()),
    }
    return betas, stats


def compute_ar(excess: pd.DataFrame, fmat: np.ndarray, betas: pd.DataFrame,
               lo: int, hi: int) -> pd.DataFrame:
    """Abnormal returns over cal[lo..hi] for the firms holding valid loadings.

    AR = (r - rf) - (alpha + beta'F), i.e. the OLS residual, so applying this over the estimation
    window must average ~0 per firm - the check that this code path and estimate_betas agree.
    """
    valid = betas[betas["exclusion_reason"] == ""]
    design = np.column_stack([np.ones(hi - lo + 1), fmat[lo:hi + 1]])
    expected = design @ valid[["alpha"] + BETA_COLUMNS].to_numpy().T
    actual = excess.iloc[lo:hi + 1][valid["permno"].to_numpy()]
    return actual - pd.DataFrame(expected, index=actual.index, columns=actual.columns)


def compute_cars(ar: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    """Sum daily ARs over each event window; NaN unless every required day is present.

    ``min_count`` set to the window length is what enforces that: the slice holds exactly that
    many rows, so a single missing AR leaves the sum NaN rather than silently cumulating a
    partial window.
    """
    offset = resolved["event_idx"] - resolved["ar_lo"]
    out = {}
    for lo, hi in EVENT_WINDOWS:
        label = _window_label(lo, hi)
        need = hi - lo + 1
        sub = ar.iloc[offset + lo:offset + hi + 1]
        if len(sub) != need:
            raise ValueError(f"event window [{lo},+{hi}] sliced {len(sub)} of {need} days")
        out[f"car_{label}"] = sub.sum(min_count=need)
        out[f"n_ar_{label}"] = sub.notna().sum()
    return pd.DataFrame(out)


def run_event(panel: pd.DataFrame, excess: pd.DataFrame, fmat: np.ndarray,
              tickers: pd.Series, resolved: dict) -> tuple[pd.DataFrame, dict]:
    """Estimate one run end to end: loadings, abnormal returns, CARs and the screen flag."""
    betas, beta_stats = estimate_betas(excess, fmat, resolved)

    ar = compute_ar(excess, fmat, betas, resolved["ar_lo"], resolved["ar_hi"])
    cars = compute_cars(ar, resolved)

    # The same beta-application path, run back over the estimation window: OLS residuals sum to
    # zero, so anything but ~0 here means compute_ar and estimate_betas disagree.
    in_sample = compute_ar(excess, fmat, betas, resolved["est_lo"], resolved["est_hi"]).mean()
    worst = float(in_sample.abs().max()) if len(in_sample) else 0.0
    if worst > IN_SAMPLE_AR_TOL:
        raise ValueError(f"{resolved['name']}: in-sample mean AR reaches {worst:.3e}, above the "
                         f"{IN_SAMPLE_AR_TOL:.0e} tolerance; beta application is inconsistent "
                         f"with estimation.")

    flag, month, screen_stats = pit_screen(panel, resolved["event_date"])

    out = betas.merge(cars, left_on="permno", right_index=True, how="left")
    out["ticker"] = out["permno"].map(tickers)
    out["run"] = resolved["name"]
    out["event"] = resolved["event"]
    out["event_date"] = resolved["event_date"]
    out["anchor_date"] = resolved["anchor_date"]
    out["est_start"] = resolved["est_start"]
    out["est_end"] = resolved["est_end"]
    out["in_screened_universe_pit"] = out["permno"].map(flag).eq(True)
    out["pit_screen_month"] = str(month)
    out = out.reindex(columns=OUTPUT_COLUMNS).sort_values("permno").reset_index(drop=True)

    incomplete = {}
    for lo, hi in EVENT_WINDOWS:
        label = _window_label(lo, hi)
        has_beta = out["exclusion_reason"] == ""
        incomplete[label] = out.loc[has_beta & out[f"car_{label}"].isna(), "permno"].tolist()

    stats = {"betas": beta_stats, "screen": screen_stats, "in_sample_worst": worst,
             "incomplete": incomplete, "ar": ar, "resolved": resolved}
    return out, stats


def write_car(frame: pd.DataFrame, path: Path) -> Path:
    """Write one run's per-firm CAR table."""
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def read_car(path: Path) -> pd.DataFrame:
    """Read a CAR table back with the rules CSV cannot carry, for Script 3 to import.

    A valid firm's exclusion_reason is the empty string, which returns as NaN; the flag columns
    are booleans that return as object once any row is blank. Both are closed here so the read
    contract lives once beside the writer, the arrangement Step 4b adopted for the cleaned-filings
    file after an unpinned dtype silently emptied a downstream filter.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run estimate_car.py first.")
    frame = pd.read_csv(path, parse_dates=["event_date", "anchor_date", "est_start", "est_end"])
    frame["exclusion_reason"] = frame["exclusion_reason"].fillna("")
    for col in ("in_screened_universe_pit", "beta_mktrf_extreme"):
        frame[col] = frame[col].astype(str).str.strip().str.lower().eq("true")
    return frame


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _check_contamination(results: dict, cal: pd.DatetimeIndex) -> None:
    """Assert each estimation window is contiguous and correctly placed relative to both events.

    Contiguity is what proves a window is *unmodified*: a subset check alone would pass on a
    window from which the imposition event had been excised.
    """
    _section("3. Estimation-window placement and cross-contamination (asserted)")
    for name, (_, stats) in results.items():
        r = stats["resolved"]
        used = cal[r["est_lo"]:r["est_hi"] + 1]
        if not used.equals(cal[(cal >= used[0]) & (cal <= used[-1])]):
            raise ValueError(f"{name}: estimation window is not contiguous on the calendar")
        if used[-1] >= r["anchor_date"]:
            raise ValueError(f"{name}: estimation window extends to or past its anchor")
        latest = stats["betas"]["max_date_used"]
        if latest is not None and latest >= r["anchor_date"]:
            raise ValueError(f"{name}: a firm has an estimation observation on "
                             f"{latest:%Y-%m-%d}, at or past the anchor")
        _say(f"  {name:<22} {used[0]:%Y-%m-%d} .. {used[-1]:%Y-%m-%d}  {len(used):>3} days, "
             f"contiguous; latest observation actually used {latest:%Y-%m-%d} < anchor "
             f"{r['anchor_date']:%Y-%m-%d}")

    widest_lo = min(lo for lo, _ in EVENT_WINDOWS)
    widest_hi = max(hi for _, hi in EVENT_WINDOWS)
    event_windows = {}
    for name, (_, stats) in results.items():
        r = stats["resolved"]
        event_windows[r["event"]] = cal[r["event_idx"] + widest_lo:r["event_idx"] + widest_hi + 1]

    # Own-event overlap is the only fatal case: it would make a run's own abnormal returns
    # in-sample OLS residuals, mechanically pulled toward zero. Overlap with *another* event's
    # window is reported, not raised. Across nine Section 301 dates it is the norm - the events sit
    # 6 to 162 trading days apart, so a later run's estimation window routinely spans earlier ones,
    # exactly as the 2025 reversal window deliberately contains the imposition. Contiguity, checked
    # above, is what proves such a window is unmodified rather than excised; a partial overlap
    # means only that the window boundary fell inside a neighbouring event, not that days were
    # removed from it.
    _say()
    _say(f"  own-event overlap must be 0 (asserted); other events' [{widest_lo},+{widest_hi}] "
         f"windows are reported, not excised:")
    for name, (_, stats) in results.items():
        r = stats["resolved"]
        used = set(cal[r["est_lo"]:r["est_hi"] + 1])
        if used & set(event_windows[r["event"]]):
            raise ValueError(f"{name}: own event window overlaps its estimation window")

        held = []
        for event, window in event_windows.items():
            if event == r["event"]:
                continue
            overlap = len(used & set(window))
            if overlap:
                held.append(f"{event} {overlap}/{len(window)}"
                            f"{' FULL' if overlap == len(window) else ' part'}")
        _say(f"  {name:<22} own 0/{len(event_windows[r['event']])}  |  "
             + (", ".join(held) if held else "no other event-window days"))


def _check_spot(results: dict, cal: pd.DatetimeIndex) -> None:
    """Print loadings, daily ARs and all three CARs for the Script 1 spot-check firms."""
    _section("9. Spot check - Script 1's named firms, every run")
    _say("  daily AR in %, excess-return space; t is trading days from the event date")
    for name, (frame, stats) in results.items():
        r, ar = stats["resolved"], stats["ar"]
        _say()
        _say(f"  [{name}]  event {r['event_date']:%Y-%m-%d}  "
             f"estimation {r['est_start']:%Y-%m-%d}..{r['est_end']:%Y-%m-%d}")
        for permno in ccd.SPOT_CHECK_PERMNOS:
            row = frame[frame["permno"] == permno]
            if row.empty:
                _say(f"    permno {permno}: not in the panel")
                continue
            row = row.iloc[0]
            _say(f"    permno {permno}  {row['ticker']}  n_obs={row['n_est_obs']:,}"
                 f"{'  EXCLUDED: ' + row['exclusion_reason'] if row['exclusion_reason'] else ''}")
            if row["exclusion_reason"]:
                continue
            _say(f"      alpha={row['alpha']:+.6f}  " + "  ".join(
                f"{f}={row[f'beta_{f}']:+.3f}" for f in FACTORS) + f"  r2={row['r2']:.3f}")
            _say("      " + "  ".join(
                f"CAR[{lo},+{hi}]={row[f'car_{_window_label(lo, hi)}'] * 100:+.2f}%"
                f" (n={int(row[f'n_ar_{_window_label(lo, hi)}'])})"
                if pd.notna(row[f"car_{_window_label(lo, hi)}"]) else
                f"CAR[{lo},+{hi}]=missing"
                for lo, hi in EVENT_WINDOWS))
            series = ar[permno]
            days = [i - r["event_idx"] for i in range(r["ar_lo"], r["ar_hi"] + 1)]
            for start in range(0, len(days), AR_PER_LINE):
                chunk = list(zip(days[start:start + AR_PER_LINE],
                                 series.iloc[start:start + AR_PER_LINE]))
                _say(f"      t{chunk[0][0]:+d}..t{chunk[-1][0]:+d}  " + " ".join(
                    f"{v * 100:+7.2f}" if pd.notna(v) else f"{'   n/a':>7}" for _, v in chunk))


def validate(results: dict, cal: pd.DatetimeIndex, cal_stats: dict,
             factors: pd.DataFrame, paths: list[Path]) -> None:
    """Assemble the consolidated validation report and write it to disk."""
    _say("=" * 78)
    _say("SECTION 7.2 EVENT STUDY - CAR ESTIMATION VALIDATION REPORT")
    _say("=" * 78)
    _say(f"Cycle         : {ccd.CYCLE}")
    _say("Events        : " + ", ".join(f"{k}={v}" for k, v in ccd.EVENT_DATES.items()))
    _say(f"Runs          : {', '.join(results)}")
    _say(f"Model         : (r - rf) = alpha + b'[{', '.join(FACTORS)}] + e, "
         f"estimation offset {EST_START_OFFSET}, min {MIN_EST_OBS} obs")
    _say(f"Inputs        : {PANEL_PATH.name}, "
         f"{', '.join(p.name for p in FF_DAILY_FILES)}")

    _section("1. Trading-day sequence and factor-calendar reconciliation")
    _say(f"  reference sequence from {PANEL_PATH.name}   {cal_stats['panel_days']:>5,} days "
         f"{cal[0]:%Y-%m-%d} .. {cal[-1]:%Y-%m-%d}")
    _say(f"  factor rows ({len(FF_DAILY_FILES)} file(s), merged)  {cal_stats['factor_rows']:>5,} "
         f"{cal_stats['factor_span'][0]:%Y-%m-%d} .. {cal_stats['factor_span'][1]:%Y-%m-%d}")
    _say(f"  working calendar after inner join on date  {cal_stats['working_days']:>5,} days")
    _say(f"  panel dates with no factor row (dropped)   {len(cal_stats['panel_only']):>5,}"
         + (f"   {_listed(cal_stats['panel_only'].strftime('%Y-%m-%d'))}"
            if len(cal_stats["panel_only"]) else ""))
    _say(f"  factor dates in span not in the panel      {len(cal_stats['ff_only']):>5,}"
         + (f"   {_listed(cal_stats['ff_only'].strftime('%Y-%m-%d'))}"
            if len(cal_stats["ff_only"]) else ""))
    if len(cal_stats["panel_only"]):
        _say("  NOTE: dropped panel dates shorten the working calendar, so trading-day offsets "
             "below are resolved against the joined sequence, not the panel's raw one.")

    _section("2. Window resolution against the calendar")
    for name, (_, stats) in results.items():
        r = stats["resolved"]
        _say(f"  [{name}]")
        _say(f"    event date            {r['event_date']:%Y-%m-%d} (index {r['event_idx']})")
        _say(f"    anchor                {r['anchor_date']:%Y-%m-%d} "
             f"({'a trading day' if r['anchor_date'] in cal else 'not a trading day'})")
        _say(f"    estimation window     {r['est_start']:%Y-%m-%d} .. {r['est_end']:%Y-%m-%d}"
             f"   [{EST_START_OFFSET}, {r['est_hi'] - r['event_idx']}] "
             f"= {r['n_est_days']:,} trading days")
        for lo, hi in EVENT_WINDOWS:
            _say(f"    event window [{lo:>3},{hi:+3}]  "
                 f"{cal[r['event_idx'] + lo]:%Y-%m-%d} .. {cal[r['event_idx'] + hi]:%Y-%m-%d}")
        _say(f"    anchor rationale      {r['anchor_note']}")

    _check_contamination(results, cal)

    _section("4. Loading estimation coverage")
    _say(f"  {'run':<22}{'firms':>8}{'valid':>8}{'excluded':>10}"
         f"{'obs min':>10}{'obs median':>12}{'obs max':>10}")
    for name, (_, stats) in results.items():
        b = stats["betas"]
        _say(f"  {name:<22}{b['n_firms']:>8,}{b['n_valid']:>8,}{b['n_excluded']:>10,}"
             f"{b['obs_min']:>10,}{b['obs_median']:>12,.0f}{b['obs_max']:>10,}")
    for name, (_, stats) in results.items():
        b = stats["betas"]
        _say(f"  {name}: {b['n_excluded']} excluded below {MIN_EST_OBS} obs "
             f"= {b['no_obs']} with no observations in the window (listed after it or delisted "
             f"before it) + {b['thin_obs']} that traded too thinly; min obs among the valid: "
             f"{b['valid_obs_min']:,}")
        if b["excluded"]:
            _say(f"    permnos: {_listed(b['excluded'])}")
    _say(f"  in-sample mean AR (must be ~0; tolerance {IN_SAMPLE_AR_TOL:.0e}): " + ", ".join(
        f"{name} {stats['in_sample_worst']:.2e}" for name, (_, stats) in results.items()))

    _section("5. Estimated loadings")
    for name, (frame, stats) in results.items():
        valid = frame[frame["exclusion_reason"] == ""]
        _say(f"  [{name}]  n = {len(valid):,}")
        _describe(valid, ["alpha"] + BETA_COLUMNS + ["r2"])
        extreme = stats["betas"]["extreme"]
        _say(f"    |beta_mktrf| > {EXTREME_BETA_MKT}: {len(extreme)} flagged, not dropped"
             + (f" - permnos {_listed(extreme)}" if extreme else ""))

    _section("6. Cumulative abnormal returns")
    _say("  AR = (r - rf) - (alpha + b'F): the OLS residual, so CARs carry no risk-free drift.")
    _say("  Had rf been left in the expected return, every CAR would have been inflated by "
         "sum(rf) over the window:")
    rf = factors.set_index("date")["rf"]
    for event in dict.fromkeys(stats["resolved"]["event"] for _, stats in results.values()):
        idx = next(s["resolved"]["event_idx"] for _, s in results.values()
                   if s["resolved"]["event"] == event)
        parts = [f"[{lo},+{hi}] {rf.loc[cal[idx + lo]:cal[idx + hi]].sum() * 100:+.3f}pp"
                 for lo, hi in EVENT_WINDOWS]
        _say(f"    {event:<8} {'  '.join(parts)}")

    for name, (frame, stats) in results.items():
        _say()
        _say(f"  [{name}]")
        _describe(frame, CAR_COLUMNS)
        for label, permnos in stats["incomplete"].items():
            if permnos:
                _say(f"    car_{label}: {len(permnos)} firms have valid loadings but an "
                     f"incomplete window, CAR left missing - {_listed(permnos)}")
        # The tails are firm-specific corporate events, not tariff responses, and a single one can
        # dominate an OLS cross-section. Reported with screen status, never dropped or winsorised:
        # both are estimation decisions for Script 3.
        for label in (_window_label(*w) for w in EVENT_WINDOWS):
            col = f"car_{label}"
            extreme = frame[frame[col].abs() > EXTREME_CAR]
            if extreme.empty:
                continue
            kept = int(extreme["in_screened_universe_pit"].sum())
            worst = extreme.loc[extreme[col].abs().sort_values(ascending=False).index]
            _say(f"    {col}: {len(extreme)} firms exceed |{EXTREME_CAR:.0%}|, "
                 f"{kept} of them inside the point-in-time screen - "
                 + ", ".join(f"{r.permno} {r.ticker} {getattr(r, col):+.1%}"
                             f"{'' if r.in_screened_universe_pit else ' [out of screen]'}"
                             for r in worst.head(MAX_LISTED).itertuples()))

    _section("7. Point-in-time sample screen (carried, not applied)")
    _say("  in_screened_universe_pit is membership in the last completed month strictly BEFORE")
    _say("  the event, not the panel's same-month flag, which would read post-event market cap.")
    _say("  Script 3 must filter on this column. Carried, never applied here: imposing the")
    _say("  section 6 screen is an estimation decision and the section 8 robustness table needs")
    _say("  the unscreened cross-section too.")
    for name, (_, stats) in results.items():
        s = stats["screen"]
        _say(f"  [{name}]  point-in-time month {s['month']}")
        _say(f"    firms with rows that month {s['firms_with_rows']:>6,}   "
             f"in screen {s['in_screen']:>6,}   out {s['out_of_screen']:>6,}")
        _say(f"    among firms trading on the event date: "
             f"point-in-time {s['pit_in_screen_at_event']:,} vs "
             f"same-month {s['same_month_in_screen']:,}  "
             f"(+{s['pit_only']} point-in-time only, -{s['same_month_only']} same-month only)")
        if s["trading_no_pit_row"]:
            _say(f"    {len(s['trading_no_pit_row'])} firms trade on the event date with no row "
                 f"in {s['month']} -> flagged False: {_listed(s['trading_no_pit_row'])}")

    _section("8. Firm coverage overlap between the two legs")
    pair = sign_flip_pair(results)
    if pair:
        (name_a, frame_a), (name_b, frame_b) = pair
        _say(f"  {'window':<14}{'both':>10}{'tighten only':>14}{'loosen only':>14}")
        for lo, hi in EVENT_WINDOWS:
            col = f"car_{_window_label(lo, hi)}"
            a = set(frame_a.loc[frame_a[col].notna(), "permno"])
            b = set(frame_b.loc[frame_b[col].notna(), "permno"])
            _say(f"  {f'[{lo},+{hi}]':<14}{len(a & b):>10,}{len(a - b):>14,}{len(b - a):>14,}")
        _say(f"  ({name_a} vs {name_b}; the joint sign-flip test in Script 3 runs on the "
             f"intersection.)")

    _check_spot(results, cal)

    _section("10. Outputs")
    for path in paths:
        _say(f"  {path.relative_to(BASE)}")
    _say(f"  {REPORT_OUT.relative_to(BASE)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_OUT.write_text("\n".join(_REPORT), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def main(cycle: str = ccd.DEFAULT_CYCLE) -> dict:
    select_cycle(cycle)
    panel = load_panel()
    factors = load_factors()

    cal = build_calendar(panel)
    aligned, cal, cal_stats = reconcile_calendar(cal, factors)
    fmat = aligned[FACTORS].to_numpy(dtype=float)

    # Excess returns, wide (date x permno): the regressand for every run, built once.
    wide = panel.pivot(index="date", columns="permno", values="ret").reindex(cal)
    excess = wide.sub(aligned.set_index("date")["rf"].reindex(cal), axis=0)
    tickers = (panel.dropna(subset=["ticker"]).sort_values("date")
               .groupby("permno")["ticker"].last())

    results, paths = {}, []
    for run in RUNS:
        resolved = resolve_run(run, cal)
        frame, stats = run_event(panel, excess, fmat, tickers, resolved)
        paths.append(write_car(frame, run["out"]))
        results[run["name"]] = (frame, stats)
        print(f"[{run['name']}] {stats['betas']['n_valid']:,} firms with loadings, "
              f"{stats['betas']['n_excluded']} excluded -> {run['out'].name}", flush=True)

    validate(results, cal, cal_stats, aligned, paths)
    print("\n".join(_REPORT))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FF5+MOM abnormal returns and CARs.")
    ap.add_argument("--cycle", choices=sorted(ccd.CYCLES), default=ccd.DEFAULT_CYCLE,
                    help="policy cycle to estimate (default: %(default)s)")
    main(ap.parse_args().cycle)
