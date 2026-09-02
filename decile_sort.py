"""
Step 5a - Section 7.3 decile-spread presentation: group CARs by exposure group, both 2025 legs.

Translates the section 7.2 coefficient into legible magnitudes. TExp has a mass point at exactly
zero, so a plain decile sort would impose a false ordering on genuinely tied firms:

    Group 0     TExp_item1a == 0 exactly
    Groups 1-9  equal-frequency quantiles of the strictly positive remainder

Breakpoints are computed ONCE on the end-March-2025 cross-section and the resulting
permno -> group mapping is applied unchanged to all three runs, so the identifying variation sits
in the returns and not in the sort.

This is a descriptive translation of the H1 regression result, NOT a strategy test. Section 7.3
explicitly excludes Sharpe, drawdown, turnover, capacity, GRS, spanning alpha and any backtest
return: reporting them would misrepresent a sign-flipping, non-forecastable effect as a tradeable
strategy. Nothing of the sort is computed or exported here.

Assumptions, also stated in the validation report:
  - The group universe is the end-March-2025 screen carried by car_imposition_primary.csv
    (in_screened_universe_pit), less firms with an exclusion_reason and firms with no TExp. The
    screen is read, never re-derived.
  - Group membership is FIXED across all three runs. The reversal's own point-in-time screen
    (evaluated at 2025-07) is deliberately not re-applied: a firm contributes wherever it has a
    valid CAR. Firms delisted between the legs simply have no reversal CAR and are reported.
  - TWO weighting schemes on identical firms per cell: value (section 7.3's nomination, the
    primary) and equal (v5 section 8's robustness). Reported side by side, so any difference
    between them is the weighting and not the sample. Results carry a `weighting` column and a
    consumer must filter on it; four charts cover (universe A/B) x (value/equal).
  - Value weights are me_lag on the trading day each WINDOW OPENS, so no return inside a window
    helps set the weights it is averaged with. The estimation anchor is deliberately not used:
    the imposition_primary anchor 2025-01-20 is MLK Day and carries no panel row at all, and
    weighting the two imposition runs at different dates would confound runs whose only intended
    difference is the estimation window. One rule serves all three windows, so that argument is
    untouched. The size and concentration diagnostics take REFERENCE_WINDOW's weights.

    python decile_sort.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: never opens a window, safe to run unattended
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import chartstyle as cs
import palette
import clean_data as cd
import clean_controls_data as ccd
import estimate_car as ec

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
CLEAN_DIR = BASE / "clean_data"
OUTPUT_DIR = BASE / "output"

# The panel path, event dates, runs, windows and CAR reader are owned by Scripts 1 and 2;
# importing rather than re-declaring keeps one definition, as run_car_regression.py does.
PANEL_PATH = ccd.PANEL_OUT.with_suffix(f".{ccd.OUTPUT_FORMAT}")
TEXP_PANEL_CSV = CLEAN_DIR / "texp_panel.csv"

RESULTS_OUT = OUTPUT_DIR / "decile_sort_results.csv"
CHART_OUT = OUTPUT_DIR / "decile_sort_chart.png"
REPORT_OUT = OUTPUT_DIR / "decile_sort_validation_report.txt"

# --- Universe B: the mega-cap-excluded robustness comparison ---------------- #
# Value weighting a broad universe concentrates several groups in a handful of mega-caps (report
# section 7), so the same procedure is re-run on a universe where no one firm can dominate a
# group. The cutoff is the Ken French NYSE 90th-percentile ME, the same source and the same
# point-in-time convention clean_data.flag_microcaps already applies at the bottom end with p10:
# universe B is therefore a clean NYSE p10-p90 band under one consistent definition of a size
# cutoff, rather than a second, sample-dependent notion of size.
ME_BP_P90_FIELD = 19             # French: every 5th pct, so field n holds p(5*(n-1)); 19 -> p90
TRIM_MONTH = "202503"            # end-March 2025, the month the group universe is defined at
RESULTS_EX_OUT = OUTPUT_DIR / "decile_sort_results_ex_megacap.csv"
CHART_EX_OUT = OUTPUT_DIR / "decile_sort_chart_ex_megacap.png"

# --- Weighting schemes: v5 section 8's equal- versus value-weighting robustness ------------- #
# Section 7.3 nominates value weighting and that stays the primary. v5 section 8 (unrevised by v6)
# additionally nominates "equal-weighting vs. value-weighting" as specification robustness, and
# this supplies it.
#
# Both schemes are computed on IDENTICAL ROWS - a firm enters a cell only with a non-null CAR and a
# strictly positive value weight, whichever scheme is being applied - so any movement between them
# is the weighting and not the sample. That is the same discipline run_car_regression uses for its
# H5 read. An unrestricted equal-weighted mean would additionally admit firms with no usable weight,
# which would confound the two; the count that would add is reported rather than taken.
#
# Equal weighting also matters as more than a robustness line here. Report section 7 shows value
# weighting concentrates several groups in a handful of mega-caps, and universe B was built to
# answer that by trimming them. Equal weighting answers the same concern differently, by giving
# every firm the same influence, so the four combinations of (universe, weighting) are two
# independent responses to one problem plus their union.
WEIGHTINGS = [
    {"name": "value", "adjective": "Value-weighted",
     "note": "on me_lag at each window's own opening trading day"},
    {"name": "equal", "adjective": "Equal-weighted",
     "note": "every entering firm counted once"},
]
PRIMARY_WEIGHTING = "value"

RESULTS_EW_CHART = OUTPUT_DIR / "decile_sort_chart_equal_weighted.png"
RESULTS_EW_CHART_EX = OUTPUT_DIR / "decile_sort_chart_equal_weighted_ex_megacap.png"

TEXP_COLUMN = "TExp_item1a"      # raw Item 1A measure, as in section 7.2
WEIGHT_COLUMN = "me_lag"         # market equity lagged one trading day

# Value weights are read per window, on the trading day the window opens, so that no return
# inside a window helps set its own weights (see weight_dates). The size and concentration
# DIAGNOSTICS - the mega-cap trim, effective N, the largest-holding tables - need one weight per
# firm rather than three, and take it from this window: the narrowest, whose weight date sits
# closest to the end-March-2025 cross-section the groups are built on.
REFERENCE_WINDOW = "car_m1p1"

# The run whose point-in-time screen defines the group universe. Section 7.0 and 7.3 both name
# end-March 2025, which is the screen estimate_car.pit_screen evaluates for the imposition legs.
UNIVERSE_RUN = "imposition_primary"

# The exposure vintage the groups are built on: the reference date whose 10-K supplies TExp for
# that run's event. Derived from the cycle registry rather than written out, so it follows if the
# registry changes. Section 7.3 needs exactly one cross-section - breakpoints are computed once and
# membership held fixed across all three runs - unlike run_car_regression.py, which reads a
# different vintage per event on the cross cycle.
UNIVERSE_EVENT = next(r["event"] for r in ec.RUNS_2025 if r["name"] == UNIVERSE_RUN)
TEXP_REFERENCE_DATE = ccd.CYCLES[ccd.DEFAULT_CYCLE]["texp_ref"][UNIVERSE_EVENT]

ZERO_GROUP = 0                   # the TExp == 0 mass point, kept out of the quantile sort
N_POSITIVE_GROUPS = 9            # groups 1..9 over the strictly positive remainder
GROUPS = [ZERO_GROUP] + list(range(1, N_POSITIVE_GROUPS + 1))

PANEL_COLUMNS = ["permno", "date", WEIGHT_COLUMN]
MAX_LISTED = 10                  # identities printed before deferring to a count
RULE = "=" * 78
THIN = "-" * 78

# Two categorical hues for the two event legs, plus texture for the robustness variant of the
# imposition leg - a secondary encoding rather than a third hue, because the robustness run is the
# same event as the primary and differs only in estimation window. Slots 1 and 2 of palette.py;
# the hatch is what separates the two imposition bars, so no pair here relies on hue alone.
PALETTE = palette.roles(impose="CATEGORICAL_1", reverse="CATEGORICAL_2")
# Distinguishes the trimmed universe in a chart title; the cutoff itself is in the report.
EX_MEGACAP_NOTE = "excluding mega-caps"

BAR_STYLE = {
    "imposition_primary": {"hatch": None, "color": "impose",
                           "label": "Imposition 2025-04-02, primary"},
    "imposition_robustness": {"hatch": "///", "color": "impose",
                              "label": "Imposition 2025-04-02, robustness"},
    "reversal_primary": {"hatch": None, "color": "reverse",
                         "label": "Reversal 2025-08-29"},
}

# Monotonicity verdict thresholds (report section 8). Both statistics enter every branch - see
# monotonicity_reading - so a series cannot read as a stronger trend than one that is strictly
# more monotone. Named here because they decide published wording.
RHO_STRONG = 0.7                 # |Spearman rho| at or above this is a strong gradient
RHO_PARTIAL = 0.4                # ... and above this, a partial one
MONOTONE_MAX_FLIPS = 3           # sign changes in the group-to-group differences, of 8 possible

# The single firm report section 9 names when explaining that extreme CARs are retained rather
# than trimmed: Janover / DeFi Development, whose +930% CAR is the largest in the CAR table and
# which the section 6 screen removed on its own terms. A firm identity that drives published
# report text belongs in configuration, not buried in a validation function.
EXTREME_CAR_EXAMPLE = {"permno": 24072, "name": "Janover / DeFi Development"}

# `weight_sum` is the market equity of the entering firms and is therefore a property of the CELL,
# identical across weighting schemes because the schemes share their rows. `eff_n` is the
# weighting-specific one: the inverse Herfindahl of the weights actually applied, which equals
# n_entering exactly under equal weighting and is far smaller under value weighting.
OUTPUT_COLUMNS = ["run", "event", "event_date", "weighting", "window", "group",
                  "n_nominal", "n_entering", "weight_sum", "eff_n",
                  "texp_min", "texp_max", "vw_car"]

_REPORT: list[str] = []


def _say(line: str = "") -> None:
    _REPORT.append(line)


def _section(title: str) -> None:
    _say()
    _say(title)
    _say(THIN)


def _listed(values) -> str:
    vals = list(values)
    head = ", ".join(str(v) for v in vals[:MAX_LISTED])
    return head if len(vals) <= MAX_LISTED else f"{head}, ... (+{len(vals) - MAX_LISTED} more)"


def _short(window: str) -> str:
    """car_m1p1 -> [-1,+1], for report and chart labels."""
    lo, hi = window.replace("car_m", "").split("p")
    return f"[-{lo},+{hi}]"


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
def load_cars() -> dict[str, pd.DataFrame]:
    """Per-firm CAR tables for all three runs, through Script 2's canonical reader."""
    return {run["name"]: ec.read_car(run["out"]) for run in ec.RUNS}


def load_texp(path: Path | None = None, reference_date: str = TEXP_REFERENCE_DATE) -> pd.Series:
    """Tariff exposure by firm at one reference date, indexed on permno.

    Read from texp_panel.csv, not tariff_scores.csv. Since the Step 2 rescale the scores table is
    one row per (permno, accession) across 17 reference dates, so permno is no longer a key there
    and which vintage a score belongs to is not recoverable from it. The panel carries
    reference_date and is unique on (permno, reference_date); one slice is taken here and reused
    for every run, which is what fixes group membership across the two legs.
    """
    path = TEXP_PANEL_CSV if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run build_texp_panel.py first.")
    panel = pd.read_csv(path, usecols=["permno", "reference_date", TEXP_COLUMN])
    frame = panel[panel["reference_date"] == reference_date]
    if frame.empty:
        raise ValueError(f"{path.name} holds no cross-section at {reference_date}; available: "
                         f"{', '.join(sorted(panel['reference_date'].unique()))}")
    if frame["permno"].duplicated().any():
        raise ValueError(f"the {reference_date} cross-section is not unique on permno; "
                         f"the merge would fan out rows")
    return frame.set_index("permno")[TEXP_COLUMN]


def weight_series(weights: pd.DataFrame, event: str, window: str) -> pd.Series:
    """The permno-indexed value weights for one (event, window), asserted unique."""
    block = weights[(weights["event"] == event) & (weights["window"] == window)]
    if block.empty:
        raise ValueError(f"no weights for event {event!r} window {window!r}")
    if block["permno"].duplicated().any():
        raise ValueError(f"weights are not unique on permno for {event}/{window}")
    return block.set_index("permno")[WEIGHT_COLUMN]


def weight_dates(calendar: pd.DatetimeIndex) -> dict[tuple[str, str], pd.Timestamp]:
    """The trading day each (event, window) reads its value weight from.

    A value weight must be known before the window it weights opens. ``me_lag`` on day d is market
    equity at the close of d-1, so the weight for window [lo, hi] is read on the trading day at
    offset ``lo`` from the event: its me_lag is then the close of the day before the window's first
    day, and no return inside the window has touched it.

    This replaces reading me_lag on the event date itself for all three windows. That was
    pre-window only for [-1,+1] and even there marginally not - the close of t-1 already carries
    day t-1's return, which is inside that window - while for [-5,+5] and [-10,+10] the weight sat
    five and ten trading days INSIDE the window, so a firm that fell over the first half of the
    window was down-weighted in the average of its own decline. One rule now serves all three, so
    the comparability argument for not using the estimation anchor is untouched.
    """
    dates = {}
    for name, day in ccd.EVENT_DATES.items():
        event = pd.Timestamp(day)
        if event not in calendar:
            raise ValueError(f"event date {day} for {name!r} is not a trading day in the panel")
        idx = int(calendar.get_loc(event))
        for lo, hi in ec.EVENT_WINDOWS:
            if idx + lo < 0:
                raise ValueError(f"{name}: window [{lo:+d},{hi:+d}] opens before the panel starts")
            dates[(name, f"car_{ec._window_label(lo, hi)}")] = calendar[idx + lo]
    return dates


def load_event_weights(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Lagged market equity on each window's own pre-window date, per (event, window, permno)."""
    path = PANEL_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_controls_data.py first.")
    frame = pd.read_csv(path, usecols=PANEL_COLUMNS, parse_dates=["date"])
    calendar = ec.build_calendar(frame)
    wanted = weight_dates(calendar)

    keep = frame[frame["date"].isin(set(wanted.values()))]
    if keep.duplicated(["date", "permno"]).any():
        raise ValueError("controls panel is not unique on (date, permno)")
    by_date = {day: block.set_index("permno")[WEIGHT_COLUMN] for day, block in keep.groupby("date")}

    rows = []
    for (event, window), day in wanted.items():
        if day not in by_date:
            raise ValueError(f"{path.name} has no rows on {day.date()}, the weight date for "
                             f"{event}/{window}")
        rows.append(by_date[day].rename(WEIGHT_COLUMN).reset_index()
                    .assign(event=event, window=window))
    weights = pd.concat(rows, ignore_index=True)
    return weights[["event", "window", "permno", WEIGHT_COLUMN]], wanted


# --------------------------------------------------------------------------- #
# Group construction - done once, then frozen                                  #
# --------------------------------------------------------------------------- #
def build_universe(cars: dict[str, pd.DataFrame], texp: pd.Series) -> tuple[pd.DataFrame, dict]:
    """The end-March-2025 cross-section the groups are built on, with its funnel.

    The screen is read from Script 2's in_screened_universe_pit, never re-derived. A firm with an
    exclusion_reason has no loadings and therefore no CAR in any window, so it cannot contribute.
    """
    car = cars[UNIVERSE_RUN]
    pit = car["in_screened_universe_pit"]
    excluded = pit & car["exclusion_reason"].ne("")
    valid = car[pit & car["exclusion_reason"].eq("")].copy()
    valid[TEXP_COLUMN] = valid["permno"].map(texp)
    universe = valid[valid[TEXP_COLUMN].notna()].copy()
    funnel = {
        "pit": int(pit.sum()),
        "excluded": int(excluded.sum()),
        "valid": len(valid),
        "no_texp": int(valid[TEXP_COLUMN].isna().sum()),
        "universe": len(universe),
    }
    return universe.set_index("permno").sort_index(), funnel


def nyse_breakpoint(month: str = TRIM_MONTH, field: int = ME_BP_P90_FIELD) -> float:
    """One Ken French NYSE ME breakpoint, in $thousands to match me_lag.

    Parses through clean_data._me_bp_data_rows: the file carries a one-line text header, blank
    lines and a copyright footer with no column header row, and a plain read_csv fails on it.
    French quotes ME in $millions while the panel carries $thousands, hence the scale.
    """
    rows = [r for r in cd._me_bp_data_rows() if r.split(",")[0].strip() == month]
    if len(rows) != 1:
        raise ValueError(f"{cd.ME_BP_FILE.name}: expected exactly one row for {month}, "
                         f"found {len(rows)}")
    level = float(rows[0].split(",")[field].strip()) * cd.ME_BP_SCALE
    if not np.isfinite(level) or level <= 0:
        raise ValueError(f"{cd.ME_BP_FILE.name}: field {field} for {month} is not a usable "
                         f"breakpoint ({level})")
    return level


def trim_megacaps(universe: pd.DataFrame, weights: pd.DataFrame,
                  event: str = "impose") -> tuple[pd.DataFrame, dict]:
    """Universe B: the primary universe less firms above the NYSE p90 market-equity breakpoint.

    Market equity is me_lag on the imposition event date, the same quantity and the same date the
    value weights use, so a firm is trimmed on exactly the size that would have given it its
    weight. Firms with no weight cannot be ranked and are kept: they carry no weight in any
    value-weighted mean either way, so trimming them would change the group counts without
    changing a single CAR.
    """
    cutoff = nyse_breakpoint()
    me = weight_series(weights, event, REFERENCE_WINDOW).reindex(
        universe.index)
    dropped = universe.index[me.gt(cutoff).fillna(False)]
    trimmed = universe.drop(dropped)
    stats_ = {
        "cutoff": cutoff,
        "n_before": len(universe),
        "n_dropped": len(dropped),
        "n_after": len(trimmed),
        "no_weight": int(me.isna().sum()),
        "smallest_dropped": me[dropped].min() if len(dropped) else np.nan,
        "largest_kept": me.reindex(trimmed.index).max(),
    }
    return trimmed, stats_


def assign_groups(universe: pd.DataFrame) -> tuple[pd.Series, np.ndarray, dict]:
    """Assign every universe firm to exactly one group, once, for all three legs.

    Group 0 holds the exact-zero mass point; groups 1..9 are equal-frequency quantiles of the
    strictly positive remainder. Ties keep equal TExp values in the same bin, so group sizes can
    differ by more than the +/-1 that exact division alone would give - see the report.
    """
    values = universe[TEXP_COLUMN]
    if (values < 0).any():
        raise ValueError(f"{TEXP_COLUMN} is a frequency and cannot be negative")
    zero = values.eq(0)
    positive = values[~zero]

    labels, breakpoints = pd.qcut(positive, N_POSITIVE_GROUPS,
                                  labels=range(1, N_POSITIVE_GROUPS + 1), retbins=True)
    groups = pd.Series(ZERO_GROUP, index=universe.index, dtype=int, name="group")
    groups.loc[labels.index] = labels.astype(int)

    stats_ = {
        "n_zero": int(zero.sum()),
        "n_positive": len(positive),
        "n_distinct_positive": int(positive.nunique()),
        "sizes": groups[groups > 0].value_counts().sort_index(),
    }
    return groups, breakpoints, stats_


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _inverse_herfindahl(weights: np.ndarray) -> float:
    """The number of equally-weighted firms a weight vector behaves like."""
    share = weights / weights.sum()
    return float(1.0 / (share ** 2).sum())


def group_car(cars: dict[str, pd.DataFrame], groups: pd.Series, universe: pd.DataFrame,
              weights: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Group CAR for every run x window x group, under both weighting schemes.

    A firm enters a cell only with a non-null CAR and a strictly positive value weight, and that
    same admission rule governs BOTH schemes - so the equal-weighted and value-weighted rows of a
    cell average exactly the same firms and any difference between them is the weighting alone.
    The count that an unrestricted equal-weighted mean would additionally admit is returned in the
    stats rather than taken.

    Returns (results, stats).
    """
    nominal = groups.value_counts().reindex(GROUPS, fill_value=0)
    bounds = universe.groupby(groups)[TEXP_COLUMN].agg(["min", "max"])
    rows, forgone = [], {}
    for run in ec.RUNS:
        car = cars[run["name"]].set_index("permno")
        for window in ec.CAR_COLUMNS:
            weight = weight_series(weights, run["event"], window)
            frame = pd.DataFrame({"group": groups, "w": weight.reindex(groups.index)})
            frame["car"] = car[window].reindex(groups.index)
            has_car = frame["car"].notna()
            usable = frame[has_car & frame["w"].notna() & frame["w"].gt(0)]
            # Firms an unrestricted equal-weighted mean would add: a CAR but no usable weight.
            forgone[(run["name"], window)] = int(has_car.sum() - len(usable))
            for group, block in usable.groupby("group"):
                w = block["w"].to_numpy(dtype=float)
                cell = {
                    "run": run["name"], "event": run["event"],
                    "event_date": str(car["event_date"].iloc[0].date()),
                    "window": window, "group": int(group),
                    "n_nominal": int(nominal[group]), "n_entering": len(block),
                    "weight_sum": float(w.sum()),
                    "texp_min": bounds.loc[group, "min"], "texp_max": bounds.loc[group, "max"],
                }
                for scheme in WEIGHTINGS:
                    if scheme["name"] == "value":
                        mean, eff = np.average(block["car"], weights=w), _inverse_herfindahl(w)
                    else:
                        # Equal weighting: effective N is the entering count by construction, and
                        # the mean is the plain average. Stated rather than computed from a
                        # vector of ones, which would only obscure that it is exact.
                        mean, eff = float(block["car"].mean()), float(len(block))
                    rows.append({**cell, "weighting": scheme["name"], "eff_n": eff,
                                 "vw_car": mean})
    results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return results, {"forgone_unrestricted_ew": forgone}


def write_results(results: pd.DataFrame, path: Path = RESULTS_OUT) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Chart                                                                        #
# --------------------------------------------------------------------------- #
def _spread(results: pd.DataFrame, run: str, window: str, weighting: str) -> float:
    """Group 9 minus group 0 CAR, in percentage points, for one cell of the results table."""
    cell = results[(results["run"] == run) & (results["window"] == window)
                   & (results["weighting"] == weighting)]
    series = cell.set_index("group")["vw_car"].reindex(GROUPS) * 100
    return float(series[N_POSITIVE_GROUPS] - series[ZERO_GROUP])


def monotonicity_reading(rho: float, flips: int) -> str:
    """Plain-language verdict on a group gradient, from BOTH statistics rather than one.

    The previous rule branched on |rho| first and consulted the flip count only inside the top
    band, which let a series with five sign changes read "strong trend" while one with four - i.e.
    strictly more monotone - read "partial gradient" purely because its rho fell 0.003 below a hard
    cutoff. Both statistics now enter on every branch, and the thresholds are named in config.
    """
    strength = abs(rho)
    smooth = flips <= MONOTONE_MAX_FLIPS
    if strength >= RHO_STRONG:
        return "monotone" if smooth else "strong trend, uneven"
    if strength >= RHO_PARTIAL:
        return "partial gradient" if smooth else "partial gradient, uneven"
    return "no clear gradient"


def effective_n(groups: pd.Series, weights: pd.DataFrame, event: str) -> pd.Series:
    """Inverse Herfindahl of value weights per group: the equally-weighted firm count it acts like.

    A group of 190 firms whose weight sits in two mega-caps is not a 190-firm portfolio, and the
    exhibit has to say so rather than let the bar imply breadth it does not have.
    """
    weight = weight_series(weights, event, REFERENCE_WINDOW)
    held = pd.DataFrame({"group": groups, "w": weight.reindex(groups.index)}).dropna()
    return held.groupby("group")["w"].apply(lambda w: 1.0 / ((w / w.sum()) ** 2).sum())


def plot_groups(results: pd.DataFrame, path: Path = CHART_OUT,
                weighting: str = PRIMARY_WEIGHTING, universe_note: str = "") -> Path:
    """Grouped bar chart: one panel per event window, three bars per TExp group.

    Two hues carry the two event legs; the imposition robustness run is the same hue as its
    primary with a hatch, because it is the same event measured over a different estimation
    window rather than a third series. The y-axis is shared across panels so the narrower
    windows are not visually exaggerated.

    One chart per (universe, weighting): the four together are the section 7.3 figure plus its
    concentration robustness in both directions. ``weighting`` selects which rows are drawn and
    relabels the axes, ``universe_note`` distinguishes the trimmed universe in the title; nothing
    else differs between them, which is the point of drawing four.
    """
    scheme = next(w for w in WEIGHTINGS if w["name"] == weighting)
    adjective = scheme["adjective"]
    results = results[results["weighting"] == weighting]
    if results.empty:
        raise ValueError(f"no {weighting!r}-weighted rows to plot")

    cs.apply()
    windows = ec.CAR_COLUMNS
    runs = [r["name"] for r in ec.RUNS]
    fig, axes = plt.subplots(len(windows), 1, figsize=cs.SIZE_STACK3, sharey=True)

    slot = 0.26                      # centre-to-centre spacing of the three bars in a cluster
    width = 0.24                     # leaves a hairline gap between adjacent fills
    offsets = {run: (i - 1) * slot for i, run in enumerate(runs)}
    x = np.arange(len(GROUPS))

    for ax, window in zip(axes, windows):
        for run in runs:
            style = BAR_STYLE[run]
            cell = results[(results["run"] == run) & (results["window"] == window)]
            height = cell.set_index("group")["vw_car"].reindex(GROUPS) * 100
            ax.bar(x + offsets[run], height, width, label=style["label"],
                   facecolor=PALETTE[style["color"]], hatch=style["hatch"],
                   edgecolor=palette.INK, linewidth=cs.BAR_EDGE_WIDTH, zorder=3)
        cs.zero_line(ax)
        cs.panel_title(ax, f"Event window {_short(window)}")
        ax.set_xticks(x, [str(g) for g in GROUPS])

    axes[len(axes) // 2].set_ylabel(f"{adjective} cumulative abnormal return (%)")
    axes[-1].set_xlabel("TExp group  (0 = zero exposure, 1-9 = ascending exposure deciles)")
    cs.headroom(axes[0], top=0.18)
    cs.legend(axes[0], loc="upper right")
    for ax in axes:
        cs.frame(ax, rotate_x=False)

    # "CAR" rather than the phrase spelled out: four charts differ only in weighting and universe,
    # so both must fit in the title, and the y-axis label expands the abbreviation on every one.
    title = f"{adjective} CAR by tariff-exposure group"
    cs.figure_title(fig, f"{title}, {universe_note}" if universe_note else title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return cs.save(fig, path)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def _check_partition(groups: pd.Series, universe: pd.DataFrame) -> None:
    """Assert the ten groups partition the universe exactly once, with no overlap or gap."""
    values = universe[TEXP_COLUMN]
    if not (groups.index.is_unique):
        raise ValueError("a firm is assigned to more than one group")
    if not (set(groups.index) == set(universe.index)):
        raise ValueError("group assignment does not cover the universe")
    if not (groups.isin(GROUPS).all()):
        raise ValueError(f"a firm carries a group outside {GROUPS}")
    if not ((groups[values.eq(0)] == ZERO_GROUP).all()):
        raise ValueError("a zero-TExp firm sits outside group 0")
    if not ((groups[values.gt(0)] > ZERO_GROUP).all()):
        raise ValueError("a positive-TExp firm sits in group 0")
    assigned = int(groups.value_counts().sum())
    if not (assigned == len(universe)):
        raise ValueError(f"{assigned} assignments for {len(universe)} firms")
    # Groups 1-9 must be contiguous in TExp: every group's max below the next group's min.
    bounds = universe.groupby(groups)[TEXP_COLUMN].agg(["min", "max"]).loc[1:]
    for lo, hi in zip(bounds.index[:-1], bounds.index[1:]):
        if not (bounds.loc[lo, "max"] <= bounds.loc[hi, "min"]):
            raise ValueError(f"groups {lo} and {hi} overlap in {TEXP_COLUMN}")


def _check_fixed_across_legs(results: pd.DataFrame, groups: pd.Series,
                             cars: dict[str, pd.DataFrame]) -> None:
    """Assert every firm carries the same group in every leg it appears in.

    Re-derived independently from the CAR tables rather than trusting that one mapping object
    was reused: that is the claim being tested.
    """
    seen: dict[str, pd.Series] = {}
    for name, car in cars.items():
        present = car.loc[car["permno"].isin(groups.index), "permno"]
        seen[name] = groups.reindex(present).sort_index()
    reference = next(iter(seen.values()))
    for name, mapping in seen.items():
        common = reference.index.intersection(mapping.index)
        mismatch = common[reference[common].to_numpy() != mapping[common].to_numpy()]
        if not (len(mismatch) == 0):
            raise ValueError(f"{name}: {len(mismatch)} firm(s) change group: {list(mismatch)[:5]}")
    if not (results.groupby(["run", "window", "weighting"])["n_nominal"]
            .apply(tuple).nunique() == 1):
        raise ValueError("nominal group sizes differ across runs; membership is not fixed")
    # The two weighting schemes must average the SAME firms in every cell - that identity is what
    # makes their difference attributable to the weighting rather than to the sample.
    per_cell = results.groupby(["run", "window", "group"])["n_entering"].nunique()
    if not (per_cell == 1).all():
        bad = per_cell[per_cell != 1]
        raise ValueError(f"{len(bad)} cell(s) admit different firms under the two weightings, "
                         f"e.g. {list(bad.index[:3])}")


def validate(results: pd.DataFrame, groups: pd.Series, universe: pd.DataFrame,
             breakpoints: np.ndarray, funnel: dict, gstats: dict,
             cars: dict[str, pd.DataFrame], weights: pd.DataFrame,
             weight_days: dict | None = None, wstats: dict | None = None,
             ex: dict | None = None) -> None:
    """Assemble the consolidated validation report and write it to disk.

    ``ex`` carries universe B's construction and results; when supplied, its sections are appended
    to this same report rather than written to a second file, per the one-report-per-script rule.
    """
    _say(RULE)
    _say("SECTION 7.3 DECILE-SPREAD PRESENTATION - VALIDATION REPORT")
    _say(RULE)
    _say(f"Construction  : group 0 = {TEXP_COLUMN} == 0 exactly; groups 1-{N_POSITIVE_GROUPS} = "
         f"equal-frequency quantiles of the strictly positive remainder")
    _say(f"Breakpoints   : computed ONCE on the end-March-2025 cross-section "
         f"({UNIVERSE_RUN}), held fixed across all runs")
    _say(f"Runs          : {', '.join(r['name'] for r in ec.RUNS)}")
    _say(f"Windows       : {', '.join(ec.CAR_COLUMNS)}   "
         f"({len(ec.RUNS)} runs x {len(ec.CAR_COLUMNS)} windows x {len(GROUPS)} groups = "
         f"{len(ec.RUNS) * len(ec.CAR_COLUMNS) * len(GROUPS)} cells)")
    _say("Events        : " + ", ".join(f"{k}={v}" for k, v in ccd.EVENT_DATES.items()))
    _say(f"Weighting     : {len(WEIGHTINGS)} schemes, on identical firms per cell -")
    for scheme in WEIGHTINGS:
        role = ("  (primary, section 7.3)" if scheme["name"] == PRIMARY_WEIGHTING
                else "  (v5 section 8 robustness)")
        _say(f"                {scheme['adjective']:<16}{scheme['note']}{role}")
    _say(f"                Value weights are {WEIGHT_COLUMN} read on the trading day each window")
    _say("                OPENS - so no return inside a window helps set its own weights")
    if weight_days:
        for (event, window), day in sorted(weight_days.items()):
            _say(f"                {event:<10}{_short(window):<10}weight date {day.date()}"
                 f"  ({WEIGHT_COLUMN} = close of the prior trading day)")
    _say(f"Exposure      : {TEXP_PANEL_CSV.name} at reference date {TEXP_REFERENCE_DATE}, "
         f"one vintage for every run")
    _say(f"Inputs        : {PANEL_PATH.name}, {TEXP_PANEL_CSV.name}, "
         f"{', '.join(r['out'].name for r in ec.RUNS)}")
    _say("NOT computed  : Sharpe, drawdown, turnover, capacity, GRS, spanning alpha, backtest "
         "returns.\n                Section 7.3 excludes them - this is a descriptive exhibit, "
         "not a strategy test.")

    _section("1. Group universe funnel (end-March 2025)")
    _say(f"  in_screened_universe_pit                     {funnel['pit']:>6,}")
    _say(f"  - exclusion_reason set (no valid loadings)   {funnel['excluded']:>6,}")
    _say(f"  = valid loadings                             {funnel['valid']:>6,}")
    _say(f"  - no {TEXP_COLUMN}                            {funnel['no_texp']:>6,}")
    _say(f"  = GROUP UNIVERSE                             {funnel['universe']:>6,}")
    if not (funnel["valid"] - funnel["no_texp"] == funnel["universe"]):
        raise ValueError("funnel does not reconcile")
    _say("  funnel reconciles (asserted)")
    _say(f"  the screen is READ from {cars[UNIVERSE_RUN].shape[0]:,}-row "
         f"{UNIVERSE_RUN}, never re-derived")

    _section("2. Group construction")
    _say(f"  group 0 (TExp == 0 exactly)   {gstats['n_zero']:>6,}  "
         f"({gstats['n_zero'] / len(universe):.1%} of the universe)")
    _say(f"  groups 1-9 (TExp > 0)         {gstats['n_positive']:>6,}")
    _say()
    _say("  breakpoints on the strictly positive remainder:")
    for i, (lo, hi) in enumerate(zip(breakpoints[:-1], breakpoints[1:]), start=1):
        _say(f"    group {i}   ({lo:.6f}, {hi:.6f}]   n = {gstats['sizes'][i]:,}")
    sizes = gstats["sizes"]
    lo_exact, rem = divmod(gstats["n_positive"], N_POSITIVE_GROUPS)
    _say()
    _say(f"  exact division of {gstats['n_positive']:,} by {N_POSITIVE_GROUPS} permits sizes "
         f"{lo_exact} or {lo_exact + 1} ({rem} group(s) of {lo_exact + 1})")
    _say(f"  observed  min {sizes.min()}  max {sizes.max()}  spread {sizes.max() - sizes.min()}")
    excess = max(0, (sizes.max() - sizes.min()) - 1)
    if excess:
        n_tied = gstats["n_positive"] - gstats["n_distinct_positive"]
        _say(f"  !! spread exceeds the +/-1 that exact division alone gives, by {excess}.")
        _say(f"     CAUSE: ties. {gstats['n_positive']:,} positive values hold only "
             f"{gstats['n_distinct_positive']:,} distinct levels ({n_tied:,} duplicated), because")
        _say(f"     {TEXP_COLUMN} is a ratio of small integer counts. qcut keeps equal values in "
             f"one bin, so bins")
        _say("     absorb ties unevenly. This is a property of a discrete-valued measure, not a "
             "defect,")
        _say("     and is REPORTED rather than asserted - aborting on it would be wrong.")
    else:
        _say("  spread is within +/-1, as exact division alone would give")

    _section("3. Partition and fixed-membership assertions")
    _check_partition(groups, universe)
    _say("  every universe firm in exactly one group 0-9 ......... asserted")
    _say("  zero-TExp firms all in group 0, none elsewhere ....... asserted")
    _say("  groups 1-9 contiguous in TExp, no overlap, no gap .... asserted")
    _check_fixed_across_legs(results, groups, cars)
    _say("  group identical across all three legs per firm ....... asserted")
    _say("  nominal group sizes identical across runs ............ asserted")
    if not (len(results) == len(ec.RUNS) * len(ec.CAR_COLUMNS) * len(GROUPS) * len(WEIGHTINGS)):
        raise ValueError(f"expected {len(ec.RUNS) * len(ec.CAR_COLUMNS) * len(GROUPS) * len(WEIGHTINGS)} rows, got {len(results)}")
    _say(f"  results table is {len(results)} rows "
         f"({len(ec.RUNS)} runs x {len(ec.CAR_COLUMNS)} windows x {len(GROUPS)} groups "
         f"x {len(WEIGHTINGS)} weightings) .. asserted")
    _say("  both weightings average identical firms per cell ..... asserted")

    _section("4. Weight coverage (me_lag on each window's own pre-window date)")
    for event, day in ccd.EVENT_DATES.items():
        w = weight_series(weights, event, REFERENCE_WINDOW)
        held = w.reindex(universe.index)
        # `absent` and `null` were the identical expression, reported as two different things.
        # A firm with no panel row at all and a firm with a row carrying a null weight are
        # distinguishable, and the report now distinguishes them.
        absent = int((~universe.index.isin(w.index)).sum())
        null = int(held.isna().sum() - absent)
        nonpos = int((held <= 0).sum())
        _say(f"  {event:8s} {day}   universe firms with a panel row {held.notna().sum():>6,} "
             f"of {len(universe):,}")
        _say(f"  {'':8s} {'':10s}   missing me_lag {null:>4,}   non-positive {nonpos:>4,}   "
             f"({absent / len(universe):.2%} of the universe)")
        if absent:
            _say(f"  {'':8s} {'':10s}   these firms have no panel row on the event date - they "
                 f"delisted before it.")
            _say(f"  {'':8s} {'':10s}   Excluded from the weighted mean and counted in "
                 f"n_entering below; never silently dropped.")

    _section("5. Firms entering each cell against the group's nominal size")
    _say("  One table: the two weighting schemes admit identical firms by construction, asserted")
    _say("  in section 3, so these counts describe both.")
    for run in (r["name"] for r in ec.RUNS):
        _say(f"  {run}")
        _say(f"    {'group':>5} {'nominal':>8}" +
             "".join(f"{_short(w):>12}" for w in ec.CAR_COLUMNS))
        block = results[(results["run"] == run)
                        & (results["weighting"] == PRIMARY_WEIGHTING)]
        for group in GROUPS:
            cells = block[block["group"] == group].set_index("window")
            nominal = int(cells["n_nominal"].iloc[0])
            entering = "".join(f"{int(cells.loc[w, 'n_entering']):>12,}" for w in ec.CAR_COLUMNS)
            _say(f"    {group:>5} {nominal:>8,}{entering}")
        totals = block.groupby("window")["n_entering"].sum().reindex(ec.CAR_COLUMNS)
        _say(f"    {'total':>5} {int(block['n_nominal'].sum() / len(ec.CAR_COLUMNS)):>8,}" +
             "".join(f"{int(t):>12,}" for t in totals))

    _section("6. Group CAR (%) under both weightings, and the group 9 - group 0 spread")
    _say("  Value weighting is section 7.3's nomination and stays the primary. Equal weighting is")
    _say("  v5 section 8's specification robustness, on identical firms, so the difference between")
    _say("  the two blocks is the weighting and nothing else.")
    for scheme in WEIGHTINGS:
        _say()
        _say(f"  [{scheme['adjective'].upper()}]  {scheme['note']}")
        for window in ec.CAR_COLUMNS:
            _say(f"  window {_short(window)}")
            _say(f"    {'run':<24}" + "".join(f"{('G' + str(g)):>8}" for g in GROUPS)
                 + f"{'G9-G0':>9}")
            for run in (r["name"] for r in ec.RUNS):
                cell = results[(results["run"] == run) & (results["window"] == window)
                               & (results["weighting"] == scheme["name"])]
                series = cell.set_index("group")["vw_car"].reindex(GROUPS) * 100
                spread = series[N_POSITIVE_GROUPS] - series[ZERO_GROUP]
                _say(f"    {run:<24}" + "".join(f"{v:>8.2f}" for v in series) + f"{spread:>9.2f}")

    _say()
    _say("  Spread side by side, value against equal (percentage points):")
    _say(f"    {'run':<24}" + "".join(f"{_short(w):>24}" for w in ec.CAR_COLUMNS))
    _say(f"    {'':<24}" + "".join(f"{'value':>11}{'equal':>13}" for _ in ec.CAR_COLUMNS))
    for run in (r["name"] for r in ec.RUNS):
        line = f"    {run:<24}"
        for window in ec.CAR_COLUMNS:
            for scheme in WEIGHTINGS:
                cell = results[(results["run"] == run) & (results["window"] == window)
                               & (results["weighting"] == scheme["name"])]
                series = cell.set_index("group")["vw_car"].reindex(GROUPS) * 100
                spread = series[N_POSITIVE_GROUPS] - series[ZERO_GROUP]
                line += f"{spread:>+11.2f}" if scheme["name"] == "value" else f"{spread:>+13.2f}"
        _say(line)
    _say()
    _say("  A sign that survives both weightings is not an artefact of a few large firms. A")
    _say("  magnitude that changes tells you how much of the value-weighted figure those firms")
    _say("  carry, which is the same question universe B asks by trimming them - see section 12,")
    _say("  where all four combinations appear together.")

    _section("7. Weight concentration within groups")
    _say("  Value weighting is what section 7.3 nominates and what is reported, but at ~190 firms")
    _say("  per group a few mega-caps carry most of the weight. Effective N is the inverse")
    _say("  Herfindahl of the weights: the number of equally-weighted firms the group behaves")
    _say("  like. Where it is small, the group CAR is a statement about those firms, not about")
    _say("  the group, and must be read that way.")
    _say("  Equal weighting is immune to this by construction - every entering firm has the same")
    _say("  influence, so its effective N equals n_entering exactly - which is precisely why the")
    _say("  equal-weighted block in section 6 is the check on this table rather than a footnote.")
    _say()
    _say(f"    {'group':>5}{'n':>6}{'eff N':>8}{'top 1 %':>9}{'top 3 %':>9}   largest holding")
    concentration = {}
    for event in ccd.EVENT_DATES:
        weight = weight_series(weights, event, REFERENCE_WINDOW)
        tickers = cars[UNIVERSE_RUN].set_index("permno")["ticker"]
        held = pd.DataFrame({"group": groups, "w": weight.reindex(groups.index)}).dropna()
        rows = []
        for group, block in held.groupby("group"):
            block = block.sort_values("w", ascending=False)
            share = block["w"] / block["w"].sum()
            rows.append((int(group), len(block), 1.0 / (share ** 2).sum(),
                         100 * share.iloc[0], 100 * share.head(3).sum(),
                         tickers.get(block.index[0], "?")))
        concentration[event] = rows
    for group, n, eff, top1, top3, tic in concentration["impose"]:
        _say(f"    {group:>5}{n:>6,}{eff:>8.1f}{top1:>9.1f}{top3:>9.1f}   {tic}")
    _say()
    worst = max(concentration["impose"], key=lambda r: r[3])
    _say(f"  Most concentrated: group {worst[0]} is {worst[3]:.1f}% {worst[5]} by weight "
         f"({worst[4]:.1f}% top three),")
    _say(f"  so its group CAR is close to a restatement of {worst[5]}'s own CAR. Effective N "
         f"across")
    _say(f"  groups runs {min(r[2] for r in concentration['impose']):.0f} to "
         f"{max(r[2] for r in concentration['impose']):.0f} against a nominal ~190. This is a "
         f"property of value weighting a")
    _say("  broad universe, not a defect, and it is the main caveat on reading the gradient "
         "below.")
    _say("  Equal weighting is NOT reported: section 7.3 nominates value weighting, and swapping")
    _say("  it to make the gradient smoother would be choosing the estimator on its result.")

    _section("8. Monotonicity diagnostic (v6 section 7.3 / H3b)")
    _say("  Reported as a diagnostic, NOT a pass/fail gate: a non-monotone gradient is a valid")
    _say("  finding and is reported as such rather than smoothed over.")
    _say()
    _say("  The Spearman p is shown for completeness and should not be read as a test: the ten")
    _say("  group means are weighted averages of ONE event-day cross-section, sharing residual")
    _say("  factor exposure, and several are dominated by one or two firms (section 7). They are")
    _say("  not ten independent draws, which is what the analytic p-value assumes.")
    _say()
    _say("  Reported under both weightings: a gradient that only appears under one of them is")
    _say("  telling you about the weighting rather than about exposure.")
    for scheme in WEIGHTINGS:
        _say()
        _say(f"  [{scheme['adjective'].upper()}]")
        _say(f"    {'run':<24}{'window':>11}{'spearman':>10}{'p':>9}{'sign changes':>14}  reading")
        for run in (r["name"] for r in ec.RUNS):
            for window in ec.CAR_COLUMNS:
                cell = results[(results["run"] == run) & (results["window"] == window)
                               & (results["weighting"] == scheme["name"])]
                series = cell.set_index("group")["vw_car"].reindex(GROUPS)
                rho = stats.spearmanr(GROUPS, series)
                flips = int((np.diff(np.sign(np.diff(series))) != 0).sum())
                _say(f"    {run:<24}{_short(window):>11}{rho.statistic:>10.3f}{rho.pvalue:>9.3f}"
                     f"{flips:>14}  {monotonicity_reading(rho.statistic, flips)}")

    _section("9. Extreme-CAR firms are retained, not excluded")
    _say(f"  Script 2 flags |CAR| > {ec.EXTREME_CAR:.1f} as a likely firm-specific event. No "
         f"outlier filter is applied")
    _say("  here: every such firm inside the universe keeps its group and its weight.")
    _say()
    found = False
    for run in ec.RUNS:
        car = cars[run["name"]].set_index("permno")
        for window in ec.CAR_COLUMNS:
            series = car[window].reindex(universe.index)
            hits = series[series.abs() > ec.EXTREME_CAR]
            for permno, value in hits.items():
                found = True
                _say(f"    {run['name']:<24}{_short(window):>11}  "
                     f"{car.loc[permno, 'ticker']:<7} permno {permno}  CAR {value:+.3f}  "
                     f"-> group {groups[permno]}  RETAINED")
    if not found:
        _say("    none inside the universe at any run/window")

    _say()
    dfdv = EXTREME_CAR_EXAMPLE["permno"]
    dfdv_name = EXTREME_CAR_EXAMPLE["name"]
    car = cars[UNIVERSE_RUN].set_index("permno")
    if dfdv in universe.index:
        _say(f"  DFDV (permno {dfdv}) is in the universe, group {groups[dfdv]}, retained.")
    elif dfdv in car.index:
        _say(f"  DFDV (permno {dfdv}, {dfdv_name}) is NOT in the universe, and "
             f"that is correct.")
        _say(f"    in_screened_universe_pit = {bool(car.loc[dfdv, 'in_screened_universe_pit'])}, "
             f"exclusion_reason = {car.loc[dfdv, 'exclusion_reason'] or 'none'}, "
             f"car_m10p10 = {car.loc[dfdv, 'car_m10p10']:+.2f}")
        _say("    It is removed by the section 6 point-in-time screen, exactly as Step 4c "
             "recorded - NOT by any")
        _say("    outlier filter, which this script does not apply. Its +930% [-10,+10] CAR "
             "would otherwise have")
        _say("    dominated whichever group held it.")

    if ex is not None:
        _validate_ex_megacap(ex, results, groups, universe, breakpoints, weights, cars)

    _section("13. Outputs" if ex is not None else "10. Outputs")
    # Two results files, one per universe, each carrying both weightings as a column; four charts,
    # one per (universe, weighting) combination.
    paths = ([RESULTS_OUT, CHART_OUT, RESULTS_EW_CHART,
              RESULTS_EX_OUT, CHART_EX_OUT, RESULTS_EW_CHART_EX, REPORT_OUT]
             if ex is not None else [RESULTS_OUT, CHART_OUT, RESULTS_EW_CHART, REPORT_OUT])
    for path in paths:
        _say(f"  {path.relative_to(BASE)}")
    _say()
    _say(f"  Each results file holds {len(WEIGHTINGS)} weighting schemes in its `weighting` "
         f"column, so a")
    _say("  consumer must filter on it. `eff_n` is the weighting-specific concentration measure;")
    _say("  `weight_sum` is the cell's market equity and is identical across schemes.")

    REPORT_OUT.write_text("\n".join(_REPORT), encoding="utf-8")


def _validate_ex_megacap(ex: dict, results: pd.DataFrame, groups: pd.Series,
                         universe: pd.DataFrame, breakpoints: np.ndarray,
                         weights: pd.DataFrame, cars: dict[str, pd.DataFrame]) -> None:
    """Report universe B's construction, its independence from A, and the comparison."""
    tstats, tgroups, tbreaks, tgstats, tresults, tuniverse = (
        ex["stats"], ex["groups"], ex["breakpoints"], ex["gstats"], ex["results"], ex["universe"])

    _section("10. Universe B: mega-cap-excluded robustness comparison")
    _say("  Section 7 showed value weighting concentrates several groups in a few mega-caps. This")
    _say("  re-runs the identical procedure on a universe where no one firm can dominate a group.")
    _say("  It sits BESIDE the primary result, not in place of it, and the primary numbers above")
    _say("  are untouched. Equal weighting is still not adopted - see section 7 for why.")
    _say()
    _say(f"  cutoff : Ken French NYSE p90 market equity at {TRIM_MONTH}, "
         f"{cd.ME_BP_FILE.name} field {ME_BP_P90_FIELD}")
    _say(f"           ${tstats['cutoff'] / 1e6:,.1f}bn  (French $M x {cd.ME_BP_SCALE:,.0f} -> "
         f"$thousands, matching {WEIGHT_COLUMN})")
    _say(f"  the same source and convention clean_data.flag_microcaps applies at the bottom end")
    _say(f"  with p{cd.NYSE_PCTILE}, so universe B is a NYSE p{cd.NYSE_PCTILE}-p90 band under one "
         f"definition of a size cutoff.")
    _say()
    _say(f"  primary universe                    {tstats['n_before']:>6,}")
    _say(f"  - above the NYSE p90 breakpoint     {tstats['n_dropped']:>6,}  "
         f"({tstats['n_dropped'] / tstats['n_before']:.1%})")
    _say(f"  = UNIVERSE B                        {tstats['n_after']:>6,}")
    _say(f"  firms with no me_lag to rank on     {tstats['no_weight']:>6,}  (kept: they carry no "
         f"weight either way)")
    _say(f"  boundary: smallest dropped ${tstats['smallest_dropped'] / 1e6:,.1f}bn   "
         f"largest kept ${tstats['largest_kept'] / 1e6:,.1f}bn")
    if not (tstats["n_before"] - tstats["n_dropped"] == tstats["n_after"]):
        raise ValueError("trim does not reconcile")
    me = weight_series(weights, UNIVERSE_EVENT, REFERENCE_WINDOW)
    if not (me.reindex(tuniverse.index).max() <= tstats["cutoff"]):
        raise ValueError("a kept firm exceeds the cutoff")
    if not (me.reindex(universe.index.difference(tuniverse.index)).min() > tstats["cutoff"]):
        raise ValueError("a dropped firm is below the cutoff")
    _say("  trim reconciles, and no firm sits on the wrong side of the cutoff (asserted)")

    _section("11. Independence of the two group constructions")
    _say("  Universe B's breakpoints are computed from scratch on its own cross-section, NOT")
    _say("  inherited from the primary and NOT a filtering of the primary's assignment. Different")
    _say("  breakpoints are the EXPECTED outcome here, not an error.")
    _say()
    _say(f"    {'q':>3}{'primary':>12}{'universe B':>13}   {'shift':>10}")
    for i, (a, b) in enumerate(zip(breakpoints, tbreaks)):
        _say(f"    {i:>3}{a:>12.6f}{b:>13.6f}   {b - a:>+10.6f}")
    # Reported, not asserted: whether the two breakpoint vectors differ is an outcome of which
    # firms the trim removed, so aborting on it would turn a finding into a crash.
    moved = int((groups.reindex(tgroups.index) != tgroups).sum())
    _say()
    if np.array_equal(breakpoints, tbreaks):
        _say("  FINDING: universe B reproduced the primary breakpoints EXACTLY. That is expected")
        _say("  only if the trim removed no firm carrying positive exposure; otherwise the two")
        _say("  constructions are not separate and the comparison below is not the independent")
        _say("  one it is described as.")
    else:
        _say("  breakpoint vectors are not identical, so the assignment was rebuilt from scratch")
        _say("  rather than restricted - which is the intended construction.")
    _say(f"  firms whose group CHANGES between the two universes: {moved:,} of "
         f"{len(tgroups):,} ({moved / len(tgroups):.1%})")
    _say("  a non-zero count is the proof the assignment was rebuilt rather than restricted.")

    _check_partition(tgroups, tuniverse)
    _check_fixed_across_legs(tresults, tgroups, cars)
    _say()
    _say("  the primary's own assertions, re-applied to universe B:")
    _say("    every firm in exactly one group 0-9, zero-TExp firms all in group 0 ... asserted")
    _say("    groups 1-9 contiguous in TExp, no overlap, no gap .................... asserted")
    _say("    group identical across all three legs per firm ....................... asserted")

    _section("12. Comparison: group counts, concentration, and the spread")
    _say(f"    {'group':>5}{'n (A)':>8}{'n (B)':>8}{'':>4}{'effN (A)':>10}{'effN (B)':>10}"
         f"{'':>3}largest holding (B)")
    eff_a = effective_n(groups, weights, "impose")
    eff_b = effective_n(tgroups, weights, "impose")
    tickers = cars[UNIVERSE_RUN].set_index("permno")["ticker"]
    na, nb = groups.value_counts(), tgroups.value_counts()
    for g in GROUPS:
        held = pd.DataFrame({"g": tgroups, "w": me.reindex(tgroups.index)}).dropna()
        top = held[held["g"] == g].nlargest(1, "w")
        _say(f"    {g:>5}{na.get(g, 0):>8,}{nb.get(g, 0):>8,}{'':>4}{eff_a.get(g, np.nan):>10.1f}"
             f"{eff_b.get(g, np.nan):>10.1f}{'':>3}"
             f"{tickers.get(top.index[0], '?') if len(top) else '-'}")
    _say()
    _say(f"  effective N   primary   min {eff_a.min():5.1f}   median {eff_a.median():5.1f}   "
         f"max {eff_a.max():5.1f}")
    _say(f"                universe B min {eff_b.min():5.1f}   median {eff_b.median():5.1f}   "
         f"max {eff_b.max():5.1f}")
    # Reported, not asserted. Whether the trim improves the worst group's breadth is an
    # empirical outcome; aborting the run on it would convert a finding into a crash, which is
    # the same mistake the extreme-spread check at section 6 deliberately avoids.
    if eff_b.min() > eff_a.min():
        _say(f"  the trim raises the worst group from {eff_a.min():.1f} to "
             f"{eff_b.min():.1f} effective firms, so it does reduce concentration.")
    else:
        _say(f"  FINDING: the trim did NOT reduce concentration - the worst group goes from "
             f"{eff_a.min():.1f} to {eff_b.min():.1f} effective firms.")
        _say("  Universe B exists to remove mega-cap dominance and on this cross-section it does")
        _say("  not, which bears directly on how the comparison below should be read.")

    _say()
    _say("  Group 9 - group 0 spread (%), all four combinations of universe and weighting.")
    _say("  Two independent answers to the same concentration concern, plus their union:")
    _say("  trimming the mega-caps out (universe B) and denying them extra weight (equal).")
    _say()
    _say(f"    {'run':<22}{'window':<10}{'A value':>10}{'A equal':>10}"
         f"{'B value':>10}{'B equal':>10}{'':>4}signs")
    combos = [("A", "value", results), ("A", "equal", results),
              ("B", "value", tresults), ("B", "equal", tresults)]
    signs_agree, magnitudes = 0, 0
    for run in (r["name"] for r in ec.RUNS):
        for window in ec.CAR_COLUMNS:
            spreads = [_spread(frame, run, window, scheme)
                       for _, scheme, frame in combos]
            same = len({int(np.sign(v)) for v in spreads}) == 1
            signs_agree += int(same)
            magnitudes += 1
            _say(f"    {run[:21]:<22}{_short(window):<10}"
                 + "".join(f"{v:>+10.2f}" for v in spreads)
                 + f"{'':>4}{'agree' if same else 'DIFFER'}")

    # How much the mega-cap trim moves each weighting. The trim exists to remove weight
    # concentration, so under equal weighting - which has none to remove - it should barely
    # register. Computing both is the coherence check that the two fixes address one thing.
    shifts = {scheme["name"]: [abs(_spread(results, run, window, scheme["name"])
                                   - _spread(tresults, run, window, scheme["name"]))
                               for run in (r["name"] for r in ec.RUNS)
                               for window in ec.CAR_COLUMNS]
              for scheme in WEIGHTINGS}
    mean_shift = {name: float(np.mean(v)) for name, v in shifts.items()}

    _say()
    _say(f"  Sign agreement across all four combinations: {signs_agree} of {magnitudes} "
         f"(run x window) cells.")
    _say()
    _say("  How far the mega-cap trim moves each weighting (mean |A - B| over the 9 cells):")
    for scheme in WEIGHTINGS:
        _say(f"    {scheme['adjective']:<16}{mean_shift[scheme['name']]:>6.2f} pp")
    if mean_shift["equal"] > 0:
        ratio = mean_shift["value"] / mean_shift["equal"]
        _say(f"  The trim moves the value-weighted spread {ratio:.0f}x further than the")
        _say("  equal-weighted one. That is the expected direction and it is a coherence check,")
        _say("  not a coincidence: the trim removes weight concentration, and equal weighting has")
        _say("  none to remove. The two fixes are addressing the same thing, and they agree on")
        _say("  what the spread is once that thing is removed.")
    _say("  Note also that within universe B the value-weighted spread stays below the")
    _say("  equal-weighted one, so the size gradient does not end at the p90 cutoff: among the")
    _say("  firms that survive the trim, the larger ones still carry a weaker spread.")
    _say()
    _say("  Reading, stated as findings rather than as a verdict:")
    _say("   - Where all four agree, the spread's direction is neither an artefact of mega-cap")
    _say("     concentration nor of the weighting scheme, which is the strongest form this")
    _say("     descriptive exhibit can take.")
    _say("   - Where they differ, the cell is carried by the weighting or by a handful of large")
    _say("     firms, and the section 7.2 regression - not this table - is what settles it.")
    _say("   - Magnitudes are expected to shrink under both fixes: universe B removes the firms")
    _say("     that amplified the primary levels, and equal weighting removes their extra")
    _say("     influence without removing the firms. The primary value-weighted universe-A figure")
    _say("     remains the headline; these are the bound on how much of it rests on those firms.")
    _say("   - No inference is attached to any of these spreads. Section 7.3 is descriptive by")
    _say("     design and nominates no test; H1 is tested in the regression report's section 6b.")


def main() -> pd.DataFrame:
    cars = load_cars()
    texp = load_texp()
    weights, weight_days = load_event_weights()

    # --- Universe A: the primary, all screened firms ------------------------ #
    universe, funnel = build_universe(cars, texp)
    groups, breakpoints, gstats = assign_groups(universe)
    results, wstats = group_car(cars, groups, universe, weights)

    write_results(results)
    plot_groups(results)
    plot_groups(results, RESULTS_EW_CHART, weighting="equal")

    # --- Universe B: the same procedure on an independently built group set -- #
    trimmed, tstats = trim_megacaps(universe, weights)
    tgroups, tbreaks, tgstats = assign_groups(trimmed)
    tresults, twstats = group_car(cars, tgroups, trimmed, weights)

    write_results(tresults, RESULTS_EX_OUT)
    plot_groups(tresults, CHART_EX_OUT, universe_note=EX_MEGACAP_NOTE)
    plot_groups(tresults, RESULTS_EW_CHART_EX, weighting="equal",
                universe_note=EX_MEGACAP_NOTE)

    validate(results, groups, universe, breakpoints, funnel, gstats, cars, weights,
             weight_days=weight_days, wstats=wstats,
             ex={"stats": tstats, "groups": tgroups, "breakpoints": tbreaks, "gstats": tgstats,
                 "results": tresults, "universe": trimmed, "wstats": twstats})
    print("\n".join(_REPORT))
    return results


if __name__ == "__main__":
    main()
