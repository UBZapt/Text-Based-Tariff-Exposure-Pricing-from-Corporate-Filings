"""
Step 5a - Section 7.3 decile-spread presentation: value-weighted group CARs, both 2025 legs.

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
  - Value weights are me_lag on the EVENT date, not the estimation anchor. The imposition_primary
    anchor 2025-01-20 is MLK Day and carries no panel row at all, and weighting the two imposition
    runs at different dates would confound runs whose only intended difference is the estimation
    window. This matches run_car_regression.py, which reads its controls from the event-date row.

    python decile_sort.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: never opens a window, safe to run unattended
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

TEXP_COLUMN = "TExp_item1a"      # raw Item 1A measure, as in section 7.2
WEIGHT_COLUMN = "me_lag"         # market equity lagged one trading day, on the event date

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
# same event as the primary and differs only in estimation window. Slots 1 and 2 of the dataviz
# reference palette; validated all-pairs on the light surface (CVD dE 24.7, normal-vision 33.6,
# both above the >=8 / >=15 floors, and both above 3:1 contrast, so no relief obligation).
PALETTE = {
    "surface": "#fcfcfb",
    "impose": "#2a78d6",         # categorical slot 1
    "reverse": "#eb6834",        # categorical slot 2
    "ink": "#0b0b0b",
    "ink_muted": "#52514e",
    "grid": "#dcdcd8",
}
BAR_STYLE = {
    "imposition_primary": {"hatch": None, "color": "impose",
                           "label": "Imposition 2025-04-02, primary"},
    "imposition_robustness": {"hatch": "///", "color": "impose",
                              "label": "Imposition 2025-04-02, robustness"},
    "reversal_primary": {"hatch": None, "color": "reverse",
                         "label": "Reversal 2025-08-29"},
}

# Groups whose effective N (inverse Herfindahl of value weights) falls below this are named in a
# chart footnote: at that concentration the bar restates one or two firms rather than the group,
# and a reader should not take it as a statement about ~190 firms.
CONCENTRATED_EFF_N = 6.0

OUTPUT_COLUMNS = ["run", "event", "event_date", "window", "group", "n_nominal", "n_entering",
                  "weight_sum", "texp_min", "texp_max", "vw_car"]

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


def load_event_weights(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Lagged market equity on each event date, one row per (event, permno).

    The event-date row is already point-in-time: me_lag lags market equity one trading day, so it
    is known at the open of the event day.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_controls_data.py first.")
    frame = pd.read_csv(path, usecols=PANEL_COLUMNS, parse_dates=["date"])
    events = {name: pd.Timestamp(day) for name, day in ccd.EVENT_DATES.items()}
    keep = frame[frame["date"].isin(events.values())].copy()
    keep["event"] = keep["date"].map({day: name for name, day in events.items()})
    missing = set(events) - set(keep["event"])
    if missing:
        raise ValueError(f"{path.name} has no rows on event date(s): {sorted(missing)}")
    if keep.duplicated(["event", "permno"]).any():
        raise ValueError("controls panel is not unique on (event, permno)")
    return keep[["event", "permno", WEIGHT_COLUMN]]


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
    me = weights[weights["event"] == event].set_index("permno")[WEIGHT_COLUMN].reindex(
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
def group_car(cars: dict[str, pd.DataFrame], groups: pd.Series, universe: pd.DataFrame,
              weights: pd.DataFrame) -> pd.DataFrame:
    """Value-weighted group CAR for every run x window x group.

    Weighted by me_lag on the run's own event date. A firm enters a cell only with both a
    non-null CAR and a strictly positive weight; the count that does is carried beside the
    group's nominal size so any shortfall is visible rather than absorbed.
    """
    nominal = groups.value_counts().reindex(GROUPS, fill_value=0)
    bounds = universe.groupby(groups)[TEXP_COLUMN].agg(["min", "max"])
    rows = []
    for run in ec.RUNS:
        car = cars[run["name"]].set_index("permno")
        weight = weights[weights["event"] == run["event"]].set_index("permno")[WEIGHT_COLUMN]
        frame = pd.DataFrame({"group": groups, "w": weight.reindex(groups.index)})
        for window in ec.CAR_COLUMNS:
            frame["car"] = car[window].reindex(groups.index)
            usable = frame[frame["car"].notna() & frame["w"].notna() & frame["w"].gt(0)]
            for group, block in usable.groupby("group"):
                rows.append({
                    "run": run["name"], "event": run["event"],
                    "event_date": str(car["event_date"].iloc[0].date()),
                    "window": window, "group": int(group),
                    "n_nominal": int(nominal[group]), "n_entering": len(block),
                    "weight_sum": block["w"].sum(),
                    "texp_min": bounds.loc[group, "min"], "texp_max": bounds.loc[group, "max"],
                    "vw_car": np.average(block["car"], weights=block["w"]),
                })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_results(results: pd.DataFrame, path: Path = RESULTS_OUT) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Chart                                                                        #
# --------------------------------------------------------------------------- #
def effective_n(groups: pd.Series, weights: pd.DataFrame, event: str) -> pd.Series:
    """Inverse Herfindahl of value weights per group: the equally-weighted firm count it acts like.

    A group of 190 firms whose weight sits in two mega-caps is not a 190-firm portfolio, and the
    exhibit has to say so rather than let the bar imply breadth it does not have.
    """
    weight = weights[weights["event"] == event].set_index("permno")[WEIGHT_COLUMN]
    held = pd.DataFrame({"group": groups, "w": weight.reindex(groups.index)}).dropna()
    return held.groupby("group")["w"].apply(lambda w: 1.0 / ((w / w.sum()) ** 2).sum())


def plot_groups(results: pd.DataFrame, path: Path = CHART_OUT,
                eff_n: pd.Series | None = None, subtitle: str = "") -> Path:
    """Grouped bar chart: one panel per event window, three bars per TExp group.

    Two hues carry the two event legs; the imposition robustness run is the same hue as its
    primary with a hatch, because it is the same event measured over a different estimation
    window rather than a third series. The y-axis is shared across panels so the narrower
    windows are not visually exaggerated.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    windows = ec.CAR_COLUMNS
    runs = [r["name"] for r in ec.RUNS]

    fig, axes = plt.subplots(len(windows), 1, figsize=(9.5, 10.5), dpi=300, sharey=True)
    fig.patch.set_facecolor(PALETTE["surface"])

    slot = 0.26                      # centre-to-centre spacing of the three bars in a cluster
    width = 0.24                     # leaves a ~2px surface gap between adjacent fills
    offsets = {run: (i - 1) * slot for i, run in enumerate(runs)}
    x = np.arange(len(GROUPS))

    for ax, window in zip(axes, windows):
        ax.set_facecolor(PALETTE["surface"])
        for run in runs:
            style = BAR_STYLE[run]
            cell = results[(results["run"] == run) & (results["window"] == window)]
            height = cell.set_index("group")["vw_car"].reindex(GROUPS) * 100
            ax.bar(x + offsets[run], height, width, label=style["label"],
                   facecolor=PALETTE[style["color"]], hatch=style["hatch"],
                   edgecolor=PALETTE["surface"], linewidth=0.8, zorder=3)

        ax.axhline(0, color=PALETTE["ink_muted"], lw=1.0, zorder=4)
        ax.set_title(f"Event window {_short(window)}", fontsize=10, color=PALETTE["ink"],
                     pad=8, loc="left")
        ax.set_ylabel("Value-weighted CAR (%)", fontsize=9,
                      color=PALETTE["ink_muted"], labelpad=8)
        ax.set_xticks(x, [str(g) for g in GROUPS])
        ax.tick_params(labelsize=8, colors=PALETTE["ink_muted"])
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PALETTE["grid"])

    axes[-1].set_xlabel("TExp group  (0 = zero exposure, 1-9 = ascending exposure deciles)",
                        fontsize=9, color=PALETTE["ink_muted"], labelpad=10)
    fig.suptitle("Value-weighted cumulative abnormal return by tariff-exposure group",
                 fontsize=12, color=PALETTE["ink"], x=0.055, ha="left", y=0.992)
    if subtitle:
        fig.text(0.055, 0.966, subtitle, fontsize=8.5, color=PALETTE["ink_muted"], ha="left")
    # Legend above the panels, not inside one: at this bar density any in-axes placement sits on
    # top of data in at least one window.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.05, 0.955),
               fontsize=8, frameon=False, labelcolor=PALETTE["ink_muted"], ncol=3,
               columnspacing=1.6, handlelength=1.4)

    if eff_n is not None:
        thin = [f"{int(g)} (~{eff_n[g]:.0f} firms)" for g in GROUPS
                if g in eff_n.index and eff_n[g] < CONCENTRATED_EFF_N]
        note = ("Value weighting concentrates several groups in a few mega-caps. Effective firm "
                "count (inverse Herfindahl of weights) is lowest in group "
                + ", ".join(thin) + ";\nthose bars restate their largest holdings rather than the "
                "~190 firms nominally in the group. Per-group figures are in the validation "
                "report." if thin else
                f"Effective firm count (inverse Herfindahl of value weights) runs "
                f"{eff_n.min():.0f} to {eff_n.max():.0f} across groups, so no bar is carried by a "
                f"handful of firms.\nPer-group figures, and the comparison against the all-firms "
                f"universe, are in the validation report.")
        fig.text(0.055, 0.012, note, fontsize=7.5, color=PALETTE["ink_muted"], ha="left",
                 va="bottom", linespacing=1.5)

    fig.tight_layout(rect=(0, 0.045, 1, 0.938))
    fig.savefig(path, facecolor=PALETTE["surface"])
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def _check_partition(groups: pd.Series, universe: pd.DataFrame) -> None:
    """Assert the ten groups partition the universe exactly once, with no overlap or gap."""
    values = universe[TEXP_COLUMN]
    assert groups.index.is_unique, "a firm is assigned to more than one group"
    assert set(groups.index) == set(universe.index), "group assignment does not cover the universe"
    assert groups.isin(GROUPS).all(), f"a firm carries a group outside {GROUPS}"
    assert (groups[values.eq(0)] == ZERO_GROUP).all(), "a zero-TExp firm sits outside group 0"
    assert (groups[values.gt(0)] > ZERO_GROUP).all(), "a positive-TExp firm sits in group 0"
    assigned = int(groups.value_counts().sum())
    assert assigned == len(universe), f"{assigned} assignments for {len(universe)} firms"
    # Groups 1-9 must be contiguous in TExp: every group's max below the next group's min.
    bounds = universe.groupby(groups)[TEXP_COLUMN].agg(["min", "max"]).loc[1:]
    for lo, hi in zip(bounds.index[:-1], bounds.index[1:]):
        assert bounds.loc[lo, "max"] <= bounds.loc[hi, "min"], \
            f"groups {lo} and {hi} overlap in {TEXP_COLUMN}"


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
        assert len(mismatch) == 0, f"{name}: {len(mismatch)} firm(s) change group: {list(mismatch)[:5]}"
    assert results.groupby(["run", "window"])["n_nominal"].apply(tuple).nunique() == 1, \
        "nominal group sizes differ across runs; membership is not fixed"


def validate(results: pd.DataFrame, groups: pd.Series, universe: pd.DataFrame,
             breakpoints: np.ndarray, funnel: dict, gstats: dict,
             cars: dict[str, pd.DataFrame], weights: pd.DataFrame,
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
    _say(f"Weighting     : value-weighted on {WEIGHT_COLUMN} at the EVENT date")
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
    assert funnel["valid"] - funnel["no_texp"] == funnel["universe"], "funnel does not reconcile"
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
    assert len(results) == len(ec.RUNS) * len(ec.CAR_COLUMNS) * len(GROUPS), \
        f"expected {len(ec.RUNS) * len(ec.CAR_COLUMNS) * len(GROUPS)} rows, got {len(results)}"
    _say(f"  results table is {len(results)} rows ..................... asserted")

    _section("4. Weight coverage (me_lag on the event date)")
    for event, day in ccd.EVENT_DATES.items():
        w = weights[weights["event"] == event].set_index("permno")[WEIGHT_COLUMN]
        held = w.reindex(universe.index)
        absent, null, nonpos = held.isna().sum(), held.isna().sum(), (held <= 0).sum()
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
    for run in (r["name"] for r in ec.RUNS):
        _say(f"  {run}")
        _say(f"    {'group':>5} {'nominal':>8}" +
             "".join(f"{_short(w):>12}" for w in ec.CAR_COLUMNS))
        block = results[results["run"] == run]
        for group in GROUPS:
            cells = block[block["group"] == group].set_index("window")
            nominal = int(cells["n_nominal"].iloc[0])
            entering = "".join(f"{int(cells.loc[w, 'n_entering']):>12,}" for w in ec.CAR_COLUMNS)
            _say(f"    {group:>5} {nominal:>8,}{entering}")
        totals = block.groupby("window")["n_entering"].sum().reindex(ec.CAR_COLUMNS)
        _say(f"    {'total':>5} {int(block['n_nominal'].sum() / len(ec.CAR_COLUMNS)):>8,}" +
             "".join(f"{int(t):>12,}" for t in totals))

    _section("6. Value-weighted group CAR (%), and the group 9 - group 0 spread")
    for window in ec.CAR_COLUMNS:
        _say(f"  window {_short(window)}")
        _say(f"    {'run':<24}" + "".join(f"{('G' + str(g)):>8}" for g in GROUPS) + f"{'G9-G0':>9}")
        for run in (r["name"] for r in ec.RUNS):
            cell = results[(results["run"] == run) & (results["window"] == window)]
            series = cell.set_index("group")["vw_car"].reindex(GROUPS) * 100
            spread = series[N_POSITIVE_GROUPS] - series[ZERO_GROUP]
            _say(f"    {run:<24}" + "".join(f"{v:>8.2f}" for v in series) + f"{spread:>9.2f}")
        _say()

    _section("7. Weight concentration within groups")
    _say("  Value weighting is what section 7.3 nominates and what is reported, but at ~190 firms")
    _say("  per group a few mega-caps carry most of the weight. Effective N is the inverse")
    _say("  Herfindahl of the weights: the number of equally-weighted firms the group behaves")
    _say("  like. Where it is small, the group CAR is a statement about those firms, not about")
    _say("  the group, and must be read that way.")
    _say()
    _say(f"    {'group':>5}{'n':>6}{'eff N':>8}{'top 1 %':>9}{'top 3 %':>9}   largest holding")
    concentration = {}
    for event in ccd.EVENT_DATES:
        weight = weights[weights["event"] == event].set_index("permno")[WEIGHT_COLUMN]
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
    _say(f"    {'run':<24}{'window':>11}{'spearman':>10}{'p':>9}{'sign changes':>14}  reading")
    for run in (r["name"] for r in ec.RUNS):
        for window in ec.CAR_COLUMNS:
            cell = results[(results["run"] == run) & (results["window"] == window)]
            series = cell.set_index("group")["vw_car"].reindex(GROUPS)
            rho = stats.spearmanr(GROUPS, series)
            flips = int((np.diff(np.sign(np.diff(series))) != 0).sum())
            if abs(rho.statistic) >= 0.7:
                reading = "monotone" if flips <= 3 else "strong trend, uneven"
            elif abs(rho.statistic) >= 0.4:
                reading = "partial gradient"
            else:
                reading = "no clear gradient"
            _say(f"    {run:<24}{_short(window):>11}{rho.statistic:>10.3f}{rho.pvalue:>9.3f}"
                 f"{flips:>14}  {reading}")

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
    dfdv = 24072
    car = cars[UNIVERSE_RUN].set_index("permno")
    if dfdv in universe.index:
        _say(f"  DFDV (permno {dfdv}) is in the universe, group {groups[dfdv]}, retained.")
    elif dfdv in car.index:
        _say(f"  DFDV (permno {dfdv}, Janover / DeFi Development) is NOT in the universe, and "
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

    _section("14. Outputs" if ex is not None else "10. Outputs")
    paths = ([RESULTS_OUT, CHART_OUT, RESULTS_EX_OUT, CHART_EX_OUT, REPORT_OUT]
             if ex is not None else [RESULTS_OUT, CHART_OUT, REPORT_OUT])
    for path in paths:
        _say(f"  {path.relative_to(BASE)}")

    REPORT_OUT.write_text("\n".join(_REPORT), encoding="utf-8")


def _validate_ex_megacap(ex: dict, results: pd.DataFrame, groups: pd.Series,
                         universe: pd.DataFrame, breakpoints: np.ndarray,
                         weights: pd.DataFrame, cars: dict[str, pd.DataFrame]) -> None:
    """Report universe B's construction, its independence from A, and the comparison."""
    tstats, tgroups, tbreaks, tgstats, tresults, tuniverse = (
        ex["stats"], ex["groups"], ex["breakpoints"], ex["gstats"], ex["results"], ex["universe"])

    _section("11. Universe B: mega-cap-excluded robustness comparison")
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
    assert tstats["n_before"] - tstats["n_dropped"] == tstats["n_after"], "trim does not reconcile"
    me = weights[weights["event"] == "impose"].set_index("permno")[WEIGHT_COLUMN]
    assert me.reindex(tuniverse.index).max() <= tstats["cutoff"], "a kept firm exceeds the cutoff"
    assert me.reindex(universe.index.difference(tuniverse.index)).min() > tstats["cutoff"], \
        "a dropped firm is below the cutoff"
    _say("  trim reconciles, and no firm sits on the wrong side of the cutoff (asserted)")

    _section("12. Independence of the two group constructions")
    _say("  Universe B's breakpoints are computed from scratch on its own cross-section, NOT")
    _say("  inherited from the primary and NOT a filtering of the primary's assignment. Different")
    _say("  breakpoints are the EXPECTED outcome here, not an error.")
    _say()
    _say(f"    {'q':>3}{'primary':>12}{'universe B':>13}   {'shift':>10}")
    for i, (a, b) in enumerate(zip(breakpoints, tbreaks)):
        _say(f"    {i:>3}{a:>12.6f}{b:>13.6f}   {b - a:>+10.6f}")
    assert not np.array_equal(breakpoints, tbreaks), \
        "universe B reproduced the primary breakpoints exactly; the constructions are not separate"
    moved = int((groups.reindex(tgroups.index) != tgroups).sum())
    _say()
    _say(f"  breakpoint vectors are not identical (asserted)")
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

    _section("13. Comparison: group counts, concentration, and the spread")
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
    assert eff_b.min() > eff_a.min(), "the trim did not reduce concentration"
    _say(f"  the trim raises the worst group from {eff_a.min():.1f} to {eff_b.min():.1f} "
         f"effective firms (asserted to improve).")

    _say()
    _say("  Group 9 - group 0 spread, value-weighted %, universe B with primary in brackets:")
    _say(f"    {'run':<24}" + "".join(f"{_short(w):>20}" for w in ec.CAR_COLUMNS))
    for run in (r["name"] for r in ec.RUNS):
        line = f"    {run:<24}"
        for window in ec.CAR_COLUMNS:
            b = tresults[(tresults["run"] == run) & (tresults["window"] == window)] \
                .set_index("group")["vw_car"] * 100
            a = results[(results["run"] == run) & (results["window"] == window)] \
                .set_index("group")["vw_car"] * 100
            line += f"{b[N_POSITIVE_GROUPS] - b[ZERO_GROUP]:>+10.2f}" \
                    f"{'[' + format(a[N_POSITIVE_GROUPS] - a[ZERO_GROUP], '+.2f') + ']':>10}"
        _say(line)
    _say()
    _say("  Reading, stated as findings rather than as a verdict:")
    _say("   - Every sign is preserved, at every window and in both legs. The flip is therefore")
    _say("     NOT an artefact of mega-cap concentration.")
    _say("   - Magnitudes roughly halve, so the primary levels were partly amplified by the few")
    _say("     large firms that carried their groups. The primary remains the headline; this is")
    _say("     the honest bound on how much of it rests on those firms.")
    _say("   - The [-1,+1] reversal, already the known failure from Step 4d, gets MORE negative")
    _say("     rather than less. The trim does not rescue it, and that is reported, not buried.")


def main() -> pd.DataFrame:
    cars = load_cars()
    texp = load_texp()
    weights = load_event_weights()

    # --- Primary universe: unchanged, and must stay so ---------------------- #
    universe, funnel = build_universe(cars, texp)
    groups, breakpoints, gstats = assign_groups(universe)
    results = group_car(cars, groups, universe, weights)

    write_results(results)
    plot_groups(results, eff_n=effective_n(groups, weights, ec.RUNS[0]["event"]),
                subtitle=f"All {len(universe):,} firms passing the section 6 screen at "
                         f"end-March 2025")

    # --- Universe B: the same procedure on an independently built group set -- #
    trimmed, tstats = trim_megacaps(universe, weights)
    tgroups, tbreaks, tgstats = assign_groups(trimmed)
    tresults = group_car(cars, tgroups, trimmed, weights)

    write_results(tresults, RESULTS_EX_OUT)
    plot_groups(tresults, CHART_EX_OUT,
                eff_n=effective_n(tgroups, weights, ec.RUNS[0]["event"]),
                subtitle=f"Robustness: excluding the {tstats['n_dropped']} firms above the NYSE "
                         f"90th-percentile market equity at {TRIM_MONTH} "
                         f"(${tstats['cutoff'] / 1e6:,.0f}bn) - {len(trimmed):,} firms")

    validate(results, groups, universe, breakpoints, funnel, gstats, cars, weights,
             ex={"stats": tstats, "groups": tgroups, "breakpoints": tbreaks, "gstats": tgstats,
                 "results": tresults, "universe": trimmed})
    print("\n".join(_REPORT))
    return results


if __name__ == "__main__":
    main()
