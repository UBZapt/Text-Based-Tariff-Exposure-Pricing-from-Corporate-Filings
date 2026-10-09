"""
Dissertation figures for the event-study results: sections 7.2, 7.6 and v6 section 5.2 item 1.

Four figures and one table, all drawn from the persisted results of the analytical stages rather
than from any recomputation of them - with one exception that is guarded rather than trusted. The
within-industry estimates are a table because as a chart they carried thirty-six intervals, most
several times the width of their point, and said only that nothing is precisely estimated inside
an industry; a table of coefficients and standard errors says that checkably. The event-time
trajectory needs abnormal returns day by day, and `data/intermediate/car_*.csv` holds only the three
cumulative CARs, so the daily ARs are rebuilt here through `estimate_car`'s own public functions.
Every piece of estimation logic stays in that module; this script contributes the assembly and
then asserts that the CARs it can re-derive reproduce the persisted tables exactly. If that glue
ever drifts from `estimate_car.main`, the assertion fires rather than the figure quietly changing.

Coefficients are drawn in native exposure units - percentage points of CAR per unit of Item 1A
tariff-sentence share - across both coefficient figures and the industry table. A per-standard-
deviation rescale reads better in isolation, but the cross-cycle sd ranges 0.0048 to 0.0122 across
events, so no single scale factor exists and a per-event one would make the panels incomparable in
a figure whose whole subject is comparability. Only the dependent variable is rescaled, from decimal
returns to percent, so that every figure in the project reads CAR on one scale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import chartstyle as cs
import clean_controls_data as ccd
import decile_sort as ds
import estimate_car as ec
import palette
import run_car_regression as rcr
from config import BASE, OUTPUT_DIR

# --------------------------------------------------------------------------- Configuration

REGRESSION_CSV = OUTPUT_DIR / "car_regression_results.csv"
CROSS_CYCLE_CSV = OUTPUT_DIR / "car_regression_results_cross_cycle.csv"
INDUSTRY_CSV = OUTPUT_DIR / "industry_split_results.csv"
DECILE_CSV = OUTPUT_DIR / "decile_sort_results.csv"

CHART_EVENT_TIME = {"value": OUTPUT_DIR / "fig_car_event_time.png",
                    "equal": OUTPUT_DIR / "fig_car_event_time_equal_weighted.png"}
CHART_COEFFICIENTS = OUTPUT_DIR / "fig_car_coefficients.png"
CHART_CROSS_CYCLE = OUTPUT_DIR / "fig_cross_cycle_coefficients.png"
TABLE_INDUSTRY = OUTPUT_DIR / "table_industry_split.txt"

TEXP_TERM = "TExp_item1a"
PRIMARY_SPEC = "primary"

# The two legs H1 is stated on. The imposition robustness run differs from its primary only in
# where the estimation window closes, so it would add a near-duplicate line to a trajectory; it
# stays in the decile chart, where three bars per group still read, and in every results table.
EVENT_TIME_RUNS = ["imposition_primary", "reversal_primary"]
INDUSTRY_RUN = "imposition_primary"

# Event days carrying an x tick. The three section 7.2 windows close at +/-1, +/-5 and +/-10, so
# ticking exactly those days marks every window boundary without spending guide lines on them.
TICK_DAYS = [-10, -5, -1, 0, 1, 5, 10]

ZERO_GROUP, TOP_GROUP = ds.ZERO_GROUP, ds.N_POSITIVE_GROUPS
TRAJECTORY_WINDOW = ec.CAR_COLUMNS[-1]          # the widest window, which spans the whole plot
CAR_TOL = 1e-12                                 # AR reassembly against the persisted CAR tables
CELL_TOL = 1e-10                                # event-time aggregation against decile_sort

# "car_m1p1" -> "[-1,+1]". Both sides derive from ec.EVENT_WINDOWS in the same order, so the
# mapping cannot drift from the columns it labels.
WINDOW_LABELS = {col: f"[{lo:+d},{hi:+d}]"
                 for col, (lo, hi) in zip(ec.CAR_COLUMNS, ec.EVENT_WINDOWS)}

PALETTE = palette.roles(high="CATEGORICAL_1", low="CATEGORICAL_2", spread="CATEGORICAL_3")
PCT = 100.0                      # CAR is a decimal return; every figure here reads in percent
COEF_LABEL = "Coefficient on TExp (% CAR per unit)"
BAR_WIDTH = 0.38                 # two bars per event window in the section 7.2 figure
GUIDE_WIDTH = 0.8                # event-day and cycle-boundary rules
IND_NAME_WIDTH = 10              # widest FF12 abbreviation is five characters


# --------------------------------------------------------------------------- Daily abnormal returns

def daily_ar(cycle: str = ccd.DEFAULT_CYCLE) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Abnormal returns day by day over the widest event window, per run.

    Assembled from estimate_car's public functions in the order main() uses them. main() is
    deliberately not called: it rewrites the three per-firm CAR tables and prints a full
    validation report, neither of which drawing a figure should cause.
    """
    ec.select_cycle(cycle)
    panel = ec.load_panel()
    factors = ec.load_factors()
    cal = ec.build_calendar(panel)
    aligned, cal, _ = ec.reconcile_calendar(cal, factors)
    fmat = aligned[ec.FACTORS].to_numpy(dtype=float)

    wide = panel.pivot(index="date", columns="permno", values="ret").reindex(cal)
    excess = wide.sub(aligned.set_index("date")["rf"].reindex(cal), axis=0)

    out = {}
    for run in ec.RUNS:
        resolved = ec.resolve_run(run, cal)
        betas, _ = ec.estimate_betas(excess, fmat, resolved)
        out[run["name"]] = (ec.compute_ar(excess, fmat, betas,
                                          resolved["ar_lo"], resolved["ar_hi"]), resolved)
    return out


def rebuild_cars(ar_by_run: dict) -> dict[str, pd.DataFrame]:
    """Each run's three window CARs, re-derived from the daily ARs by estimate_car's own summer."""
    return {name: ec.compute_cars(ar, resolved) for name, (ar, resolved) in ar_by_run.items()}


def reconcile_cars(rebuilt: dict, stored: dict[str, pd.DataFrame]) -> float:
    """Assert the rebuilt CARs reproduce every persisted one, and return the worst gap.

    This is the guard on the assembly in daily_ar. compute_cars is the same function estimate_car
    writes its tables with, so a mismatch can only mean the inputs it was handed differ - a
    changed calendar, a changed panel, or an assembly step that has since moved inside main().
    """
    worst = 0.0
    for name, frame in rebuilt.items():
        table = stored[name].set_index("permno")
        for window in ec.CAR_COLUMNS:
            a, b = frame[window], table[window].reindex(frame.index)
            if a.isna().ne(b.isna()).any():
                raise ValueError(f"{name}/{window}: the rebuilt abnormal returns disagree with "
                                 f"the stored CAR table on which firms have a complete window")
            worst = max(worst, float((a - b).abs().max()))
    if worst > CAR_TOL:
        raise ValueError(f"rebuilt abnormal returns reproduce the stored CARs only to {worst:.2e}, "
                         f"above the {CAR_TOL:.0e} tolerance; the assembly in daily_ar has drifted "
                         f"from estimate_car.main")
    return worst


# --------------------------------------------------------------------------- Event-time aggregation

def _cell_mean(car: pd.Series, groups: pd.Series, weight: pd.Series,
               scheme: str) -> pd.Series:
    """Group means of one CAR column under section 7.3's admission rule and weighting schemes.

    A firm enters only with a non-null CAR and a strictly positive value weight, whichever scheme
    is applied, so the two schemes average identically the same firms - decile_sort's discipline,
    repeated here because the reconciliation is only meaningful if the rule is the same one.
    """
    frame = pd.DataFrame({"group": groups, "w": weight.reindex(groups.index),
                          "car": car.reindex(groups.index)})
    usable = frame[frame["car"].notna() & frame["w"].notna() & frame["w"].gt(0)]
    if scheme == "equal":
        return usable.groupby("group")["car"].mean()
    totals = (usable.assign(weighted=usable["car"] * usable["w"])
              .groupby("group")[["weighted", "w"]].sum())
    return totals["weighted"] / totals["w"]


def event_time_caar(ar: pd.DataFrame, resolved: dict, groups: pd.Series, weight: pd.Series,
                    scheme: str) -> pd.DataFrame:
    """Cumulative average abnormal return by event day, for the zero and top exposure groups.

    Admission is section 7.3's rule read at the widest window: a complete set of abnormal returns
    across the whole span, and a strictly positive value weight. Because compute_cars requires
    every day of a window to be present, that set is exactly the one the [-10,+10] decile cell
    averages, which is what lets the two figures reconcile.

    Weights are held at the widest window's own opening-day market equity for all twenty-one days,
    so no return inside the plot helps set the weights it is averaged under.
    """
    offset = resolved["event_idx"] - resolved["ar_lo"]
    days = np.arange(ar.shape[0]) - offset
    complete = ar.columns[ar.notna().all(axis=0)]
    usable = [p for p in complete if p in groups.index
              and pd.notna(weight.get(p)) and weight.get(p) > 0]
    cum = ar[usable].cumsum(axis=0)

    out = {}
    for group in (ZERO_GROUP, TOP_GROUP):
        members = [p for p in usable if groups[p] == group]
        if not members:
            raise ValueError(f"group {group} has no firm with a complete abnormal-return series")
        w = (weight.reindex(members).to_numpy(dtype=float) if scheme == "value"
             else np.ones(len(members)))
        out[group] = cum[members].to_numpy() @ w / w.sum()
    frame = pd.DataFrame(out, index=pd.Index(days, name="day"))
    frame["spread"] = frame[TOP_GROUP] - frame[ZERO_GROUP]
    return frame


def reconcile_cells(rebuilt: dict, groups: pd.Series, weights: pd.DataFrame,
                    decile: pd.DataFrame) -> float:
    """Assert this script's aggregation reproduces every decile_sort cell, and return the worst gap.

    Run across every (run, window, group, weighting) decile_sort published, so the trajectory
    figure and the decile figure are proven to be one set of numbers seen two ways rather than
    asserted to be. The event-time lines are the same aggregation walked day by day.
    """
    worst = 0.0
    for name, frame in rebuilt.items():
        event = next(r["event"] for r in ec.RUNS if r["name"] == name)
        for window in ec.CAR_COLUMNS:
            weight = ds.weight_series(weights, event, window)
            for scheme in ("value", "equal"):
                mine = _cell_mean(frame[window], groups, weight, scheme)
                published = decile[(decile["run"] == name) & (decile["window"] == window)
                                   & (decile["weighting"] == scheme)].set_index("group")["vw_car"]
                if sorted(mine.index) != sorted(published.index):
                    raise ValueError(f"{name}/{window}/{scheme}: group membership differs from "
                                     f"decile_sort's")
                worst = max(worst, float((mine - published.reindex(mine.index)).abs().max()))
    if worst > CELL_TOL:
        raise ValueError(f"event-time aggregation reproduces decile_sort's cells only to "
                         f"{worst:.2e}, above the {CELL_TOL:.0e} tolerance")
    return worst


# --------------------------------------------------------------------------- Results readers

def load_coefficients(path, runs: list[str]) -> pd.DataFrame:
    """The primary-specification TExp coefficient for each (run, window), in the run order given."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run run_car_regression.py first.")
    frame = pd.read_csv(path)
    keep = frame[(frame["spec"] == PRIMARY_SPEC) & (frame["term"] == TEXP_TERM)
                 & (frame["run"].isin(runs))].copy()
    missing = set(runs) - set(keep["run"])
    if missing:
        raise ValueError(f"{path.name} holds no primary {TEXP_TERM} rows for {sorted(missing)}")
    keep["stars"] = keep["stars"].fillna("")
    keep["run"] = pd.Categorical(keep["run"], categories=runs, ordered=True)
    keep["window"] = pd.Categorical(keep["window"], categories=ec.CAR_COLUMNS, ordered=True)
    return keep.sort_values(["run", "window"])


def load_industry(path=INDUSTRY_CSV) -> pd.DataFrame:
    """Within-FF12 coefficients for the imposition leg, ordered by the narrowest window's fit.

    One order across all three panels, so a group can be followed down the figure. Groups below
    run_car_regression.MIN_INDUSTRY_N carry no estimate and sort last.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run run_car_regression.py first.")
    frame = pd.read_csv(path)
    keep = frame[frame["run"] == INDUSTRY_RUN].copy()
    keep["stars"] = keep["stars"].fillna("")
    keep["skipped"] = keep["skipped"].fillna("")
    order = (keep[keep["window"] == ec.CAR_COLUMNS[0]]
             .sort_values("coef", na_position="last")["ff12"].tolist())
    keep["ff12"] = pd.Categorical(keep["ff12"], categories=order, ordered=True)
    keep["window"] = pd.Categorical(keep["window"], categories=ec.CAR_COLUMNS, ordered=True)
    return keep.sort_values(["window", "ff12"])


# --------------------------------------------------------------------------- Figures

def plot_event_time(caars: dict[str, pd.DataFrame], events: dict[str, str], scheme: str, path):
    """Two panels, imposition and reversal, cumulating abnormal return across the event window."""
    adjective = next(w["adjective"] for w in ds.WEIGHTINGS if w["name"] == scheme)
    fig, axes = plt.subplots(1, len(EVENT_TIME_RUNS), figsize=cs.SIZE_PAIR, sharey=True)

    for ax, run in zip(axes, EVENT_TIME_RUNS):
        frame = caars[run] * 100
        cs.zero_line(ax)
        ax.axvline(0, color=PALETTE["ink_muted"], lw=GUIDE_WIDTH, zorder=1)
        ax.plot(frame.index, frame[TOP_GROUP], color=PALETTE["high"], lw=1.7,
                label=f"Group {TOP_GROUP}, highest exposure")
        ax.plot(frame.index, frame[ZERO_GROUP], color=PALETTE["low"], lw=1.7,
                label=f"Group {ZERO_GROUP}, zero exposure")
        ax.plot(frame.index, frame["spread"], color=PALETTE["spread"], lw=1.4, ls="--",
                label=f"Spread, group {TOP_GROUP} less group {ZERO_GROUP}")
        ax.set_xticks(TICK_DAYS)
        ax.set_xlim(float(min(frame.index)), float(max(frame.index)))
        cs.headroom(ax, top=0.10)
        ax.set_xlabel("Trading days relative to the event")
        cs.panel_title(ax, f"{_leg(run)}, {events[run]}")

    axes[0].set_ylabel(f"{adjective} cumulative abnormal return (%)")
    for ax in axes:
        cs.frame(ax, rotate_x=True)
    cs.figure_title(fig, f"{adjective} cumulative abnormal return by exposure group")
    # The three keys are common to both panels and there is no corner of either that does not
    # cover data, so the legend sits in the band between the title and the panels.
    cs.figure_legend(fig, *axes[0].get_legend_handles_labels(), ncol=3, y=0.895)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return cs.save(fig, path)


def plot_coefficients(coefs: pd.DataFrame, events: dict[str, str], path=CHART_COEFFICIENTS):
    """Section 7.2 / H1: the TExp coefficient in each leg at each of the three event windows."""
    fig, ax = plt.subplots(figsize=cs.SIZE_WIDE)
    x = np.arange(len(ec.CAR_COLUMNS))
    colours = {EVENT_TIME_RUNS[0]: PALETTE["high"], EVENT_TIME_RUNS[1]: PALETTE["low"]}
    blocks = {run: coefs[coefs["run"] == run].set_index("window").reindex(ec.CAR_COLUMNS)
              for run in EVENT_TIME_RUNS}

    for i, run in enumerate(EVENT_TIME_RUNS):
        pos = x + (i - 0.5) * BAR_WIDTH
        ax.bar(pos, blocks[run]["coef"] * PCT, BAR_WIDTH, color=colours[run],
               edgecolor=palette.INK, linewidth=cs.BAR_EDGE_WIDTH, zorder=3,
               label=f"{_leg(run)}, {events[run]}")
        cs.whiskers(ax, pos, blocks[run]["coef"] * PCT, blocks[run]["se"] * PCT)

    cs.zero_line(ax)
    ax.set_xticks(x, [WINDOW_LABELS[w] for w in ec.CAR_COLUMNS])
    ax.set_xlabel("Event window (trading days)")
    ax.set_ylabel(COEF_LABEL)
    for i, run in enumerate(EVENT_TIME_RUNS):
        cs.value_labels(ax, x + (i - 0.5) * BAR_WIDTH, blocks[run]["coef"] * PCT,
                        blocks[run]["se"] * PCT, blocks[run]["stars"])
    cs.headroom(ax, bottom=0.22)
    cs.legend(ax, loc="lower left")
    cs.frame(ax, rotate_x=False)
    cs.figure_title(fig, "Tariff-exposure coefficient, imposition versus reversal")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return cs.save(fig, path)


def plot_cross_cycle(cc: pd.DataFrame, baseline: pd.DataFrame, events: dict[str, str],
                     path=CHART_CROSS_CYCLE):
    """Section 7.6 / H4: the same coefficient at nine 2018-19 events, with the 2025 legs alongside.

    A line with a marker per event, one panel per event window. The events are ordered
    chronologically and unevenly spaced in time, so the segments between markers carry no estimate
    of their own; they are there to make the path of the coefficient across the two cycles
    followable, which eleven separate bars per panel were not.

    Chronological order puts the in-sample 2025 pair at the right-hand end on its own; a dotted
    rule marks where the out-of-sample cycle ends. The events whose predicted sign is positive are
    shaded rather than given a second colour, since the sign is a property of the event and holds
    across all three panels.
    """
    cc = cc.copy()
    cc["label"] = pd.to_datetime(cc["event"].str.extract(r"(\d{8})$")[0],
                                 format="%Y%m%d").dt.strftime("%Y-%m-%d")
    base = baseline.copy()
    base["label"] = base["run"].map(events)
    both = pd.concat([cc, base], ignore_index=True)
    order = sorted(both["label"].unique())
    x = np.arange(len(order))
    divider = cc["label"].nunique() - 0.5
    loosening = [i for i, name in enumerate(order)
                 if both.loc[both["label"].eq(name), "expected_sign"].iloc[0] > 0]

    fig, axes = plt.subplots(len(ec.CAR_COLUMNS), 1, figsize=cs.SIZE_STACK3, sharex=True)
    for ax, window in zip(axes, ec.CAR_COLUMNS):
        block = both[both["window"] == window].set_index("label").reindex(order)
        for i in loosening:
            ax.axvspan(i - 0.5, i + 0.5, facecolor=PALETTE["grid"], lw=0, zorder=1)
        cs.whiskers(ax, x, block["coef"] * PCT, block["se"] * PCT)
        ax.plot(x, block["coef"] * PCT, color=PALETTE["high"], lw=1.4, marker="o",
                markersize=4.0, zorder=6)
        cs.zero_line(ax)
        ax.axvline(divider, color=PALETTE["ink_muted"], lw=GUIDE_WIDTH, ls=":", zorder=2)
        ax.set_xticks(x, order)
        ax.set_xlim(-0.6, len(order) - 0.4)
        cs.panel_title(ax, f"Event window {WINDOW_LABELS[window]}")
        cs.star_labels(ax, x, block["coef"] * PCT, block["se"] * PCT, block["stars"])

    axes[len(axes) // 2].set_ylabel(COEF_LABEL)
    axes[-1].set_xlabel("Event date  (2018-19 Section 301 cycle, then the 2025 IEEPA cycle)")
    handles = [plt.Line2D([], [], color=PALETTE["high"], lw=1.4, marker="o", markersize=4.0,
                          label="Coefficient on TExp, with its 95% interval"),
               Patch(facecolor=PALETTE["grid"], label="Loosening event, predicted positive")]
    cs.headroom(axes[0], top=0.30)
    for ax in axes:
        cs.frame(ax)
    cs.figure_title(fig, "Tariff-exposure coefficient across policy events, 2018 to 2025")
    cs.figure_legend(fig, handles, [h.get_label() for h in handles], ncol=2, y=0.958)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return cs.save(fig, path)


def industry_table(ind: pd.DataFrame, event_date: str) -> list[str]:
    """v6 section 5.2 item 1 as a table: the coefficient estimated inside each FF12 group.

    A table rather than a chart. Twelve groups over three windows is thirty-six estimates, eight
    of which are the same group repeated, and half carry intervals several times the width of the
    point; read as bars the panel said only that nothing is precisely estimated within an
    industry, which is exactly the sentence a table of coefficients and standard errors makes
    checkable.
    """
    order = list(ind["ff12"].cat.categories)
    heads = "".join(f"{WINDOW_LABELS[w]:^24}" for w in ec.CAR_COLUMNS)
    lines = [f"{'':<{IND_NAME_WIDTH}}{heads}".rstrip(),
             f"{'Industry':<{IND_NAME_WIDTH}}" + "".join(
                 f"{'n':>5}{'coef':>11}{'s.e.':>8}" for _ in ec.CAR_COLUMNS),
             "-" * (IND_NAME_WIDTH + 24 * len(ec.CAR_COLUMNS))]

    for group in order:
        cells = ""
        for window in ec.CAR_COLUMNS:
            row = ind[(ind["window"] == window) & (ind["ff12"] == group)].iloc[0]
            if row["skipped"]:
                cells += f"{int(row['n']):>5}{row['skipped']:>11}{'':>8}"
            else:
                cells += (f"{int(row['n']):>5}"
                          f"{row['coef'] * PCT:>+9.1f}{row['stars']:<2}"
                          f"{'(' + format(row['se'] * PCT, '.1f') + ')':>8}")
        lines.append(f"{group:<{IND_NAME_WIDTH}}{cells}".rstrip())

    lines += ["", f"Imposition leg, {INDUSTRY_RUN.split('_')[1]} run, event {event_date}.",
              COEF_LABEL + ", HC1 standard errors.",
              "Stars: " + "  ".join(f"{mark} p < {level:g}"
                                    for level, mark in sorted(rcr.STARS.items())) + ".",
              f"Groups below {rcr.MIN_INDUSTRY_N} firms carry no estimate."]
    return lines


def write_industry_table(ind: pd.DataFrame, event_date: str, path=TABLE_INDUSTRY) -> Path:
    """Write the industry table as fixed-width text, titled as the figure it replaces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    title = "Tariff-exposure coefficient within each industry group"
    lines = [title, "=" * len(title), ""] + industry_table(ind, event_date)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _leg(run: str) -> str:
    """'imposition_primary' -> 'Imposition'."""
    return run.split("_")[0].capitalize()


# --------------------------------------------------------------------------- Entry point

def main() -> list:
    cs.apply()
    ec.select_cycle(ccd.DEFAULT_CYCLE)

    cars = ds.load_cars()
    universe, _ = ds.build_universe(cars, ds.load_texp())
    groups, _, _ = ds.assign_groups(universe)
    weights, _ = ds.load_event_weights()
    events = {run["name"]: str(cars[run["name"]]["event_date"].iloc[0].date()) for run in ec.RUNS}

    ar_by_run = daily_ar()
    rebuilt = rebuild_cars(ar_by_run)
    worst_car = reconcile_cars(rebuilt, cars)
    worst_cell = reconcile_cells(rebuilt, groups, weights, pd.read_csv(DECILE_CSV))
    print(f"rebuilt abnormal returns reproduce the stored CARs to {worst_car:.1e} and "
          f"decile_sort's cells to {worst_cell:.1e}", flush=True)

    paths = []
    for scheme in ("value", "equal"):
        caars = {}
        for run in EVENT_TIME_RUNS:
            ar, resolved = ar_by_run[run]
            weight = ds.weight_series(weights, resolved["event"], TRAJECTORY_WINDOW)
            caars[run] = event_time_caar(ar, resolved, groups, weight, scheme)
        paths.append(plot_event_time(caars, events, scheme, CHART_EVENT_TIME[scheme]))

    coefs = load_coefficients(REGRESSION_CSV, EVENT_TIME_RUNS)
    paths.append(plot_coefficients(coefs, events))
    paths.append(plot_cross_cycle(
        load_coefficients(CROSS_CYCLE_CSV, sorted(ccd.CYCLES["cross_cycle"]["events"])),
        coefs, events))
    paths.append(write_industry_table(load_industry(), events[INDUSTRY_RUN]))

    for path in paths:
        print(f"wrote {path.relative_to(BASE)}", flush=True)
    return paths


if __name__ == "__main__":
    main()
