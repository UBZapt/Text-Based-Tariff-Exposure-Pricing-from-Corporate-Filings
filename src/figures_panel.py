"""
Dissertation figures for the full-panel results: sections 7.4 and 7.5.

Three figures. The two regime charts take their threshold from `fama_macbeth_pricing`'s
`epu_thresholds()` rather than recomputing a percentile, so the shaded months are the ones that
actually did the splitting: tau is a property of the EPU series over 2017-2026, not of the 96
estimated months, and the two differ (tau_75 is 252.6 on the series against 231.1 on the estimated
months). The thresholds returned are checked against the published regime table before anything is
drawn.

The section 7.5 headline is the premium chart: the estimated monthly premium on tariff exposure,
with the high-EPU months shaded behind it. It replaced a bar chart of the six regime splits, which
put the regime means side by side but hid the fact that the elevated bucket is a handful of
clustered months rather than a condition that recurs across the sample. The EPU chart beside it
carries the series and the thresholds the shading comes from, and `fama_macbeth_pricing`'s own
chart carries the same premium shaded by policy episode instead.

The cross-section chart exists because v6 section 7.4 and section 5.2 item 6 both require the
number of firms entering each monthly cross-section and the average within-month R-squared to be
reported. It also makes the two thin months at the start of 2018 visible, which no table row does
as directly.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.patches import Patch

import chartstyle as cs
import fama_macbeth_pricing as fmp
import palette
from config import BASE, OUTPUT_DIR

# --------------------------------------------------------------------------- Configuration

LAMBDA_CSV = OUTPUT_DIR / "fm_lambda_panel.csv"
REGIME_CSV = OUTPUT_DIR / "fm_epu_regime_results.csv"

CHART_EPU = OUTPUT_DIR / "fig_epu_regime.png"
CHART_PREMIUM = OUTPUT_DIR / "fig_epu_regime_premium.png"
CHART_CROSS_SECTION = OUTPUT_DIR / "fig_fm_cross_section.png"

PRIMARY_SPEC = "full"            # the section 7.4 specification, with FS and FF12 effects
TAU_TOL = 1e-9                   # thresholds re-derived here against the published regime table
SHADE_ALPHA = 0.8
EPISODE_TICK = 0.028             # floor-tick height, in axes fraction
MAX_YTICKS = 6                   # rotated labels stack, so a dense scale runs together

PALETTE = palette.roles(series="CATEGORICAL_1")
PCT = 100.0


# --------------------------------------------------------------------------- Readers

def load_lambdas(path=LAMBDA_CSV, spec: str = PRIMARY_SPEC) -> pd.DataFrame:
    """The monthly Fama-MacBeth panel for one specification, month-ordered."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run fama_macbeth_pricing.py first.")
    frame = pd.read_csv(path)
    frame = frame[frame["spec"] == spec].copy()
    if frame.empty:
        raise ValueError(f"{path.name} holds no rows for specification {spec!r}")
    frame["ym"] = pd.PeriodIndex(frame["ym"], freq="M")
    return frame.sort_values("ym")


def load_regimes(path=REGIME_CSV) -> pd.DataFrame:
    """The section 7.5 regime table, with its star column normalised for labelling."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run fama_macbeth_pricing.py first.")
    frame = pd.read_csv(path)
    frame["stars"] = frame["stars"].fillna("")
    return frame


def epu_window(regimes: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    """The EPU series over the tau window, and the thresholds, checked against the regime table.

    epu_thresholds owns the definition; this only confirms that the numbers about to be drawn are
    the ones the published split used, so a change to TAU_WINDOW or to the EPU input cannot leave
    the figure showing a boundary the results were not computed at.
    """
    taus, series, _ = fmp.epu_thresholds()
    published = (regimes[regimes["split"] == "epu"]
                 .drop_duplicates("tau_pct").set_index("tau_pct")["tau"])
    for pct, tau in taus.items():
        gap = abs(tau - float(published[float(pct)]))
        if gap > TAU_TOL:
            raise ValueError(f"tau at p{pct} re-derives to {tau:.6f} against the published "
                             f"{published[float(pct)]:.6f}, a gap of {gap:.2e}")
    return series[series["ym"].between(*fmp.TAU_WINDOW)].copy(), taus


def high_epu_spans(series: pd.DataFrame, tau: float) -> list[tuple]:
    """The high-EPU months as contiguous (start, end) timestamp pairs.

    Runs of adjacent months are merged into one span rather than shaded month by month, so a
    six-month episode reads as one period and a lone month is not lost between two edges. Each
    span opens at the start of its first month and closes at the end of its last, so the shading
    covers exactly the months the split assigned to the elevated regime.
    """
    high = series.loc[series["epu_news"] > tau, "ym"].sort_values().tolist()
    spans = []
    for month in high:
        if spans and month == spans[-1][1] + 1:
            spans[-1][1] = month
        else:
            spans.append([month, month])
    return [(lo.to_timestamp(how="start"), hi.to_timestamp(how="end")) for lo, hi in spans]


# --------------------------------------------------------------------------- Figures

def plot_epu_regime(series: pd.DataFrame, taus: dict[int, float], episodes: list[dict],
                    path=CHART_EPU):
    """Section 7.5: the EPU series, its three reported thresholds, and the high-EPU months.

    The index is a strictly positive level with no meaningful reference point of its own, so the
    scale starts at zero. Autoscaling put the floor near 60 and left the line running off the
    bottom of the frame, which reads as data continuing below the axis.

    The shaded months come from `high_epu_spans`, the same helper the premium figure uses, so the
    two cannot disagree about which months the split called elevated. A per-point fill drew an
    isolated month too narrow to see, and there are three of them.
    """
    x = series["ym"].dt.to_timestamp(how="end")
    y = series["epu_news"].to_numpy(dtype=float)
    primary = taus[fmp.PRIMARY_TAU]

    fig, ax = plt.subplots(figsize=cs.SIZE_WIDE)
    for lo, hi in high_epu_spans(series, primary):
        ax.axvspan(lo, hi, facecolor=PALETTE["grid"], alpha=SHADE_ALPHA, lw=0, zorder=1)
    for episode in episodes:
        for day in episode["events"]:
            ax.plot([day, day], [0, EPISODE_TICK], transform=ax.get_xaxis_transform(),
                    color=PALETTE["ink"], lw=0.9, zorder=4, clip_on=False)

    ax.plot(x, y, color=PALETTE["series"], lw=1.6, zorder=5, label="EPU index")
    ax.axhline(primary, color=PALETTE["ink"], lw=cs.AXIS_WIDTH, zorder=3,
               label=f"p{fmp.PRIMARY_TAU} = {primary:.0f}")
    for pct in sorted(set(fmp.TAU_PERCENTILES) - {fmp.PRIMARY_TAU}):
        ax.axhline(taus[pct], color=PALETTE["ink_muted"], ls=":", lw=1.0, zorder=3,
                   label=f"p{pct} = {taus[pct]:.0f}")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, y.max() * 1.08)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.set_ylabel("Economic Policy Uncertainty index")
    ax.set_xlabel("Month")
    handles = [Patch(facecolor=PALETTE["grid"], alpha=SHADE_ALPHA,
                     label=f"Months above p{fmp.PRIMARY_TAU}")]
    cs.frame(ax, max_yticks=MAX_YTICKS)
    cs.figure_title(fig, "US policy uncertainty and the high-uncertainty regime")
    cs.figure_legend(fig, *_with(ax, handles), ncol=3, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return cs.save(fig, path)


def plot_regime_premium(lambdas: pd.DataFrame, series: pd.DataFrame, taus: dict[int, float],
                        path=CHART_PREMIUM):
    """Section 7.5 / H3: the monthly exposure premium against the high-uncertainty months.

    The same series `fama_macbeth_pricing` draws against the policy episodes, shaded instead by
    the EPU split the H3 test is run on, so the figure and the test partition the months
    identically. The regime means and their Newey-West tests are in fm_epu_regime_results.csv.
    """
    x = lambdas["ym"].dt.to_timestamp(how="end")
    y = lambdas["lambda_texp"].to_numpy(dtype=float) * PCT
    primary = taus[fmp.PRIMARY_TAU]

    fig, ax = plt.subplots(figsize=cs.SIZE_WIDE)
    for lo, hi in high_epu_spans(series, primary):
        ax.axvspan(lo, hi, facecolor=PALETTE["grid"], alpha=SHADE_ALPHA, lw=0, zorder=1)
    cs.zero_line(ax)
    ax.plot(x, y, color=PALETTE["series"], lw=1.6, zorder=5,
            label="$\\lambda_{1,t}$, estimated month by month")

    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.set_ylabel("$\\lambda_{1,t}$   (% monthly return per s.d.)")
    ax.set_xlabel("Month")
    handles = [Patch(facecolor=PALETTE["grid"], alpha=SHADE_ALPHA,
                     label=f"High-uncertainty months, EPU above p{fmp.PRIMARY_TAU} "
                           f"({primary:.0f})")]
    cs.frame(ax, max_yticks=MAX_YTICKS)
    cs.figure_title(fig, "Monthly tariff-exposure premium and the high-uncertainty regime")
    cs.figure_legend(fig, *_with(ax, handles), ncol=2, y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return cs.save(fig, path)


def plot_cross_section(lambdas: pd.DataFrame, path=CHART_CROSS_SECTION):
    """Section 7.4 diagnostics: the size and fit of each monthly cross-section."""
    x = lambdas["ym"].dt.to_timestamp(how="end")
    fig, axes = plt.subplots(2, 1, figsize=cs.SIZE_STACK2, sharex=True)

    for ax, column, label in ((axes[0], "n_firms", "Firms in the cross-section"),
                              (axes[1], "r2", "Within-month $R^2$")):
        ax.plot(x, lambdas[column], color=PALETTE["series"], lw=1.5, zorder=3)
        ax.set_ylabel(label)
        ax.set_xlim(x.min(), x.max())
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    axes[-1].set_xlabel("Month")
    for ax in axes:
        cs.frame(ax)
    cs.figure_title(fig, "Size and fit of each monthly Fama-MacBeth cross-section")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return cs.save(fig, path)


def _with(ax, extra: list) -> tuple[list, list]:
    """Existing plotted handles plus hand-built ones, as a (handles, labels) pair."""
    handles, labels = ax.get_legend_handles_labels()
    handles = handles + extra
    return handles, labels + [h.get_label() for h in extra]


# --------------------------------------------------------------------------- Entry point

def main() -> list:
    cs.apply()
    regimes = load_regimes()
    series, taus = epu_window(regimes)
    lambdas = load_lambdas()

    paths = [plot_epu_regime(series, taus, fmp.episode_spans()),
             plot_regime_premium(lambdas, series, taus),
             plot_cross_section(lambdas)]
    print(f"tau reproduces the published regime table at all "
          f"{len(fmp.TAU_PERCENTILES)} percentiles", flush=True)
    for path in paths:
        print(f"wrote {path.relative_to(BASE)}", flush=True)
    return paths


if __name__ == "__main__":
    main()
