"""
Dissertation figures for the collinearity question v6 section 5.2 ranks first among threats:
"Tariff exposure is correlated with industry, size, and global supply-chain proxies. Fatal if
unaddressed."

Three correlation grids and one scatter grid. All are built from the samples the regressions were
actually fitted on, not from a looser join: the event-study frames come from
`run_car_regression.run_cycle`, which is the same code path section 7.2 uses and therefore carries
its listwise deletion, and the panel grid comes from `fama_macbeth_pricing.load_panel`, which
applies Script 7a's exclusion ladder and asserts the result carries no nulls.

The third grid answers the same question of the risk model rather than of the controls: whether
TExp is a repackaging of the FF5+MOM loadings the abnormal returns are already purged of. It costs
nothing to draw, because `estimate_car` persists each firm's six loadings in the CAR table and the
event-study frame is built on top of it, so no beta is re-estimated here.

The scatter grid is the same event-study frame read pairwise. A correlation cell reports one number
per pair and cannot distinguish a linear relation from a mass point plus a tail, which is exactly
the shape TExp has: a third of the sample sits at exactly zero. The scatters show it.

Pearson, not Spearman: collinearity in OLS is a question about linear dependence among the
regressors, which is what a variance inflation factor reads and what Pearson measures. The
dependent variable is deliberately excluded from every grid - a raw bivariate TExp-CAR correlation
invites being read as the result, when the result is a coefficient conditional on foreign sales,
four controls and twelve industry effects.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chartstyle as cs
import fama_macbeth_pricing as fmp
import palette
import run_car_regression as rcr

# --------------------------------------------------------------------------- Configuration

BASE = rcr.BASE
OUTPUT_DIR = BASE / "output"

CHART_EVENT_STUDY = OUTPUT_DIR / "fig_corr_event_study.png"
CHART_FM_PANEL = OUTPUT_DIR / "fig_corr_fm_panel.png"
CHART_FACTOR_LOADINGS = OUTPUT_DIR / "fig_corr_factor_loadings.png"
CHART_SCATTER_GRID = OUTPUT_DIR / "fig_texp_scatter_grid.png"

# The cross-section every event-study exhibit here is read on: the primary imposition leg at the
# narrowest window, which is the section 7.2 headline regression.
EVENT_STUDY_CYCLE = "2025"
EVENT_STUDY_RUN = "imposition_primary"

# Display names, in the order the design matrix lists the regressors. Keyed on the event-study
# column names; the panel's standardised exposure column is mapped onto the same first slot.
LABELS = {
    rcr.TEXP_COLUMN: "TExp",
    rcr.FS_COLUMN: "Foreign sales",
    "ln_me_lag": "Log market equity",
    "bm": "Book-to-market",
    "lev": "Leverage",
    "mom12": "Prior 12m return",
}

# The FF5+MOM loadings, as estimate_car names them and in the order it estimates them.
FACTOR_LABELS = {rcr.TEXP_COLUMN: "TExp"} | {
    f"beta_{factor}": name for factor, name in
    zip(rcr.ec.FACTORS, ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"])}

DARK_TEXT_BELOW = 0.60           # |r| above this needs light type to stay legible on the fill
SCATTER_COLUMNS = 3              # five pairs over two rows; the sixth cell is removed
POINT_SIZE = 7
POINT_ALPHA = 0.28               # the TExp = 0 mass is handled with alpha, never with jitter

PALETTE = palette.roles(points="CATEGORICAL_2", fit="CATEGORICAL_1")


def event_study_frame() -> tuple[pd.DataFrame, str]:
    """The section 7.2 estimation frame for the primary imposition leg at the narrowest window.

    Returned whole rather than column-selected, because three exhibits read different columns of
    the one frame and re-running run_cycle for each would fit the section 7.2 regressions three
    times over to obtain the same sample.
    """
    cycle = rcr.run_cycle(EVENT_STUDY_CYCLE, quiet=True)
    per_window = next(w for name, _, w in cycle["fits"] if name == EVENT_STUDY_RUN)
    frame = per_window[rcr.ec.CAR_COLUMNS[0]]["frame"]
    return frame, f"{len(frame):,} firms"


def fm_panel_frame() -> tuple[pd.DataFrame, str]:
    """The section 7.4 estimation sample, through the reader that owns its exclusion ladder."""
    _, sample = fmp.load_panel()
    columns = [fmp.TEXP_COLUMN, fmp.FS_COLUMN] + fmp.CONTROLS
    frame = sample[columns].rename(columns={fmp.TEXP_COLUMN: rcr.TEXP_COLUMN})
    return frame, f"{len(frame):,} firm-months"


def plot_grid(frame: pd.DataFrame, labels: dict[str, str], title: str, path):
    """One Pearson correlation grid, annotated per cell on a fixed [-1, 1] scale.

    The scale is fixed rather than fitted to the data so the grids are directly comparable: a cell
    that reads pale in one and pale in another really does carry the same correlation.
    """
    corr = frame[list(labels)].corr(method="pearson")
    names = [labels[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=cs.SIZE_HEATMAP)
    image = ax.imshow(corr.to_numpy(), cmap=cs.diverging_cmap(), vmin=-1, vmax=1)

    for i in range(len(names)):
        for j in range(len(names)):
            value = corr.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=cs.TICK_SIZE,
                    color=palette.SURFACE if abs(value) > DARK_TEXT_BELOW else palette.INK)

    ax.set_xticks(np.arange(len(names)), names)
    ax.set_yticks(np.arange(len(names)), names)
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
    bar.outline.set_linewidth(cs.AXIS_WIDTH)
    bar.outline.set_edgecolor(palette.INK)
    bar.set_label("Pearson correlation", fontsize=cs.LABEL_SIZE)
    bar.ax.tick_params(labelsize=cs.TICK_SIZE)

    cs.frame(ax, box=True, rotate_x=True, rotate_y=False)
    cs.figure_title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return cs.save(fig, path)


def plot_scatter_grid(frame: pd.DataFrame, title: str, path=CHART_SCATTER_GRID):
    """TExp against each other regressor, one small panel per pair with its OLS fit.

    Exposure is on the x axis of every panel, as in the foreign-sales scatter, so the five read as
    one exhibit and the mass point at TExp = 0 lines up down the column. Each panel's fitted line
    is the bivariate OLS of that regressor on TExp; the multivariate coefficient it is not is the
    subject of the regression tables, not of a scatter.
    """
    pairs = [c for c in LABELS if c != rcr.TEXP_COLUMN]
    rows = -(-len(pairs) // SCATTER_COLUMNS)
    fig, axes = plt.subplots(rows, SCATTER_COLUMNS,
                             figsize=(cs.SIZE_PAIR[0], 2.5 * rows + 0.6))
    flat = axes.ravel()

    x = frame[rcr.TEXP_COLUMN].to_numpy(dtype=float)
    for ax, column in zip(flat, pairs):
        y = frame[column].to_numpy(dtype=float)
        ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA, color=PALETTE["points"],
                   linewidths=0, zorder=3)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color=PALETTE["fit"], lw=1.5, zorder=4)
        cs.panel_title(ax, f"{LABELS[column]}   r = {np.corrcoef(x, y)[0, 1]:+.2f}")
        ax.set_ylabel(LABELS[column])
        ax.set_xlim(0, x.max())
        cs.frame(ax, rotate_x=False, max_yticks=5)

    for ax in flat[len(pairs):]:
        ax.remove()
    for ax in flat[len(pairs) - SCATTER_COLUMNS:len(pairs)]:
        ax.set_xlabel("TExp")

    cs.figure_title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return cs.save(fig, path)


def main() -> list:
    cs.apply()
    event, event_n = event_study_frame()
    panel, panel_n = fm_panel_frame()

    paths = [plot_grid(event, LABELS, f"Regressor correlations, 2025 event study ({event_n})",
                       CHART_EVENT_STUDY),
             plot_grid(panel, LABELS, f"Regressor correlations, monthly panel ({panel_n})",
                       CHART_FM_PANEL),
             plot_grid(event, FACTOR_LABELS,
                       f"Tariff exposure against FF5+MOM loadings ({event_n})",
                       CHART_FACTOR_LOADINGS),
             plot_scatter_grid(event, f"Tariff exposure against each regressor ({event_n})")]
    for path in paths:
        print(f"wrote {path.relative_to(BASE)}", flush=True)
    return paths


if __name__ == "__main__":
    main()
