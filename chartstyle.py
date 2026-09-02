"""
The one place the project's chart *furniture* is defined, as `palette` is for its colours.

Every figure in the dissertation is formatted against one worked example, whose look is R base
graphics: a white ground, sans-serif type, a bold centred title, no gridlines, tick labels rotated
ninety degrees, and a boxed legend. Encoding that once means a module draws its data and calls
`frame` and `save`; it never restates the convention, and a revision to the look is one edit
rather than six.

The one departure from that example is that the axis lines span the full limits and therefore meet
at the origin corner. The example draws each spine only across its tick range, which leaves a gap
below the lowest tick and to the left of the first one; on a time series whose first month falls
before the first year tick that gap reads as missing data rather than as a style.

The figures carry a title, axis labels, data labels and a legend, and nothing else. Descriptive
captions belong above the figure in the dissertation document, not inside the image, so there is
deliberately no helper here for a subtitle or a footnote.
"""

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: never opens a window, safe to run unattended
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                  # noqa: E402
from matplotlib.ticker import MaxNLocator                              # noqa: E402

import palette                                                         # noqa: E402

# --------------------------------------------------------------------------- Configuration

DPI = 300

# Figure sizes in inches. The wide single panel is the worked example's ~2:1 shape; the others
# hold that width so a document laying several out in sequence keeps one column measure.
SIZE_WIDE = (7.0, 3.6)           # one panel, time series or a single row of bars
SIZE_PAIR = (8.0, 3.6)           # two panels side by side
SIZE_STACK2 = (7.0, 5.6)         # two panels stacked, shared x
SIZE_STACK3 = (7.0, 8.0)         # three panels stacked, one per event window
SIZE_HEATMAP = (5.4, 4.6)        # square-ish correlation grid

TITLE_SIZE = 11
PANEL_TITLE_SIZE = 9.5
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8

AXIS_WIDTH = 0.8
BAR_EDGE_WIDTH = 0.25            # bar outlines: present so a hatch reads, too thin to frame a fill
TICK_LENGTH = 4
ROTATE_X_MIN_CHARS = 3           # labels this long or longer are rotated, as the example's are
CI_Z = 1.959963984540054         # two-sided 95%, the interval every whisker in the project draws


def apply() -> None:
    """Set the rcParams the worked example's look implies. Idempotent; call once per script."""
    plt.rcParams.update({
        "figure.facecolor": palette.SURFACE,
        "axes.facecolor": palette.SURFACE,
        "savefig.facecolor": palette.SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.grid": False,
        "axes.edgecolor": palette.INK,
        "axes.linewidth": AXIS_WIDTH,
        "axes.labelcolor": palette.INK,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "text.color": palette.INK,
        "xtick.color": palette.INK,
        "ytick.color": palette.INK,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.width": AXIS_WIDTH,
        "ytick.major.width": AXIS_WIDTH,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": palette.INK,
        "legend.facecolor": palette.SURFACE,
    })


# --------------------------------------------------------------------------- Axis furniture

def frame(ax, *, box: bool = False, rotate_x: bool | str = "auto",
          rotate_y: bool = True, max_yticks: int | None = None) -> None:
    """Apply the project's axis treatment. Call last, after every artist and limit is set.

    The left and bottom spines span the full axis limits, so they meet at the origin corner. An
    earlier version bounded each spine to its tick range, in the reference example's style; on a
    monthly series starting mid-year that left a visible gap between the two lines, which reads as
    a stretch of missing data rather than as a convention.

    `box=True` draws all four spines, which is what the example uses for small multiples;
    single panels get the two-spine form. `max_yticks` thins a dense y axis: rotated tick labels
    stack along the axis rather than across it, so a scale that reads cleanly upright can run its
    labels together once turned.
    """
    if max_yticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))
    if box:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(palette.INK)
            ax.spines[side].set_linewidth(AXIS_WIDTH)
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side, limits in (("left", ax.get_ylim()), ("bottom", ax.get_xlim())):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(palette.INK)
            ax.spines[side].set_linewidth(AXIS_WIDTH)
            ax.spines[side].set_bounds(min(limits), max(limits))

    if rotate_y:
        plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    if rotate_x == "auto":
        labels = [t.get_text() for t in ax.get_xticklabels()]
        rotate_x = any(len(t) >= ROTATE_X_MIN_CHARS for t in labels)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")


def figure_title(fig, text: str) -> None:
    """The figure's one title: bold, centred, black - the example's convention."""
    fig.suptitle(text, fontsize=TITLE_SIZE, fontweight="bold", color=palette.INK)


def panel_title(ax, text: str) -> None:
    """A sub-panel's label, plain weight so the figure title stays the only bold element."""
    ax.set_title(text, fontsize=PANEL_TITLE_SIZE, fontweight="normal", color=palette.INK)


def legend(ax, handles=None, labels=None, *, loc: str = "upper right", **kw):
    """A boxed legend: thin black border, opaque white fill, square corners."""
    args = (handles, labels) if handles is not None else ()
    leg = ax.legend(*args, loc=loc, **kw)
    leg.get_frame().set_linewidth(AXIS_WIDTH)
    return leg


def figure_legend(fig, handles, labels, *, ncol: int = 3, y: float = 0.90):
    """A boxed legend centred under the figure title, for keys shared across panels.

    A panel-level legend has to sit inside one panel's data area, and on the event-time figure it
    covered the first ten days of the imposition trajectory whichever corner it was placed in.
    Centred between the title and the panels it belongs to both and hides neither. `y` is a figure
    fraction, so the caller reserves the band for it in `tight_layout`'s rect.
    """
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)
    leg.get_frame().set_linewidth(AXIS_WIDTH)
    return leg


def zero_line(ax, *, axis: str = "y") -> None:
    """The solid black reference at zero, which makes a sign change legible."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(0, color=palette.INK, lw=AXIS_WIDTH, zorder=1)


def whiskers(ax, x, coef, se, *, color=palette.INK, z: float = CI_Z, capsize: float = 3.0):
    """Two-sided 95% confidence bars on point or bar estimates."""
    return ax.errorbar(x, coef, yerr=[z * s for s in se], fmt="none", ecolor=color,
                       elinewidth=AXIS_WIDTH, capsize=capsize, capthick=AXIS_WIDTH, zorder=5)


def headroom(ax, *, top: float = 0.0, bottom: float = 0.0) -> None:
    """Widen the y limits by a share of the current range, to clear a legend or a label."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - span * bottom, hi + span * top)


def outside_labels(ax, x, coef, se, marks, *, z: float = CI_Z, pad_frac: float = 0.03) -> None:
    """Data labels placed clear of each estimate's confidence bar.

    Placed above a positive estimate and below a negative one, so a label never sits on the zero
    line where the two signs meet. The limits are then widened to hold whatever was placed: a
    label on the widest interval in a panel otherwise lands on the axis and is read as clipped.
    `pad_frac` is a share of the axis range, so the gap holds whatever the units.
    """
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * pad_frac
    placed = []
    for xi, c, s, mark in zip(x, coef, se, marks):
        if not isinstance(mark, str) or not mark.strip() or np.isnan(c) or np.isnan(s):
            continue
        up = c >= 0
        placed.append((xi, c + z * s + pad if up else c - z * s - pad, up, mark))
    if not placed:
        return
    ys = [y for _, y, _, _ in placed]
    ax.set_ylim(min(lo, min(ys) - pad), max(hi, max(ys) + pad))
    for xi, y, up, mark in placed:
        ax.annotate(mark, (xi, y), ha="center", va="bottom" if up else "top",
                    fontsize=TICK_SIZE, color=palette.INK)


def star_labels(ax, x, coef, se, stars, **kw) -> None:
    """Significance stars alone, where the estimate itself is read off the axis."""
    outside_labels(ax, x, coef, se, stars, **kw)


def value_labels(ax, x, coef, se, stars, *, fmt: str = "{:+.1f}", **kw) -> None:
    """The estimate and its stars as one data label, for a chart read value by value.

    A bar chart of three or four estimates is quoted in the text, so the number belongs on the
    bar; a chart of eleven is read for its shape and takes `star_labels` instead.
    """
    marks = ["" if np.isnan(c) else fmt.format(c) + (s if isinstance(s, str) else "")
             for c, s in zip(coef, stars)]
    outside_labels(ax, x, coef, se, marks, **kw)


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue-white-red, for correlation grids read on a fixed [-1, 1] scale."""
    return LinearSegmentedColormap.from_list(
        "project_diverging", [palette.CATEGORICAL_1, palette.SURFACE, palette.DIVERGE_WARM])


def save(fig, path: Path) -> Path:
    """Write at print resolution on the white ground and release the figure."""
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=palette.SURFACE)
    plt.close(fig)
    return path
