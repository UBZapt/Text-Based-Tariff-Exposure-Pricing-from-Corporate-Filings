"""Step 7b of 2 - section 7.4 Fama-MacBeth pricing test (H2, H5).

One cross-sectional OLS per calendar month on the panel Script 7a assembled,

    ret_i,t = lambda_0,t + lambda_1,t*texp_z + c_t*FS + gamma_t'X + delta_ind + e_i,t

with every regressor dated t-1. The monthly slopes {lambda_1,t} are collected and their
time-series mean tested against zero with Newey-West standard errors. This is not an event study:
the dependent variable is the ordinary monthly return and every month enters, quiet or not.

The design predicts a weak mean and says so in advance - averaging a handful of large event-month
slopes against ninety-odd near-zero ones attenuates it by construction - so the informative output
is the time series with the tariff episodes marked, not the average, and the reported comparison is
against the section 7.2 event-window coefficients rather than against zero alone.

H5 is read off the second specification: the same months, the same firms, with the foreign-sales
control and the FF12 industry effects removed. Only the regressor set changes, so any movement in
lambda_1_bar is the specification and not the sample.

PolRisk is excluded, per the constraint note at the head of section 7: the Hassan series ends
March 2021 and including it would truncate the panel past the events the dissertation is built on.

    python src/fama_macbeth_pricing.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import hypergeom, ttest_ind

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.dates import DateFormatter, YearLocator            # noqa: E402

import chartstyle as cs                                            # noqa: E402
import palette
import build_fm_panel as bfp                                       # noqa: E402
import clean_controls_data as ccd                                  # noqa: E402
import run_car_regression as rcr                                   # noqa: E402
from config import BASE, OUTPUT_DIR                                # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
PANEL_CSV = bfp.PANEL_OUT
TEXP_PANEL_CSV = bfp.TEXP_PANEL_CSV
CAR_RESULTS_CSV = OUTPUT_DIR / "car_regression_results.csv"       # section 7.2, for the comparison
SIGNFLIP_CSV = OUTPUT_DIR / "signflip_test_results.csv"           # section 7.2 per-sample TExp sd

# Which section 7.2 event supplies the exposure vintage the per-standard-deviation conversion is
# taken from. Named here rather than as a bare "2025-04-02" inside the reader, so it follows the
# cycle registry if that changes rather than silently desynchronising the reported units.
SECTION_72_EVENT = "impose"
EPU_CSV = bfp.EPU_CSV
LAMBDA_OUT = OUTPUT_DIR / "fm_lambda_panel.csv"
CHART_OUT = OUTPUT_DIR / "fm_lambda_chart.png"
REGIME_OUT = OUTPUT_DIR / "fm_epu_regime_results.csv"
HEADLINE_OUT = OUTPUT_DIR / "fm_headline_results.csv"
REPORT_OUT = OUTPUT_DIR / "fama_macbeth_validation_report.txt"

TEXP_COLUMN = bfp.TEXP_COLUMN                 # texp_z, standardised within (vintage x subsample)
TEXP_RAW = bfp.TEXP_RAW
FS_COLUMN = bfp.FS_COLUMN
CONTROLS = bfp.CONTROLS                       # ln_me_lag, bm, lev, mom12
INDUSTRY_COLUMN = bfp.INDUSTRY_COLUMN
FF12_REFERENCE = rcr.FF12_REFERENCE           # omitted dummy; shifts the intercept, never lambda_1

# The two specifications, estimated on identical rows. The second is the H5 read: does the slope
# survive the inclusion of foreign-sales share and industry effects?
SPECS = [
    {"name": "full", "fs": True, "ff12": True},
    {"name": "no_fs_no_ff12", "fs": False, "ff12": False},
]
PRIMARY_SPEC = "full"

# Newey-West lag on the second stage. Six is the design's nomination; the grid is reported beside
# it so the choice is visibly not driving the result. Lag 0 leaves heteroskedasticity only.
NW_LAGS = 6
NW_LAG_GRID = (0, 3, 6, 12)

# statsmodels' HAC finite-sample switches, stated rather than inherited. All three are the library
# defaults, so nothing here changes a number - the point is that a reader can see the convention
# instead of having to know what statsmodels does when they are omitted.
#
#   use_correction=False  no n/(n-k) scaling of the sandwich. At T = 96 the factor is 1.005 and
#                         immaterial; at the T = 5 and T = 9 regime buckets it would be 1.12 and
#                         1.06, so those errors are optimistic by about that much.
#   adjust_df=False       statsmodels applies no cluster-style degrees-of-freedom adjustment to HAC.
#   use_t=False           inference is standard-normal, not t(T-1). Also immaterial at T = 96 and
#                         not at T = 5.
#
# They are left at the defaults deliberately: the primary series is 96 months, where all three are
# negligible, and switching them only for the small buckets would put two conventions in one table.
# The consequence for the sub-floor rows is that their p-values are optimistic on BOTH counts, and
# the report says so beside the hac_reliable flag.
HAC_KWDS = {"use_correction": False, "adjust_df": False, "use_t": False}

# Degrees-of-freedom floor, not a sample choice: the full specification carries 18 parameters, so
# 50 firms leaves ~32 residual degrees of freedom. Non-binding on this panel (the thinnest month
# has 97 firms) and set from the parameter count rather than from which months it would remove.
MIN_FIRMS_PER_MONTH = 50

# Reported as a sensitivity, never as the primary sample. Two months - 2018-01 and 2018-02 - sit
# below it because Compustat's geographic-segment file begins at datadate 2017-01-31, so firms
# still on FY2016 fundamentals have no foreign-sales row.
THIN_MONTH_FLOOR = 200
SMALL_INDUSTRY_CELL = 5           # industry-months below this contribute little identifying variation
N_LARGEST_CHECK = 10              # largest |lambda_1,t| months tested for episode concentration

LAMBDA_COLUMNS = ["ym", "spec", "lambda_texp", "se", "t", "p", "stars", "lambda_fs",
                  "n_firms", "r2", "adj_r2", "epu_lag", "episode"]

# --- section 7.5: EPU regime conditioning (H3) ------------------------------ #
# tau is the percentile of the EPU series over 2017-2026, per section 7.5 - not over
# the 96 estimated months. The series ends 2026-05, so the realised window is 113 months and
# includes 17 that sit outside the Fama-MacBeth sample; the sample-window alternative is reported
# as a contrast and not used. Fixed here, before estimation, and never searched over.
TAU_WINDOW = (pd.Period("2017-01", freq="M"), pd.Period("2026-12", freq="M"))
TAU_PERCENTILES = (50, 75, 90)     # 75 is section 7.5's nomination; 50 and 90 are its robustness
PRIMARY_TAU = 75

# Below this many months a Newey-West standard error at NW_LAGS is not to be trusted: at p90 the
# high bucket holds 9 months and NW-6 returns a standard error *smaller* than the iid one, which is
# a finite-sample artefact rather than a finding. statsmodels raises nothing, so the flag is what
# surfaces it. Set at 4 x NW_LAGS from the lag length, not from which regimes it happens to catch,
# and it flags rather than suppresses - the iid error is reported alongside.
MIN_REGIME_MONTHS_FOR_HAC = 4 * NW_LAGS

# Section 7.5 asks whether a high-EPU premium is only 2020. One named calendar year answers it
# without asserting a pandemic date range this project has nowhere else defined.
COVID_YEAR = 2020
MONTHS_PER_LINE = 8               # month lists wrap rather than run off the report width

REGIME_COLUMNS = ["split", "tau_pct", "tau", "regime", "n_months", "series_contiguous",
                  "lambda_bar", "lambda_bar_pct",
                  "nw_se", "nw_t", "nw_p", "stars", "iid_se", "hac_reliable", "diff",
                  "welch_t", "welch_p", "welch_df", "welch_stars",
                  "hac_diff_se", "hac_diff_t", "hac_diff_p", "hac_diff_stars"]
OUTSIDE_LABEL = "outside both episodes"

# Chart. One data series, so only slot 1 and the neutrals are used and there is no legend - the
# title names the series. Axis furniture comes from chartstyle.
PALETTE = palette.roles(series="CATEGORICAL_1")
EPISODE_ALPHA = 0.8
EPISODE_TICK = 0.028              # floor-tick height, in axes fraction
N_LABELLED_EXTREMES = 2           # direct labels are selective by design; never one per point
MAX_YTICKS = 6                    # rotated labels stack, so a dense scale runs together
Y_PAD_LOW, Y_PAD_HIGH = 0.24, 0.24   # headroom the extreme and episode labels sit in

RULE = "=" * 78

# The reporting helpers are local rather than imported: rcr's _say appends to rcr's own report
# list, and rerouting it would mean editing a script whose section 7.2 output must stay
# byte-identical. rcr.stars, rcr.significance_label and rcr._listed are pure and reused as-is.
_REPORT: list[str] = []


def _say(line: str = "") -> None:
    _REPORT.append(line)


def _section(title: str) -> None:
    _say()
    _say(title)
    _say("-" * len(title))


def _describe(frame: pd.DataFrame, cols: list[str], width: int = 13) -> None:
    """Distribution and missingness for a set of columns, in rcr._describe's layout."""
    _say(f"  {'variable':<16}{'n':>7}{'null':>7}" + "".join(
        f"{h:>{width}}" for h in ("mean", "sd", "min", "median", "max")))
    for col in cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        values = series.dropna()
        cells = "".join(format(x, f">{width},.4f") for x in
                        (values.mean(), values.std(), values.min(),
                         values.median(), values.max()))
        _say(f"  {col:<16}{len(values):>7,}{int(series.isna().sum()):>7,}{cells}")


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_panel(path: Path = PANEL_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full candidate panel and the estimation sample it reconciles to.

    Both are returned because the report's funnel is a property of the candidates, not of the
    survivors: Script 7a wrote every excluded firm-month with its reason precisely so this script
    needs no second source for it.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run build_fm_panel.py first.")
    panel = pd.read_csv(path, dtype={"accession": str, "gvkey": str},
                        parse_dates=["date_t", "pit_date", "filing_date", "datadate"])
    panel["ym"] = pd.PeriodIndex(panel["ym"], freq="M")
    panel["exclusion_reason"] = panel["exclusion_reason"].fillna("")

    sample = panel[panel["exclusion_reason"].eq("")].copy()
    needed = [TEXP_COLUMN, FS_COLUMN, INDUSTRY_COLUMN] + CONTROLS + ["ret"]
    if sample[needed].isna().any().any():
        null_cols = [c for c in needed if sample[c].isna().any()]
        raise ValueError(f"the estimation sample carries nulls in {null_cols}; Script 7a's "
                         f"exclusion ladder should have removed those rows")
    if sample.duplicated(["permno", "ym"]).any():
        raise ValueError("the estimation sample is not unique on (permno, month)")
    return panel, sample


def load_reference_sd(path: Path = TEXP_PANEL_CSV) -> tuple[float, int]:
    """Cross-sectional sd of raw TExp in the whole section 7.2 exposure vintage.

    Section 7.2 reports its coefficient in raw units and instructs the reader to multiply by the
    sd of its own cross-section to read it per standard deviation. Two candidate cross-sections
    exist and they differ: the whole 2025-04-02 vintage (2,868 firms) and the estimation sample
    section 7.2 actually fits (about 1,554 firms after the FS and control exclusions), whose sd is
    roughly 4.5% higher. This returns the vintage-wide figure and the report carries both, because
    the estimation sample differs per window and per leg while the vintage does not - so only the
    vintage gives one stable conversion factor for a table that spans them all.

    The reference date is the registry's own, not a literal: this must follow the cycle definition
    that decides which 10-K the event study reads.
    """
    reference = ccd.CYCLES[rcr.BASELINE_CYCLE]["texp_ref"][SECTION_72_EVENT]
    vintage = pd.read_csv(path, usecols=["reference_date", TEXP_RAW])
    cross = vintage.loc[vintage["reference_date"] == reference, TEXP_RAW]
    if cross.empty:
        raise ValueError(f"{path.name} holds no cross-section at {reference}; "
                         f"available: {sorted(vintage['reference_date'].unique())}")
    return float(cross.std(ddof=1)), len(cross)


def sample_reference_sd(path: Path = SIGNFLIP_CSV) -> dict[str, float]:
    """sd of raw TExp inside section 7.2's own estimation samples, per window.

    The event study's H1 test records this per leg and window, so it is read rather than
    recomputed. Reported beside the vintage-wide figure so the two conversion factors are visibly
    different numbers instead of one number presented as both. Absent file returns {} - this is a
    reporting nicety, not an input the test depends on.
    """
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "sd_texp_left" not in frame.columns:
        return {}
    rows = frame[frame["label"].str.startswith("b^", na=False)]
    return {str(window): float(block["sd_texp_left"].iloc[0])
            for window, block in rows.groupby("window")}


def episode_spans() -> list[dict]:
    """The two tariff episodes, derived from the cycle registry rather than restated.

    clean_controls_data.CYCLES is the single definition of what a policy cycle is, so the shaded
    spans are its own event dates: the cross-cycle registry's first escalation to its last
    de-escalation, and the 2025 cycle's imposition to its reversal.
    """
    cross = sorted(pd.Timestamp(d) for d in ccd.CROSS_CYCLE_EVENTS.values())
    primary = sorted(pd.Timestamp(d) for d in ccd.CYCLES["2025"]["events"].values())
    return [{"label": "2018-19 Section 301", "events": cross,
             "start": cross[0], "end": cross[-1]},
            {"label": "2025 IEEPA", "events": primary,
             "start": primary[0], "end": primary[-1]}]


def label_episodes(months: pd.PeriodIndex, episodes: list[dict]) -> pd.Series:
    """Tag each month with the episode containing it, at month granularity."""
    labels = pd.Series("", index=range(len(months)), dtype=object)
    for episode in episodes:
        inside = ((months >= episode["start"].to_period("M"))
                  & (months <= episode["end"].to_period("M")))
        labels[inside] = episode["label"]
    return labels


# --------------------------------------------------------------------------- #
# First stage: one cross-section per month                                    #
# --------------------------------------------------------------------------- #
def monthly_design(frame: pd.DataFrame, spec: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Regressors for one month, with the FF12 reference category dropped by name.

    Dropping by name rather than position keeps the intercept comparable across months even when
    an industry's membership changes, the convention rcr.design_matrix sets. lambda_1 is invariant
    to which category is omitted; the intercept is not.
    """
    regressors = [TEXP_COLUMN] + ([FS_COLUMN] if spec["fs"] else []) + CONTROLS
    design = frame[regressors].astype(float)
    if spec["ff12"]:
        dummies = pd.get_dummies(frame[INDUSTRY_COLUMN], prefix=INDUSTRY_COLUMN, dtype=float)
        reference = f"{INDUSTRY_COLUMN}_{FF12_REFERENCE}"
        if reference not in dummies.columns:
            raise ValueError(f"reference industry {FF12_REFERENCE!r} is absent from the "
                             f"cross-section; present: {sorted(frame[INDUSTRY_COLUMN].unique())}")
        design = pd.concat([design, dummies.drop(columns=reference)], axis=1)
    design["_cons"] = 1.0
    return frame["ret"].astype(float), design


def run_spec(sample: pd.DataFrame, spec: dict,
             episodes: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate one specification month by month, returning (slope panel, full coefficient panel).

    The whole coefficient vector is kept on the single pass rather than re-fitting to report the
    controls: the panel is the same, so a second pass would only be a chance for the two to differ.
    Errors here are the ordinary OLS ones - the within-month standard error is context, and
    inference on the premium is the second stage's job.
    """
    rows, params, thin = [], [], []
    for month, frame in sample.groupby("ym", sort=True):
        if len(frame) < MIN_FIRMS_PER_MONTH:
            thin.append((str(month), len(frame)))
            continue
        y, design = monthly_design(frame, spec)
        fit = sm.OLS(y, design, missing="raise").fit()
        rows.append({
            "ym": month, "spec": spec["name"],
            "lambda_texp": float(fit.params[TEXP_COLUMN]),
            "se": float(fit.bse[TEXP_COLUMN]), "t": float(fit.tvalues[TEXP_COLUMN]),
            "p": float(fit.pvalues[TEXP_COLUMN]),
            "stars": rcr.stars(float(fit.pvalues[TEXP_COLUMN])),
            "lambda_fs": float(fit.params[FS_COLUMN]) if spec["fs"] else np.nan,
            "n_firms": int(fit.nobs), "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "epu_lag": float(frame["epu_lag"].iloc[0]),
        })
        # Industry dummies are dropped from the coefficient panel: their membership changes month
        # to month, so a time-series mean of one dummy is not a quantity with an interpretation.
        params.append({term: float(value) for term, value in fit.params.items()
                       if not term.startswith(f"{INDUSTRY_COLUMN}_")})
    if thin:
        raise ValueError(f"{len(thin)} month(s) below MIN_FIRMS_PER_MONTH={MIN_FIRMS_PER_MONTH}: "
                         f"{thin}; the floor is a degrees-of-freedom guard and should not bind")

    out = pd.DataFrame(rows)
    out["episode"] = label_episodes(pd.PeriodIndex(out["ym"]), episodes).to_numpy()
    return out[LAMBDA_COLUMNS], pd.DataFrame(params)


# --------------------------------------------------------------------------- #
# Second stage: Newey-West on the time series of slopes                       #
# --------------------------------------------------------------------------- #
def newey_west(values: pd.Series, lags: int = NW_LAGS) -> dict:
    """Time-series mean of the monthly slopes, with a Newey-West standard error.

    A constant-only OLS on {lambda_1,t} with a HAC covariance is exactly the Fama-MacBeth second
    stage: the coefficient is the mean and its standard error carries the autocorrelation
    correction. The finite-sample switches are named in HAC_KWDS rather than inherited silently.

    The caller is responsible for handing this a CONTIGUOUS monthly series: a Bartlett kernel
    reads row adjacency as month adjacency, so a series with holes in it gets the wrong weights.
    regime_fit exists because the regime means used to violate that.
    """
    array = np.asarray(values, dtype=float)
    fit = sm.OLS(array, np.ones((len(array), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, **HAC_KWDS})
    return {"mean": float(fit.params[0]), "se": float(fit.bse[0]), "t": float(fit.tvalues[0]),
            "p": float(fit.pvalues[0]), "n_months": len(array), "lags": lags}


def control_means(params: pd.DataFrame) -> pd.DataFrame:
    """lambda_bar and its Newey-West test for every term, not just exposure.

    Lets the report show whether the controls behave as the asset-pricing literature expects,
    which is the evidence that the monthly cross-sections are estimating something real.
    """
    rows = []
    for term in params.columns:
        test = newey_west(params[term])
        rows.append({"term": term, **test, "stars": rcr.stars(test["p"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Section 7.5: EPU regime conditioning (H3)                                   #
# --------------------------------------------------------------------------- #
def epu_thresholds(path: Path | None = None) -> tuple[dict[int, float], pd.DataFrame, dict]:
    """tau at each percentile of the EPU series over TAU_WINDOW.

    Taken from the series itself over 2017-2026, per section 7.5, rather than from the EPU values
    the 96 estimated months happen to carry: tau is a property of the uncertainty environment, not
    of this test's sample window, and computing it on the sample would let it move whenever the
    panel does. numpy.percentile with its default linear interpolation - the convention is named
    because a different one shifts tau slightly and with it the borderline months.
    """
    path = EPU_CSV if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_data.py first.")
    series = pd.read_csv(path, parse_dates=["date"])
    series["ym"] = series["date"].dt.to_period("M")

    window = series[series["ym"].between(*TAU_WINDOW)]
    if window.empty:
        raise ValueError(f"{path.name} holds no EPU observations inside {TAU_WINDOW}")
    if window["epu_news"].isna().any():
        raise ValueError(f"{path.name} carries {int(window['epu_news'].isna().sum())} null EPU "
                         f"values inside the tau window; a percentile over gaps is not the "
                         f"stated tau")

    taus = {pct: float(np.percentile(window["epu_news"], pct)) for pct in TAU_PERCENTILES}
    return taus, series, {"span": (str(window["ym"].min()), str(window["ym"].max())),
                          "n_months": len(window)}


def verify_epu_lag(primary: pd.DataFrame, series: pd.DataFrame) -> None:
    """Assert the panel's epu_lag really is the EPU of month t-1.

    build_fm_panel attached it, so this checks the convention rather than re-deriving it: a second
    implementation here would be one more thing able to disagree with the panel it classifies.
    """
    lagged = {period + 1: value for period, value
              in zip(series["ym"], series["epu_news"], strict=True)}
    expected = np.array([lagged.get(period, np.nan)
                         for period in pd.PeriodIndex(primary["ym"], freq="M")])
    if np.isnan(expected).any():
        raise ValueError(f"{EPU_CSV.name} does not cover every sample month lagged one period")
    if not np.allclose(primary["epu_lag"].to_numpy(dtype=float), expected, atol=1e-9):
        raise ValueError("the panel's epu_lag is not the EPU of month t-1")


def welch(high: pd.Series, low: pd.Series) -> dict:
    """Difference in regime means with unequal variances - the test section 7.5 nominates.

    Welch treats the monthly slopes within a regime as independent draws. They are not, which is
    why each regime mean also carries a Newey-West error and why hac_difference is reported beside
    this rather than instead of it.
    """
    result = ttest_ind(high.to_numpy(dtype=float), low.to_numpy(dtype=float), equal_var=False)
    return {"welch_t": float(result.statistic), "welch_p": float(result.pvalue),
            "welch_df": float(result.df)}


def regime_fit(values: pd.Series, indicator: pd.Series) -> dict:
    """Every regime statistic from ONE HAC fit on the UN-SPLIT series.

    lambda_t = a + b*1{high} + e, fitted with Newey-West over the series as ordered, so the
    Bartlett kernel sees genuine month adjacency. Then a is the low-regime mean, a + b the
    high-regime mean and b the difference, each standard error read off the joint covariance by
    t_test. OLS makes b identically the difference in group means, so no point estimate changes;
    what changes is that Welch's independence assumption is replaced by the same autocorrelation
    correction the levels carry.

    This replaces calling newey_west() on frame.loc[indicator] and frame.loc[~indicator]
    separately. Those subsets are scattered across the calendar - the p75 high-EPU bucket holds
    eight months of 2020, two of early 2021, two from the whole 2018-19 trade war and ten from
    2025 - so a lag-6 Bartlett kernel was giving observations three years apart the weight of
    consecutive months. Only the standard errors were affected; every mean is unchanged.
    """
    y = values.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), indicator.to_numpy(dtype=float)])
    fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS, **HAC_KWDS})
    tests = {"low": fit.t_test([[1.0, 0.0]]), "high": fit.t_test([[1.0, 1.0]])}
    out = {"diff": float(fit.params[1]), "hac_diff_se": float(fit.bse[1]),
           "hac_diff_t": float(fit.tvalues[1]), "hac_diff_p": float(fit.pvalues[1])}
    for key, test in tests.items():
        out[key] = {"mean": float(np.squeeze(test.effect)), "se": float(np.squeeze(test.sd)),
                    "t": float(np.squeeze(test.tvalue)), "p": float(np.squeeze(test.pvalue))}
    return out


def is_contiguous(periods: pd.Series) -> bool:
    """Whether these months form an unbroken monthly run, which HAC on them assumes."""
    ordinals = pd.PeriodIndex(periods, freq="M").astype("int64").to_numpy()
    return bool(len(ordinals) > 1 and np.all(np.diff(np.sort(ordinals)) == 1))


def regime_split(frame: pd.DataFrame, indicator: pd.Series, split: str, high_label: str,
                 low_label: str, tau_pct=np.nan, tau: float = np.nan) -> list[dict]:
    """Two rows - high regime, then low - for one binary classification of the monthly slopes.

    Generic over the indicator on purpose. The EPU splits and the episode splits then carry
    identical statistics from identical code, which is what makes section 7.5's "the episode-based
    split should be sharper than the EPU-based one" a comparison rather than an impression.
    """
    indicator = indicator.astype(bool)
    high, low = frame.loc[indicator, "lambda_texp"], frame.loc[~indicator, "lambda_texp"]
    if len(high) + len(low) != len(frame):
        raise ValueError(f"{split}: the indicator does not partition the {len(frame)} months")
    if high.empty or low.empty:
        raise ValueError(f"{split}: a regime is empty ({len(high)} high, {len(low)} low)")

    fitted = regime_fit(frame["lambda_texp"], indicator)
    contiguous = is_contiguous(frame["ym"])
    shared = {"split": split, "tau_pct": tau_pct, "tau": tau, "series_contiguous": contiguous,
              **welch(high, low),
              **{k: fitted[k] for k in ("diff", "hac_diff_se", "hac_diff_t", "hac_diff_p")}}
    shared["welch_stars"] = rcr.stars(shared["welch_p"])
    shared["hac_diff_stars"] = rcr.stars(shared["hac_diff_p"])

    # A dummy regression's slope IS the difference in group means. If these disagree, the indicator
    # and the two subsamples are not the same partition - which no covariance choice would reveal.
    if not np.isclose(shared["diff"], high.mean() - low.mean(), atol=1e-12):
        raise ValueError(f"{split}: HAC dummy coefficient {shared['diff']:.10f} does not equal the "
                         f"difference in means {high.mean() - low.mean():.10f}")
    # The two regime means must recombine to the frame's own mean, weighted by month count.
    recombined = (high.sum() + low.sum()) / len(frame)
    if not np.isclose(recombined, frame["lambda_texp"].mean(), atol=1e-12):
        raise ValueError(f"{split}: regimes recombine to {recombined:.10f} against the sample's "
                         f"{frame['lambda_texp'].mean():.10f}")

    rows = []
    for label, values, key in ((high_label, high, "high"), (low_label, low, "low")):
        level = fitted[key]
        # The fitted level must equal the subset's own mean - the same partition check, applied to
        # the levels rather than the difference.
        if not np.isclose(level["mean"], values.mean(), atol=1e-12):
            raise ValueError(f"{split}/{label}: fitted level {level['mean']:.10f} does not equal "
                             f"the subset mean {values.mean():.10f}")
        iid = sm.OLS(values.to_numpy(dtype=float), np.ones((len(values), 1))).fit()
        rows.append({**shared, "regime": label, "n_months": len(values),
                     "lambda_bar": level["mean"], "lambda_bar_pct": level["mean"] * 100,
                     "nw_se": level["se"], "nw_t": level["t"], "nw_p": level["p"],
                     "stars": rcr.stars(level["p"]),
                     "iid_se": float(iid.bse[0]),
                     "hac_reliable": len(values) >= MIN_REGIME_MONTHS_FOR_HAC})
    return rows


def regime_table(primary: pd.DataFrame, taus: dict[int, float]) -> pd.DataFrame:
    """Every section 7.5 split: the three EPU thresholds, then episode membership.

    Three classifications of the same 96 slopes. The EPU splits and the episode-binary split are
    partitions of the whole sample; each per-episode split is estimated on that episode's months
    against the months outside both, so the 2018-19 comparison is not contaminated by 2025 and
    vice versa.
    """
    rows = []
    for pct in TAU_PERCENTILES:
        rows += regime_split(primary, primary["epu_lag"] > taus[pct], "epu",
                             "high EPU", "low EPU", pct, taus[pct])

    episode = primary["episode"].fillna("")
    rows += regime_split(primary, episode.ne(""), "episode",
                         "in a tariff episode", OUTSIDE_LABEL)

    for label in sorted(episode[episode.ne("")].unique()):
        subset = primary[episode.eq("") | episode.eq(label)]
        rows += regime_split(subset, subset["episode"].fillna("").eq(label),
                             f"episode:{label}", label, OUTSIDE_LABEL)

    return pd.DataFrame(rows)[REGIME_COLUMNS]


def write_regime_results(table: pd.DataFrame, path: Path = REGIME_OUT) -> Path:
    """The machine-readable form of the section 10 and 11 tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def headline_results(lambdas: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """lambda_bar and its Newey-West test - the H2 statistic - in machine-readable form.

    This existed only as report prose and a chart subtitle, so anyone rebuilding a table from the
    output files got the per-month slopes and the regime split but not the single number H2 is
    about. Carries both specifications (the H5 read), the whole NW lag grid so the lag choice is
    visibly not driving anything, and every control's own lambda_bar.
    """
    rows = []
    for spec in SPECS:
        series = lambdas.loc[lambdas["spec"] == spec["name"], "lambda_texp"]
        for lags in sorted({*NW_LAG_GRID, NW_LAGS}):
            test = newey_west(series, lags=lags)
            rows.append({"quantity": "lambda_texp_bar", "spec": spec["name"], "term": TEXP_COLUMN,
                         "nw_lags": lags, "primary": lags == NW_LAGS and spec["name"] ==
                         PRIMARY_SPEC, **test, "mean_pct": test["mean"] * 100,
                         "stars": rcr.stars(test["p"])})
    for row in control_means(params).itertuples(index=False):
        rows.append({"quantity": "term_lambda_bar", "spec": PRIMARY_SPEC, "term": row.term,
                     "nw_lags": NW_LAGS, "primary": False, "mean": row.mean, "se": row.se,
                     "t": row.t, "p": row.p, "n_months": row.n_months, "lags": row.lags,
                     "mean_pct": row.mean * 100, "stars": row.stars})
    return pd.DataFrame(rows)


def write_headline_results(table: pd.DataFrame, path: Path = HEADLINE_OUT) -> Path:
    """The H2 headline, which previously reached no output file at all."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Chart                                                                       #
# --------------------------------------------------------------------------- #
def plot_lambda(lambdas: pd.DataFrame, episodes: list[dict],
                path: Path = CHART_OUT) -> Path:
    """The section 7.4 figure: the monthly exposure premium, with the policy episodes marked.

    Coefficients are drawn in percentage points a month per standard deviation of exposure, which
    is the unit the text quotes; the panel's own CSV keeps them in native decimals. Direct labels
    go on the extremes only: a value beside every one of ninety-six points is unreadable and goes
    unread. The mean and its Newey-West test are reported in fm_headline_results.csv and in the
    caption, not inside the frame.
    """
    frame = lambdas.sort_values("ym")
    x = pd.PeriodIndex(frame["ym"]).to_timestamp(how="end")
    y = frame["lambda_texp"].to_numpy() * 100

    cs.apply()
    fig, ax = plt.subplots(figsize=cs.SIZE_WIDE)

    for episode in episodes:
        ax.axvspan(episode["start"], episode["end"], facecolor=PALETTE["grid"],
                   alpha=EPISODE_ALPHA, lw=0, zorder=1)
        # Policy dates as short ticks on the floor of the axes: the span says when the episode
        # ran, these say which dates drove it, without nine vertical rules across the data.
        for day in episode["events"]:
            ax.plot([day, day], [0, EPISODE_TICK], transform=ax.get_xaxis_transform(),
                    color=PALETTE["ink"], lw=0.9, zorder=4, clip_on=False)

    cs.zero_line(ax)
    ax.plot(x, y, color=PALETTE["series"], lw=1.6, zorder=5)

    # Headroom set from the data rather than left to autoscale: the extreme labels and the
    # episode labels both live in it, and without it they collide with the floor ticks.
    span = y.max() - y.min()
    ax.set_ylim(y.min() - Y_PAD_LOW * span, y.max() + Y_PAD_HIGH * span)

    for idx in frame["lambda_texp"].abs().nlargest(N_LABELLED_EXTREMES).index:
        month, value = frame.at[idx, "ym"], frame.at[idx, "lambda_texp"] * 100
        ax.annotate(f"{month}   {value:+.2f}",
                    xy=(pd.Period(month, freq="M").to_timestamp(how="end"), value),
                    xytext=(0, 10 if value > 0 else -15), textcoords="offset points",
                    ha="center", va="bottom" if value > 0 else "top",
                    fontsize=cs.TICK_SIZE, color=PALETTE["ink"], zorder=6)

    ax.set_ylabel("$\\lambda_{1,t}$   (% monthly return per s.d.)")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.set_xlim(x.min() - pd.Timedelta(days=20), x.max() + pd.Timedelta(days=20))

    top = ax.get_ylim()[1]
    for episode in episodes:
        middle = episode["start"] + (episode["end"] - episode["start"]) / 2
        ax.annotate(episode["label"], xy=(middle, top), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=cs.TICK_SIZE, color=PALETTE["ink"], zorder=6)

    cs.frame(ax, max_yticks=MAX_YTICKS)
    cs.figure_title(fig, "Monthly cross-sectional premium on tariff exposure")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return cs.save(fig, path)


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def _regime_block(rows: pd.DataFrame) -> None:
    """One table per split: the two regime means, then their difference under both tests."""
    for (split, tau_pct), pair in rows.groupby(["split", "tau_pct"], dropna=False, sort=False):
        lead = pair.iloc[0]
        title = split if pd.isna(tau_pct) else f"{split} p{int(tau_pct)}"
        _say(f"  {title}" + ("" if pd.isna(lead["tau"]) else f"   tau {lead['tau']:.2f}"))
        _say(f"    {'regime':<24}{'months':>8}{'lambda_bar':>13}{'as %':>9}{'NW se':>10}"
             f"{'t':>8}{'p':>8}")
        for row in pair.itertuples(index=False):
            _say(f"    {row.regime:<24}{row.n_months:>8}{row.lambda_bar:>13.5f}"
                 f"{row.lambda_bar_pct:>9.3f}{row.nw_se:>10.5f}{row.nw_t:>8.2f}"
                 f"{row.nw_p:>8.3f}{rcr.stars(row.nw_p):>4}")
        _say(f"    {'difference (high - low)':<24}{'':>8}{lead['diff']:>13.5f}"
             f"{lead['diff'] * 100:>9.3f}")
        _say(f"      Welch  t = {lead['welch_t']:+.2f}  df = {lead['welch_df']:.1f}  "
             f"p = {lead['welch_p']:.3f}  - {rcr.significance_label(lead['welch_p'])}")
        _say(f"      HAC    se = {lead['hac_diff_se']:.5f}  t = {lead['hac_diff_t']:+.2f}  "
             f"p = {lead['hac_diff_p']:.3f}  - {rcr.significance_label(lead['hac_diff_p'])}")
        for row in pair[~pair["hac_reliable"].astype(bool)].itertuples(index=False):
            _say(f"      CAVEAT: '{row.regime}' holds {row.n_months} months against a "
                 f"{MIN_REGIME_MONTHS_FOR_HAC}-month floor for a")
            _say(f"      {NW_LAGS}-lag HAC error. Its NW se {row.nw_se:.5f} against an iid "
                 f"{row.iid_se:.5f} is a")
            _say("      finite-sample artefact, not a precision gain. Indicative only.")
        if not bool(lead["series_contiguous"]):
            _say("      CAVEAT: this split is estimated on a series with CALENDAR GAPS - it")
            _say("      excludes the other episode's months by construction. A Bartlett kernel")
            _say(f"      reads row adjacency as month adjacency, so its {NW_LAGS}-lag weights are")
            _say("      wrong across each gap. The means are unaffected; the errors are")
            _say("      approximate. The full-sample splits above carry no such gap.")
        _say()


def _h3_verdict(regimes: pd.DataFrame, split: str, tau_pct) -> list[str]:
    """H3 has two limbs: the regimes differ, and |lambda_bar| is the larger in the high regime."""
    pair = regimes[(regimes["split"] == split) & (regimes["tau_pct"] == tau_pct)]
    high, low = pair.iloc[0], pair.iloc[1]
    bigger = "larger" if abs(high["lambda_bar"]) > abs(low["lambda_bar"]) else "no larger"
    return [f"the difference is {rcr.significance_label(float(high['hac_diff_p']))} on the HAC "
            f"test, and |lambda_bar| is {bigger} in the",
            f"high regime - {abs(high['lambda_bar']) * 100:.3f}% against "
            f"{abs(low['lambda_bar']) * 100:.3f}%. H3 needs both limbs and has one."]


def _sharper_split(regimes: pd.DataFrame) -> list[str]:
    """Rank the conditioning variables by how sharply each separates the monthly slopes."""
    lead = regimes.drop_duplicates(["split", "tau_pct"])
    ordered = lead.sort_values("hac_diff_t", key=lambda col: col.abs(), ascending=False)
    lines = []
    for row in ordered.itertuples(index=False):
        label = row.split if pd.isna(row.tau_pct) else f"{row.split} p{int(row.tau_pct)}"
        lines.append(f"{label:<36}|diff| {abs(row.diff) * 100:>6.3f}%   "
                     f"|t| {abs(row.hac_diff_t):>5.2f}   p {row.hac_diff_p:>6.3f}")
    return lines



def write_report(panel: pd.DataFrame, sample: pd.DataFrame, lambdas: pd.DataFrame,
                 params: pd.DataFrame, episodes: list[dict], regimes: pd.DataFrame,
                 taus: dict[int, float], tau_stats: dict, paths: list[Path]) -> None:
    """The section 7.4 validation report, in the house style of the other 7.x reports."""
    primary = lambdas[lambdas["spec"] == PRIMARY_SPEC].sort_values("ym")
    tests = {spec["name"]: newey_west(lambdas.loc[lambdas["spec"] == spec["name"],
                                                  "lambda_texp"]) for spec in SPECS}
    head = tests[PRIMARY_SPEC]
    ref_sd, ref_n = load_reference_sd()

    _say(RULE)
    _say("SECTION 7.4 FAMA-MACBETH PRICING TEST - VALIDATION REPORT")
    _say(RULE)
    _say(f"Specification : ret_i,t = l0_t + l1_t*{TEXP_COLUMN} + c_t*{FS_COLUMN} + "
         f"gamma_t'[{', '.join(CONTROLS)}] + FF12 + e_i,t")
    _say(f"Estimator     : one cross-sectional OLS per month; mean of {{l1_t}} tested with "
         f"Newey-West ({NW_LAGS} lags)")
    _say(f"Sample        : section 7.0 random 1,000-firm subsample, {bfp.SAMPLE_START} .. "
         f"{bfp.SAMPLE_END} ({head['n_months']} monthly cross-sections)")
    _say(f"Exposure      : {TEXP_COLUMN}, nine April-2 vintages, standardised within "
         f"(vintage x subsample)")
    _say(f"Screen        : in_screen_lag - section 6 membership at t-1")
    _say(f"Std errors    : ordinary OLS within month; Newey-West on the second stage")
    _say(f"Inputs        : {PANEL_CSV.name}, {TEXP_PANEL_CSV.name}, {CAR_RESULTS_CSV.name}")

    _section("1. Specification choices")
    _say(f"  dependent var        ret, the ordinary CRSP monthly return, taken from the")
    _say(f"                       unscreened file so a firm screened in at t-1 that breaks $1")
    _say(f"                       during t still contributes the return it actually earned")
    _say(f"  exposure measure     {TEXP_COLUMN} - raw {TEXP_RAW} standardised within each of the")
    _say(f"                       nine full_panel vintages on the subsample that enters. Raw")
    _say(f"                       TExp's cross-sectional sd rises 2.7x and its zero share falls")
    _say(f"                       from 72% to 16% across the vintages, so a mean of raw slopes")
    _say(f"                       would mix two scales; section 4 defines TExp standardised.")
    _say(f"  vintage in effect    a y-04-02 vintage covers return months May y .. April y+1, and")
    _say(f"                       is carried no further: its newest filing is dated y-04-01, so")
    _say(f"                       it is public before the first month opens")
    _say(f"  point-in-time        every regressor is resolved at pit_date, the last calendar day")
    _say(f"                       of month t-1; only the return is dated t")
    _say(f"  industry FE          {INDUSTRY_COLUMN} from the firm's SIC at t-1, reference "
         f"category {FF12_REFERENCE!r}")
    _say(f"  specifications       " + ", ".join(
        f"{s['name']} (FS {'in' if s['fs'] else 'out'}, FF12 {'in' if s['ff12'] else 'out'})"
        for s in SPECS))
    _say(f"                       both on identical rows, so any movement in l1_bar is the")
    _say(f"                       specification and not the sample")
    _say(f"  PolRisk              excluded, per the constraint note at the head of section 7:")
    _say(f"                       the Hassan series ends March 2021 and would truncate the panel")
    _say(f"  not run here         the EPU regime split is section 7.5; epu_lag is carried in the")
    _say(f"                       lambda panel so that test needs no rebuild")

    _section("2. Exposure vintages actually used")
    _say("  Nine April-2 reference dates, sliced from the seventeen in texp_panel.csv. The other")
    _say("  eight belong to the cross-cycle event study. Two of these nine were also pulled at")
    _say("  scope full, so the panel's own z column is standardised over the wrong population")
    _say("  there and is recomputed on the subsample - see build_fm_panel.load_texp.")
    vintage_rows = pd.read_csv(TEXP_PANEL_CSV, usecols=["reference_date"])
    vintage_rows = vintage_rows["reference_date"].value_counts()
    _say(f"  {'vintage':<13}{'in panel':>10}{'firms':>7}{'months':>8}{'firm-months':>13}"
         f"{'raw sd':>10}{'z mean':>8}{'z sd':>7}")
    for vintage, group in sample.groupby("reference_date"):
        _say(f"  {vintage:<13}{vintage_rows[vintage]:>10,}{group['permno'].nunique():>7,}"
             f"{group['ym'].nunique():>8}{len(group):>13,}{group[TEXP_RAW].std():>10.5f}"
             f"{group[TEXP_COLUMN].mean():>8.3f}{group[TEXP_COLUMN].std():>7.3f}")
    _say("  'in panel' is the whole vintage cross-section in texp_panel.csv; 'firms' is what this")
    _say("  test used. The two scope-full vintages contribute no more firms than any other - the")
    _say("  subsample restriction is applied to the CRSP read and asserted again on the vintage")
    _say("  slice, so a non-subsample firm has no row to join to and could not enter. Nothing")
    _say("  here can be weighted toward 2018 or 2025 by universe size.")
    _say("  The z moments here are over the estimation sample, not the vintage cross-section they")
    _say("  were computed on, so they need not be exactly (0, 1); Script 7a asserts that they are")
    _say("  on the population that defines them.")
    age = ((sample["date_t"] - sample["filing_date"]).dt.days / 30.44)
    _say(f"  Exposure age at the return month: median {age.median():.1f} months, "
         f"max {age.max():.1f}.")
    _say("  That maximum is what annual refresh means - a April y+1 month scored by a 10-K filed")
    _say("  in the spring of year y - and is the convention section 4 nominates, not a defect.")

    _section("3. Exclusion ladder and funnel reconciliation")
    _say("  One reason per excluded firm-month, assigned in precedence order by Script 7a, so the")
    _say("  counts sum exactly to the candidate rows. Asserted there, re-checked here.")
    counts = panel.loc[panel["exclusion_reason"].ne(""), "exclusion_reason"].value_counts()
    _say(f"    {len(panel):,} candidate firm-months -> {len(sample):,} estimated")
    for why, n in counts.items():
        _say(f"       - {why:<52}{n:>9,}")
    if len(sample) + int(counts.sum()) != len(panel):
        raise ValueError("the funnel does not reconcile against the candidate row count")
    _say(f"       = reconciles: {len(sample):,} + {int(counts.sum()):,} = {len(panel):,}")
    fs_cost = int(counts.filter(like="no_fs:").sum())
    _say(f"  Requiring the H5 foreign-sales control costs {fs_cost:,} firm-months "
         f"({fs_cost / len(panel):.1%} of candidates).")
    _say("  Both specifications pay it, because they are estimated on the same rows; the")
    _say("  alternative would confound the specification change with a sample change.")

    _section("4. Monthly cross-sections")
    sizes = primary.set_index("ym")["n_firms"]
    _say(f"  {'year':<8}{'months':>8}{'min':>8}{'mean':>8}{'max':>8}{'mean R2':>10}"
         f"{'mean adj R2':>13}")
    for year, group in primary.groupby(pd.PeriodIndex(primary["ym"]).year):
        _say(f"  {year:<8}{len(group):>8}{group['n_firms'].min():>8,}"
             f"{group['n_firms'].mean():>8,.0f}{group['n_firms'].max():>8,}"
             f"{group['r2'].mean():>10.3f}{group['adj_r2'].mean():>13.3f}")
    _say(f"  {'all':<8}{len(primary):>8}{sizes.min():>8,}{sizes.mean():>8,.0f}{sizes.max():>8,}"
         f"{primary['r2'].mean():>10.3f}{primary['adj_r2'].mean():>13.3f}")
    last = sample[sample["ym"] == sample["ym"].max()]
    n_params = monthly_design(last, next(s for s in SPECS if s["name"] == PRIMARY_SPEC))[1].shape[1]
    _say(f"  Floor MIN_FIRMS_PER_MONTH={MIN_FIRMS_PER_MONTH} - a degrees-of-freedom guard set")
    _say(f"  from the specification's {n_params} parameters, not from any month - binds on none.")
    _say("  Cross-sections grow from 97 to 721 firms because the section 7.0 draw was taken at")
    _say("  end-March 2025, so more of the 1,000 are listed in later years - the acknowledged")
    _say("  cost of that reference date. Fama-MacBeth averages the monthly slopes with equal")
    _say("  weight per month, so this changes how precisely each month is estimated, not how")
    _say("  much each month counts toward lambda_1_bar. Weighting by firm-months instead would")
    _say(f"  give {np.average(primary['lambda_texp'], weights=primary['n_firms']) * 100:+.3f}% "
         f"against the reported {primary['lambda_texp'].mean() * 100:+.3f}%.")
    thin = sizes[sizes < THIN_MONTH_FLOOR]
    _say(f"  Months below {THIN_MONTH_FLOOR} firms: {len(thin)}"
         + (f" ({', '.join(f'{m} ({n})' for m, n in thin.items())})" if len(thin) else ""))
    _say("  Those months are thin because Compustat's geographic-segment file begins at datadate")
    _say("  2017-01-31, so firms still reporting FY2016 fundamentals in early 2018 carry no")
    _say("  foreign-sales row. Their cross-sections are also selected toward early filers, which")
    _say("  is a composition caveat and not only a precision one. Section 7 reports the mean")
    _say("  without them; nothing is imputed.")
    small = sample.groupby(["ym", INDUSTRY_COLUMN]).size()
    _say(f"  Industry-months with fewer than {SMALL_INDUSTRY_CELL} firms: "
         f"{int((small < SMALL_INDUSTRY_CELL).sum()):,} of {len(small):,}. A dummy on a")
    _say("  one-firm cell absorbs that firm rather than identifying anything, which costs a")
    _say("  degree of freedom but cannot bias l1.")

    _section("5. Estimation sample - descriptives and correlations")
    _describe(sample, [TEXP_COLUMN, TEXP_RAW, FS_COLUMN] + CONTROLS + ["ret"])
    _say(f"  Firms entering at least one month: {sample['permno'].nunique():,} of 1,000 drawn.")
    _say()
    _say(f"  Pearson correlations against {TEXP_COLUMN} (pooled over firm-months):")
    for col in [FS_COLUMN] + CONTROLS:
        _say(f"    {col:<16}{sample[TEXP_COLUMN].corr(sample[col]):>8.4f}")
    _say("  The FS correlation is the H5 concern in one number: exposure and multinational status")
    _say("  move together, which is why FS and industry sit in the primary specification.")

    _section("6. Fama-MacBeth results")
    _say(f"  Mean of {{l1_t}} against zero, Newey-West {NW_LAGS} lags. Coefficients are decimal")
    _say(f"  monthly return per standard deviation of exposure; the percentage is the same number.")
    _say(f"  {'spec':<16}{'l1_bar':>11}{'% / month':>11}{'NW se':>10}{'t':>8}{'p':>8}"
         f"{'':>4}{'months':>8}")
    for spec in SPECS:
        test = tests[spec["name"]]
        _say(f"  {spec['name']:<16}{test['mean']:>11.5f}{test['mean'] * 100:>11.3f}"
             f"{test['se']:>10.5f}{test['t']:>8.2f}{test['p']:>8.3f}"
             f"{rcr.stars(test['p']):>4}{test['n_months']:>8}")
    _say(f"  Verdict on H2: l1_bar is {rcr.significance_label(head['p'])}.")
    _say()
    _say("  H5 read: the exposure premium with and without the foreign-sales control and the")
    _say(f"  industry effects, on identical rows. l1_bar moves from "
         f"{tests['no_fs_no_ff12']['mean'] * 100:+.3f}% to {head['mean'] * 100:+.3f}% per s.d.")
    _say(f"  when they enter - a change of "
         f"{(head['mean'] - tests['no_fs_no_ff12']['mean']) * 100:+.3f} points - and keeps its "
         f"sign. A premium merely")
    _say("  restating multinational status or sector membership would not survive their")
    _say("  inclusion. Neither estimate is distinguishable from zero, though, so this reads as")
    _say("  consistency rather than as a passed test.")
    _say()
    _say(f"  Every term of the {PRIMARY_SPEC} specification, same second stage:")
    _say(f"  {'term':<16}{'lambda_bar':>13}{'NW se':>10}{'t':>8}{'p':>8}")
    for row in control_means(params).itertuples(index=False):
        _say(f"  {row.term:<16}{row.mean:>13.5f}{row.se:>10.5f}{row.t:>8.2f}{row.p:>8.3f}"
             f"{row.stars:>4}")

    _section("7. Robustness of the second stage")
    _say(f"  Newey-West lag, {PRIMARY_SPEC} specification:")
    _say(f"    {'lags':<8}{'se':>11}{'t':>8}{'p':>8}")
    for lags in NW_LAG_GRID:
        test = newey_west(primary["lambda_texp"], lags)
        _say(f"    {lags:<8}{test['se']:>11.5f}{test['t']:>8.2f}{test['p']:>8.3f}")
    extreme = primary["lambda_texp"].abs().idxmax()
    without = newey_west(primary.drop(index=extreme)["lambda_texp"])
    _say(f"  Excluding the single largest |l1_t| month ({primary.at[extreme, 'ym']}, "
         f"{primary.at[extreme, 'lambda_texp'] * 100:+.2f}%):")
    _say(f"    l1_bar {without['mean'] * 100:+.3f}% per s.d., t = {without['t']:.2f}, "
         f"p = {without['p']:.3f} over {without['n_months']} months")
    keep = primary[primary["n_firms"] >= THIN_MONTH_FLOOR]
    trimmed = newey_west(keep["lambda_texp"])
    _say(f"  Excluding months below {THIN_MONTH_FLOOR} firms ({len(primary) - len(keep)} months):")
    _say(f"    l1_bar {trimmed['mean'] * 100:+.3f}% per s.d., t = {trimmed['t']:.2f}, "
         f"p = {trimmed['p']:.3f} over {trimmed['n_months']} months")

    _section("8. Against the section 7.2 event-window coefficients")
    _say("  The design's nominated comparison. Section 7.2 reports b in raw TExp units and")
    _say(f"  instructs the reader to multiply by the sd of its own cross-section; that")
    _say(f"  cross-section is the whole 2025-04-02 vintage, {ref_n:,} firms, sd {ref_sd:.6f}.")
    sample_sds = sample_reference_sd()
    if sample_sds:
        _say("  Two conversion factors exist and they are not the same number. The vintage-wide sd")
        _say("  above is one stable figure for a table spanning several windows and both legs; the")
        _say("  sd inside each regression's own estimation sample, after the FS and control")
        _say("  exclusions, is higher - so the per-s.d. column below is conservative by that much:")
        for window, sd in sample_sds.items():
            _say(f"    {rcr._short(window):<12}estimation-sample sd {sd:.6f}  "
                 f"({sd / ref_sd - 1:+.1%} against the vintage)")
        _say("  The vintage figure is used throughout, and this is the size of the understatement.")
    if CAR_RESULTS_CSV.exists():
        car = pd.read_csv(CAR_RESULTS_CSV)
        car = car[(car["term"] == TEXP_RAW) & (car["spec"] == "primary")]
        _say(f"  {'run':<24}{'window':<12}{'b raw':>10}{'b per s.d.':>13}{'as %':>9}"
             f"{'as % (own sd)':>16}")
        for row in car.itertuples(index=False):
            own = sample_sds.get(row.window)
            own_cell = "-" if own is None else f"{row.coef * own * 100:.2f}"
            _say(f"  {row.run:<24}{rcr._short(row.window):<12}{row.coef:>10.4f}"
                 f"{row.coef * ref_sd:>13.5f}{row.coef * ref_sd * 100:>9.2f}{own_cell:>16}")
        widest = car[car["window"] == "car_m10p10"]["coef"].abs().max() * ref_sd
        _say(f"  Largest event-window effect: {widest * 100:.2f}% over up to 21 trading days,")
        _say(f"  against a mean monthly premium of {head['mean'] * 100:+.3f}%. An event-window")
        _say(f"  coefficient roughly {widest / abs(head['mean']):.0f} times the unconditional")
        _say("  monthly mean is what the design predicted in advance, and it is the comparison")
        _say("  section 7.4 exists to make: averaging a handful of large event-month slopes")
        _say("  against ninety-odd quiet ones attenuates the mean toward zero by construction.")
        _say("  A weak mean here is therefore not a refutation of H1. Nor is it, on its own,")
        _say("  positive evidence for episodic pricing: for that the monthly series would have")
        _say("  to concentrate on the episodes, and section 9 finds only a weak tilt.")
    else:
        _say(f"  {CAR_RESULTS_CSV.name} absent; run run_car_regression.py to populate this table.")

    _section("9. Episode months against the rest (descriptive)")
    _say("  A read of the figure, not a test: the formal regime split is section 7.5.")
    _say(f"  {'group':<24}{'months':>8}{'mean l1':>11}{'as %':>9}{'sd':>10}")
    for label, group in primary.groupby(primary["episode"].replace("", "outside both episodes")):
        _say(f"  {label:<24}{len(group):>8}{group['lambda_texp'].mean():>11.5f}"
             f"{group['lambda_texp'].mean() * 100:>9.3f}{group['lambda_texp'].std():>10.5f}")
    _say("  Episode spans are taken from clean_controls_data.CYCLES, not restated:")
    for episode in episodes:
        _say(f"    {episode['label']:<24}{episode['start']:%Y-%m-%d} to "
             f"{episode['end']:%Y-%m-%d}   ({len(episode['events'])} policy dates)")
    _say("  Each span runs from its cycle's first event to its last, so it contains both the")
    _say("  tightening and the loosening legs. Its mean therefore nets a predicted-negative")
    _say("  against a predicted-positive month and is not a directional statement; that is what")
    _say("  the figure and section 7.2 are for, and what section 7.5 tests formally.")
    _say()
    _say("  Does the figure show the concentration the design looked for? Section 7.4 nominates")
    _say("  it as this section's most informative output, to show 'whether the monthly")
    _say("  coefficients spike around policy events and sit near zero otherwise'.")
    episode_months = int(primary["episode"].ne("").sum())
    largest = primary.loc[primary["lambda_texp"].abs().nlargest(N_LARGEST_CHECK).index]
    inside = int(largest["episode"].ne("").sum())
    tail = float(hypergeom.sf(inside - 1, len(primary), episode_months, N_LARGEST_CHECK))
    _say(f"    episode months                     {episode_months} of {len(primary)} "
         f"({episode_months / len(primary):.0%})")
    _say(f"    largest |l1_t| months inside them  {inside} of {N_LARGEST_CHECK}, against "
         f"{N_LARGEST_CHECK * episode_months / len(primary):.1f} expected by chance")
    _say(f"    tail probability of {inside} or more     {tail:.3f}  hypergeometric; descriptive")
    _say(f"                                       only - {N_LARGEST_CHECK} was fixed in config, "
         f"but this is not a")
    _say("                                       pre-registered test")
    _say(f"    largest of all                     {primary.at[extreme, 'ym']}, "
         f"{primary.at[extreme, 'lambda_texp'] * 100:+.2f}%, "
         f"{'inside' if primary.at[extreme, 'episode'] else 'outside'} both episodes")
    _say("  So: partly. The largest monthly slopes do tilt toward the episodes - roughly 1.7")
    _say("  times the chance rate - but the tilt is not distinguishable from chance at this")
    _say("  count, the single largest month of all falls outside both, and the dispersion of")
    _say("  l1_t inside the episodes is barely above the dispersion outside them. The clean")
    _say("  pattern the design hoped to see - spikes at the events, quiet elsewhere - is not")
    _say("  what the monthly series delivers.")
    _say("  The episodic reading therefore rests mainly on the section 7.2 event-window")
    _say("  coefficients and their distance from this mean, with the monthly series offering")
    _say("  weak corroboration rather than independent support. Write it up that way. Per")
    _say("  section 5.2 item 6, subsample noise cannot be separated from a genuine absence of")
    _say("  unconditional pricing here, and this null must not be presented as evidence of no")
    _say("  effect.")

    _section("10. EPU regime conditioning (H3)")
    _say("  A post-hoc classification of the same 96 monthly slopes, not a second estimation: no")
    _say("  cross-section is re-fitted, so nothing above can move. H3 is that tariff risk is priced")
    _say("  only when policy uncertainty makes it salient, which is what would reconcile a flat")
    _say("  unconditional mean with a premium concentrated in high-EPU months.")
    _say(f"  tau is the percentile of the EPU series over {tau_stats['span'][0]} .. "
         f"{tau_stats['span'][1]} ({tau_stats['n_months']} months), per")
    _say("  section 7.5 - the series, not the estimated months - by numpy.percentile with linear")
    _say("  interpolation. Fixed before estimation and not searched over.")
    _say(f"    {'percentile':<14}{'tau':>10}{'high months':>14}{'low months':>13}")
    for pct in TAU_PERCENTILES:
        pair = regimes[(regimes["split"] == "epu") & (regimes["tau_pct"] == pct)]
        primary_mark = "  <- section 7.5 primary" if pct == PRIMARY_TAU else ""
        _say(f"    p{pct:<13}{taus[pct]:>10.2f}"
             f"{int(pair.iloc[0]['n_months']):>14}{int(pair.iloc[1]['n_months']):>13}"
             f"{primary_mark}")
    sample_taus = {pct: float(np.percentile(primary["epu_lag"], pct)) for pct in TAU_PERCENTILES}
    _say("  For contrast only, not used: computed on the 96 estimated months instead, tau would")
    _say("  be " + ", ".join(f"p{p} {sample_taus[p]:.2f}" for p in TAU_PERCENTILES) + ".")
    _say()
    _say("  EPU is lagged one month before classification. The panel already carries epu_lag as")
    _say("  the EPU of t-1; this section asserts that rather than re-deriving it, and asserts the")
    _say("  tau window has no gaps. The first sample month, 2018-01, takes 2017-12 - inside the")
    _say("  series - so no month is lost at the boundary.")
    _say()
    _regime_block(regimes[regimes["split"] == "epu"])
    _say(f"  Verdict on H3 at the nominated p{PRIMARY_TAU} threshold:")
    for line in _h3_verdict(regimes, "epu", PRIMARY_TAU):
        _say(f"  {line}")
    _say()
    _say("  Read the three thresholds together, because they do not line up the way H3 implies.")
    epu = regimes[regimes["split"] == "epu"].drop_duplicates("tau_pct")
    _say(f"    {'threshold':<12}{'difference':>12}{'as %':>9}{'HAC t':>8}{'HAC p':>8}"
         f"{'Welch p':>10}")
    for row in epu.itertuples(index=False):
        _say(f"    p{int(row.tau_pct):<11}{row.diff:>12.5f}{row.diff * 100:>9.3f}"
             f"{row.hac_diff_t:>8.2f}{row.hac_diff_p:>8.3f}{row.welch_p:>10.3f}")
    _say("  The difference *falls monotonically as the threshold rises* - largest at the median,")
    _say("  gone by the 90th percentile. If the premium tracked the intensity of policy")
    _say("  uncertainty the ordering would run the other way: the most extreme months should")
    _say("  separate most sharply, not least. Two readings are available and this panel cannot")
    _say("  settle between them. Either the median split is the only one whose buckets are both")
    _say("  large enough to detect anything, making the ordering a power pattern rather than a")
    _say("  mechanism; or the high buckets at p75 and p90 are diluted by what is in them, which is")
    _say("  what section 12 finds - drop the 2020 months and the p75 high-regime mean roughly")
    _say("  doubles. The second reading is the more interesting and the less well supported, since")
    _say("  every sub-bucket it rests on sits below the HAC floor.")
    _say()
    _say(f"  So the honest headline is the p{PRIMARY_TAU} row, and it does not support H3. The")
    _say("  median split is significant - +0.224% per s.d., HAC p = 0.021, Welch p = 0.080 - and it")
    _say(f"  is tempting, but promoting it over the pre-nominated p{PRIMARY_TAU} would be choosing "
         f"the threshold")
    _say("  on the result, which is exactly what fixing tau in advance exists to prevent. Report")
    _say("  the median split as the robustness line section 7.5 asked for, say plainly that it is")
    _say("  the strongest of the three, and let the reader see the ordering.")

    _section("11. Episode split beside the EPU split")
    _say("  Section 7.5 robustness, in its own words: 'If the conditioning is really about *tariff*")
    _say("  salience rather than general policy uncertainty, the episode-based split should be")
    _say("  sharper than the EPU-based one.' Same function, same statistics, so the two are")
    _say("  directly comparable. Each per-episode row compares that episode's months against the")
    _say("  months outside both, so 2018-19 is not contaminated by 2025 or the reverse.")
    _say()
    _regime_block(regimes[regimes["split"].str.startswith("episode")])
    _say("  Which conditioning is sharper, by |difference| / its HAC standard error:")
    for line in _sharper_split(regimes):
        _say(f"    {line}")
    _say()
    _say("  Section 7.5's expectation is not borne out. Every EPU split above the 90th percentile")
    _say("  separates the slopes more sharply than any episode split does, and no episode split")
    _say("  approaches significance. On this panel the episode indicator is the *weaker*")
    _say("  conditioning variable, not the sharper one.")
    _say("  Two things make that less surprising than it first reads. Each span runs from its")
    _say("  cycle's first event to its last, so it contains both the tightening and the loosening")
    _say("  legs and its mean nets a predicted-negative month against a predicted-positive one -")
    _say("  the point section 9 makes. And the 2025 span holds 5 months and 2018-19 holds 23, so")
    _say("  both sit under the HAC floor. The episode split as section 7.5 specifies it is")
    _say("  therefore not a powerful test of tariff salience, and its null should not be read as")
    _say("  evidence that tariff salience does not matter - section 7.2, where the legs are")
    _say("  separated and the window is days rather than months, is where that question is")
    _say("  actually answered.")

    _section("12. What is in the high-EPU bucket")
    _say("  Section 7.5 requires this: 'If the high-EPU bucket is dominated by COVID months rather")
    _say("  than trade-policy months, say so - a premium that appears only in 2020 is a different")
    _say("  finding from one that appears whenever policy uncertainty is elevated.' Composition is")
    _say("  reported by calendar year and by episode membership, both already defined, rather than")
    _say("  against an invented COVID date range.")
    for pct in TAU_PERCENTILES:
        high = primary[primary["epu_lag"] > taus[pct]]
        by_year = high.groupby(pd.PeriodIndex(high["ym"], freq="M").year).size()
        _say()
        _say(f"  p{pct} (tau {taus[pct]:.2f}) - {len(high)} high months, "
             f"{int(high['episode'].fillna('').ne('').sum())} of them inside a tariff episode")
        _say("    by year: " + ", ".join(f"{year} {n}" for year, n in by_year.items()))
        months = [str(month) for month in high["ym"]]
        for start in range(0, len(months), MONTHS_PER_LINE):
            label = "    months:  " if start == 0 else " " * 13
            _say(label + ", ".join(months[start:start + MONTHS_PER_LINE]))
    _say()
    _say(f"  Is the high-EPU premium only {COVID_YEAR}? Section 7.5 asks that directly, so the high")
    _say(f"  regime's mean is recomputed with the {COVID_YEAR} months dropped. One named calendar")
    _say("  year, not an invented pandemic window.")
    _say(f"    {'threshold':<12}{'high n':>8}{'lambda_bar %':>14}{'ex-' + str(COVID_YEAR) + ' %':>12}"
         f"{'n left':>8}{'t':>7}{'p':>8}")
    for pct in TAU_PERCENTILES:
        high = primary[primary["epu_lag"] > taus[pct]]
        kept = high[pd.PeriodIndex(high["ym"], freq="M").year != COVID_YEAR]
        full_test = newey_west(high["lambda_texp"])
        ex_test = newey_west(kept["lambda_texp"]) if len(kept) > 1 else None
        thin = "  (!)" if len(kept) < MIN_REGIME_MONTHS_FOR_HAC else ""
        cells = (f"{ex_test['mean'] * 100:>12.3f}{len(kept):>8}{ex_test['t']:>7.2f}"
                 f"{ex_test['p']:>8.3f}{thin}" if ex_test else
                 f"{'-':>12}{len(kept):>8}{'-':>7}{'-':>8}")
        _say(f"    p{pct:<11}{len(high):>8}{full_test['mean'] * 100:>14.3f}{cells}")
    flagged = sum(1 for pct in TAU_PERCENTILES
                  if len(primary[(primary["epu_lag"] > taus[pct])
                                 & (pd.PeriodIndex(primary["ym"], freq="M").year != COVID_YEAR)])
                  < MIN_REGIME_MONTHS_FOR_HAC)
    _say(f"  (!) marks a bucket below the {MIN_REGIME_MONTHS_FOR_HAC}-month floor for a "
         f"{NW_LAGS}-lag HAC error - {flagged} of {len(TAU_PERCENTILES)} here.")
    _say()
    _say(f"  The premium is not a {COVID_YEAR} artefact - it is larger without that year, not")
    _say("  smaller. At the nominated p75 the high-regime mean roughly doubles, 0.169% to 0.322%,")
    _say("  and at the median it rises 0.159% to 0.195%. On the point estimates, COVID months were")
    _say("  *diluting* the high-EPU premium rather than producing it: 2020 was a period of extreme")
    _say("  policy uncertainty in which tariff exposure specifically was not what moved, which is")
    _say("  a coherent story rather than a puzzle.")
    _say("  The accompanying t-statistics are not evidence, though. Two of those three buckets sit")
    _say(f"  below the {MIN_REGIME_MONTHS_FOR_HAC}-month floor, so p = 0.001 at p75 ex-{COVID_YEAR} "
         f"is the same finite-sample")
    _say("  artefact flagged in section 10, not a stronger result. Read the table as point")
    _say("  estimates only.")
    _say()
    _say("  What the composition undermines is the *interpretation*, and that survives whatever the")
    _say("  coefficients do. At p75 the high bucket is 8 COVID-2020 months and 2 from early 2021")
    _say("  against 2 from the whole 2018-19 trade war; at p90 the trade war contributes nothing.")
    _say("  News-based EPU in 2018-19 was simply not extreme by 2017-2026 standards, dwarfed by")
    _say("  COVID and by 2025. So a premium in this bucket is a statement about elevated policy")
    _say("  uncertainty in general - and it is not a COVID artefact, since dropping 2020 leaves it")
    _say("  larger - but it is not evidence about *tariff* salience, because the months in which")
    _say("  tariff policy actually moved are largely not in the bucket. That distinction is the")
    _say("  one section 7.5 asks for, and section 11's episode split is the test that addresses")
    _say("  it directly.")

    _section("13. Assumptions and deviations")
    _say("  1. Sample opens 2018-01, not 2017-01. Monthly Returns.csv begins 2017-01-31 and no")
    _say("     earlier return history exists in the project, so the twelve months ending t-1 are")
    _say("     first complete at 2018-01. TExp is independently unavailable before 2017-05. The")
    _say("     design nominates ~110 monthly cross-sections; 96 are available.")
    _say("  2. Exposure standardised within vintage, where sections 7.2 and 7.3 use raw TExp.")
    _say("     Section 4 defines TExp standardised, and a mean of raw monthly slopes would mix")
    _say("     scales that differ by 2.7x across the panel. Section 8 converts 7.2 to the same")
    _say("     units so the two remain comparable.")
    _say("  3. FF12 is point-in-time and time-varying, where section 7.2 fixes one label per")
    _say("     firm. That convention exists to hold industry effects identical across the two")
    _say("     legs of one event study; 96 independent cross-sections have no such pair.")
    _say("  4. A vintage is carried forward twelve months and no further. Unlimited carry-forward")
    _say("     would add 120 firm-vintage-years (+1.7%) at the cost of scoring a month with a")
    _say("     filing already known to be more than a year stale.")
    _say("  5. No winsorising or trimming, matching the section 7.2 primary specification. The")
    _say("     extreme-month and thin-month sensitivities in section 7 stand in for it.")
    _say("  6. Within-month errors are ordinary OLS. Inference is the second stage's, and")
    _say("     Fama-MacBeth takes only the point estimate from each month.")
    _say("  7. Version B of the design (Fama-MacBeth on rolling TExp-factor betas) is not run;")
    _say("     v6 drops it with the long-short portfolio it depended on.")
    _say("  Section 7.5 (sections 10-12) adds:")
    _say(f"  8. tau is a percentile of the EPU series over {tau_stats['span'][0]} .. "
         f"{tau_stats['span'][1]}, per section 7.5,")
    _say("     not of the EPU values the estimated months carry. The window therefore")
    _say(f"     includes {tau_stats['n_months'] - len(primary)} months outside the "
         f"Fama-MacBeth sample. The sample-window alternative")
    _say("     is reported in section 10 as a contrast and is not used.")
    _say("  9. numpy.percentile with its default linear interpolation. Named because another")
    _say("     quantile convention moves tau slightly and with it the borderline months.")
    _say(" 10. EPU is lagged one month, and that lag was applied when the panel was built rather")
    _say("     than here. Section 10 asserts the panel's epu_lag equals a fresh one-month lag of")
    _say("     the series, so the convention is verified without a second implementation of it.")
    _say(" 11. The regime split runs on the primary specification only. The no_fs_no_ff12 slopes")
    _say("     are in fm_lambda_panel.csv and are one constant away, but H3 asks about the")
    _say("     specification H2 was tested on.")
    _say(" 12. Welch is the named difference test and stays the headline. It treats the monthly")
    _say("     slopes within a regime as independent draws, which they are not, so a HAC")
    _say("     difference - the same correction the regime means carry - is reported beside it.")
    _say("     The two point estimates are identical by construction; only the errors differ.")
    _say(f" 13. Regimes below {MIN_REGIME_MONTHS_FOR_HAC} months are flagged rather than dropped, "
         f"with their iid error shown.")
    _say("     The floor is 4 x the NW lag, set from the lag length and not from which regimes it")
    _say("     catches. Nothing is re-estimated in sections 10-12: they classify the same 96")
    _say("     slopes, so no number in sections 1-9 can move.")
    _say(" 14. Each regime's mean and NW error now come from ONE HAC fit on the un-split series")
    _say("     (regime_fit), not from applying the HAC estimator to the high and low subsets")
    _say("     separately. The subsets are scattered across the calendar - the p75 high bucket")
    _say("     holds eight months of 2020, two of early 2021, two from the whole 2018-19 trade")
    _say("     war and ten from 2025 - and a Bartlett kernel reads row adjacency as month")
    _say("     adjacency, so observations years apart were being given consecutive-month weights.")
    _say("     Every regime mean is unchanged to machine precision (a dummy regression's fitted")
    _say("     levels ARE the group means, asserted); only the errors move, by 0.95x to 1.09x.")
    _say("     The two per-episode splits still carry calendar gaps by construction and are")
    _say("     flagged in their own blocks; series_contiguous records it per row.")
    _say(" 15. The HAC finite-sample switches are named in HAC_KWDS rather than inherited")
    _say("     silently: use_correction, adjust_df and use_t are all False, which are the")
    _say("     statsmodels defaults, so no number changes. They are left there deliberately - at")
    _say("     96 months all three are negligible, and switching them only for the small buckets")
    _say("     would put two conventions in one table. The consequence is that the sub-floor rows'")
    _say("     p-values are optimistic on two counts at once: no small-sample scaling and a")
    _say("     normal rather than t reference distribution.")
    _say(" 16. Two per-standard-deviation conversion factors exist and section 8 reports both.")
    _say("     The vintage-wide sd is used throughout because it is one stable figure across")
    _say("     windows and legs; the estimation sample's own sd is 4.5% higher, so the reported")
    _say("     per-s.d. figures are conservative by that much rather than wrong.")
    _say(" 17. lambda_bar and its Newey-West test now reach a file, fm_headline_results.csv. The")
    _say("     H2 statistic previously existed only as report prose and a chart subtitle, so a")
    _say("     reader rebuilding tables from the outputs got H3 but not H2.")

    _section("14. Outputs")
    for path in paths:
        _say(f"  {path.relative_to(BASE)}")

    REPORT_OUT.write_text("\n".join(_REPORT), encoding="utf-8")


def main() -> pd.DataFrame:
    panel, sample = load_panel()
    episodes = episode_spans()

    fitted = {spec["name"]: run_spec(sample, spec, episodes) for spec in SPECS}
    lambdas = pd.concat([slopes for slopes, _ in fitted.values()], ignore_index=True)
    params = fitted[PRIMARY_SPEC][1]
    primary = lambdas[lambdas["spec"] == PRIMARY_SPEC]
    expected = (bfp.SAMPLE_END - bfp.SAMPLE_START).n + 1
    if len(primary) != expected:
        raise ValueError(f"estimated {len(primary)} monthly cross-sections, expected {expected}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lambdas.assign(ym=lambdas["ym"].astype(str)).to_csv(LAMBDA_OUT, index=False)
    chart = plot_lambda(primary, episodes)

    headline_csv = write_headline_results(headline_results(lambdas, params))

    # Section 7.5 - a classification of the slopes just estimated, not a second estimation.
    taus, epu_series, tau_stats = epu_thresholds()
    verify_epu_lag(primary, epu_series)
    regimes = regime_table(primary, taus)
    regime_csv = write_regime_results(regimes)

    write_report(panel, sample, lambdas, params, episodes, regimes, taus, tau_stats,
                 [LAMBDA_OUT, chart, headline_csv, regime_csv, REPORT_OUT])
    print("\n".join(_REPORT))
    return lambdas


if __name__ == "__main__":
    main()
