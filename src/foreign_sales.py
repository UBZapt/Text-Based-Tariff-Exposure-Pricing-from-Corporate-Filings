"""
Step 3a - Compustat geographic-segment cleaning, foreign-sales share (FS), and TExp validation.

Cleans the raw Compustat geographic-segment file to one as-first-reported snapshot per
firm-datadate, aggregates it to a firm-year foreign-sales share, links it to PERMNO through the
CCM bridge, merges it onto the scored tariff-exposure panel, and reports the TExp-FS correlations
required by research design section 7.1 (and the FS control used for H5 in section 7.2).

    FS = foreign_sales / total_sales,  total_sales = domestic + foreign + reconciling rows

The segment file carries no company-total row (geotp is only 2/3 plus non-geographic sid=99
rows), so the denominator is built from every row in the firm-datadate group and the
domestic + foreign vs total reconciliation is reported per firm-year as recon_gap.

Reads the 2025-04-02 cross-section of data/clean/texp_panel.csv; the scoring pipeline itself is
not touched.

    python src/foreign_sales.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import chartstyle as cs
import palette
from config import BASE, CLEAN_DIR, OUTPUT_DIR, RAW_DIR

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
SEGMENT_FILE = RAW_DIR / "Compustat Geographic segment data.csv"
BRIDGE_CSV = CLEAN_DIR / "clean_firm_bridge.csv"
FS_OUT = CLEAN_DIR / "foreign_sales_share.csv"

# Exposure is read from the reference-date panel at ONE vintage, not from tariff_scores.csv.
# Since the Step 2 rescale the scores table is one row per (permno, accession) across 17 reference
# dates, so joining it on (permno, fiscal_year) does not fail - it silently widens this validation
# from one cross-section to nine fiscal years (2,110 firm-years to 9,383). The measure validation
# reported in section 7.1 is anchored on the 2025 event, so it takes the 2025 cross-section and
# only that: one FS-TExp correlation, on the sample the event study actually uses.
# 2025-04-02 is clean_controls_data.CYCLES["2025"]["texp_ref"]["impose"]; load_texp_vintage raises
# and lists the alternatives if the panel ever stops carrying it.
TEXP_PANEL_CSV = CLEAN_DIR / "texp_panel.csv"
TEXP_REFERENCE_DATE = "2025-04-02"
# CSV rather than Parquet throughout: Windows Smart App Control blocks pyarrow's DLLs.
# cik and accession must come back as strings or their zero padding is lost, exactly as gvkey
# would be - the failure this module already documents for the unpadded segment export.
TEXP_READ_DTYPES = {"cik": str, "accession": str}
SCATTER_OUT = OUTPUT_DIR / "texp_fs_scatter.png"
REPORT_OUT = OUTPUT_DIR / "fs_validation_report.txt"

DATE_FORMAT = "%d/%m/%Y"          # the Compustat export is day-first
VALUE_FIELD = "sales"             # segment sales; revts agrees on 97.8% of rows (reported)
DOMESTIC_GEOTP = "2"              # Compustat: country of INCORPORATION, not the US (reported)
FOREIGN_GEOTP = "3"
GVKEY_WIDTH = 6                   # the bridge zero-pads gvkey; the segment file does not
OPEN_END = pd.Timestamp("2099-12-31")   # still-active CCM link sentinel
RECON_TOL = 0.01                  # |(dom+for)-total| / |total| above this is a material gap
SALES_COLS = ["domestic_sales", "foreign_sales", "other_sales"]

HEADLINE_MEASURE = "TExp_item1a"  # scatter exhibit and primary reported correlation
TEXP_MEASURES = ["TExp_item1a", "TExp_rest", "TExp_combined"]

# Labels counted as a US-domestic segment in the incorporation-country diagnostic only; they
# never affect a score. Compustat assigns geotp==2 by country of incorporation, so a
# Bermuda-registered filer books its US sales as foreign unless CRSP's USIncFlg screen has
# already removed it upstream.
US_DOMESTIC_LABELS = ("united states", "u.s", "us", "usa", "domestic", "puerto rico")
REGIONAL_DOMESTIC_LABELS = ("north america", "americas", "the americas", "north american")

OUTPUT_COLUMNS = [
    "gvkey", "permno", "fiscal_year", "datadate", "domestic_segment",
    "domestic_sales", "foreign_sales", "other_sales", "total_sales", "FS",
    "n_foreign_segments", "domestic_sales_missing", "recon_gap",
]

# Grey for the mass of points, blue for the line drawn through them: the emphasis colour goes on
# the one element the eye should find first, as the formatting reference does.
PALETTE = palette.roles(points="CATEGORICAL_2", fit="CATEGORICAL_1")

_REPORT: list[str] = []


def _say(line: str = "") -> None:
    """Append one line to the single consolidated validation report."""
    _REPORT.append(line)


def _section(title: str) -> None:
    """Start a titled report section."""
    _say()
    _say(title)
    _say("-" * len(title))


# --------------------------------------------------------------------------- #
# Load and de-duplicate the raw segment file                                  #
# --------------------------------------------------------------------------- #
def load_segments() -> pd.DataFrame:
    """Read the raw geographic-segment file with identifier and date types pinned.

    gvkey is zero-padded to the bridge's width here rather than at join time: the export carries
    4-, 5- and 6-character gvkeys, so an unpadded merge silently keeps under a third of the rows
    instead of failing.
    """
    if not SEGMENT_FILE.exists():
        raise FileNotFoundError(f"{SEGMENT_FILE.name} not found; it is a required raw input.")
    seg = pd.read_csv(SEGMENT_FILE, dtype=str)
    seg["gvkey"] = seg["gvkey"].str.strip().str.zfill(GVKEY_WIDTH)
    seg["datadate"] = pd.to_datetime(seg["datadate"], format=DATE_FORMAT)
    seg["srcdate"] = pd.to_datetime(seg["srcdate"], format=DATE_FORMAT)
    seg["sales"] = pd.to_numeric(seg[VALUE_FIELD], errors="coerce")
    seg["revts"] = pd.to_numeric(seg["revts"], errors="coerce")
    return seg


def filter_first_reported(seg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the as-first-reported snapshot of each firm-datadate: srcdate == datadate.

    Compustat re-presents a segment year in each later filing's comparatives, so a firm-datadate
    carries up to three srcdate vintages. Returns (kept rows, the firm-datadate groups holding no
    as-first-reported row at all, which are lost at this step).
    """
    kept = seg[seg["srcdate"] == seg["datadate"]].copy()
    all_groups = seg[["gvkey", "datadate"]].drop_duplicates()
    kept_groups = kept[["gvkey", "datadate"]].drop_duplicates()
    lost = all_groups.merge(kept_groups, on=["gvkey", "datadate"], how="left", indicator=True)
    return kept, lost[lost["_merge"] == "left_only"].drop(columns="_merge")


def verify_dedup(kept: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test the two pre-conditions the srcdate filter is being trusted to deliver.

    (a) No duplicate sid within a firm-datadate group, and (b) exactly one domestic (geotp==2)
    row per group. Returns the FAILING groups for both checks. They are reported separately and
    never dropped or averaged: a failure means the filter is not a valid dedup rule for that
    vintage and the aggregation below would silently double-count.
    """
    sid_stats = kept.groupby(["gvkey", "datadate"])["sid"].agg(["size", "nunique"])
    dup_sid = sid_stats[sid_stats["size"] != sid_stats["nunique"]].reset_index()

    is_domestic = kept["geotp"] == DOMESTIC_GEOTP
    dom_count = is_domestic.groupby([kept["gvkey"], kept["datadate"]]).sum().rename("n_domestic")
    bad_domestic = dom_count[dom_count != 1].reset_index()
    return dup_sid, bad_domestic


# --------------------------------------------------------------------------- #
# Firm-year aggregation and FS                                                #
# --------------------------------------------------------------------------- #
def build_firm_year(kept: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate segment rows to one row per firm-year.

    Sales are summed with min_count=1 so a bucket whose rows are all null stays NaN instead of
    collapsing to a spurious zero that would clear the total_sales > 0 screen. n_foreign_segments
    counts foreign ROWS, which is what separates "no foreign segment reported" (a legitimate
    FS = 0) from "foreign segments reported but every sales value null" (unrecoverable).
    fiscal_year is the calendar year of datadate, matching the TExp convention rather than
    Compustat's fyear rule. Returns (panel, the firm-years that carried two datadates).
    """
    bucket = np.where(kept["geotp"] == DOMESTIC_GEOTP, "domestic_sales",
                      np.where(kept["geotp"] == FOREIGN_GEOTP, "foreign_sales", "other_sales"))
    work = kept.assign(bucket=bucket)
    grouped = work.groupby(["gvkey", "datadate", "bucket"])

    values = grouped["sales"].sum(min_count=1).unstack("bucket").reindex(columns=SALES_COLS)
    counts = (grouped.size().unstack("bucket")
              .reindex(columns=SALES_COLS, index=values.index).fillna(0))

    panel = values.reset_index()
    panel["n_foreign_segments"] = counts["foreign_sales"].to_numpy().astype(int)
    panel["fiscal_year"] = panel["datadate"].dt.year

    labels = (kept.loc[kept["geotp"] == DOMESTIC_GEOTP, ["gvkey", "datadate", "snms"]]
              .rename(columns={"snms": "domestic_segment"}))
    panel = panel.merge(labels, on=["gvkey", "datadate"], how="left")

    dup_mask = panel.duplicated(["gvkey", "fiscal_year"], keep=False)
    multi = (panel.loc[dup_mask, ["gvkey", "fiscal_year", "datadate"]]
             .sort_values(["gvkey", "fiscal_year", "datadate"]))
    panel = (panel.sort_values("datadate")
             .drop_duplicates(["gvkey", "fiscal_year"], keep="last")
             .reset_index(drop=True))
    return panel, multi


def compute_fs(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute total_sales, FS and the reconciliation gap, dropping only the stated cases.

    total_sales sums every row in the group, reconciling (sid=99 Corporate/Eliminations) rows
    included, so recon_gap = |(domestic + foreign) - total| / |total| measures exactly what those
    rows contribute. Nothing is imputed: firm-years are dropped only where total_sales is
    null/zero/negative (FS undefined) or foreign sales are unrecoverable, each counted separately.
    """
    panel = panel.copy()
    panel["total_sales"] = panel[SALES_COLS].sum(axis=1, min_count=1)
    panel["domestic_sales_missing"] = panel["domestic_sales"].isna()

    total = panel["total_sales"]
    rules = {
        "total_sales null": total.isna(),
        "total_sales zero": total == 0,
        "total_sales negative": total < 0,
        "foreign unrecoverable (segments reported, all sales null)":
            (panel["n_foreign_segments"] > 0) & panel["foreign_sales"].isna(),
    }
    dropped = pd.Series(False, index=panel.index)
    counts = {}
    for reason, mask in rules.items():      # first matching reason wins, so counts never overlap
        fresh = mask.fillna(False) & ~dropped
        counts[reason] = int(fresh.sum())
        dropped |= fresh

    fs = panel[~dropped].copy()
    fs["FS"] = fs["foreign_sales"].fillna(0.0) / fs["total_sales"]
    fs["recon_gap"] = ((fs["domestic_sales"].fillna(0.0) + fs["foreign_sales"].fillna(0.0)
                        - fs["total_sales"]).abs() / fs["total_sales"].abs())
    return fs.reset_index(drop=True), counts


# --------------------------------------------------------------------------- #
# Linking                                                                     #
# --------------------------------------------------------------------------- #
def link_permno(fs: pd.DataFrame, bridge: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach PERMNO via the CCM bridge, requiring the fiscal year-end inside the link interval.

    The bridge is already restricted to primary links with the still-active 'E' sentinel mapped to
    an open end date (clean_data.load_link / build_bridge), so only date validity is applied here.
    A firm-year matching several links is resolved the way the EDGAR pull resolves a CIK: LINKPRIM
    'P' first, then the latest interval.
    """
    b = bridge.copy()
    b["gvkey"] = b["gvkey"].str.strip().str.zfill(GVKEY_WIDTH)
    b["linkdt"] = pd.to_datetime(b["linkdt"])
    b["linkenddt"] = pd.to_datetime(b["linkenddt"]).fillna(OPEN_END)
    b["_prim"] = (b["linkprim"] == "P").astype(int)

    m = fs.merge(b[["gvkey", "permno", "linkdt", "linkenddt", "_prim"]], on="gvkey", how="left")
    has_link = m["permno"].notna()
    in_window = has_link & (m["linkdt"] <= m["datadate"]) & (m["linkenddt"] >= m["datadate"])

    keys = ["gvkey", "fiscal_year"]
    with_link_keys = m.loc[has_link, keys].drop_duplicates()
    linked_keys = m.loc[in_window, keys].drop_duplicates()

    linked = m[in_window].sort_values(["_prim", "linkenddt", "linkdt"])
    ambiguous = int(linked.loc[linked.duplicated(keys, keep=False), keys]
                    .drop_duplicates().shape[0])
    linked = linked.drop_duplicates(keys, keep="last").copy()
    linked["permno"] = linked["permno"].astype(int)

    link_stats = {
        "firm_years": len(fs),
        "no_ccm_link": len(fs) - len(with_link_keys),
        "outside_link_interval": len(with_link_keys) - len(linked_keys),
        "linked": len(linked),
        "ambiguous_resolved": ambiguous,
    }
    return linked.drop(columns=["linkdt", "linkenddt", "_prim"]), link_stats


def load_texp_vintage(path: Path | None = None,
                      reference_date: str = TEXP_REFERENCE_DATE) -> pd.DataFrame:
    """The scored TExp cross-section at one reference date, one row per firm.

    Sliced before anything downstream sees it, so the correlation, the scatter and the report all
    describe the same single vintage rather than a pool of nine.
    """
    path = TEXP_PANEL_CSV if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run build_texp_panel.py first.")
    panel = pd.read_csv(path, dtype=TEXP_READ_DTYPES)
    frame = panel[panel["reference_date"] == reference_date]
    if frame.empty:
        raise ValueError(f"{path.name} holds no cross-section at {reference_date}; available: "
                         f"{', '.join(sorted(panel['reference_date'].unique()))}")
    if frame["permno"].duplicated().any():
        raise ValueError(f"the {reference_date} cross-section is not unique on permno")
    return frame.reset_index(drop=True)


def merge_texp(fs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the FS panel to the scored TExp panel on (permno, fiscal_year)."""
    sc = scores.copy()
    sc["permno"] = sc["permno"].astype(int)
    sc["fiscal_year"] = sc["fiscal_year"].astype(int)
    keep = ["permno", "fiscal_year", "cik", "accession"] + TEXP_MEASURES
    return sc[keep].merge(fs, on=["permno", "fiscal_year"], how="inner")


# --------------------------------------------------------------------------- #
# Correlations and exhibit                                                    #
# --------------------------------------------------------------------------- #
def correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman between each raw TExp measure and FS, pooled and by fiscal year."""
    samples = [("pooled", merged)] + [(str(y), g) for y, g in merged.groupby("fiscal_year")]
    rows = []
    for measure in TEXP_MEASURES:
        for label, sub in samples:
            pair = sub[[measure, "FS"]].dropna()
            row = {"measure": measure, "sample": label, "n": len(pair),
                   "pearson_r": np.nan, "pearson_p": np.nan,
                   "spearman_rho": np.nan, "spearman_p": np.nan}
            if len(pair) >= 3:
                pr = stats.pearsonr(pair[measure], pair["FS"])
                sr = stats.spearmanr(pair[measure], pair["FS"])
                row.update(pearson_r=pr.statistic, pearson_p=pr.pvalue,
                           spearman_rho=sr.statistic, spearman_p=sr.pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_scatter(merged: pd.DataFrame):
    """Write the pooled TExp-FS scatter with an OLS fit; returns the fit for the report.

    Two elements, so a legend: it identifies the point cloud and the fitted line without a block
    of statistics inside the frame, which belongs in the caption. The FS = 0 mass is handled with
    alpha rather than jitter, which would displace real values.
    """
    pair = merged[[HEADLINE_MEASURE, "FS"]].dropna()
    x, y = pair[HEADLINE_MEASURE].to_numpy(), pair["FS"].to_numpy()
    fit = stats.linregress(x, y)

    cs.apply()
    fig, ax = plt.subplots(figsize=cs.SIZE_HEATMAP)
    ax.scatter(x, y, s=9, alpha=0.30, color=PALETTE["points"], linewidths=0, zorder=3,
               label=f"Firm-years (n = {len(pair):,})")
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, fit.intercept + fit.slope * grid, color=PALETTE["fit"], lw=1.8, zorder=4,
            label=f"OLS fit, slope {fit.slope:.2f}")

    ax.set_xlabel(f"Tariff exposure, {HEADLINE_MEASURE} (share of Item 1A sentences)")
    ax.set_ylabel("Foreign sales share, FS")
    cs.legend(ax, loc="upper right")
    cs.frame(ax)
    cs.figure_title(fig, "Tariff exposure against foreign-sales share")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cs.save(fig, SCATTER_OUT)
    return fit


# --------------------------------------------------------------------------- #
# Validation report                                                           #
# --------------------------------------------------------------------------- #
def _describe_domestic_labels(merged: pd.DataFrame) -> None:
    """Report how often the geotp==2 segment is not actually the United States.

    Compustat defines the domestic segment by country of incorporation, so a foreign-registered
    filer books US sales as foreign and FS inverts. Reported, not corrected: CRSP's USIncFlg
    screen upstream already removes almost all such filers.
    """
    label = merged["domestic_segment"].fillna("").str.strip().str.lower().str.rstrip(".")
    is_us = label.str.startswith(US_DOMESTIC_LABELS)
    is_regional = label.isin(REGIONAL_DOMESTIC_LABELS)
    other = merged[~is_us & ~is_regional]
    n = len(merged)
    _say(f"  US-labelled domestic segment      {int(is_us.sum()):>7,} ({is_us.mean():.1%})")
    _say(f"  North America / Americas          {int(is_regional.sum()):>7,} "
         f"({is_regional.mean():.1%})")
    _say(f"  other label                       {len(other):>7,} ({len(other) / n:.1%})")
    if len(other):
        top = other["domestic_segment"].value_counts().head(8)
        _say("  most common other labels: "
             + ", ".join(f"{name} ({cnt})" for name, cnt in top.items()))
    _say("  (geotp==2 is the country of incorporation, not the US; reported, not corrected)")


def write_report(seg: pd.DataFrame, kept: pd.DataFrame, lost: pd.DataFrame,
                 dup_sid: pd.DataFrame, bad_domestic: pd.DataFrame, panel: pd.DataFrame,
                 multi: pd.DataFrame, drop_counts: dict, fs: pd.DataFrame,
                 link_stats: dict, linked: pd.DataFrame, scores: pd.DataFrame,
                 merged: pd.DataFrame, corr: pd.DataFrame, fit) -> None:
    """Assemble the single consolidated validation report and write it to disk."""
    _say("=" * 78)
    _say("FOREIGN-SALES SHARE (FS) - VALIDATION REPORT")
    _say("=" * 78)
    _say(f"Source      : {SEGMENT_FILE.name}")
    _say(f"Definition  : FS = foreign_sales / total_sales, "
         f"total_sales = domestic + foreign + reconciling")
    _say(f"Value field : {VALUE_FIELD}")

    _section("1. Segment funnel (every exclusion attributed)")
    n_groups_all = seg[["gvkey", "datadate"]].drop_duplicates().shape[0]
    n_groups_kept = kept[["gvkey", "datadate"]].drop_duplicates().shape[0]
    _say(f"  raw rows                                   {len(seg):>9,}")
    _say(f"  rows with srcdate == datadate              {len(kept):>9,}")
    _say(f"  firm-datadate groups, raw                  {n_groups_all:>9,}")
    _say(f"  firm-datadate groups, as-first-reported    {n_groups_kept:>9,}")
    _say(f"    - lost: no srcdate == datadate row       {len(lost):>9,}")
    if len(lost):
        by_year = lost["datadate"].dt.year.value_counts().sort_index()
        _say(f"      by year: {by_year.to_dict()}")
    _say(f"  firm-years after collapsing datadates      {len(panel):>9,}")
    for reason, n in drop_counts.items():
        _say(f"    - dropped {reason:<48s} {n:>7,}")
    _say(f"  = FS panel                                 {len(fs):>9,}")
    total_dropped = sum(drop_counts.values())
    if len(fs) + total_dropped != len(panel):
        raise ValueError(f"funnel does not reconcile: {len(fs):,} kept + {total_dropped:,} "
                         f"dropped != {len(panel):,} firm-years")
    _say(f"  reconciles: {len(fs):,} + {total_dropped:,} = {len(panel):,}")

    _section("2. De-duplication pre-conditions (srcdate == datadate as the dedup rule)")
    _say(f"  (a) groups with a duplicate sid            {len(dup_sid):>9,}")
    if len(dup_sid):
        _say("      FAILING groups (not dropped, not averaged):")
        for r in dup_sid.head(10).itertuples(index=False):
            _say(f"        gvkey={r.gvkey} datadate={r.datadate:%Y-%m-%d} "
                 f"rows={r.size} distinct sid={r.nunique}")
    _say(f"  (b) groups without exactly one geotp==2    {len(bad_domestic):>9,} "
         f"of {n_groups_kept:,}")
    if len(bad_domestic):
        _say("      FAILING groups (not dropped, not averaged):")
        for r in bad_domestic.head(10).itertuples(index=False):
            _say(f"        gvkey={r.gvkey} datadate={r.datadate:%Y-%m-%d} "
                 f"n_domestic={r.n_domestic}")
    if not len(dup_sid) and not len(bad_domestic):
        _say("      both hold on this file: the srcdate filter is a valid dedup rule here")
    n_multi = multi[["gvkey", "fiscal_year"]].drop_duplicates().shape[0]
    _say(f"  firm-years carrying two datadates          {n_multi:>9,} (later datadate kept)")
    for r in multi.head(10).itertuples(index=False):
        _say(f"      gvkey={r.gvkey} fiscal_year={r.fiscal_year} datadate={r.datadate:%Y-%m-%d}")

    _section("3. Source-field agreement (sales vs revts)")
    both = kept.dropna(subset=["sales", "revts"])
    _say(f"  rows with both present                     {len(both):>9,}")
    _say(f"  sales == revts                             {(both['sales'] == both['revts']).mean():>9.1%}")
    _say(f"  rows with both null                        {int(kept['sales'].isna().sum()):>9,}")

    _section("4. Sanity check: domestic + foreign vs total_sales")
    gap = fs["recon_gap"]
    _say(f"  exact (gap == 0)                           {(gap < 1e-9).mean():>9.1%}")
    _say(f"  gap > {RECON_TOL:.0%}                                    {int((gap > RECON_TOL).sum()):>9,}")
    _say(f"  gap > 5%                                   {int((gap > 0.05).sum()):>9,}")
    _say(f"  max gap                                    {gap.max():>9.3f}")
    _say("  (the gap is exactly the reconciling sid=99 Corporate/Eliminations contribution)")
    _say(f"  firm-years with a null domestic row        {int(fs['domestic_sales_missing'].sum()):>9,}")
    _say("  (flagged as domestic_sales_missing, not dropped: with this denominator a null")
    _say("   domestic row omits domestic sales from the total and inflates FS)")

    _section("5. FS distribution (panel, before the TExp merge)")
    s = fs["FS"]
    _say(f"  n {len(s):,} | min {s.min():.4f} | p10 {s.quantile(.1):.4f} | median {s.median():.4f}"
         f" | p90 {s.quantile(.9):.4f} | max {s.max():.4f}")
    _say(f"  FS == 0 (no foreign segment reported or zero foreign sales) {(s == 0).mean():.1%}")
    _say(f"  FS < 0 {int((s < 0).sum()):,} | FS > 1 {int((s > 1).sum()):,} "
         f"(retained: negative and eliminating segment sales are genuine, not imputed away)")
    _say(f"  fiscal years {int(fs['fiscal_year'].min())}-{int(fs['fiscal_year'].max())}")
    clean = fs.loc[~fs["domestic_sales_missing"], "FS"]
    _say(f"  excluding domestic_sales_missing rows: n {len(clean):,} | median "
         f"{clean.median():.4f} | p90 {clean.quantile(.9):.4f} | FS == 0 {(clean == 0).mean():.1%}")
    _say("  (those rows lack a domestic figure, so their total omits domestic sales and FS runs")
    _say("   high; they are almost all foreign-incorporated filers the CRSP screen excludes)")

    _section("6. gvkey -> permno link (CCM bridge, date-valid)")
    _say(f"  FS firm-years                              {link_stats['firm_years']:>9,}")
    _say(f"    - gvkey absent from the bridge           {link_stats['no_ccm_link']:>9,}")
    _say(f"    - fiscal year-end outside link interval  {link_stats['outside_link_interval']:>9,}")
    _say(f"  = linked                                   {link_stats['linked']:>9,} "
         f"({link_stats['linked'] / link_stats['firm_years']:.1%} match rate)")
    _say("  (the bridge covers only the cleaned CRSP analysis universe - US common equity,")
    _say("   price > $1, above the NYSE p10 size breakpoint - so a gvkey absent from it is a")
    _say("   firm outside the sample, not a broken link. Compustat segments span 6,770 gvkeys")
    _say("   against the bridge's 4,354.)")
    _say(f"  firm-years matching >1 link (resolved on LINKPRIM then latest interval) "
         f"{link_stats['ambiguous_resolved']:,}")

    _section("7. Merge to the scored TExp panel")
    _say(f"  TExp vintage                              {TEXP_REFERENCE_DATE:>10}  "
         f"(one cross-section, not the pooled panel)")
    _say(f"  TExp scored firms at that vintage           {len(scores):>9,}")
    _say(f"  FS firm-years with a permno                {len(linked):>9,}")
    _say(f"  = inner join on (permno, fiscal_year)      {len(merged):>9,} "
         f"({len(merged) / len(scores):.1%} of scored)")
    _say(f"  by fiscal year: {merged['fiscal_year'].value_counts().sort_index().to_dict()}")
    unmatched = len(scores) - len(merged)
    _say(f"  scored firms with no FS match              {unmatched:>9,}")
    if merged.duplicated(["permno", "fiscal_year"]).any():
        raise ValueError("merged panel has duplicate (permno, fiscal_year) rows")

    _section("8. Measurement-validity diagnostic: the domestic segment label")
    _describe_domestic_labels(merged)

    _section("9. TExp vs FS correlations")
    _say(f"  {'measure':16s} {'sample':>8} {'n':>7} {'pearson r':>11} {'p':>9} "
         f"{'spearman':>11} {'p':>9}")
    for r in corr.itertuples(index=False):
        _say(f"  {r.measure:16s} {r.sample:>8} {r.n:>7,} {r.pearson_r:>11.4f} "
             f"{r.pearson_p:>9.2e} {r.spearman_rho:>11.4f} {r.spearman_p:>9.2e}")
    _say(f"  headline measure: {HEADLINE_MEASURE}")
    _say(f"  pooled OLS FS on {HEADLINE_MEASURE}: slope {fit.slope:.4f} "
         f"(s.e. {fit.stderr:.4f}), intercept {fit.intercept:.4f}, R2 {fit.rvalue ** 2:.4f}")
    by_year = merged["fiscal_year"].value_counts().sort_index()
    if len(by_year) > 1 and by_year.min() < 0.1 * by_year.max():
        small = by_year.idxmin()
        _say(f"  NOTE: the FY{small} cell is {by_year.min():,} firm-years against "
             f"{by_year.max():,} in FY{by_year.idxmax()}. It holds only filers whose fiscal")
        _say("        period ended in early 2025 and who filed before the reference date, so it")
        _say("        is a fiscal year-end subsample rather than a second year of data; its")
        _say("        correlation is not comparable with the pooled figure.")

    _section("Outputs")
    for path in (FS_OUT, SCATTER_OUT, REPORT_OUT):
        _say(f"  {path.relative_to(BASE)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(_REPORT) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def _up_to_date(outputs, label: str, force: bool) -> bool:
    """True when every output already exists and the caller has not passed --force.

    Cleaning is deterministic in its inputs, so re-deriving an output that is already on disk
    costs I/O and produces the same bytes. The stages that stream per document (clean_filings,
    score_filings) have always resumed; this gives the whole-file stages the same courtesy, with
    an explicit override rather than an implicit one.
    """
    missing = [p for p in outputs if not p.exists()]
    if force or missing:
        if missing and not force:
            print(f"{label}: rebuilding - missing "
                  f"{', '.join(p.name for p in missing)}")
        return False
    print(f"{label}: already built, not regenerated. Outputs:")
    for p in outputs:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:,.1f} MB)")
    print("  Pass --force to rebuild.")
    return True


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    ap = argparse.ArgumentParser(description="Foreign-sales share and the TExp-FS validation.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the outputs already exist")
    args = ap.parse_args(argv)
    if _up_to_date([FS_OUT, SCATTER_OUT, REPORT_OUT], "foreign_sales", args.force):
        return None
    for path in (BRIDGE_CSV, TEXP_PANEL_CSV):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; run the upstream pipeline first.")

    seg = load_segments()
    kept, lost = filter_first_reported(seg)
    dup_sid, bad_domestic = verify_dedup(kept)
    panel, multi = build_firm_year(kept)
    fs, drop_counts = compute_fs(panel)

    bridge = pd.read_csv(BRIDGE_CSV, dtype=str)
    linked, link_stats = link_permno(fs, bridge)

    scores = load_texp_vintage()
    merged = merge_texp(linked, scores)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    linked.reindex(columns=OUTPUT_COLUMNS).to_csv(FS_OUT, index=False)

    corr = correlations(merged)
    fit = plot_scatter(merged)
    write_report(seg, kept, lost, dup_sid, bad_domestic, panel, multi, drop_counts, fs,
                 link_stats, linked, scores, merged, corr, fit)
    print("\n".join(_REPORT))
    return merged


if __name__ == "__main__":
    main()
