"""
Step 2c - assemble the reference-date panel of tariff-exposure scores.

Joins the raw per-document scores from tariff_scores.csv onto the pull log's
(permno, reference_date, accession) rows, applies the contemporaneity rule, and standardises
each measure WITHIN each reference date. The result is one row per firm per reference date -
the panel v6 sections 7.4 and 7.5 consume.

Why standardisation lives here and not in score_filings: a z-score is only defined relative to
a cross-section, and the cross-section here is a reference date. tariff_scores.csv is keyed on
the document and deliberately has no reference_date column, because the same 10-K is the most
recent filing at several dates and carries the same raw score at each. Pooling the
standardisation across 17 dates would mix vintages into one distribution.

    python src/build_texp_panel.py
    python src/build_texp_panel.py --status     # per-date counts, writes nothing
"""

import argparse

import pandas as pd

import run_report

import score_filings
from config import BASE, CLEAN_DIR, EDGAR_LOG as LOG_CSV, INTERMEDIATE_DIR, OUTPUT_DIR

REPORT_OUT = OUTPUT_DIR / "texp_panel_validation_report.txt"
SCORES_CSV = CLEAN_DIR / "tariff_scores.csv"
PANEL_OUT = CLEAN_DIR / "texp_panel.csv"
DIAG_OUT = INTERMEDIATE_DIR / "texp_panel_diagnostics.csv"

# Maximum age of the fiscal period behind a filing, measured to the reference date in exact
# calendar months. Replaces score_filings.FISCAL_YEARS = (2024, 2025), which was correct for a
# single 2025 reference date but cannot generalise: a calendar-year gap rule collapses at year
# boundaries and would discard 79% of the 2020-01-15 cross-section, whose FY2018 filings are
# the correct point-in-time choice because the FY2019 10-Ks were not yet filed.
#
# 15 months chosen on the corpus, not by analogy: the 99.9th percentile of period-to-reference
# staleness is 15.0 months, a 15-month bound retains 99.1-100% at every one of the 17 reference
# dates, and at 2025-04-02 it reproduces the old FISCAL_YEARS filter exactly (2,995 of 3,039
# rows). What it excludes are transition-period and fiscal-year-end-change filings.
MAX_PERIOD_STALENESS_MONTHS = 15

TEXP_COLUMNS = score_filings.TEXP_COLUMNS
OUTPUT_COLUMNS = (
    ["permno", "cik", "accession", "reference_date", "filing_date", "period_of_report",
     "fiscal_year", "period_staleness_days"]
    + TEXP_COLUMNS + [f"{c}_z" for c in TEXP_COLUMNS]
)


def load_pull_rows(log: pd.DataFrame) -> pd.DataFrame:
    """One row per (permno, reference_date) for every successful pull, with its accession.

    The log is append-only and not deduplicated - a firm retried across runs keeps every row,
    and no_cik firms accumulate one row per reference date - so identical duplicates are
    collapsed here. Verified in-session: all 147 duplicate groups are exact copies.
    """
    ok = log[log["found_10k"].astype(str).str.strip().str.lower() == "true"].copy()
    ok = ok.drop_duplicates(["permno", "reference_date", "accession"])
    # The log is read wholly as str to protect cik's zero padding; permno is a numeric
    # identifier and must match the scores table's int64 for the join to bind at all.
    ok["permno"] = pd.to_numeric(ok["permno"]).astype("int64")
    return ok[["permno", "cik", "accession", "reference_date",
               "filing_date", "period_of_report"]]


def attach_scores(pull_rows: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                         pd.DataFrame]:
    """Join raw scores onto the pull rows, returning (matched, unmatched).

    Joined on (permno, accession) rather than accession alone: a joint 10-K filed by a parent
    and a subsidiary carries a separate score row per PERMNO. Unmatched rows are returned
    rather than dropped silently - they are filings that exist in the pull but were excluded
    at scoring (no Item 1A) or have not been scored yet, and the caller reports both.
    """
    scores = scores.assign(permno=pd.to_numeric(scores["permno"]).astype("int64"))
    joined = pull_rows.merge(
        scores[["permno", "accession", "fiscal_year"] + TEXP_COLUMNS],
        on=["permno", "accession"], how="left", validate="many_to_one")
    unmatched = joined[joined["TExp_item1a"].isna()]
    return joined[joined["TExp_item1a"].notna()].copy(), unmatched


def apply_contemporaneity(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop firm-dates whose fiscal period is staler than MAX_PERIOD_STALENESS_MONTHS.

    Exact calendar months via DateOffset, so the bound does not drift with month length.
    Returns (kept, dropped); the caller reports the drop rather than absorbing it.
    """
    ref = pd.to_datetime(panel["reference_date"])
    period = pd.to_datetime(panel["period_of_report"], errors="coerce")
    panel = panel.assign(period_staleness_days=(ref - period).dt.days)
    within = period + pd.DateOffset(months=MAX_PERIOD_STALENESS_MONTHS) >= ref
    # A period that will not parse cannot be shown to be contemporaneous, so it is dropped
    # rather than admitted by a comparison that silently evaluates False.
    within = within & period.notna()
    return panel[within].copy(), panel[~within].copy()


def add_within_date_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Standardise each measure within its reference date.

    A reference date is one cross-section, so its own mean and standard deviation are the only
    meaningful centring for that date. A date whose measure has zero dispersion yields NaN
    rather than a divide-by-zero; NaN raw scores stay NaN and are excluded from both moments.
    """
    for col in TEXP_COLUMNS:
        grouped = panel.groupby("reference_date")[col]
        sd = grouped.transform("std")
        panel[f"{col}_z"] = (panel[col] - grouped.transform("mean")) / sd.where(sd > 0)
    return panel


def report(panel: pd.DataFrame, unmatched: pd.DataFrame, stale: pd.DataFrame) -> None:
    """Print the per-date panel report and assert the standardisation held."""
    print("=" * 78)
    print("TEXP PANEL")
    print("=" * 78)
    print(f"{'reference_date':<16}{'firms':>8}{'TExp_item1a mean':>19}"
          f"{'z mean':>10}{'z sd':>8}{'zero share':>12}")
    for date, g in panel.groupby("reference_date"):
        z = g["TExp_item1a_z"].dropna()
        print(f"{date:<16}{len(g):>8,}{g['TExp_item1a'].mean():>19.5f}"
              f"{z.mean():>10.2e}{z.std():>8.3f}{(g['TExp_item1a'] == 0).mean():>11.1%}")

    print(f"\nPanel rows {len(panel):,} across {panel['reference_date'].nunique()} "
          f"reference dates, {panel['permno'].nunique():,} distinct firms.")
    print(f"Dropped: {len(unmatched):,} unscored/no-Item-1A, {len(stale):,} beyond "
          f"{MAX_PERIOD_STALENESS_MONTHS} months of period staleness.")

    # Standardisation is checked, not assumed: a silent grouping error would leave a z column
    # whose per-date moments are not (0, 1) and nothing downstream would notice.
    for col in TEXP_COLUMNS:
        stats = panel.groupby("reference_date")[f"{col}_z"].agg(["mean", "std"]).dropna()
        if len(stats) and (stats["mean"].abs().max() > 1e-9
                           or (stats["std"] - 1).abs().max() > 1e-6):
            raise ValueError(f"{col}_z is not standardised within reference_date: "
                             f"worst mean {stats['mean'].abs().max():.2e}, "
                             f"worst sd {stats['std'].max():.6f}")
    print("Verified: every reference date's z columns have mean 0 and sd 1 within date.")


def _up_to_date(outputs, label: str, force: bool) -> bool:
    """True when every output already exists and the caller has not passed --force."""
    missing = [p for p in outputs if not p.exists()]
    if force or missing:
        if missing and not force:
            print(f"{label}: rebuilding - missing {', '.join(p.name for p in missing)}")
        return False
    print(f"{label}: already built, not regenerated. Outputs:")
    for p in outputs:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:,.1f} MB)")
    print("  Pass --force to rebuild.")
    return True


def main() -> pd.DataFrame | None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the outputs already exist")
    ap.add_argument("--status", action="store_true",
                    help="report per-date counts and exit without writing the panel")
    args = ap.parse_args()
    if not args.status and _up_to_date([PANEL_OUT, DIAG_OUT, REPORT_OUT], 'build_texp_panel', args.force):
        return None

    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"{SCORES_CSV.name} not found; run score_filings.py first.")
    log = pd.read_csv(LOG_CSV, dtype=str).fillna("")
    scores = pd.read_csv(SCORES_CSV, dtype={"cik": str, "accession": str})

    pull_rows = load_pull_rows(log)
    matched, unmatched = attach_scores(pull_rows, scores)
    panel, stale = apply_contemporaneity(matched)
    panel = add_within_date_z(panel).reindex(columns=OUTPUT_COLUMNS)
    panel = panel.sort_values(["reference_date", "permno"])

    with run_report.capture(REPORT_OUT, title="STEP 2C - TEXP PANEL VALIDATION"):
        report(panel, unmatched, stale)
    if args.status:
        return None

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_OUT, index=False)
    print(f"\nWrote {PANEL_OUT.relative_to(BASE)} "
          f"({len(panel):,} rows x {len(panel.columns)} cols).")
    drops = pd.concat([unmatched.assign(drop_reason="unscored_or_no_item_1a"),
                       stale.assign(drop_reason="period_too_stale")], ignore_index=True)
    drops.to_csv(DIAG_OUT, index=False)
    print(f"Wrote {DIAG_OUT.relative_to(BASE)} ({len(drops):,} rows) - excluded firm-dates.")
    return panel


if __name__ == "__main__":
    main()
