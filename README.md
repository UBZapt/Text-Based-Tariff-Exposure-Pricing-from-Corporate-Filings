# Dissertation
> **Status:** this file is still a stub. `CLAUDE.md` requires it to be a full
> reproducibility document (dependencies, credentials, WRDS source-to-field mappings,
> end-to-end run instructions, assumptions, fallbacks, deviations). Only the EDGAR pull
> caching/rate-limiting behaviour is documented so far.

## EDGAR pull: submissions cache and rate limiting

### SEC fair-access compliance

`EDGAR_USER_AGENT` must be set (shell or gitignored `.env`) to `AppName ContactEmail`, as
SEC fair access requires; `edgar_pull.py` warns if the placeholder is still in use.

Request spacing is enforced by `_throttle()` inside `_get`, so the floor applies per
*request* — retries included — rather than per firm. With `REQUEST_SLEEP = 0.15` the
ceiling is **6.7 req/s against SEC's 10 req/s limit**; measured end-to-end rate is
~1.8 req/s once network latency is included. Firms answered from cache issue no request
and therefore consume no rate budget.

Previously the spacing came from two `time.sleep(REQUEST_SLEEP)` calls in `pull_one`, one
of which fired even when the filing was already cached and no request had been made. Both
were removed in favour of the `_get`-level gate, which is strictly more conservative: no
code path can now issue an unspaced request.

### Submissions cache (`submissions_cache/`)

One gzipped file per CIK, `CIK##########.json.gz`, wrapping the SEC submissions payload
with the timestamp it was fetched at:

```json
{"fetched_at": "<ISO-8601>", "submissions": { ... }}
```

The submissions file is an append-only filing history and does not depend on the reference
date, so one fetch serves an entire multi-date batch. A cached copy is reused for reference
date `ref` **only if `fetched_at >= ref`** — everything filed up to `ref-1` is then
guaranteed present. Otherwise it is refetched, and the newer copy (valid for that `ref` and
every earlier one) replaces it. Corrupt, truncated or old-format files are deleted and
refetched.

**This cannot change which filing is selected.** `select_10k` computes the selection locally
from whichever copy it is handed; the cache only changes where the filing history was read
from. Verified by comparing cached-vs-fresh selections across 5 CIKs × 3 reference dates:
0 mismatches. Before this change every reference date refetched a byte-identical document.

Effect on the 9-date `cross_cycle` batch: submissions requests fall from ~38,300
(4,258 firms × 9 dates) to ~4,258, cutting roughly 6.5 hours of request spacing to ~45 min.
Cache size is ~26 KB/firm, ~109 MB for the full 4,213-CIK bridge. The directory is
gitignored and is a pure cache — deleting it costs only refetch time.

Unaffected: `STALENESS_DAYS = 364`, the exact `form == "10-K"` match, the
`[ref-364, ref-1]` selection window, `MIN_FILING_BYTES`, the throttle-marker rejection,
and `filings_raw/` retention (still keyed on accession, still pruned against
`bridge_ciks()`).

## Step 2 pipeline: cleaning, scoring, and the reference-date panel

Across 17 reference dates the pull log's 31,142 successful rows cover only 13,983 distinct
firm-documents. Cleaning and raw scoring are properties of a *document*, not of the reference
date that selected it, so both run once per accession and are shared by every date that
selected it. Only the cross-sectional standardisation is date-dependent.

```
filings_raw/*.html
   |  clean_filings.py        once per accession, resumable
   v
clean_text/<accession>.json.gz  +  clean_data/clean_filings.csv (metadata only)
   |  score_filings.py        once per accession, resumable
   v
clean_data/tariff_scores.csv    raw TExp, no reference_date
   |  build_texp_panel.py     join pull log + contemporaneity + z within date
   v
clean_data/texp_panel.csv       one row per firm per reference date (v6 7.4/7.5)
```

Run in order; each stage has `--status` (reports done/remaining, does no work) and resumes
after an interruption, losing at most one document:

```
python clean_filings.py
python score_filings.py
python build_texp_panel.py
```

### Cleaned-text store (`clean_text/`)

One gzipped `{"item_1a": ..., "rest": ...}` document per accession. The text was previously
carried as two columns of `clean_filings.csv`, which at 380 KB of prose per filing was 1.14 GB
for one reference date and would have been ~5.3 GB across all 17 - a table that can be neither
built nor re-read without exhausting memory. Both stages now stream one document at a time, so
memory is flat in corpus size. Gitignored; ~1.45 GB gzipped at full scale. `clean_filings.csv`
keeps every non-text column, so identifiers, character counts and cleaning flags are unchanged.

`migrate_clean_filings.py` performed the one-off re-shaping of the original 2,987 rows. It
re-parses no HTML, so no score could move; all 2,988 rows were verified against their recorded
character counts (1,130,523,512 characters intact) before the old file was replaced, and the
original is kept as `clean_filings.pre-migration.csv`.

### Deviations from the single-vintage implementation

Dated 2026-09-01, on completion of the `cross_cycle` and `full_panel` pulls.

1. **`FISCAL_YEARS = (2024, 2025)` removed from `score_filings.py`**; the contemporaneity rule
   is now `build_texp_panel.MAX_PERIOD_STALENESS_MONTHS = 15`, applied per reference date.
   Scoring runs once per document and cannot know which date will consume the result, and the
   old pair would have dropped 12,779 of 13,980 accessions.

   15 months was chosen on the corpus rather than by analogy. The 99.9th percentile of
   period-to-reference staleness is 15.0 months; a 15-month bound retains 99.1-100% at every
   one of the 17 dates; and at 2025-04-02 it reproduces `FISCAL_YEARS=(2024, 2025)` exactly
   (2,995 of 3,039 pull rows). A calendar-year gap rule was rejected: it collapses at year
   boundaries and would discard 79% of the 2020-01-15 cross-section, whose FY2018 filings are
   the correct point-in-time choice because the FY2019 10-Ks were not yet filed.

2. **`TExp_*_z` removed from `tariff_scores.csv`; standardisation moved to
   `texp_panel.csv`, computed within each reference date.** A z-score is defined only relative
   to a cross-section, and that cross-section is a reference date - a column
   `tariff_scores.csv` deliberately does not have, because one 10-K is the most recent filing
   at several dates and carries the same raw score at each. Pooling 17 vintages into one
   distribution was the alternative. Nothing downstream read these columns
   (`decile_sort.py:75`, `foreign_sales.py:55`, `run_car_regression.py:43` all use raw
   `TExp_item1a`), and the 2,868 pre-existing rows were verified score-for-score identical
   after the column removal.

3. **Scope check sourced from `edgar_pull.bridge_ciks()`** instead of the CIK set resolved at
   `REFERENCE_DATE`. `_resolve_cik` picks one link row per reference date, so a firm linked in
   2018 but not 2025 was being dropped as `cik_outside_universe` purely because of when the
   scope happened to be evaluated. This is the same trap `CLAUDE.md` records for `prune_stale`.

4. **Both stages made resumable and streaming.** Rows are appended and flushed per document
   (`open_clean_log`, `open_scores_log`), mirroring `edgar_pull.open_log`. `score_filings` no
   longer materialises the corpus's ~26 million sentences: `tokenise_filing` and
   `score_section` keep their separation of concerns but are driven from one loop
   (`score_stream`) that discards a filing's sentences once scored. The removed-term audit,
   short-sentence tally and en-dash diagnostic are accumulated in that pass rather than each
   taking its own full pass over the text. **No change to the measure**: `clean_filing`,
   `split_item_1a`, `score_section` and the term list are untouched.

### Item 1A detection by vintage

`clean_filings.validate` reports the detection rate either side of FY2020
(`VINTAGE_SPLIT_YEAR`) and flags a gap wider than `VINTAGE_GAP_TOLERANCE = 5%`. The section
regexes were tuned on 2024-vintage documents and pre-2020 filings largely predate inline XBRL,
so the research design requires this rate be reported per vintage; a materially worse early
vintage is a finding for the write-up, not something to patch away silently. Baseline is 97.1%
on the 2025 corpus.
