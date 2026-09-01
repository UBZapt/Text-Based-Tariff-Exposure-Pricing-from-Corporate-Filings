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

## Step 4/6 event study: the `--cycle` switch and the two policy cycles

The three event-study scripts run against either policy cycle, selected by one flag. `2025` is the
default and reproduces every pre-existing output; `cross_cycle` is the section 7.6 out-of-sample
leg. Run them in order:

```
python clean_controls_data.py [--cycle 2025|cross_cycle]   # controls panel
python estimate_car.py        [--cycle 2025|cross_cycle]   # FF5+MOM loadings, ARs, CARs
python run_car_regression.py  [--cycle 2025|cross_cycle]   # cross-sectional regressions
```

Cycle configuration lives in one place, `clean_controls_data.CYCLES`, shaped like
`edgar_pull.SCOPES`/`BATCHES`. Scripts 2 and 3 import it rather than re-declaring dates or paths,
and each has a `select_cycle()` that rebinds its own cycle-dependent constants off it.

| | `2025` (section 7.2) | `cross_cycle` (section 7.6) |
| --- | --- | --- |
| events | impose 2025-04-02, reverse 2025-08-29 | 7 Section 301 escalations 2018-03-01 … 2019-08-23, plus de-escalations 2019-10-11 (primary) and 2020-01-15 (Phase One, robustness) |
| daily returns | `Daily returns.csv` | `CRSP Daily returns cross cycle.csv` + `CRSp Daily returns cross cycle pt2.csv` |
| factors | `Fama French daily.csv` | `FF5+MOM daily Cross cycle.csv` + `FF5 + MOM daily cross cycle pt2.csv` |
| estimation window | per-run policy anchors (−51, −34, −66 trading days) | uniform `[-252, -11]`, all nine runs |
| outputs | unsuffixed | suffixed `_cross_cycle` |

Escalation dates are Bruno, Goltz & Luyten (2024, *European Financial Management*) Table 3, used
exactly as published; taking their event list removes the discretion in date selection and makes
the comparison to their results direct.

### Merging the two-part daily exports

Where a cycle lists more than one returns or factor file they are concatenated and reconciled by
the code that already handles CRSP's own repeated rows — no separate merge step. `dedupe_daily`
drops exact duplicate rows and **raises** on a `(permno, date)` pair carrying differing values;
`estimate_car.load_factors` does the same on `date`. The 2017 extensions overlap the originals by
61 trading days, and that overlap is byte-identical (174,902 CRSP rows across 2,903 PERMNOs, zero
differences on any field; zero disagreement on any FF factor), so the merge is silent. A future
re-pull that disagreed would fail loudly instead of being resolved by read order.

### Which 10-K each event uses

TExp is read from `clean_data/texp_panel.csv`, **sliced to each event's `edgar_pull` reference
date**, never from `tariff_scores.csv`. Since Step 2 was rescaled across 17 reference dates the
scores table holds one row per `(permno, accession)` — 13,380 rows over 3,762 firms — so a
permno-keyed read of it is ambiguous and the vintage a score belongs to is not recoverable from it.
`texp_panel.csv` carries `reference_date` and is unique on `(permno, reference_date)`.

`decile_sort.py` takes a single vintage this way — 2025-04-02, since §7.3 fixes breakpoints on one
cross-section and holds group membership constant across both legs. `foreign_sales.py` takes the
same vintage: the project reports **one** TExp–FS correlation, measured on the 2025 event's
cross-section (§7.1), because that is the sample the event study uses. It joins on
`(permno, fiscal_year)` rather than permno, so the rescale had not made it fail — it had silently
widened the correlation sample from 2,110 firm-years to 9,383 across FY2017–FY2025. Restricting it
restored every reported figure exactly: headline Pearson r = 0.2865, Spearman 0.3801, n = 2,110,
and `foreign_sales_share.csv` byte-identical.

Restricting it also surfaced a latent bug: `build_texp_panel.attach_scores` never selected
`fiscal_year` from the scores table, so `texp_panel.csv` declared the column and left it null in
all 29,474 rows. Inert where the column was only carried along, fatal to a join keyed on it. Fixed
at source; the rebuilt panel is identical on every TExp and staleness value.

Cross-cycle event dates and pull reference dates coincide by construction, so event `2018-03-01`
uses the 10-K selected at reference date `2018-03-01` — filed inside `[ref−364, ref−1]`, strictly
before the event, no look-ahead. Both 2025 legs share the single `2025-04-02` vintage, because H1
requires exposure held fixed across the imposition and reversal legs and there is no 2025-08-29
pull. `load_texp_reasons` is sliced the same way, against `output/texp_panel_diagnostics.csv` and
the reference-date slice of `edgar_pull_log.csv`.

### The H4 stability test

Run with a non-baseline cycle, Script 3 re-estimates the baseline cycle and fits

```
CAR = a + b*TExp + B*cc + d*(TExp x cc) + c*FS + gamma'X + FF12 dummies + e     H0: d = 0
```

on the per-event primary estimation samples themselves, so every section 7.2 exclusion already
applies and the pooled test cannot admit a firm the per-event regressions dropped. Reported
pooled per leg (tightening: 2025 imposition against all seven escalations; loosening: 2025 reversal
against the primary de-escalation) and pairwise, one cross-cycle date at a time, at each of the
three event windows. Output: `output/stability_test_results_cross_cycle.csv` and section 10b of the
regression report.

### Deviations, dated 2026-09-01

1. **Estimation window `[-252, -11]` for the cross cycle**, in place of per-run policy anchors.
   Nine events admit no defensible per-event anchor — the Section 301 process ran continuously from
   January 2018, so every candidate date sits inside the repricing it is meant to exclude, and
   choosing nine would reintroduce the date discretion section 7.6 rules out. `-11` is the latest
   uniform close leaving the widest event window `[-10,+10]` free of its own estimation window; all
   nine resolve to a full 242 days with zero own-event overlap, asserted in the report.

2. **Standard errors clustered on permno, on the pooled stability test only.** Stacking two cycles
   puts each firm in the sample once per event, so the per-event independence assumption does not
   carry over. All 27 per-event cross-cycle regressions keep `COV_TYPE = "nonrobust"`, identical to
   the section 7.2 run. Section 7.2 still nominates White HC for the reported table; that remains
   outstanding on both cycles.

3. **Controls and FF12 effects constrained equal across cycles** in the pooled regression; only
   TExp is interacted, per section 7.6's "a cycle interaction". A fully interacted model is a
   different and much weaker test.

4. **`estimate_car._check_contamination` no longer raises on cross-event overlap**, only on
   own-event overlap and on a non-contiguous window. With nine dates 6 to 162 trading days apart a
   later run's estimation window routinely spans earlier events — as the 2025 reversal window
   deliberately contains the imposition — and two overlaps are partial. Contiguity is what proves a
   window unmodified; partial overlap means only that its boundary fell inside a neighbouring
   event. Overlaps are now reported full/partial/none rather than failing the run.

5. **PolRisk is not included** in the cross-cycle specification, though the Hassan series covers
   2017-2020 and section 7.6 nominates it. Adding a control absent from the 2025 specification
   would break the specification identity the out-of-sample claim rests on. Descoped by
   instruction, along with the lexicon/BEA-Census stability check.

### Known limitation: foreign-sales coverage at the two earliest events

`Compustat Geographic segment data.csv` begins at datadate 2017-01-31, so firms still reporting
FY2016 fundamentals in early 2018 have no segment row and drop out on the FS control. FS matches
**1,185 firms at 2018-03-01 and 1,759 at 2018-03-22, against 2,093–2,157 at every later
cross-cycle date and 2,893 at 2025-04-02**; regression *n* at 2018-03-01 is 917 against 1,242–1,431
elsewhere. Nothing is imputed — absence from the segment file is not evidence of a domestic-only
firm — so the effect is a thinner cross-section at those two dates, reported rather than absorbed.
Re-pulling segments back to FY2015 would close it.

### Standing implementation trap

**Never bind a cycle-dependent path as a default argument.** Python evaluates defaults once, at
definition, so `def load_panel(path=PANEL_PATH)` keeps pointing at whichever cycle was active on
import and silently reads the wrong panel after `select_cycle`. `load_panel`,
`load_event_controls`, `load_factors`, `write_results` and `write_stability` all take `None` and
resolve from the module global at call time.
