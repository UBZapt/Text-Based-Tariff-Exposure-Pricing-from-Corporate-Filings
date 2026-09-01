# Dissertation
> **Status:** partial. `CLAUDE.md` requires this to be a full reproducibility document
> (dependencies, credentials, WRDS source-to-field mappings, end-to-end run instructions,
> assumptions, fallbacks, deviations). Documented so far: the EDGAR pull, the Step 2 cleaning
> and scoring pipeline, the Step 4/6 event-study cycles, and the Step 7/8 Fama-MacBeth test and
> its EPU regime split - each with its source mappings, assumptions and dated deviations.
> **Every §7.x test in the research design is now implemented.**
>
> **Still missing, and required:** the dependency list and Python version, `.env` and WRDS
> credential expectations, Step 1 (`clean_data.py`) source-to-field mappings for the CRSP and
> Compustat exports, and a single end-to-end run order covering every script from
> `clean_data.py` to `fama_macbeth_pricing.py`.

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

## Step 7: section 7.4 Fama-MacBeth pricing test (H2, H5)

Two scripts, split the way `build_texp_panel.py` and `run_car_regression.py` are: assemble the
monthly panel, then estimate. Run in order.

```
python build_fm_panel.py [--status]      # monthly firm panel, every candidate row + its reason
python fama_macbeth_pricing.py           # 96 monthly cross-sections, Newey-West, figure, report
```

| Path | Contents |
| --- | --- |
| `clean_data/fm_panel.csv` | 86,009 candidate firm-months x 31 cols; `exclusion_reason` empty on the 53,745 that estimate |
| `output/fm_lambda_panel.csv` | 192 rows - monthly lambda_1, its within-month se/t/p, lambda_FS, firm count, R2, lagged EPU, episode label, per specification |
| `output/fm_lambda_chart.png` | the section 7.4 figure: lambda_1,t over 96 months with both tariff episodes shaded |
| `output/fama_macbeth_validation_report.txt` | eleven sections, house style |

**No existing script was modified.** Both scripts import `clean_controls_data`, `clean_data` and
`run_car_regression` read-only and reuse their functions, so the section 7.2 and 7.6 outputs cannot
have moved; `git status` showing only two new files is the proof, which is why no `--cycle 2025`
re-verification was needed.

### Which filings the test uses, and why it cannot read the panel own z column

`texp_panel.csv` pools all **17** `edgar_pull` reference dates. This test uses the **nine** that are
the `full_panel` batch - April 2 each year 2017-2025, pulled at scope `subsample` - intersected with
the 1,000 PERMNOs in `output/full_panel_firm_sample.csv`. The other eight dates belong to the
cross-cycle event study.

Two of those nine, **2018-04-02 and 2025-04-02**, were *also* pulled at scope `full` for section
7.6, so they hold 2,289 and 2,868 rows against the 560 and 958 subsample firms present. The panel
`TExp_item1a_z` is therefore standardised over the wrong population at exactly those two vintages -
the subsample slice has mean 0.045/0.073 and sd 1.078/1.041 rather than (0, 1), with individual z
values moving by up to 0.70 - and those two vintages cover **20 of the 96 sample months**.
`build_fm_panel.load_texp` recomputes the z within (reference date x subsample) and asserts mean 0
and sd 1 per vintage to `build_texp_panel` own tolerances.

Standardising at all is required rather than cosmetic. Raw `TExp_item1a` cross-sectional sd rises
**2.7x** across the nine vintages (0.00471 in 2017 to 0.01265 in 2025) and its zero share falls from
**72% to 16%**, so one raw unit is a 2.7x larger move in exposure in 2017 than in 2025 and a
time-series mean of raw slopes would silently mix the two scales. Section 4 of the research design
defines TExp standardised for the same reason. Section 8 of the report converts the section 7.2
coefficients into the same units - multiplying by the sd of their own 2,868-firm cross-section,
0.012149 - so the two sections stay comparable.

**The full-scope rows at those two vintages never enter.** The subsample restriction is applied to
the CRSP monthly read, so a non-subsample firm has no row for a vintage to join to, and
`load_texp` asserts `issubset` on the vintage slice as well. At 2018-04-02 the panel holds 2,289
rows and this test uses 550 (425 estimate); at 2025-04-02 it holds 2,868 and uses 950 (733
estimate) - in line with the 330-728 of the other seven vintages, and the largest monthly
cross-section anywhere is 721 firms. The 20 months those two vintages serve hold 19.6% of
firm-months against 20.8% of months, so they are marginally *under*-represented rather than over.
Report section 2 shows the whole-vintage row count beside the count used, so the containment is
visible rather than asserted.

Separately, cross-sections do grow from 97 firms (2018-01) to 721 (2025), because the §7.0 draw was
taken at end-March 2025 and more of the 1,000 are listed in later years - the acknowledged cost of
that reference date. Fama-MacBeth averages the monthly slopes with **equal weight per month**, so
this affects how precisely each month is estimated, not how much each month counts toward
`lambda_1_bar`. Firm-month weighting would give +0.034% against the reported +0.046%.

A `y-04-02` vintage covers return months **May y through April y+1**: its newest filing is dated
`y-04-01`, so it is public before the first month opens, and `filing_date <= pit_date` is asserted
on every retained row. It is carried no further. Unlimited carry-forward would add 120
firm-vintage-years (+1.7%) at the cost of scoring a month with a filing already known to be more
than a year stale.

### Point-in-time construction

Every regressor is resolved at `pit_date`, the last calendar day of month *t-1*; the only month-*t*
quantity is the dependent variable.

| Variable | Source and rule |
| --- | --- |
| `ret` | raw `Monthly Returns.csv`, month *t*. Taken from the unscreened file so a firm screened in at *t-1* that breaks $1 during *t* still contributes the return it earned - selecting the dependent variable on the screen would be selection on the outcome |
| `in_screen_lag` | presence in `clean_returns.csv` at *t-1*. Presence *is* the section 6 screen; reading it one month back is the `estimate_car.pit_screen` rule |
| `me_lag`, `ln_me_lag` | `MthCap` at *t-1*, joined on the previous **calendar** month rather than by `shift`, which across a listing gap would import market equity from several months earlier |
| `bm`, `lev` | `ccd.merge_fundamentals` as-of `pit_date` with `allow_exact_matches=False`, then `ccd.build_ratios` - the same validated `available_date` gate section 7.2 uses |
| `mom12` | `ccd.add_momentum`, evaluated at month *t* because its window for month *j* is `[j-12, j-1]`, which is already the characteristic known at *t-1* |
| `FS` | `foreign_sales_share.csv` on the panel own point-in-time `(gvkey, datadate)`, exactly as `run_car_regression.build_sample` joins it |
| `ff12` | the firm own SIC at *t-1*, remapped monthly through `ccd.parse_ff12` |
| `epu_lag` | `clean_epu.csv` at *t-1*, carried only - section 7.5 consumes it, this test does not |

`clean_data.COMMON_EQUITY_FILTER` is applied to the raw monthly read so the universe definition
matches the `clean_returns.csv` one. Without it ten firm-months carry a second CRSP row -
`PrimaryExch` X, `SICCD` 0, null security metadata, the stub written the month a listing moves or
ends - which repeat the return and cap exactly but would fan out every downstream merge.

### Deviations, dated 2026-09-01

1. **Sample is 2018-01 to 2025-12, 96 monthly cross-sections**, against the design "approximately
   110". `Monthly Returns.csv` begins 2017-01-31 and no earlier return history exists anywhere in
   the project, so the twelve months ending *t-1* are first complete at 2018-01; TExp is
   independently unavailable before 2017-05. The eight months this costs are 2017-05 to 2017-12.
   Recovering them needs a CRSP monthly export for 2016; the momentum grid is derived from the
   file own span, so no code change would be required.
2. **Exposure standardised within vintage**, where sections 7.2 and 7.3 use raw TExp. Reasons above.
3. **FF12 is point-in-time and time-varying**, where `ccd.assign_ff12` fixes one label per firm from
   a snapshot before the active cycle first event - which for cycle 2025 would apply a 2025
   classification to a 2018 cross-section. That convention exists to hold industry effects
   identical across the two legs of one event study; 96 independent cross-sections have no such
   pair to protect.
4. **Both specifications are estimated on identical rows.** The H5 variant drops FS and the industry
   dummies from the right-hand side but not the rows requiring them, so any movement in
   `lambda_1_bar` is the specification and not the sample. Requiring FS costs 13,752 firm-months
   (16.0% of candidates) in both.
5. **`MIN_FIRMS_PER_MONTH = 50`** is a degrees-of-freedom floor set from the specification 18
   parameters, not from any month. It binds on none. Thin months are reported and carried, with a
   stated sensitivity, rather than removed at a threshold chosen after seeing which months it hits.
6. **PolRisk excluded**, per the constraint note at the head of section 7. **Version B** of the
   design (Fama-MacBeth on rolling TExp-factor betas) is not run; v6 drops it with the long-short
   portfolio it depended on.

### Known limitation: two thin months in early 2018

`Compustat Geographic segment data.csv` begins at datadate 2017-01-31, so firms still reporting
FY2016 fundamentals in early 2018 have no segment row and drop on the FS control - the same gap
recorded above for the two earliest cross-cycle events. Cross-sections are **97 firms at 2018-01
and 106 at 2018-02**, against 240 by March, 402 by May and 685-721 through 2025. Those two months
are also selected toward early filers, which is a composition caveat and not only a precision one.
The report gives `lambda_1_bar` with and without them; nothing is imputed.

### How to read the outputs

`lambda_1_bar = +0.046%` per month per standard deviation of exposure, Newey-West 6-lag t = 0.95,
p = 0.344 - **weak and insignificant, which is what the design predicted in advance and must not be
written up as a refutation of H1**. H5: the slope moves from +0.065% to +0.046% when FS and the
industry effects enter, keeping its sign, though neither estimate is distinguishable from zero, so
that reads as consistency rather than as a passed test.

The figure is the nominated output of the section, and it does **not** show the pattern the design
hoped for. The largest monthly slopes tilt toward the episodes - 5 of the 10 largest against 2.9
expected by chance, hypergeometric tail 0.124 - but the tilt is not distinguishable from chance, the
single largest month of all (2022-11, +1.69%) falls outside both episodes, and the dispersion of
`lambda_1,t` inside the episodes is barely above the dispersion outside. The episodic reading
therefore rests mainly on the section 7.2 event-window coefficients and their distance from this
mean, with the monthly series offering weak corroboration rather than independent support. Per
section 5.2 item 6, subsample noise cannot be separated from a genuine absence of unconditional
pricing, and this null must not be presented as evidence of no effect.

### §7.5 EPU regime conditioning (H3)

Sections 10-12 of the same validation report, produced by the same run. Not a second estimation: a
post-hoc classification of the 96 monthly lambda_1,t values, so no cross-section is re-fitted and
`fm_lambda_panel.csv` is byte-identical either side of the addition.

| Path | Contents |
| --- | --- |
| `output/fm_epu_regime_results.csv` | 12 rows - 3 tau x 2 regimes, plus 3 episode splits x 2. One row per (split, tau, regime), with the pair's test statistics on both rows so a single row is self-describing |
| `output/fama_macbeth_validation_report.txt` | gains sections 10 (EPU split), 11 (episode split beside it), 12 (bucket composition); the old 10 and 11 renumbered 13 and 14 |

**tau.** The percentile of the EPU series over **2017-01 to 2026-05** (113 months - the series ends
May 2026), per §7.5: a property of the uncertainty environment, not of this test's sample window.
`numpy.percentile` with its default linear interpolation, named because another quantile convention
moves tau and with it the borderline months. p50 = 175.87, p75 = 252.59 (§7.5's primary),
p90 = 371.25, giving 48/48, 22/74 and 9/87 high/low months across the 96 estimated. Computed on the
96 estimated months instead, tau would be 176.04 / 231.13 / 350.11 - reported as a contrast, not
used. Fixed before estimation; nothing searches over thresholds.

**The lag was applied upstream.** `build_fm_panel` attached EPU of *t-1* to month *t*, so §7.5
asserts that the panel's `epu_lag` equals a fresh one-month lag of the series rather than
re-deriving it - a second implementation here would be one more thing able to disagree with the
panel it classifies. No boundary loss: the first sample month, 2018-01, takes 2017-12.

**One function, two conditioning variables.** `regime_split` is generic over any boolean indicator,
so the three EPU splits and the three episode splits carry identical statistics from identical code.
That is what makes §7.5's "the episode-based split should be sharper than the EPU-based one" a
comparison rather than two tables side by side. Each per-episode split compares that episode's
months against the months outside **both**, so 2018-19 is not contaminated by 2025 or the reverse.

**Two difference tests, deliberately.** **Welch** is the nominated test and the headline. It treats
the monthly slopes within a regime as independent draws, which they are not - which is why each
regime mean carries a Newey-West error - so a **HAC difference** is reported beside it:
`lambda_t = a + b*1{high} + e` with NW 6 lags, the same correction the levels carry. The point
estimates are identical by construction (a dummy regression's slope *is* the difference in means,
asserted); only the errors differ.

**Small-regime caveat.** `MIN_REGIME_MONTHS_FOR_HAC = 4 x NW_LAGS = 24`, set from the lag length and
not from which regimes it catches. Below it a Newey-West error at 6 lags is unreliable: the p90 high
bucket holds 9 months and returns se = 0.00134 against an iid 0.00247, and a HAC SE *below* the iid
SE is a finite-sample artefact. statsmodels raises nothing, so the flag is the only thing that
surfaces it. Four of the twelve rows carry it (p75 high 22 months, p90 high 9, 2018-19 23, 2025 5);
none is suppressed, and the iid error is printed alongside.

#### How to read the outputs

**H3 is not supported at the nominated p75 threshold.** High-EPU lambda_bar = +0.169% per s.d.
against +0.010% low; difference +0.159%, Welch p = 0.305, HAC p = 0.234. One of H3's limbs holds
(|lambda_bar| is larger in the high regime), the other does not.

**The thresholds run backwards.** The difference is +0.224% at p50 (HAC p = 0.021), +0.159% at p75
(p = 0.234), +0.020% at p90 (p = 0.876) - falling monotonically as the threshold rises, where a
premium tracking uncertainty intensity would rise. Either only the median split has both buckets
large enough to detect anything (a power pattern, not a mechanism), or the higher buckets are
diluted by their content; every sub-bucket the second reading rests on is below the HAC floor, so
the panel cannot settle it. **The median split is significant and must not become the headline** -
promoting it over the pre-nominated p75 would be choosing the threshold on the result, which is
exactly what fixing tau in advance prevents. Report it as the robustness line §7.5 asked for.

**The high-EPU bucket is not a tariff bucket** - the §7.5-mandated disclosure. At p75 its 22 months
are 8 from COVID-2020, 2 from early 2021, 10 from 2025, and **2 from the whole 2018-19 trade war**;
at p90 the trade war contributes **none**. News-based EPU in 2018-19 was not extreme by 2017-2026
standards, dwarfed by COVID and 2025. A premium here is therefore about elevated policy uncertainty
in general, not about tariff salience.

**It is not a COVID artefact.** Dropping the 2020 months makes the high-regime mean *larger* at
every threshold - p50 0.159% to 0.195%, p75 0.169% to 0.322%, p90 0.065% to 0.157% - so on the point
estimates COVID months were diluting the premium rather than producing it. The accompanying
t-statistics are not evidence: those buckets hold 37, 14 and 5 months, two below the floor.

**The episode split is the weaker conditioning variable, not the sharper one**, contrary to §7.5's
expectation. Every EPU split above p90 separates the slopes more sharply, and no episode split
approaches significance (binary p = 0.474, 2018-19 p = 0.310, 2025 p = 0.513). Each span runs from
its cycle's first event to its last, so it holds both the tightening and loosening legs and its mean
nets a predicted-negative month against a predicted-positive one; and both episodes sit under the
HAC floor at 23 and 5 months. Its null is not evidence that tariff salience does not matter - §7.2,
where the legs are separated and the window is days rather than months, is where that is answered.
