# Dissertation — Is Tariff Exposure Priced?

Text-based tariff exposure from 10-K filings, tested against the 2025 impose-and-reverse
sequence and the 2018–19 Section 301 cycle. Methodology lives in
`Tariff_Factor_Research_Design_v5.md` and `Research_Design_v6_revised_sections.md` (v6 revises
v5's §0, §1, §2.1, §2.3, §3.2, §3.3, §4, §5.2 items 1+6 and §7.0–§7.6; v5 governs elsewhere).
This file is the reproducibility document: what to install, how to run it, where things land,
and every place the code departs from the design.

> ### The EDGAR filing pull is not to be re-run
> `edgar_pull.py` and the 13,975 cached filings in `filings_raw/` are complete and frozen. The
> pull spans 17 reference dates and ~31,000 successful request rows against SEC's rate limit;
> re-running it costs hours, and nothing downstream needs it to. Every stage below reads the
> cached corpus. `edgar_pull.py` is also the one module deliberately left untouched by the
> 2026-09-01 audit — see "Audit, 2026-09-01" at the end.

## Requirements

- **Python ≥ 3.10** (PEP 604 `X | None` appears in runtime-evaluated annotations). Results
  reported here were produced on **3.12.10**.
- `pip install -r requirements.txt` — pandas, numpy, scipy, statsmodels, matplotlib, nltk,
  beautifulsoup4, lxml, requests, openpyxl. Exact pinned versions are in the file.
- **NLTK sentence data, once:** `python -c "import nltk; nltk.download('punkt_tab')"`.
  `score_filings.ensure_punkt` will fetch it on demand and raises a clear error if the network
  is unavailable; this is the pipeline's only runtime download.
- **No database access is required.** Every WRDS extract is a static CSV exported by hand and
  read from the repository root. No script authenticates against WRDS.

### Credentials

Copy `.env.example` to `.env` and set `EDGAR_USER_AGENT` to `AppName ContactEmail`, as SEC fair
access requires. `.env` is gitignored. It is needed **only** if you deliberately extend the
filing corpus; no analytical stage reads it. The two `WRDS_*` lines in the template are
commented out and unused.

## Required inputs

These are licensed or public extracts, all gitignored, all read from the repository root. A
fresh clone must supply them; the pipeline raises a named `FileNotFoundError` for each.

| File | Source | Used by |
| --- | --- | --- |
| `Monthly Returns.csv` | CRSP monthly (WRDS) | `clean_data`, `clean_controls_data`, `build_fm_panel` |
| `Daily returns.csv` | CRSP daily (WRDS) | `clean_controls_data --cycle 2025` |
| `CRSP Daily returns cross cycle.csv` + `CRSp Daily returns cross cycle pt2.csv` | CRSP daily (WRDS) | `clean_controls_data --cycle cross_cycle` |
| `PERMNO - GVKEY - CIK.csv` | CCM link table (WRDS) | `clean_data` |
| `BM and Lev.csv` | Compustat fundamentals (WRDS) | `clean_controls_data`, `build_fm_panel` |
| `Compustat Geographic segment data.csv` | Compustat segments (WRDS) | `foreign_sales` |
| `FF5_MOM_Factors.csv` | Ken French monthly | `clean_data` |
| `Fama French daily.csv` | Ken French daily | `estimate_car --cycle 2025` |
| `FF5+MOM daily Cross cycle.csv` + `FF5 + MOM daily cross cycle pt2.csv` | Ken French daily | `estimate_car --cycle cross_cycle` |
| `ME_Breakpoints.csv` | Ken French NYSE size breakpoints (**public**) | `clean_data` (p10 micro-cap), `decile_sort` (p90 mega-cap) |
| `Siccodes12.txt` | Ken French FF12 definitions (**public**) | `clean_controls_data`, `build_fm_panel` |
| `US_Policy_Uncertainty_Data.xlsx` | policyuncertainty.com (**public**) | `clean_data` |
| `bigram_list.json` | this project's tariff lexicon (**tracked in git**) | `score_filings` |

The four public files are caught by `.gitignore`'s blanket `*.csv` / `*.txt` / `*.xlsx` rules,
so a fresh clone gets neither them nor the licensed data. Only `bigram_list.json` — the measure
itself — is version-controlled, which is deliberate: it is the one input that is this project's
own work rather than a third-party extract.

## Running the pipeline

Every stage is idempotent: it checks for its own outputs and skips, printing what it found.
Pass `--force` to rebuild (`--redraw` / `--rebuild` for the two fixed firm lists). So the block
below is safe to paste in full — on a warm tree it verifies rather than recomputes.

```bash
python clean_data.py                                #  ~5 s   cleaned CRSP/CCM/factors/EPU
python clean_filings.py                             #  ~1 min re-validates the cached corpus
python score_filings.py                             #  ~1 s   re-validates existing scores
python build_texp_panel.py                          #  ~1 s   TExp panel, z within reference date
python foreign_sales.py                             # ~20 s   FS control + §7.1 validation
python clean_controls_data.py --cycle 2025           # ~25 s   357 MB controls panel
python estimate_car.py        --cycle 2025           # ~30 s   FF5+MOM loadings, ARs, CARs
python run_car_regression.py  --cycle 2025           # ~10 s   §7.2 + H1 test + FF12 split
python clean_controls_data.py --cycle cross_cycle    # ~35 s   722 MB controls panel
python estimate_car.py        --cycle cross_cycle    # ~60 s
python run_car_regression.py  --cycle cross_cycle    # ~15 s   §7.6 + H4 + H1 test
python decile_sort.py                               # ~20 s   §7.3, both universes
python build_fm_panel.py                            #  ~2 min monthly panel
python fama_macbeth_pricing.py                      #  ~3 s   §7.4 + §7.5
python figures_event_study.py                       #  ~6 s   4 figures + 1 table for §7.2, §7.6, §5.2
python figures_panel.py                             #  ~2 s   3 figures for §7.4, §7.5
python figures_collinearity.py                      #  ~6 s   3 correlation grids + 1 scatter grid
python build_results_workbook.py                    #  ~2 s   merges every results table
```

Cold from raw inputs, `clean_filings.py` takes **~105 minutes** (13,972 documents) and
`score_filings.py` **~9 minutes**. Both are fully resumable — they append and flush per
document and lose at most one on an interruption — so an interrupted run is restarted by
re-issuing the same command. `--status` on either reports progress and writes nothing.

**Order constraints, not stylistic preferences:**

1. **`--cycle 2025` must complete through `run_car_regression.py` before any `cross_cycle`
   stage.** `run_car_regression.py --cycle cross_cycle` re-estimates the 2025 cycle in-process
   to build the H4 pooled test's baseline samples, so the 2025 controls panel and the three 2025
   CAR tables must be on disk.
2. **`decile_sort.py` and `fama_macbeth_pricing.py` have no `--cycle` flag.** Both are 2025-only
   by construction and read the unsuffixed artifacts.
3. **The three `figures_*.py` scripts run after every analytical stage**, since each reads the
   persisted results. `figures_event_study.py` additionally reconciles against
   `output/decile_sort_results.csv` and against the three 2025 CAR tables, and raises if either
   disagrees, so `decile_sort.py` and `estimate_car.py --cycle 2025` must have completed.
4. **`build_results_workbook.py` runs last**, since it reads every results CSV. It reads no figure
   and is unaffected by them.

### The two fixed firm lists, and the bootstrap they sit in

`persist_2025_universe.py` (2,140 firms) and `sample_full_panel_firms.py` (the 1,000-firm draw,
`SEED = 20250402`) are **written once and reused unchanged**. Both now refuse to overwrite
without an explicit flag, and **neither is in the normal run order** — they are already built.

This matters because of a genuine circularity the code does not otherwise state.
`estimate_car.py --cycle 2025` rewrites `intermediate/car_imposition_primary.csv`, which is the
sole input to `persist_2025_universe.py`; and the 1,000-firm sample is drawn as *indices into the
universe's enumeration order*. So re-running `persist_2025_universe.py` after a CRSP refresh
would silently rewrite the pool, and the recorded seed would then select a **different** 1,000
firms from a pool that no longer matches the `N=2140` in the sample file's own header. Nothing
downstream detects that. Hence the guards. If you ever do rebuild the universe, re-draw the
sample immediately (`--rebuild` then `--redraw`) and rebuild `intermediate/fm_panel.csv`.

Full cold order, including the pull (**for reference only — do not run**):

```
clean_data.py
  → edgar_pull.py --scope full  (2025-04-02 bootstrap)
  → clean_controls_data.py --cycle 2025 → estimate_car.py --cycle 2025
  → persist_2025_universe.py → sample_full_panel_firms.py
  → edgar_pull.py --batch cross_cycle --scope full      (cross_cycle FIRST, per CLAUDE.md)
  → edgar_pull.py --batch full_panel --scope subsample
  → clean_filings.py → score_filings.py → build_texp_panel.py → foreign_sales.py
  → the analytical stages, in the order given above
```

## Where things land

| Directory | Contents | Tracked? |
| --- | --- | --- |
| `output/` | **Final results only**: results CSVs, the merged workbook, 14 validation reports, 16 figures, and the two fixed firm lists | no |
| `intermediate/` | Derived analytical panels and per-row audit tables: `controls_panel{,_cross_cycle}.csv`, 12 per-firm `car_*.csv`, `fm_panel.csv`, `scoring_diagnostics.csv`, `texp_panel_diagnostics.csv` | no |
| `clean_data/` | Cleaned source data: returns, bridge, factors, EPU, filings metadata, TExp scores and panel, FS | no |
| `clean_text/` | 13,972 gzipped `{item_1a, rest}` documents — the cleaned corpus | no |
| `filings_raw/`, `submissions_cache/` | Raw SEC HTML and cached submissions payloads. **Read-only.** | no |
| `logs/` | Launcher logs from long unattended runs. Written by the shell, read by no script. | no |

`output/results_workbook.xlsx` is the single human-facing deliverable: one sheet per reported
table plus a contents sheet naming each table's design section and source CSV. The source CSVs
**remain** beside it — they are the machine-readable form, diffing them is how every numerical
change in this project is verified, and `car_regression_results.csv` is a genuine *input* to
`fama_macbeth_pricing.py` (report §8 converts §7.2 coefficients into comparable units).

The two `controls_panel` files are excluded from the workbook at 1.1M and 2.2M rows — past
Excel's 1,048,576-row limit — as are the per-firm CAR tables and the diagnostics.

### Reports

Every stage writes a persisted report to `output/`. The six that assemble one deliberately
(`clean_controls_data`, `estimate_car`, `run_car_regression`, `decile_sort`, `foreign_sales`,
`fama_macbeth_pricing`) buffer into a `_REPORT` list. The seven that build their output with
plain `print` are teed to disk by `run_report.capture`, so the console text and the file are
identical — see `run_report.py` for why that route was taken rather than rewriting ~150 call
sites.

### Figures

Sixteen PNGs at 300 dpi and one fixed-width text table, all drawn from persisted pipeline output.
They are formatted against one worked example — R base graphics: white ground, sans-serif, bold
centred title, no gridlines, tick labels rotated ninety degrees, boxed legend. The palette is a
muted set — a desaturated dark blue, a mid grey and a desaturated dark green — with a solid black
reference line at zero; the correlation grids' warm end is a muted red. `chartstyle.py` is the
single definition of that furniture and `palette.py` of its colours; no drawing module restates
either.

Two details depart from the worked example, both because it misread on this project's data. The
left and bottom axis lines span the full limits and therefore **meet at the origin corner**: the
example bounds each spine to its tick range, which on a monthly series starting mid-year left a
gap between the two lines that reads as missing data. And bar outlines are drawn at
`BAR_EDGE_WIDTH = 0.25` rather than at axis weight, thin enough for a hatch to read without
framing every fill in black.

**Every figure carries a title, axis labels with units, data labels and a legend, and nothing
else.** Descriptive captions belong above the figure in the dissertation document, not inside the
image, so there is deliberately no subtitle or footnote helper. Where a caption previously lived
inside a PNG the information was already in a results CSV or a validation report; nothing was lost.

| Figure | Design section | Written by |
| --- | --- | --- |
| `fig_car_event_time{,_equal_weighted}.png` | §7.2 / H3b — cumulative CAAR by event day, both legs | `figures_event_study` |
| `fig_car_coefficients.png` | §7.2 / H1 — b per leg per window, 95% CI | `figures_event_study` |
| `fig_cross_cycle_coefficients.png` | §7.6 / H4 — b at nine 2018–19 events plus the 2025 pair, as a line per window | `figures_event_study` |
| `table_industry_split.txt` | v6 §5.2 item 1 — b within each FF12 group, coefficient and s.e. per window | `figures_event_study` |
| `fig_epu_regime_premium.png` | §7.5 / H3 — the monthly premium with the high-EPU months shaded | `figures_panel` |
| `fig_epu_regime.png` | §7.5 — EPU series, its three thresholds, the high-EPU months | `figures_panel` |
| `fig_fm_cross_section.png` | §7.4 + v6 §5.2 item 6 — firms per month, within-month R² | `figures_panel` |
| `fig_corr_event_study.png`, `fig_corr_fm_panel.png` | v6 §5.2 item 1 — regressor correlations | `figures_collinearity` |
| `fig_corr_factor_loadings.png` | v6 §5.2 item 1 — TExp against the FF5+MOM loadings | `figures_collinearity` |
| `fig_texp_scatter_grid.png` | v6 §5.2 item 1 — TExp against each regressor, five panels | `figures_collinearity` |
| `decile_sort_chart{,_equal_weighted}{,_ex_megacap}.png` | §7.3 — group CAR, four (universe × weighting) | `decile_sort` |
| `texp_fs_scatter.png` | §7.1 — TExp against foreign-sales share | `foreign_sales` |
| `fm_lambda_chart.png` | §7.4 — the monthly premium, episodes marked | `fama_macbeth_pricing` |

**Two reconciliations run before any figure is drawn, and raise rather than warn.** The event-time
trajectory needs abnormal returns day by day, which no file holds — `intermediate/car_*.csv` carries
only the three cumulative CARs. `figures_event_study.py` therefore rebuilds them through
`estimate_car`'s public functions (it does **not** call `main()`, which would rewrite the CAR tables,
and does not edit that module), then asserts (i) the CARs it can re-derive match the persisted tables
to `1e-12`, measured worst case **4.4e-16**, and (ii) its group aggregation reproduces all 180 cells
of `decile_sort_results.csv` to `1e-10`, measured worst case **1.4e-16**. `figures_panel.py` likewise
asserts that the EPU thresholds it shades on match the `tau` column of `fm_epu_regime_results.csv`.

Coefficients are drawn in percentage points of CAR per unit of exposure, not per standard deviation:
the cross-cycle sd ranges 0.0048 to 0.0122 across the nine events, so no single scale factor exists
and a per-event one would make the panels incomparable in a figure whose subject is comparability.

Assessed and deliberately **not** produced, with reasons, in `build_notes.md` Step 9: a long-horizon
return series (v6 retires the portfolio programme and §7.3 bars performance exhibits), a bar chart of
λ̄₁, a lexicon term-frequency chart, a per-regime firm count, a CAR-against-TExp scatter, and pipeline
diagnostics already covered by the validation reports. Two exhibits were **withdrawn** on 2026-09-02:
`fig_epu_regime_coefficients.png`, the six-split regime bar chart, whose H3 content
`fig_epu_regime_premium.png` now carries month by month; and `fig_industry_split.png`, replaced by
`table_industry_split.txt` because thirty-six estimates with intervals several times the width of
their point say only that nothing is precisely estimated inside an industry, which a table of
coefficients and standard errors states checkably.

#### Visual revision, dated 2026-09-02

A presentation pass on the whole set. No estimate, sample or standard error changed: re-running
`foreign_sales.py --force`, `decile_sort.py` and `fama_macbeth_pricing.py` left all of their
results CSVs and validation reports `cmp`-identical, and `figures_event_study.py`'s two
reconciliations still pass at 4.4e-16 and 1.4e-16. What changed:

| Change | Reason |
| --- | --- |
| Palette moved from R's saturated primaries to a muted dark blue / mid grey / dark green, and `ZERO` (red, dashed) was retired in favour of a solid black line at zero drawn in `INK` | the primaries print harshly and, at line weight, read as brighter than the data they carry; a dashed red rule at zero was read as a second data series rather than as the axis |
| `chartstyle.frame` spans both spines to the axis limits instead of bounding them to the tick range | the tick-range form left a gap at the origin corner on any series whose first observation falls before the first tick, which reads as missing data |
| Bar outlines dropped from `AXIS_WIDTH` to `BAR_EDGE_WIDTH = 0.25` | thin enough for the decile chart's hatch to read, not thick enough to frame every fill in black |
| `chartstyle.figure_legend` added, and the two event-time panels and both regime charts use it | a panel-level legend covered data in whichever corner it was placed; centred under the title it belongs to both panels and hides neither |
| `chartstyle.value_labels` added, and `fig_car_coefficients.png` labels each bar with its estimate and stars | four estimates quoted in the text, so the number belongs on the bar; `star_labels` stays for the eleven-event figure, read for shape |
| `fig_cross_cycle_coefficients.png` redrawn as a line with a marker per event, loosening events shaded rather than given a second colour | eleven bars per panel × three panels did not let the coefficient's path across the two cycles be followed; the predicted sign is a property of the event and holds across all three panels |
| `fig_epu_regime.png`'s y scale pinned to zero, and its shading moved onto `high_epu_spans` | autoscaling put the floor near 60, leaving the line running off the bottom of the frame; the per-point fill drew an isolated high-EPU month too narrow to see, and there are three of them. The helper is shared with the premium chart, so the two figures cannot disagree about which months the split called elevated |
| `fig_epu_regime_premium.png` added; `fig_epu_regime_coefficients.png` withdrawn | §7.5's headline is the monthly premium against the high-uncertainty months. The bar chart put the six regime means side by side but hid that the elevated bucket is a handful of clustered months rather than a recurring condition; the means and their Newey-West tests remain in `fm_epu_regime_results.csv` |
| `table_industry_split.txt` added; `fig_industry_split.png` withdrawn | see above |
| `fig_corr_factor_loadings.png` added | asks the collinearity question of the risk model rather than of the controls: whether TExp repackages the FF5+MOM loadings the abnormal returns are already purged of. Free to draw — `estimate_car` persists each firm's six loadings in the CAR table, so no beta is re-estimated |
| `fig_texp_scatter_grid.png` added | a correlation cell reports one number per pair and cannot distinguish a linear relation from a mass point plus a tail, which is the shape TExp has |

## Configuration a re-runner would change

All of these are declared in a marked `Configuration` block at the top of their module.

| Constant | Module | Meaning |
| --- | --- | --- |
| `SEED = 20250402` | `sample_full_panel_firms` | the §7.0 draw; also recorded in the output file's `#` header |
| `COV_TYPE = "HC1"` | `run_car_regression` | §7.2 standard errors (see the SE note below) |
| `STABILITY_COV_TYPE`, `MIN_EVENT_CLUSTERS = 30` | `run_car_regression` | pooled-test clustering and the floor below which the event dimension is dropped |
| `MIN_INDUSTRY_N = 30` | `run_car_regression` | size floor for a within-FF12-group fit |
| `EVENT_WINDOWS = [(-1,1), (-5,5), (-10,10)]` | `estimate_car` | the three CAR windows |
| `CYCLES` | `clean_controls_data` | **the single definition of a policy cycle** — dates, inputs, expected signs, estimation-window rule, output suffix |
| `NW_LAGS = 6`, `HAC_KWDS` | `fama_macbeth_pricing` | Newey-West lag and its finite-sample switches |
| `TAU_PERCENTILES`, `PRIMARY_TAU = 75` | `fama_macbeth_pricing` | §7.5 EPU thresholds |
| `MIN_FIRMS_PER_MONTH = 50` | `fama_macbeth_pricing` | degrees-of-freedom floor; binds on no month |
| `MAX_PERIOD_STALENESS_MONTHS = 15` | `build_texp_panel` | contemporaneity rule, per reference date |
| `CARRY_MONTHS = 12`, `SAMPLE_START/END` | `build_fm_panel` | exposure carry-forward bound and the panel window |
| `REFERENCE_WINDOW`, `RHO_STRONG`, `MONOTONE_MAX_FLIPS` | `decile_sort` | weights for the size diagnostics, and the monotonicity verdict thresholds |
| `SURFACE`, `CATEGORICAL_1..3`, `DIVERGE_WARM`, `INK`, `GRID` | `palette` | **the single definition of every chart colour** — six modules build role names from it |
| `DPI`, `SIZE_*`, `CI_Z`, `AXIS_WIDTH`, `BAR_EDGE_WIDTH`, `ROTATE_X_MIN_CHARS` | `chartstyle` | **the single definition of the chart furniture** — resolution, figure sizes, the 95% interval every whisker draws, the bar outline weight, and the axis treatment |

---

# Per-stage documentation

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
pull. `load_texp_reasons` is sliced the same way, against `intermediate/texp_panel_diagnostics.csv` and
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
   carry over. All 27 per-event cross-cycle regressions use the same `COV_TYPE` as the section 7.2
   run, which since the 2026-09-01 audit is `HC1` - the White heteroskedasticity-robust errors
   section 7.2 nominates. The pooled test's clustering was also revisited; see the audit section.

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
| `intermediate/fm_panel.csv` | 86,009 candidate firm-months x 31 cols; `exclusion_reason` empty on the 53,745 that estimate |
| `output/fm_lambda_panel.csv` | 192 rows - monthly lambda_1, its within-month se/t/p, lambda_FS, firm count, R2, lagged EPU, episode label, per specification |
| `output/fm_lambda_chart.png` | the section 7.4 figure: lambda_1,t over 96 months with both tariff episodes shaded |
| `output/fm_headline_results.csv` | lambda_1_bar with its Newey-West inference - the H2 statistic - per specification and across the NW lag grid. Added by the 2026-09-01 audit; it previously reached no file |
| `output/fm_panel_validation_report.txt` | the panel stage's own record, teed to disk by `run_report` |
| `output/fama_macbeth_validation_report.txt` | fourteen sections, house style |

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

---

## Audit, 2026-09-01

A full-folder review of methodology conformance, statistical correctness, repo hygiene and
publication safety. `edgar_pull.py` and `filings_raw/` were held read-only throughout and are
unmodified. Every numerical change below was applied one at a time and diffed against a captured
baseline, so each movement is attributable to a single cause.

### Standard errors: HC1, and a conflict inside the design

`run_car_regression.COV_TYPE` was `"nonrobust"`, with a comment conceding that §7.2 nominates
White HC. It is now **`HC1`**.

Both v5 §7.2 and v6 §7.2 say "White heteroskedasticity-robust". But **v5 §8 — a section v6 does
not revise — instead says event-study regressions are "double-clustered by firm and industry".**
The design therefore nominates two mutually exclusive estimators for the same regression. §7.2
governs, on two grounds: it is the specific, revised instruction for exactly this regression, and
clustering is inapt on a single cross-section, where each firm appears once (so firm-clustering
reduces to HC) and the 12 FF12 groups are already absorbed as fixed effects (far too few
clusters). The conflict is recorded rather than resolved silently.

Effect: `coef`, `n`, `r2` and `adj_r2` are **identical** on all 639 + 1,917 result rows — OLS
point estimates do not depend on the covariance — while `se`, `t`, `p` and `stars` move. One star
changes (`imposition_robustness` at `[-1,+1]`, p 0.1033 → 0.0980). The headline coefficients are
essentially unmoved, which is worth reporting: the result is robust to the estimator the design
actually specifies. This deliberately supersedes CLAUDE.md's byte-identity invariant for
`--cycle 2025`, replaced by the narrower and more informative *coef/n/r² identical, se/t/p
changed*.

### H1 now has a test statistic

The sign flip — which v6 §3.3 calls "the primary causal evidence and the core contribution" —
was evaluated by comparing two signs. `signflip_matrix` printed `b` per leg with a Y/N verdict,
and the report declared `sign flip HOLDS` from `np.sign` alone; at `[-5,+5]` it did so on
coefficients with p = 0.807 and p = 0.349. No p-value on the *difference between legs* existed in
any output file. Worse, `estimate_car.py` wrote into a published report that "the joint sign-flip
test in Script 3 runs on the intersection" — a test that did not exist, on a sample that was not
used.

**`signflip_tests` is an addition to both design documents** — each states H1 as a pair of
predictions on two separately estimated coefficients and nominates no test of their difference.
Nothing was removed to make room for it. Output: `output/signflip_test_results{,_cross_cycle}.csv`
and report §6b. Section 6's wording is now "signs ALL MATCH", explicitly carrying no inference,
pointing at §6b.

Result (2025 cycle, δ = b_reversal − b_imposition, H1 predicts δ > 0):

| window | b_impose | b_reverse | δ | p (2-sided) | p (1-sided) |
| --- | --- | --- | --- | --- | --- |
| `[-1,+1]` | −0.1881 | −0.0543 | +0.1339 | 0.298 | 0.149 |
| `[-5,+5]` | −0.0411 | +0.1731 | +0.2142 | 0.369 | 0.184 |
| `[-10,+10]` | −0.4930 | +0.2964 | **+0.7894** | **0.032** | 0.016 |

δ is correctly signed at every window and significant at 5% at the widest — and the trajectory
0.298 → 0.369 → 0.032 is the H3b digestion prediction, now with inference attached rather than
eyeballed. The 2018–19 cycle's own legs give δ = +0.173 (p = 0.056), +0.252, +0.338: correctly
signed at all three, out-of-sample.

### H4: the pooled test no longer re-estimates b^2025

The old pooled regression constrained FS, all controls and all FF12 dummies equal across cycles,
interacting only TExp. The consequence, measured: **the quantity labelled `b^2025` took eight
values from −0.119 to −0.235 against §7.2's −0.188**, and on the loosening leg it was −0.146
(p = 0.021) where §7.2 reports −0.054 (p = 0.495) — the restriction manufactured a significant,
wrong-signed reversal coefficient out of a null, and *that* value drove the reported "the two
cycles genuinely differ at `[-1,+1]`" finding.

The design is now **block diagonal**: every regressor, the constant and the industry dummies
included, enters once per group and nothing is constrained across them. The blocks are orthogonal
by construction, so each group's `b` is exactly its own per-event estimate — asserted at
`POOLED_COEF_TOL = 1e-9`, measured worst case 2e-15 — and the stack exists only to supply the
joint covariance δ's standard error needs. The Wald test on δ is valid either way; only this
version leaves the reported `b`s reconcilable with §7.2.

| leg | window | δ (old) | p (old) | δ (new) | p (new) |
| --- | --- | --- | --- | --- | --- |
| tightening | `[-1,+1]` | +0.100 | 0.316 | +0.117 | 0.313 |
| tightening | `[-5,+5]` | +0.022 | 0.894 | −0.124 | 0.492 |
| tightening | `[-10,+10]` | +0.234 | 0.333 | +0.191 | 0.489 |
| loosening | `[-1,+1]` | +0.379 | **<0.001** | +0.173 | 0.068 |
| loosening | `[-5,+5]` | +0.115 | 0.543 | +0.035 | 0.876 |
| loosening | `[-10,+10]` | +0.012 | 0.964 | −0.039 | 0.899 |

H₀: δ = 0 is now rejected nowhere at 5%. That is a **stronger** H4 result — directional
consistency with no detectable difference between cycles — and the previous headline
"the cycles differ" was substantially an artefact of the restriction.

A per-standard-deviation column was added alongside the raw coefficients. Raw TExp's
cross-sectional sd ranges 0.00478 (2018-04) to 0.01215 (2025-04), a 2.5× spread, so a raw
equality test is not scale-invariant; the per-sd column makes the comparison legible without
re-estimating anything.

### Two-way clustering: added, then rejected on evidence

Event date was added as a second clustering dimension for the pooled tests, since every firm in one
cross-section shares that day's common shock and permno clustering leaves that uncorrected. It was
then **withdrawn as primary**, because the fitted two-way error came back at roughly a **quarter**
of the permno-only error at eight event dates. Two-way clustering adds a covariance component and
should widen an interval; a four-fold narrowing is the Cameron-Gelbach-Miller estimator failing on
too few clusters. Reporting it would have turned an insignificant cross-cycle difference into
p < 0.001 on a standard error known to be wrong by a factor of four.

`MIN_EVENT_CLUSTERS = 30` is the conventional floor and is met nowhere in this project (nine event
dates is the widest stack). Both errors are reported on every row — `se_permno`, `se_twoway`, with
`cov_type` naming the primary — so the collapse is visible. **Cross-sectional dependence within an
event date is an acknowledged, uncorrected limitation of the pooled tests.**

### Shanken correction: not applicable, and its absence is correct

v5 §8 says "FM regressions: Shanken correction mandatory", but v5 §7.4 scopes that to **Version B**
("mandatory for Version B: it adjusts for the errors-in-variables bias introduced by using
estimated (rather than true) betas in Stage 2") and says of Version A that "no first-pass beta
estimation is required — TExp is a directly observable characteristic". v6 §7.4 drops Version B.

Traced in code: `estimate_car.estimate_betas` produces FF5+MOM loadings consumed **only** by
`compute_ar` to form residuals. They are written to the CAR tables as diagnostics and never appear
in `design_matrix`'s regressor list. `fama_macbeth_pricing`'s regressor is `texp_z`, a
text-derived score. **No two-pass errors-in-variables setup exists anywhere in the pipeline**, so
Shanken corrects a bias that is not present. Adding it would be wrong, not conservative.

### v6 §5.2 item 1: within-industry coefficients

v6 calls this "the most important robustness check in the paper — run it before anything else" and
it was not implemented. `industry_split` now estimates `b` separately inside each FF12 group, on
the 2025 cycle, dropping the industry dummies (each sample is one industry by construction — which
is why `design_matrix` gained `include_industry`, since the reference-category check fires on
eleven of twelve otherwise). Groups below `MIN_INDUSTRY_N = 30` are reported as skipped, never
pooled to rescue. Output: `output/industry_split_results.csv`, report §7b.

**The result matters: 53 of 99 estimated within-group fits carry the predicted sign** — close to a
coin flip. The imposition `[-1,+1]` effect is carried by BusEq (−0.51\*) and Enrgy (−1.51\*), with
Chems significantly the *wrong* way (+0.65\*). So the effect is **concentrated, not diffuse across
sectors**, which per v6 "is still a result, but it is a different result and should not be
presented as a general one". Read it as a concentration diagnostic, not twelve independent tests:
these are subsamples of one cross-section and no multiple-testing adjustment is applied.

### §7.5: Newey-West on scattered months

Each regime's mean and NW error came from applying the HAC estimator to `frame.loc[indicator]` and
`frame.loc[~indicator]` separately. Those subsets are scattered across the calendar — the p75
high-EPU bucket holds eight months of 2020, two of early 2021, two from the whole 2018–19 trade war
and ten from 2025 — and a Bartlett kernel reads row adjacency as month adjacency, so observations
years apart were getting consecutive-month weights.

`regime_fit` now takes everything from **one HAC fit on the un-split contiguous series**:
`λ_t = a + b·1{high}`, with `a` the low mean, `a + b` the high mean and `b` the difference, each
error read off the joint covariance by `t_test`. Every regime mean is unchanged to machine
precision (a dummy regression's fitted levels *are* the group means, asserted); only the errors
move, by 0.95× to 1.09×. The headline p75 high-EPU error widens 7% and its p goes 0.102 → 0.126.
`fm_lambda_panel.csv` is **byte-identical** — the monthly slopes were never involved.

The two per-episode splits still carry calendar gaps by construction (each excludes the other
episode's months) and are now flagged per row via `series_contiguous`, with the caveat printed in
their own report blocks.

The HAC finite-sample switches are named in `HAC_KWDS` rather than inherited silently:
`use_correction`, `adjust_df` and `use_t` are all `False`, which are the statsmodels defaults, so
no number changed. They are left there deliberately — at 96 months all three are negligible, and
switching them only for the small buckets would put two conventions in one table. The consequence,
now stated, is that the sub-floor rows' p-values are optimistic on two counts at once: no
small-sample scaling and a normal rather than *t* reference distribution.

### §7.3: value weights were measured inside the window

`decile_sort` weighted on `me_lag` read from the event-date panel row. For `[-1,+1]` that is
marginally pre-window; for `[-5,+5]` and `[-10,+10]` it sat **five and ten trading days inside the
window**, so a firm that fell over the first half of the window was down-weighted in the average
of its own decline. Weights are now read on the trading day each window **opens**
(`weight_dates`), so `me_lag` is the close of the day before the window's first day and no return
inside a window helps set its own weights. One rule serves all three windows, leaving intact the
existing argument for not using the estimation anchor.

Effect is real but small, and every sign survives: imposition spreads −1.55/−1.83/−2.90% →
−1.57/−1.95/−2.94%, reversal −0.33/+1.93/+3.46% → −0.33/+1.93/+3.36%. In universe B the trim now
sees slightly different market equity, so two firms cross the NYSE p90 cutoff differently and the
group bounds shift accordingly.

Also in `decile_sort`: the monotonicity verdict label branched on |ρ| first and consulted the flip
count only inside the top band, so a series with five sign changes read "strong trend" while one
with four — strictly more monotone — read "partial gradient", purely because its ρ fell 0.003 below
a cutoff. `monotonicity_reading` now takes both statistics on every branch, with the thresholds in
config. The Spearman p is explicitly labelled as not a test: the ten group means are weighted
averages of one event-day cross-section, several dominated by one or two firms, and are not ten
independent draws.

### Inference reaching output files

`λ̄₁` with its Newey-West error — the single statistic H2 is about — existed only as report prose
and a chart subtitle. It now has `output/fm_headline_results.csv`, carrying both specifications
and the whole NW lag grid.

The per-standard-deviation conversion used the 2,868-firm vintage sd (0.012149) where §7.2's own
estimation sample sd is ≈0.0127, understating by 4.5%. The vintage figure is retained — it is the
only stable factor across windows and legs — and report §8 now prints both, so the understatement
is quantified rather than hidden.

Per the brief, **no non-standard-error robustness check was added**, and no existing one removed.

### Repo hygiene

`output/` went from **1,048 MB to 4.7 MB** and now holds only final results. Derived panels and
per-row audit tables moved to a new `intermediate/`, added to `.gitignore`. One constant changed
per file — `clean_controls_data.PANEL_OUT` alone repoints both controls panels, since the three
consumers derive their path from it.

Deleted, all verified to have zero references in any `.py` including the out-of-scope pull:

| Target | Reclaimed | Basis |
| --- | --- | --- |
| `clean_data/clean_filings.pre-migration.csv` | 1,135 MB | `migrate_clean_filings.py` states the deletion precondition; verified met — 13,972 text files = 13,972 distinct accessions, 0 partials, live schema carries no `item_1a_text` |
| `output/cleaning_diagnostics.csv` | 2.6 MB | **`cmp`-identical** to `clean_data/clean_filings.csv` — `write_diagnostics` wrote the whole cleaned table, whose schema already *is* the diagnostics schema — and nothing read it |
| `CIKs.txt`, `PERMNOs.txt`, `clean_data/tariff_scores.pre-panel-backup.csv` | 585 KB | zero references anywhere; schemas proven superseded |
| `__pycache__/`, 14 dead `logs/` files | 650 KB | no `.py` touches `logs/`; 5 stale `.pid`, 1 `.logname` pointing at a file that does not exist, 3 empty orphan `.err.log`, 5 pre-timestamp legacy names |
| `edgar_pull_log.pre-migration.csv` | 7 KB | `edgar_pull.migrate_log` returns early once `reference_date` exists on the log — which it does — so `LOG_BACKUP` is written-once and never read, and the migration cannot recur. A 73-row backup of a 49,742-row live file |
| `__pycache__/`, 26 → 12 `logs/` files | 1.0 MB | Build cache (regenerates). Of the launcher logs, the 14 covering the **analytical** stages were removed: each of those stages now writes its own validation report, and the logged figures predate the audit, so keeping them alongside the new reports risks someone quoting a superseded number |

**Deliberately kept in `logs/`:** the four `pull-*` logs and the `clean-*` / `score-*` pair. Those
record operations that are unrepeatable or expensive — the frozen EDGAR pull, and the 105-minute
cleaning and 9-minute scoring runs — and they carry per-document progress detail that the warm
re-validation reports cannot reconstruct. Nothing reads them; they are provenance.

**Deliberately kept elsewhere, with reasons.** `migrate_clean_filings.py` is inert
(`already_migrated()` refuses to run) and functionally dead, but it is the audit trail for how
`clean_text/` came to exist and for the character-count verification this README cites, and it sits
inside the cleaned-filings area that is out of scope for deletion. `clean_data/clean_ff5_mom.csv`
has no consumer, but it is one of the four outputs `clean_data.py`'s skip check tests for — deleting
it would silently force a full rebuild of that stage on the next run — so it stays until the
generator itself is retired.

**Left in place and flagged.** `edgar_pull_log.pre-migration.csv` has no reader, but the literal
appears in `edgar_pull.py`, which was out of scope to read — it cannot be cleared safely.
`output/event_study_firm_universe.csv` and `output/full_panel_firm_sample.csv` stay in `output/`
because `edgar_pull.read_firm_list` reads them and CLAUDE.md pins those paths; they legitimately
qualify as the "reproducibility artifacts" CLAUDE.md permits there.
**`clean_data/clean_ff5_mom.csv` is an orphan output** — written by `clean_data.clean_ff5_mom`,
read by nothing, since `estimate_car` reads the raw daily factor files instead. Flagged rather
than deleted: removing the generator would also remove the documented `rf` handling.

Other fixes: `persist_2025_universe.py` gained the overwrite guard its downstream sample already
had (highest-severity hygiene finding — the file that must be immutable had none while the draw
from it did); `--force` skip-if-exists added to the five whole-file stages that recomputed
unconditionally; the cycle-dependent default-argument trap CLAUDE.md forbids closed at
`decile_sort.load_event_weights` and `persist_2025_universe.load_screened_permnos`;
`load_texp_reasons` now reads its two sources once per process instead of eighteen times per
cross-cycle run; 20 bare `assert`s converted to `raise ValueError` (they vanish under `python -O`,
and the other modules already used raise for the same class of check); three result-dependent
assertions that would have aborted on a legitimate empirical finding converted to reported
findings; `decile_sort`'s report renumbered (it skipped section 10); buried literals promoted to
config, including a hardcoded PERMNO 170 lines into a validation function that drives published
report text.

Duplication that carried a correctness risk was consolidated: `palette.py` is now the single
definition of the chart colours that three modules each held their own copy of under different
role names, so a palette revision is one edit rather than three coordinated ones across four
published figures. All four regenerated **byte-identically**, which was the proof that *that* change
was behaviour-preserving.

> **Superseded, 2026-09-02.** The figures build replaced the palette values and restyled every
> chart, so the four decile PNGs no longer regenerate byte-identically and that particular check no
> longer applies — it was evidence for the consolidation, not a standing invariant. The consolidation
> itself is what made the restyle one edit. The invariant that replaces it is narrower and stronger:
> re-running `foreign_sales.py`, `decile_sort.py` and `fama_macbeth_pricing.py --force` leaves all
> **nine** of their results CSVs and validation reports `cmp`-identical, and only the PNGs move. The
> docstring's measured CVD ΔE was removed rather than restated, since it was true of the old values
> and is not true of the new ones; see `build_notes.md` Step 9.

The audit found **no dead code**: ~150 functions checked by AST load-reference, every one with a
call site; zero commented-out blocks, zero unused imports, zero unused module constants, zero
TODO/FIXME markers.

> **Corrected, 2026-09-02.** The figures build re-ran that sweep and found one exception the audit
> missed: **`decile_sort._listed` has no call site**, and `git show HEAD:decile_sort.py` confirms it
> had none before either. Left in place and flagged rather than removed, since deleting it is a
> hygiene fix unrelated to figures. Every other function, import and module constant in the eight
> files that build touched is still referenced.

### Publication safety

**Nothing sensitive has ever been committed.** `git log --all --diff-filter=A --name-only` over
the full history returns the same paths as `git ls-files`: `.gitignore`, `README.md`,
`bigram_list.json` and the Python sources. No CSV, no `.env`, no `output/` artifact, no design
document has ever entered git, so **no history rewrite is needed** — worth stating plainly in a
data-management declaration. Zero absolute paths in source (every path derives from
`BASE = Path(__file__).resolve().parent`); zero credentials; the only `OneDrive` mentions are
comments about file locking.

Fixed: `.claude/settings.local.json` held a real email address inside a Bash allow-rule (untracked
and gitignored, so never a publication risk) — that rule and a standing pre-approved
`rm -f .env` rule were removed, and `.gitignore` widened from the single filename to `.claude/`
so anything later added to that directory is not tracked by default. `.env.example` added (the
`!.env.example` negation already existed but the file did not). `requirements.txt` added, with
`!requirements.txt` ahead of the blanket `*.txt` rule that would otherwise have swallowed it.

**Flagged, and yours to decide:** all commits are authored with a personal email and there is no
repo-local git identity. That is not fixable by ignoring anything — if the repository is published
and pseudonymity is required, it needs a local identity set before the next commit and a rewrite
of the existing author fields.

### Methodology gaps: flagged, not resolved

Each is a live requirement of a section v6 did **not** revise, or a v6 requirement descoped by
prior instruction. None was silently implemented, and none was silently dropped.

| # | Requirement | Source | Status |
| --- | --- | --- | --- |
| M1 | Harvey-Liu-Zhu adjusted t-thresholds for multiple testing | v5 §8 | Not implemented; zero matches repo-wide. The starred surface is wide — 30 δ-tests, 27×4 per-event coefficients, 99 within-industry fits, 12 regime rows — and in the H4 pairwise block four `**` hits appear against ≈1.5 expected by chance at 5% over 24 tests. Now noted in the regression report's deviations. |
| M2 | 120-day filing-staleness robustness | v5 §7.1 | Not implemented. The code's rule is different in kind: `MAX_PERIOD_STALENESS_MONTHS = 15`, period-to-reference, not filing-to-event. |
| M3 | Anticipation windows `[-10,+1]` and `[0,+1]` | v5 §5.2 item 5 | Not implemented; `EVENT_WINDOWS` is the three symmetric windows v6 §7.2 names. |
| M4 | PolRisk as a §7.6 control | v5 §7.6, **v6 §7.6** | Descoped by instruction. v6 calls it "a genuine strength of the out-of-sample leg". The Hassan data is not in the project, and adding a control absent from the 2025 specification would break the specification identity the out-of-sample claim rests on. Now disclosed in the regression report itself, which previously did not mention it. |
| M5 | 2018-vintage lexicon vs BEA/Census SIC import intensity | v5 §7.1, **v6 §7.6** | Descoped by instruction. **v6 says this "governs interpretation" of H4**, so the 27 cross-cycle regressions are reported without the measurement gate the design places in front of them. Now disclosed in the report. |
| M7 | Placebo events (FOMC, non-farm payrolls) | v5 §8 | Not implemented. |
| M8 | §7.1 sample scope | v6 §7 table says "full universe"; `CLAUDE.md` mandates one 2025 cross-section | Code follows CLAUDE.md, which reads as a deliberate prior override — but CLAUDE.md disclaims methodology authority, so this needs a ruling. Unchanged; the pooled nine-vintage correlation is technically available. |
| M9 | Carry-forward refresh boundary | v6 §4 / §7.4 say "until the next annual filing arrives" | The code refreshes every firm on 1 May regardless of its own filing date, up to an 11-month lag for non-December fiscal year-ends. Documented in `build_fm_panel`'s comments but not previously framed as a deviation. |

**M6 (equal- vs value-weighted deciles) is closed** — see the §7.3 subsection above. The numbering is left as it was so the audit's own references stay valid.

### §7.3: equal- versus value-weighting (closes M6)

v5 §8 nominates "equal-weighting vs. value-weighting" as specification robustness and v6 does not
revise that section, so this was a live requirement rather than an addition. `decile_sort` now
reports both schemes.

**Both are computed on identical firms.** A firm enters a cell only with a non-null CAR and a
strictly positive value weight, whichever scheme is applied, so any difference between the two is
the weighting and not the sample — the same discipline `run_car_regression` uses for its H5 read.
The identity is asserted per cell, not assumed. An unrestricted equal-weighted mean would
additionally admit firms with no usable weight; the count that would add is reported rather than
taken.

Outputs. Each results file keeps one row per (run, window, group, **weighting**) — 180 rows, not 90
— so **a consumer must filter on `weighting`**. Two new columns: `eff_n`, the inverse Herfindahl of
the weights actually applied (equal to `n_entering` exactly under equal weighting), and the
existing `weight_sum`, which is the cell's market equity and therefore identical across schemes.
Four charts, one per (universe, weighting):

| Chart | Universe | Weighting |
| --- | --- | --- |
| `decile_sort_chart.png` | all screened firms | value (the §7.3 primary) |
| `decile_sort_chart_equal_weighted.png` | all screened firms | equal |
| `decile_sort_chart_ex_megacap.png` | ex-NYSE-p90 | value |
| `decile_sort_chart_equal_weighted_ex_megacap.png` | ex-NYSE-p90 | equal |

**The result.** Group 9 − group 0 spread, in percentage points:

| run | window | A value | A equal | B value | B equal |
| --- | --- | --- | --- | --- | --- |
| imposition | `[-1,+1]` | −1.57 | −1.51 | −0.69 | −1.49 |
| imposition | `[-5,+5]` | −1.95 | −0.99 | −0.88 | −0.90 |
| imposition | `[-10,+10]` | −2.94 | −2.10 | −1.27 | −2.03 |
| reversal | `[-1,+1]` | −0.33 | −0.70 | −1.00 | −0.71 |
| reversal | `[-5,+5]` | +1.93 | +1.28 | +0.67 | +1.45 |
| reversal | `[-10,+10]` | +3.36 | +2.82 | +1.87 | +3.14 |

**All nine (run × window) cells agree in sign across all four combinations.** The direction of the
spread — including the flip between legs — is therefore an artefact of neither mega-cap
concentration nor the weighting scheme, which is the strongest form a descriptive exhibit can take.
The known `[-1,+1]` reversal failure is wrong-signed in all four, so nothing here rescues it either.

Two further readings, both computed in report §12 rather than asserted:

- **The trim moves the value-weighted spread 12× further than the equal-weighted one** (mean
  |A − B| of 1.16 pp against 0.10 pp). That is a coherence check, not a coincidence: universe B
  exists to remove weight concentration, and equal weighting has none to remove. The two fixes are
  addressing the same thing and they agree on what the spread is once it is removed.
- **Within universe B the value-weighted spread stays below the equal-weighted one**, so the size
  gradient does not end at the p90 cutoff — among the firms that survive the trim, the larger ones
  still carry a weaker spread.

Value weighting remains the §7.3 primary and the headline figure; equal weighting is the robustness
line. No inference is attached to any of these spreads: §7.3 is descriptive by design and nominates
no test, and H1 is tested in the regression report's §6b.

### Retired methodology: zero live violations

v6 retires the v5 §7.3 portfolio programme. A full sweep for Sharpe, drawdown, turnover,
capacity, GRS, spanning regressions, long-short legs, portfolio return series, rolling TExp betas
and factor-return construction found **no live computation of any of them**. Every keyword hit is
either deliberate exclusion prose — which is the compliance evidence and was kept — or a false
positive (`alpha` = the FF5+MOM market-model intercept; "sharper" from §7.5's own wording;
"spanning" as an English verb). One stale item was corrected: `clean_data` justified retaining the
risk-free rate "for Sharpe and excess-return computation". `rf` *is* still needed — `estimate_car`
forms excess returns as `r − rf` — so only the stated reason was a v5 artifact.

### Universe conformance: correct throughout

`output/full_panel_firm_sample.csv` is read by exactly one module, `build_fm_panel`. §7.2, §7.3
and §7.6 use the full screened universe via `estimate_car.pit_screen`; §7.4 and §7.5 restrict to
the 1,000-firm draw, with an `issubset` raise as a second guard. v6's own warning — "do not apply
the subsample to the event study" — is honoured. TExp vintage conventions match each section:
per-event for §7.2/§7.6, one fixed cross-section for §7.3, carried forward monthly for §7.4.
