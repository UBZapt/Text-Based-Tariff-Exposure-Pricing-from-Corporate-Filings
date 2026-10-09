# Is Tariff Exposure Priced?

Code for an MSFE dissertation. It measures how exposed each US-listed firm is to tariffs from the
text of its annual report, then tests whether that exposure is reflected in returns:

- **Exposure (TExp):** the share of sentences in a 10-K's *Item 1A Risk Factors* section that
  mention a tariff or trade-policy term. It is scored from the filing available before each date
  studied.
- **Event study:** abnormal returns around the April 2025 tariff imposition and the August 2025
  court ruling against it, repeated out of sample on the 2018–19 Section 301 escalations.
- **Pricing tests:** a decile sort of event returns, and monthly Fama–MacBeth regressions over
  2018–2025, split by economic policy uncertainty.

## Repository layout

```
.
├── run_pipeline.py        runs every stage below, in order
├── requirements.txt
├── .env.example           template for the SEC contact header
├── src/
│   ├── config.py          every input and output directory, defined once
│   ├── bigram_list.json   the tariff term list
│   └── *.py               one module per pipeline stage
├── data/                  not tracked: your inputs and everything derived from them
│   ├── raw/               the input files listed below
│   ├── filings/           downloaded 10-Ks, cleaned text, SEC metadata cache
│   ├── clean/             cleaned panels and exposure scores
│   └── intermediate/      analysis panels and per-firm abnormal returns
└── output/                not tracked: results tables, figures and validation reports
```

## Requirements

- Python 3.10 or later. The reported results were produced on Python 3.12.10.
- `pip install -r requirements.txt` (pinned versions).
- NLTK sentence data, once: `python -c "import nltk; nltk.download('punkt_tab')"`.

No database connection is needed. Every input is a file you place in `data/raw/`.

## Data

The data are licensed or third-party and are **not included**. Export the files below into
`data/raw/` under exactly these names. CRSP and Compustat files are WRDS exports; CRSP files use
the CIZ (v2) format. The date ranges are those the reported results used.

| File | Source | Columns the code reads | Range used |
| --- | --- | --- | --- |
| `Monthly Returns.csv` | CRSP monthly stock file | `PERMNO, PrimaryExch, USIncFlg, IssuerType, SecurityType, SecuritySubType, ShareType, SICCD, Ticker, YYYYMM, MthCalDt, MthPrc, MthCap, ShrOut, MthRet` | 2017-01 to 2026-03 |
| `Daily returns.csv` | CRSP daily stock file | `PERMNO, PERMCO, Ticker, DlyCalDt, DlyPrc, DlyRet, ShrOut` | 2020-01 to 2026-03 |
| `CRSP Daily returns cross cycle.csv`, `CRSp Daily returns cross cycle pt2.csv` | CRSP daily stock file, as two exports | as above | 2017-01 to 2020-02 |
| `PERMNO - GVKEY - CIK.csv` | CRSP/Compustat Merged link history, with CIK | `gvkey, cik, LINKPRIM, LINKTYPE, LPERMNO, LINKDT, LINKENDDT` | full history |
| `BM and Lev.csv` | Compustat annual fundamentals | `gvkey, cik, conm, datadate, fyear, at, ceq, dlc, dltt, pstk, pstkl, pstkr, pstkrv, txditc` | FY2016 onward |
| `Compustat Geographic segment data.csv` | Compustat historical segments (geographic) | `gvkey, datadate, srcdate, sid, geotp, sales` (dates day-first) | 2017-01 onward |
| `FF5_MOM_Factors.csv` | Fama–French five factors + momentum, monthly | `dateff, mktrf, smb, hml, rmw, cma, rf, umd` | 2017-01 onward |
| `Fama French daily.csv` | the same factors, daily | `date, mktrf, smb, hml, rmw, cma, rf, umd` | 2024-01 onward |
| `FF5+MOM daily Cross cycle.csv`, `FF5 + MOM daily cross cycle pt2.csv` | the same factors, daily, as two exports | as above | 2017-01 to 2020-03 |
| `ME_Breakpoints.csv` | Kenneth R. French Data Library, NYSE market-equity breakpoints | file as published | 2017 onward |
| `Siccodes12.txt` | Kenneth R. French Data Library, 12-industry SIC definitions | file as published | – |
| `US_Policy_Uncertainty_Data.xlsx` | policyuncertainty.com, US monthly index | sheet `Main News Index`: `Year, Month, News_Based_Policy_Uncert_Index` | 2017 onward |

Where a cycle has two exports, they are merged automatically. Overlapping rows must agree exactly,
or the run stops with an error.

The 10-K filings are downloaded from SEC EDGAR by the pipeline itself.

## SEC access

SEC requires every request to carry a contact. Copy `.env.example` to `.env` and set
`EDGAR_USER_AGENT` to `AppName your@email`. `.env` is ignored by git. It is only needed for the
filing download.

## Running

```bash
python run_pipeline.py --cold      # first run: every stage, including the filing download
python run_pipeline.py             # later runs: everything after the download
python run_pipeline.py --list      # the stages, in order
python run_pipeline.py --from decile_sort
python run_pipeline.py --only figures_panel,workbook
python run_pipeline.py --force     # rebuild outputs that already exist
```

Each stage is a script in `src/` and can also be run on its own, for example
`python src/estimate_car.py --cycle cross_cycle`. Stages skip work whose outputs already exist.

The filing download and the cleaning and scoring of filings are resumable: an interrupted run
continues where it stopped. The download is rate-limited below SEC's fair-access ceiling, so the
first run takes several hours. `python src/edgar_pull.py --status`,
`python src/clean_filings.py --status` and `python src/score_filings.py --status` report
progress without doing any work.

## Stages

Code comments refer to stages by section number (§) and to the hypotheses below by label (H).

| Stage | Script | What it does | § |
| --- | --- | --- | --- |
| `clean_data` | `clean_data.py` | Keeps CRSP common equity priced above $1 and drops NYSE bottom-decile micro-caps. Builds the PERMNO–GVKEY–CIK link table, and cleans the factors and the uncertainty index. | 6 |
| `pull_*` | `edgar_pull.py` | For each firm and date studied, downloads the most recent 10-K filed in the 364 days before that date. | – |
| `universe`, `sample` | `persist_2025_universe.py`, `sample_full_panel_firms.py` | Fixes the 2025 event-study universe (2,140 firms) and draws the 1,000-firm subsample used by the monthly tests. | 7.0 |
| `clean_filings` | `clean_filings.py` | Strips markup and splits Item 1A from the rest of each filing. | 7.1 |
| `score_filings` | `score_filings.py` | Splits sentences and scores TExp against `bigram_list.json`. | 7.1 |
| `texp_panel` | `build_texp_panel.py` | One score per firm per date, standardised within date. | 7.1 |
| `foreign_sales` | `foreign_sales.py` | Foreign-sales share from Compustat segments, and its correlation with TExp. | 7.1 |
| `controls_*`, `car_*` | `clean_controls_data.py`, `estimate_car.py` | Daily panel with size, book-to-market, leverage, momentum and industry; FF5 + momentum abnormal returns and CARs over [−1,+1], [−5,+5] and [−10,+10] days. | 7.2, 7.6 |
| `regression_*` | `run_car_regression.py` | Regresses CAR on TExp with controls and industry effects. Tests the sign flip between legs (H1) and stability across cycles (H4). | 7.2, 7.6 |
| `decile_sort` | `decile_sort.py` | Average CAR by exposure group. Descriptive only. | 7.3 |
| `fm_panel`, `fama_macbeth` | `build_fm_panel.py`, `fama_macbeth_pricing.py` | Monthly Fama–MacBeth regressions of returns on TExp (H2, H5), split by uncertainty regime (H3). | 7.4, 7.5 |
| `figures_*`, `workbook` | `figures_*.py`, `build_results_workbook.py` | Figures and a single Excel workbook of every results table. | – |

The `_2025` stages cover the 2025 events: tariff imposition on 2025-04-02 and the Federal Circuit
ruling on 2025-08-29. The `_cc` stages cover the 2018–19 cycle: seven Section 301 escalations, as
dated by Bruno, Goltz and Luyten (2024, Table 3), plus de-escalations on 2019-10-11 and
2020-01-15. Event dates are defined once, in `CYCLES` in `src/clean_controls_data.py`.

| Label | Prediction |
| --- | --- |
| H1 | Exposed firms fall on the imposition and recover on the reversal (the coefficient flips sign). |
| H2 | Exposure carries an unconditional monthly return premium. |
| H3 | That premium is larger when policy uncertainty is high. H3b: the event reaction builds across wider windows. |
| H4 | The 2025 relation holds in the 2018–19 cycle. |
| H5 | The premium is not explained by foreign sales or industry. |

## Outputs

Everything lands in `output/`:

- `results_workbook.xlsx`: every results table on its own sheet, with a contents sheet.
- `*_results.csv`: the same tables in machine-readable form.
- `fig_*.png`, `decile_sort_chart*.png`, `fm_lambda_chart.png` and `texp_fs_scatter.png`: the
  figures.
- `*_validation_report.txt`: one per stage, recording sample sizes, exclusions and checks.
- `event_study_firm_universe.csv` and `full_panel_firm_sample.csv`: the two fixed firm lists.

## Reproducibility notes

- **Fixed firm lists.** The two firm lists are written once. `persist_2025_universe.py` and
  `sample_full_panel_firms.py` refuse to overwrite them unless given `--rebuild` or `--redraw`. The
  subsample is drawn with `numpy.random.default_rng(20250402)` from the universe in ascending PERMNO
  order. If you rebuild the universe, redraw the sample.
- **No look-ahead.** Every exposure score comes from a filing dated before the date it is used
  for. Every monthly regressor is dated the month before the return.
- **Built-in checks.** Stages check their own inputs and stop with an error rather than continue
  on inconsistent data. The figure scripts check that what they draw matches the results tables.

## References

**Data**

- Center for Research in Security Prices (CRSP®), The University of Chicago Booth School of
  Business. Accessed via Wharton Research Data Services (WRDS).
- Compustat, S&P Global Market Intelligence. Accessed via WRDS.
- Kenneth R. French Data Library: factor returns, NYSE breakpoints and industry definitions.
- Economic Policy Uncertainty index, policyuncertainty.com; Baker, Bloom and Davis (2016).
- U.S. Securities and Exchange Commission, EDGAR.

**Methods**

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty.
  *Quarterly Journal of Economics*, 131(4), 1593–1636.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly.
  (NLTK, used for sentence splitting.)
- Bruno, Goltz and Luyten (2024). *European Financial Management*. Table 3: Section 301 escalation
  dates.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway
  clustering. *Journal of Business & Economic Statistics*, 29(2), 238–249.
- Campbell, J. L., Chen, H., Dhaliwal, D. S., Lu, H., & Steele, L. B. (2014). The information
  content of mandatory risk factor disclosures in corporate filings. *Review of Accounting
  Studies*, 19(1), 396–455. Item 1A extraction approach.
- Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1),
  57–82.
- Davis, J. L., Fama, E. F., & French, K. R. (2000). Characteristics, covariances, and average
  returns: 1929 to 1997. *Journal of Finance*, 55(1), 389–406. Book-equity definition.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial
  Economics*, 116(1), 1–22.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal
  of Political Economy*, 81(3), 607–636.
- Hassan, T. A., Hollander, S., van Lent, L., & Tahoun, A. (2019). Firm-level political risk:
  Measurement and effects. *Quarterly Journal of Economics*, 134(4), 2135–2202. The
  sentence-level, length-scaled construction TExp follows.
- Loughran, T., & McDonald, B. (2016). Textual analysis in accounting and finance: A survey.
  *Journal of Accounting Research*, 54(4), 1187–1230. 10-K parsing conventions.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic
  Literature*, 35(1), 13–39.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test
  for heteroskedasticity. *Econometrica*, 48(4), 817–838.
