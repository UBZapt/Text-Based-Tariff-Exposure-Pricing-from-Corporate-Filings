"""
Final stage - merge every reported results table into one Excel workbook, one sheet per table.

The pipeline writes each result as its own CSV, which is what downstream code reads and what makes
the outputs diffable. This adds a single human-facing deliverable on top: one file a supervisor or
marker can open, with a contents sheet naming each table's design section and source file.

The source CSVs are NOT removed. They stay because they are the machine-readable form, because
diffing them is how every numerical change in this project is verified, and because one of them -
car_regression_results.csv - is a genuine input to fama_macbeth_pricing.py.

Deliberately excluded, as cleaned data and intermediate panels are not themselves reported:
  - data/intermediate/controls_panel*.csv  1.1M and 2.2M rows, past Excel's 1,048,576-row limit
  - data/intermediate/car_*.csv            per-firm CARs; the regressions' input, not a reported table
  - data/intermediate/*_diagnostics.csv    per-document and per-firm-date audit tables
  - data/clean/*                           cleaned source data
  - output/*.txt, output/*.png             validation reports and figures keep their own formats

    python src/build_results_workbook.py
"""

import argparse
from pathlib import Path

import pandas as pd

from config import BASE, OUTPUT_DIR

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
WORKBOOK_OUT = OUTPUT_DIR / "results_workbook.xlsx"

# Excel caps sheet names at 31 characters and forbids : \ / ? * [ ]. Names are chosen short
# enough to survive that without truncation, and ordered as the dissertation reports them.
#
# `read_kwargs` carries per-file reader options. full_panel_firm_sample.csv leads with a `#`
# provenance header, so it needs comment="#".
SHEETS = [
    {"sheet": "7.2 CAR regressions", "file": "car_regression_results.csv",
     "section": "7.2", "about": "Cross-sectional CAR regressions, 2025 cycle (H1, H5)"},
    {"sheet": "7.2 H1 sign-flip test", "file": "signflip_test_results.csv",
     "section": "7.2", "about": "Formal test of b_reversal - b_imposition (H1)"},
    {"sheet": "5.2 within-industry b", "file": "industry_split_results.csv",
     "section": "5.2 item 1", "about": "TExp coefficient estimated separately by FF12 group"},
    {"sheet": "7.3 decile spread", "file": "decile_sort_results.csv",
     "section": "7.3", "about": "Group CARs, full screened universe; value- and equal-weighted "
                                "on identical firms (v5 s8 robustness) - filter `weighting`"},
    {"sheet": "7.3 decile ex-megacap", "file": "decile_sort_results_ex_megacap.csv",
     "section": "7.3", "about": "Same, excluding firms above NYSE p90 market equity; both "
                                "weightings - filter `weighting`"},
    {"sheet": "7.4 FM headline", "file": "fm_headline_results.csv",
     "section": "7.4", "about": "lambda_1_bar with Newey-West inference (H2), and the lag grid"},
    {"sheet": "7.4 FM monthly lambdas", "file": "fm_lambda_panel.csv",
     "section": "7.4", "about": "One cross-sectional slope per month, per specification"},
    {"sheet": "7.5 EPU regimes", "file": "fm_epu_regime_results.csv",
     "section": "7.5", "about": "lambda_1_bar split by EPU regime and by tariff episode (H3)"},
    {"sheet": "7.6 CAR regressions", "file": "car_regression_results_cross_cycle.csv",
     "section": "7.6", "about": "Per-event regressions, 2018-19 Section 301 cycle"},
    {"sheet": "7.6 H4 stability", "file": "stability_test_results_cross_cycle.csv",
     "section": "7.6", "about": "Pooled test of H0: b^2025 = b^2018-19"},
    {"sheet": "7.6 H1 sign-flip test", "file": "signflip_test_results_cross_cycle.csv",
     "section": "7.6", "about": "Sign-flip difference test on the 2018-19 cycle's own legs"},
    {"sheet": "7.1 term hits", "file": "term_hits.csv",
     "section": "7.1", "about": "Per-bigram match counts across the scored corpus"},
    {"sheet": "7.0 subsample draw", "file": "full_panel_firm_sample.csv",
     "section": "7.0", "about": "The fixed 1,000-firm draw used by 7.4 and 7.5",
     "read_kwargs": {"comment": "#"}},
]

CONTENTS_SHEET = "Contents"
MAX_SHEET_NAME = 31
EXCEL_MAX_ROWS = 1_048_576 - 1        # one row is the header
ENGINE = "openpyxl"                   # xlsxwriter is not installed; do not switch to it


def collect(sheets: list[dict], output_dir: Path) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Read each source CSV, returning the frames and one contents row per sheet.

    A missing source is recorded and skipped rather than raising: the workbook is a presentation
    layer, and a partial pipeline should still produce one that says what is absent.
    """
    frames, contents = {}, []
    for spec in sheets:
        name = spec["sheet"]
        if len(name) > MAX_SHEET_NAME:
            raise ValueError(f"sheet name {name!r} exceeds Excel's {MAX_SHEET_NAME}-char limit")
        path = output_dir / spec["file"]
        row = {"Sheet": name, "Design section": spec["section"], "Contents": spec["about"],
               "Source file": spec["file"]}
        if not path.exists():
            contents.append({**row, "Rows": 0, "Status": "SOURCE MISSING - sheet omitted"})
            continue
        frame = pd.read_csv(path, **spec.get("read_kwargs", {}))
        if len(frame) > EXCEL_MAX_ROWS:
            contents.append({**row, "Rows": len(frame),
                             "Status": f"too large for one sheet ({len(frame):,} rows) - omitted"})
            continue
        frames[name] = frame
        contents.append({**row, "Rows": len(frame), "Status": "included"})
    return frames, contents


def write_workbook(frames: dict[str, pd.DataFrame], contents: list[dict],
                   path: Path | None = None) -> Path:
    """One sheet per table, contents first so the workbook opens on an index."""
    path = WORKBOOK_OUT if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(contents)[["Sheet", "Design section", "Contents", "Rows", "Status",
                                    "Source file"]]
    with pd.ExcelWriter(path, engine=ENGINE) as writer:
        index.to_excel(writer, sheet_name=CONTENTS_SHEET, index=False)
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return path


def verify(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    """Re-read the written workbook and check every sheet against its source frame.

    Cheap, and it is the only thing that establishes the workbook actually carries the numbers
    rather than a truncated or silently re-typed copy of them.
    """
    book = pd.read_excel(path, sheet_name=None)
    expected = {CONTENTS_SHEET, *frames}
    if set(book) != expected:
        raise ValueError(f"workbook sheets {sorted(book)} do not match expected {sorted(expected)}")
    for name, frame in frames.items():
        got = book[name]
        if len(got) != len(frame):
            raise ValueError(f"sheet {name!r}: {len(got)} rows written for {len(frame)} in source")
        if list(got.columns) != list(frame.columns):
            raise ValueError(f"sheet {name!r}: column names do not round-trip")


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                    help="directory holding the results CSVs (default: %(default)s)")
    args = ap.parse_args()

    frames, contents = collect(SHEETS, args.output_dir)
    path = write_workbook(frames, contents)
    verify(path, frames)

    print(f"Wrote {path.relative_to(BASE)}")
    print(f"  {len(frames)} result sheets plus '{CONTENTS_SHEET}', "
          f"{sum(len(f) for f in frames.values()):,} data rows total")
    for row in contents:
        flag = "" if row["Status"] == "included" else f"   <-- {row['Status']}"
        print(f"  {row['Sheet']:<24}{row['Rows']:>7,} rows   {row['Source file']}{flag}")
    print("  verified: every sheet round-trips to its source's row count and column names")
    return path


if __name__ == "__main__":
    main()
