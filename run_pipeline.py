"""
Run the pipeline end to end, or any contiguous part of it, in dependency order.

Each stage is a script in src/ and runs in its own process: the event-study modules rebind
module-level state when a cycle is selected, so chaining them in one interpreter would carry that
state from one stage into the next. Stages skip work whose outputs already exist; --force passes
through to the stages that support rebuilding.

Two kinds of stage are excluded unless --cold is given:
  pull  downloads 10-K filings from SEC EDGAR (rate-limited, resumable, needs EDGAR_USER_AGENT)
  once  writes a fixed firm list exactly once; re-running it only rewrites its report

    python run_pipeline.py --list              # stages in order, with their kind
    python run_pipeline.py --cold              # fresh clone: every stage, filings included
    python run_pipeline.py                     # re-run the analysis from the cached filings
    python run_pipeline.py --from decile_sort  # resume from a stage
    python run_pipeline.py --only figures_panel,workbook
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

# (name, script, arguments, kind, accepts --force). Order is the dependency order.
STAGES = [
    ("clean_data",          "clean_data.py",              [],                          "", True),
    ("pull_2025",           "edgar_pull.py",              ["--all"],                   "pull", False),
    ("controls_2025",       "clean_controls_data.py",     ["--cycle", "2025"],         "", True),
    ("car_2025",            "estimate_car.py",            ["--cycle", "2025"],         "", False),
    ("universe",            "persist_2025_universe.py",   [],                          "once", False),
    ("sample",              "sample_full_panel_firms.py", [],                          "once", False),
    ("pull_cross_cycle",    "edgar_pull.py",              ["--batch", "cross_cycle"],  "pull", False),
    ("pull_full_panel",     "edgar_pull.py",              ["--batch", "full_panel"],   "pull", False),
    ("clean_filings",       "clean_filings.py",           [],                          "", False),
    ("score_filings",       "score_filings.py",           [],                          "", False),
    ("texp_panel",          "build_texp_panel.py",        [],                          "", True),
    ("foreign_sales",       "foreign_sales.py",           [],                          "", True),
    ("regression_2025",     "run_car_regression.py",      ["--cycle", "2025"],         "", False),
    ("controls_cc",         "clean_controls_data.py",     ["--cycle", "cross_cycle"],  "", True),
    ("car_cc",              "estimate_car.py",            ["--cycle", "cross_cycle"],  "", False),
    ("regression_cc",       "run_car_regression.py",      ["--cycle", "cross_cycle"],  "", False),
    ("decile_sort",         "decile_sort.py",             [],                          "", False),
    ("fm_panel",            "build_fm_panel.py",          [],                          "", True),
    ("fama_macbeth",        "fama_macbeth_pricing.py",    [],                          "", False),
    ("figures_event_study", "figures_event_study.py",     [],                          "", False),
    ("figures_panel",       "figures_panel.py",           [],                          "", False),
    ("figures_collinearity", "figures_collinearity.py",   [],                          "", False),
    ("workbook",            "build_results_workbook.py",  [],                          "", False),
]
NAMES = [stage[0] for stage in STAGES]


def select(cold: bool, start: str | None, only: list[str] | None) -> list[tuple]:
    """The stages to run, in pipeline order."""
    unknown = [name for name in (only or []) + ([start] if start else []) if name not in NAMES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. See --list.")
    if only:
        return [stage for stage in STAGES if stage[0] in only]
    chosen = STAGES[NAMES.index(start):] if start else STAGES
    return [stage for stage in chosen if cold or not stage[3]]


def run(stages: list[tuple], force: bool) -> None:
    """Run each stage as its own process, stopping at the first failure."""
    for name, script, args, _, accepts_force in stages:
        command = [sys.executable, str(SRC_DIR / script), *args]
        if force and accepts_force:
            command.append("--force")
        print(f"\n=== {name}: {' '.join([f'src/{script}', *command[2:]])}", flush=True)
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            raise SystemExit(f"Stage '{name}' failed; fix it and resume with --from {name}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cold", action="store_true",
                    help="include the filing download and the one-time firm-list stages")
    ap.add_argument("--from", dest="start", metavar="STAGE", help="start at this stage")
    ap.add_argument("--only", metavar="STAGE[,STAGE]", help="run only these stages")
    ap.add_argument("--force", action="store_true",
                    help="rebuild outputs that already exist, where a stage supports it")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    args = ap.parse_args()

    if args.list:
        for name, script, stage_args, kind, _ in STAGES:
            print(f"{name:<22}{kind:<6}{' '.join([script, *stage_args])}")
        return
    only = args.only.split(",") if args.only else None
    run(select(args.cold, args.start, only), args.force)


if __name__ == "__main__":
    main()
