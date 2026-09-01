"""
Step 1e - draw the fixed random 1,000-firm subsample for the full-panel tests.

Research Design v6 section 7.0: the Fama-MacBeth (7.4) and EPU regime tests (7.5) need TExp
refreshed annually across 2017-2026, roughly nine filing-years per firm. At full-universe scale
that is ~40,000 filing-years, so those two tests run on a random subsample of n = 1,000 drawn
once from the end-March-2025 screened universe.

The draw is fixed ONCE and reused across all ten annual reference dates. Redrawing per year
would introduce artificial entry and exit unrelated to any economic process, so this module
refuses to overwrite an existing sample without --redraw.

The subsample applies to sections 7.4 and 7.5 only. The event study (7.2) and the cross-cycle
validation (7.6) run on the full universe.

    python sample_full_panel_firms.py
    python sample_full_panel_firms.py --redraw     # only to deliberately replace the draw
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import run_report

import persist_2025_universe as pu

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"

UNIVERSE_CSV = pu.UNIVERSE_OUT
SAMPLE_OUT = OUTPUT_DIR / "full_panel_firm_sample.csv"
REPORT_OUT = OUTPUT_DIR / "full_panel_sample_validation_report.txt"

# Recorded seed. Arbitrary in value - it is the imposition event date read as an integer - but
# fixed and stated, which is the section 7.0 requirement: the draw must be exactly reproducible.
SEED = 20250402
SAMPLE_SIZE = 1_000

# The recorded enumeration order. Ascending PERMNO is the ordering edgar_pull.resolve_firms and
# every other module in the project already uses, so the index a firm receives here is
# reproducible from the universe file alone without carrying a permutation.
ORDER_COLUMN = "permno"

OUTPUT_COLUMNS = ["permno", "gvkey", "cik_2025"]
RULE = "=" * 74


def _say(line: str = "") -> None:
    print(line)


def _section(title: str) -> None:
    _say()
    _say(RULE)
    _say(title)
    _say(RULE)


# --------------------------------------------------------------------------- #
# Draw                                                                         #
# --------------------------------------------------------------------------- #
def draw_sample(universe: pd.DataFrame, seed: int = SEED,
                n: int = SAMPLE_SIZE) -> tuple[pd.DataFrame, np.ndarray]:
    """Draw n distinct firms from the enumerated universe using a recorded seed.

    Enumerates the universe in ascending PERMNO order, draws n distinct indices without
    replacement, and returns the sampled rows sorted back into PERMNO order alongside the
    drawn indices. Sorting the output does not affect which firms were drawn - the draw is
    over indices into the enumerated order, which is fixed before the draw.
    """
    enumerated = universe.sort_values(ORDER_COLUMN, kind="mergesort").reset_index(drop=True)
    if n > len(enumerated):
        raise ValueError(f"cannot draw {n:,} firms from a universe of {len(enumerated):,}.")
    idx = np.random.default_rng(seed).choice(len(enumerated), size=n, replace=False)
    sample = enumerated.iloc[np.sort(idx)][OUTPUT_COLUMNS].reset_index(drop=True)
    return sample, idx


def write_sample(sample: pd.DataFrame, n_universe: int, seed: int = SEED,
                 path: Path = SAMPLE_OUT) -> Path:
    """Write the sample with the draw provenance as a leading comment block.

    The seed lives in the file itself, not only in this module, so the artefact is
    self-describing if it is ever read without the code. Readers must pass comment='#'.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    header = (f"# full-panel firm subsample - Research Design v6 section 7.0\n"
              f"# seed={seed} draw_date={date.today().isoformat()} "
              f"N={n_universe} n={len(sample)}\n"
              f"# source={UNIVERSE_CSV.name} order=ascending_{ORDER_COLUMN}\n"
              f"# fixed once; reused unchanged across all annual reference dates\n")
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(header)
        sample.to_csv(fh, index=False)
    return path


def read_sample(path: Path = SAMPLE_OUT) -> pd.DataFrame:
    """Canonical reader: skips the provenance comments, keeps identifiers zero-padded."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run sample_full_panel_firms.py first.")
    frame = pd.read_csv(path, dtype=str, comment="#").fillna("")
    frame["permno"] = frame["permno"].astype(int)
    return frame


def read_provenance(path: Path = SAMPLE_OUT) -> str:
    """Return the recorded seed/draw-date line, for reporting by other modules."""
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# seed="):
            return line.lstrip("# ").strip()
    return ""


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(sample: pd.DataFrame, universe: pd.DataFrame, idx: np.ndarray) -> None:
    """Check the draw is distinct, contained, reproducible, and identifier-complete."""
    _section("VALIDATION")
    _say(f"Universe (N)          : {len(universe):,}  [{UNIVERSE_CSV.name}]")
    _say(f"Drawn (n)             : {len(sample):,}")
    _say(f"Seed                  : {SEED}")
    _say(f"Enumeration order     : ascending {ORDER_COLUMN}")

    if not len(set(idx)) == len(idx) == SAMPLE_SIZE:
        raise ValueError("draw is not n distinct indices")
    if len(sample) != SAMPLE_SIZE:
        raise ValueError("sample row count does not match the draw")
    if not set(sample["permno"]) <= set(universe["permno"]):
        raise ValueError("sample is not a subset of the universe")
    if not sample["permno"].is_unique:
        raise ValueError("duplicate PERMNO in the sample")
    _say("Distinct indices      : yes (asserted)")
    _say("Subset of universe    : yes (asserted)")

    # Re-running the generator must reproduce the identical draw from the seed alone.
    repeat, _ = draw_sample(universe)
    if repeat["permno"].tolist() != sample["permno"].tolist():
        raise ValueError("draw is not reproducible")
    _say("Reproducible from seed: yes (asserted, re-drawn and compared)")

    no_cik = sample.loc[sample["cik_2025"] == "", "permno"]
    _say(f"Sampled without a CIK : {len(no_cik)} "
         f"{list(no_cik) if len(no_cik) else '(these will log as no_cik)'}")
    _say(f"Unique CIKs           : {sample.loc[sample['cik_2025'] != '', 'cik_2025'].nunique():,}")

    _say()
    _say(f"Sampling rate         : {len(sample) / len(universe):.1%} of the screened universe")
    _say(f"PERMNO range          : {sample['permno'].min():,} .. {sample['permno'].max():,} "
         f"(universe {universe['permno'].min():,} .. {universe['permno'].max():,})")


def print_assumptions() -> None:
    _section("ASSUMPTIONS (this run)")
    for line in [
        f"Eligible universe = every firm in {UNIVERSE_CSV.name}, i.e. the section 6 screen at "
        f"end-March 2025. Firms whose filings cannot be scored are dropped downstream by the "
        f"scoring module, not gated here - the pull layer does not duplicate a filter another "
        f"step already owns.",
        f"Enumeration order = ascending {ORDER_COLUMN}, recorded so the index-to-firm mapping is "
        f"reproducible from the universe file alone.",
        f"Seed = {SEED}, fixed and recorded here, in the output file header, in build_notes.md "
        f"and in CLAUDE.md. Draw is without replacement, so the {SAMPLE_SIZE:,} firms are distinct.",
        "The draw is made ONCE and reused unchanged across all ten annual reference dates. "
        "Redrawing per year would create artificial entry and exit unrelated to any economic "
        "process (section 7.0 step 3), so an existing sample file is never silently overwritten.",
        "A sampled firm not listed in an earlier panel year simply has no observations then. "
        "That is absence, not a gap, and is not backfilled (section 7.0 step 4).",
        "The subsample governs sections 7.4 and 7.5 only. The event study (7.2) and the "
        "cross-cycle validation (7.6) run on the full universe.",
    ]:
        _say(f"  - {line}")


def main() -> pd.DataFrame:
    with run_report.capture(REPORT_OUT,
                            title="STEP 1E - FULL-PANEL SUBSAMPLE DRAW"):
        return _run()


def _run() -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Draw the fixed full-panel firm subsample.")
    ap.add_argument("--redraw", action="store_true",
                    help="replace an existing sample file (the draw is meant to be fixed once)")
    args = ap.parse_args()

    if SAMPLE_OUT.exists() and not args.redraw:
        _section("SAMPLE ALREADY DRAWN - NOT REGENERATED")
        existing = read_sample()
        _say(f"  {SAMPLE_OUT.relative_to(BASE)}  ({len(existing):,} firms)")
        _say(f"  {read_provenance()}")
        _say()
        _say("  The draw is fixed once and reused across every reference date. Pass --redraw "
             "only to\n  deliberately replace it, which invalidates any panel already built "
             "on the old sample.")
        return existing

    universe = pu.read_universe(UNIVERSE_CSV)
    sample, idx = draw_sample(universe)
    path = write_sample(sample, len(universe))
    validate(sample, universe, idx)
    print_assumptions()
    _section("OUTPUT")
    _say(f"  {path.relative_to(BASE)}  ({len(sample):,} rows x {len(sample.columns)} cols)")
    return sample


if __name__ == "__main__":
    main()
