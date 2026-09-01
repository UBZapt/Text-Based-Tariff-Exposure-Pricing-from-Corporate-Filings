"""
Step 1e - persist the 2025 event-study firm universe as a fixed, reusable list.

Research Design v6 section 7.0 draws the full-panel subsample from "all firms passing the
section 6 screens at the point-in-time cutoff of end-March 2025". That set is already computed
- estimate_car.pit_screen() evaluates the screen at 2025-03, the last completed calendar month
strictly before the imposition event, and carries it as in_screened_universe_pit. This module
persists it so the draw and every downstream reference to "the 2025 universe" read one file
rather than each recomputing a screen.

The list is NOT the pull scope. edgar_pull.py --scope full continues to iterate the bridge:
the bridge is ever-qualifying across 2017-2026 while this list is point-in-time at one date,
and filtering the 2018-19 event pull through a 2025 screen would impose a survivorship filter
the design does not ask for.

    python persist_2025_universe.py
"""

from pathlib import Path

import pandas as pd

import edgar_pull as ep
import estimate_car as ec

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"

# The imposition run carries the end-March-2025 screen. The reversal run's screen is evaluated
# at 2025-07 and is deliberately not used here: section 7.0 names end-March 2025.
CAR_FILE = ec.RUNS[0]["out"]
SCREEN_COLUMN = "in_screened_universe_pit"

UNIVERSE_OUT = OUTPUT_DIR / "event_study_firm_universe.csv"

# The reference date the existing 2025 pull ran at. Identifiers are resolved at this date so
# cik_2025 reproduces what that pull actually used.
REF_2025 = pd.Timestamp(ep.DEFAULT_REF_DATE)

OUTPUT_COLUMNS = ["permno", "gvkey", "cik_2025"]
MAX_LISTED = 20          # identities printed before deferring to a count
RULE = "=" * 74


def _say(line: str = "") -> None:
    print(line)


def _section(title: str) -> None:
    _say()
    _say(RULE)
    _say(title)
    _say(RULE)


def _listed(values) -> str:
    vals = list(values)
    head = ", ".join(str(v) for v in vals[:MAX_LISTED])
    return head if len(vals) <= MAX_LISTED else f"{head}, ... (+{len(vals) - MAX_LISTED} more)"


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def load_screened_permnos(path: Path = CAR_FILE) -> list[int]:
    """Read the CAR table and return the PERMNOs inside the end-March-2025 screen.

    Read through estimate_car.read_car rather than pd.read_csv: the screen flag returns as
    object once any row is blank, and a truthiness test on that would silently keep every row.
    """
    car = ec.read_car(path)
    return sorted(int(p) for p in car.loc[car[SCREEN_COLUMN], "permno"])


def resolve_identifiers(permnos: list[int],
                        ref: pd.Timestamp = REF_2025) -> tuple[pd.DataFrame, list[int]]:
    """Attach gvkey and CIK by calling the same resolver the 2025 pull used.

    edgar_pull.resolve_firms applies _resolve_cik per PERMNO at this reference date, so cik_2025
    is audit-faithful by construction rather than copied from the log. Returns the frame and any
    PERMNOs the bridge cannot resolve.
    """
    firms = {f["permno"]: f for f in ep.resolve_firms(ep.BRIDGE_CSV, ref)}
    rows, unresolved = [], []
    for permno in permnos:
        firm = firms.get(permno)
        if firm is None:
            unresolved.append(permno)
            continue
        rows.append({"permno": permno, "gvkey": firm["gvkey"], "cik_2025": firm["cik"]})
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), unresolved


def write_universe(frame: pd.DataFrame, path: Path = UNIVERSE_OUT) -> Path:
    """Write the universe list, zero-padded identifiers preserved as text."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def read_universe(path: Path = UNIVERSE_OUT) -> pd.DataFrame:
    """Canonical reader: keeps gvkey and CIK zero-padded, PERMNO integer.

    Read as text or the 6-digit gvkey and 10-digit CIK become integers and stop matching the
    bridge and the log - the identical hazard Step 4b documented for the cleaned-filings file.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run persist_2025_universe.py first.")
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["permno"] = frame["permno"].astype(int)
    return frame


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(frame: pd.DataFrame, permnos: list[int], unresolved: list[int]) -> None:
    """Report resolution, CIK agreement against the log, and 2025 pull coverage.

    The coverage check is the definition-of-done item: it establishes, rather than assumes,
    that the existing 2025 pull already covered every firm in this universe.
    """
    _section("VALIDATION")

    _say(f"Source            : {CAR_FILE.name} [{SCREEN_COLUMN}], screen at 2025-03")
    _say(f"Screened PERMNOs  : {len(permnos):,}")
    _say(f"Resolved in bridge: {len(frame):,} of {len(permnos):,}")
    if unresolved:
        _say(f"!! UNRESOLVED     : {len(unresolved)} {_listed(unresolved)}")
    else:
        _say("Unresolved        : 0")

    no_cik = frame.loc[frame["cik_2025"] == "", "permno"]
    _say(f"Without a CIK     : {len(no_cik)} "
         f"{_listed(no_cik) if len(no_cik) else '(these log as no_cik)'}")
    ciks = set(frame.loc[frame["cik_2025"] != "", "cik_2025"].str.zfill(10))
    _say(f"Unique CIKs       : {len(ciks):,}  "
         f"({len(frame) - len(no_cik) - len(ciks):,} PERMNOs share a CIK with another)")

    # -- cik_2025 against what the pull actually recorded ------------------- #
    log = ep.load_log()
    same_ref = log[log["reference_date"] == str(REF_2025.date())].copy()
    _section("CIK AGREEMENT vs edgar_pull_log.csv @ " + str(REF_2025.date()))
    if same_ref.empty:
        _say("!! no log rows at this reference date; agreement not checkable.")
        return
    same_ref["permno"] = same_ref["permno"].astype(int)
    logged = same_ref.drop_duplicates("permno", keep="last").set_index("permno")["cik"]
    checked = frame[frame["permno"].isin(logged.index)]
    mismatch = [int(r.permno) for r in checked.itertuples(index=False)
                if r.cik_2025.zfill(10) != str(logged[int(r.permno)]).strip().zfill(10)]
    _say(f"PERMNOs present in the log : {len(checked):,} of {len(frame):,}")
    _say(f"cik_2025 mismatches        : {len(mismatch)} {_listed(mismatch) if mismatch else ''}")
    assert not mismatch, "cik_2025 disagrees with the log; the audit column is not faithful."

    # -- coverage: did the 2025 pull reach every firm in this universe? ----- #
    _section("2025 PULL COVERAGE OF THIS UNIVERSE  (definition-of-done check)")
    logged_ciks = {c.strip().zfill(10) for c in same_ref["cik"] if c.strip()}
    covered = ciks & logged_ciks
    _say(f"Universe CIKs attempted at {REF_2025.date()} : {len(covered):,} of {len(ciks):,} "
         f"({len(covered) / len(ciks):.1%})")
    _say(f"Universe CIKs NOT attempted                : {len(ciks - logged_ciks):,} "
         f"{_listed(sorted(ciks - logged_ciks))}")

    absent = sorted(set(frame["permno"]) - set(same_ref["permno"]))
    shared = [p for p in absent
              if frame.loc[frame["permno"] == p, "cik_2025"].iloc[0].zfill(10) in logged_ciks]
    _say()
    _say(f"PERMNOs absent from the log : {len(absent)}")
    _say(f"  covered under another PERMNO (select_gap_firms de-duplicates on CIK) : "
         f"{len(shared)} {_listed(shared)}")
    _say(f"  genuinely unattempted                                              : "
         f"{len(absent) - len(shared)} {_listed([p for p in absent if p not in shared])}")
    _say()
    if not (ciks - logged_ciks):
        _say("=> Every CIK in this universe was attempted by the existing 2025 pull. The "
             "cross-cycle\n   and full-panel batches therefore extend a list the 2025 event "
             "study already covered.")
    else:
        _say("!! Some universe CIKs were never attempted at 2025-04-02 - investigate before "
             "drawing.")


def print_assumptions() -> None:
    _section("ASSUMPTIONS (this run)")
    for line in [
        f"The 2025 event-study universe is {CAR_FILE.name} [{SCREEN_COLUMN}] - the section 6 "
        f"screen evaluated at 2025-03, the last completed month strictly before the imposition "
        f"event. It is not recomputed here.",
        "The reversal run's screen (2025-07) is not used: section 7.0 names end-March 2025.",
        f"gvkey and cik_2025 are resolved by edgar_pull.resolve_firms at {REF_2025.date()}, the "
        f"same function and date the existing pull used, so cik_2025 reproduces it rather than "
        f"copying it. It is an audit column only - every pull re-resolves the CIK per "
        f"(PERMNO, reference_date).",
        "This file is the draw pool for sample_full_panel_firms.py and the audit reference for "
        "edgar_pull.print_status. It is NOT a pull scope: --scope full remains the bridge.",
    ]:
        _say(f"  - {line}")


def main() -> pd.DataFrame:
    permnos = load_screened_permnos()
    frame, unresolved = resolve_identifiers(permnos)
    path = write_universe(frame)
    validate(frame, permnos, unresolved)
    print_assumptions()
    _section("OUTPUT")
    _say(f"  {path.relative_to(BASE)}  ({len(frame):,} rows x {len(frame.columns)} cols)")
    return frame


if __name__ == "__main__":
    main()
