"""
One-off migration: move cleaned text out of clean_filings.csv into the per-accession store.

clean_filings.csv originally carried item_1a_text and rest_text as columns, which at
single-vintage scale (2,987 accessions) was a 1.14 GB file. Across all 17 reference dates the
corpus is 13,980 accessions, and the same schema would be a ~5.3 GB CSV that cannot be built
or re-read without exhausting memory.

This script re-shapes the existing rows into the new layout. It does NOT re-parse any HTML, so
no character of extracted text and no downstream score can change. Every row is verified
against the character counts it already recorded before the old file is replaced.

Runs once; refuses to run again if clean_filings.csv is already in the slim schema.

    python migrate_clean_filings.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

import clean_filings as cf

CHUNK_ROWS = 200          # ~76 MB of text per chunk at 380 KB/row
OLD_TEXT_COLUMNS = ["item_1a_text", "rest_text"]
OLD_DTYPES = {**cf.READ_DTYPES, "item_1a_text": str, "rest_text": str}
# The pre-migration file is kept rather than overwritten, matching the convention
# edgar_pull.py already uses for edgar_pull_log.pre-migration.csv. Safe to delete once the
# text store has been used by a scoring run.
BACKUP = cf.CLEAN_OUT.with_name("clean_filings.pre-migration.csv")


def already_migrated(path: Path) -> bool:
    """True if clean_filings.csv no longer carries the text columns."""
    header = pd.read_csv(path, nrows=0).columns
    return not any(c in header for c in OLD_TEXT_COLUMNS)


def migrate(path: Path) -> pd.DataFrame:
    """Write each row's text to the store and return the slim metadata table.

    Character counts are checked per row as they are written: the stored counts were computed
    at cleaning time from the same strings, so a mismatch means the original CSV was already
    lossy and the migration must not proceed.
    """
    kept, n_rows, mismatches = [], 0, []
    for chunk in pd.read_csv(path, dtype=OLD_DTYPES, chunksize=CHUNK_ROWS):
        for col in OLD_TEXT_COLUMNS:
            chunk[col] = chunk[col].fillna("")
        for r in chunk.itertuples(index=False):
            if len(r.item_1a_text) != r.item_1a_char_count:
                mismatches.append((r.accession, "item_1a",
                                   len(r.item_1a_text), r.item_1a_char_count))
            if len(r.rest_text) != r.rest_char_count:
                mismatches.append((r.accession, "rest",
                                   len(r.rest_text), r.rest_char_count))
            cf.write_clean_text(r.accession, r.item_1a_text, r.rest_text)
        kept.append(chunk.drop(columns=OLD_TEXT_COLUMNS))
        n_rows += len(chunk)
        print(f"[progress] {n_rows:,} rows migrated", flush=True)

    if mismatches:
        for acc, sect, got, want in mismatches[:10]:
            print(f"  {acc} {sect}: text is {got:,} chars, row records {want:,}")
        raise ValueError(f"{len(mismatches)} section(s) disagree with their recorded character "
                         f"count; the source CSV is lossy. Migration aborted, nothing replaced.")
    return pd.concat(kept, ignore_index=True).reindex(columns=cf.OUTPUT_COLUMNS)


def verify(meta: pd.DataFrame) -> int:
    """Re-read every migrated document and confirm both sections survived the round trip.

    Runs on all rows, not a sample: this is the only check that would catch a lossy format
    migration, and it runs before the old file is discarded.
    """
    bad, chars = [], 0
    for r in meta.itertuples(index=False):
        item_1a, rest = cf.read_clean_text(r.accession)
        if len(item_1a) != r.item_1a_char_count or len(rest) != r.rest_char_count:
            bad.append(r.accession)
        chars += len(item_1a) + len(rest)
    if bad:
        raise ValueError(f"{len(bad)} document(s) did not round-trip through the text store: "
                         f"{bad[:10]}. Migration aborted, nothing replaced.")
    return chars


def main() -> None:
    path = cf.CLEAN_OUT
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; nothing to migrate.")
    if already_migrated(path):
        print(f"{path.name} is already in the slim schema - nothing to do.")
        return

    size_before = path.stat().st_size / 1e6
    print(f"Migrating {path.name} ({size_before:,.0f} MB) into "
          f"{cf.CLEAN_TEXT_DIR.name}/\n")
    cf.clear_partials()

    meta = migrate(path)
    chars = verify(meta)

    tmp = path.with_suffix(".slim.tmp")
    meta.to_csv(tmp, index=False)
    os.replace(path, BACKUP)          # only after verify() passed
    os.replace(tmp, path)

    store_mb = sum(p.stat().st_size for p in cf.CLEAN_TEXT_DIR.glob("*.json.gz")) / 1e6
    print(f"\nVerified {len(meta):,} documents, {chars:,} characters intact.")
    print(f"{path.name}: {size_before:,.0f} MB -> {path.stat().st_size / 1e6:,.1f} MB "
          f"({len(meta.columns)} cols, text removed)")
    print(f"{cf.CLEAN_TEXT_DIR.name}/: {len(meta):,} files, {store_mb:,.0f} MB gzipped")
    print(f"Original kept as {BACKUP.name} - delete once a scoring run has used the store.")


if __name__ == "__main__":
    sys.exit(main())
