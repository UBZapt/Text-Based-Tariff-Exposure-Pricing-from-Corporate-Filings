"""
Step 2a — Clean raw EDGAR 10-K HTML into Item 1A and rest-of-document text blocks.

Reads the cached filings listed in edgar_pull_log.csv, strips HTML/XBRL markup,
financial tables and repeated page furniture, and splits each filing's prose into
Item 1A ("Risk Factors") and everything else. Paragraph boundaries are preserved so
the downstream sentence tokeniser is not corrupted.

Cleaning only: no sentence splitting, tokenisation, lowercasing, stopword removal,
deduplication, bigram or keyword logic (those belong to the tokenisation and scoring
steps).

Keyed on the document, not the pull row. Across 17 reference dates the pull log's 31,142
successful rows cover 13,983 distinct firm-documents, so each accession is parsed once and
shared by every reference date that selected it. Cleaned text goes to clean_text/ as one
gzipped file per accession; clean_filings.csv carries metadata only. Both are append-only and
flushed per document, so an interrupted run resumes and loses at most one filing.

    python clean_filings.py             # clean whatever is outstanding
    python clean_filings.py --status    # done/remaining, no work, no network
"""

import argparse
import csv
import gzip
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

import edgar_pull

# Inline-XBRL filings carry an XML declaration but are parsed as HTML deliberately:
# the HTML tree builder is what tolerates the malformed markup these documents contain.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
CLEAN_DIR = BASE / "clean_data"
OUTPUT_DIR = BASE / "output"
LOG_CSV = BASE / "edgar_pull_log.csv"
FILINGS_DIR = BASE / "filings_raw"     # must match edgar_pull.FILINGS_DIR
CLEAN_OUT = CLEAN_DIR / "clean_filings.csv"
DIAG_OUT = OUTPUT_DIR / "cleaning_diagnostics.csv"
# One gzipped {item_1a, rest} document per accession. The text lives here rather than in
# CLEAN_OUT because the full corpus is ~380 KB of prose per filing: as columns that is a
# ~5.3 GB CSV which cannot be built or re-read without exhausting memory, while per-accession
# files keep both cleaning and scoring at constant memory and make either stage resumable.
CLEAN_TEXT_DIR = BASE / "clean_text"

# CSV rather than Parquet: Windows Smart App Control (Enforcement) blocks pyarrow's unsigned
# native DLLs, so no Parquet file on this machine can be read. See build_notes.md Step 4b.
# Identifier columns must be read back as strings or their zero padding is silently lost, and
# the flag column must not become NaN when a filing legitimately holds an empty string.
READ_DTYPES = {"cik": str, "accession": str, "form_type": str, "filing_date": str,
               "period_of_report": str, "cleaning_flags": str, "source_path": str}
FILL_EMPTY_COLUMNS = ["cleaning_flags"]
TEXT_KEYS = ("item_1a", "rest")

PARSER = "lxml"          # pinned: html.parser builds a different tree on malformed markup
FORM_TYPE = "10-K"       # constant by construction (edgar_pull.select_10k matches exactly)

DROP_TAGS = ["script", "style", "ix:header", "ix:hidden"]
# Newlines are injected at block boundaries only; inline <span>s must concatenate
# untouched or inline-XBRL splits words ("RIS\nK FACTORS").
BLOCK_TAGS = ["p", "div", "br", "tr", "td", "th", "li",
              "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"]

HEADING_TABLE_MAX_CHARS = 60          # a table this short holding only an Item heading is kept
HEADING_TABLE_RE = re.compile(r"(?i)^item\s*\d+[a-c]?\b")
BOILERPLATE_MAX_CHARS = 100           # page furniture is short; body paragraphs are not
BOILERPLATE_MIN_REPEATS = 5

_PART = r"(?:part\s*[i1]\s*[\.\-]?\s*)?"
ITEM_1A_START = re.compile(
    rf"(?im)^\s*{_PART}item\s*1a\b[\.\:\-–—\s]*risk\s*factors")
ITEM_1A_END = re.compile(rf"(?im)^\s*{_PART}item\s*(?:1b|1c|2)\b")

SHORT_ITEM_1A_CHARS = 2000
PROGRESS_EVERY = 100                  # filings between progress/ETA lines
# Item 1A detection is reported either side of this fiscal year, because the section regexes
# were tuned on 2024-vintage documents and pre-2020 filings largely predate inline XBRL. A gap
# wider than the tolerance is a finding for the write-up, not something to patch away silently.
VINTAGE_SPLIT_YEAR = 2020
VINTAGE_GAP_TOLERANCE = 0.05
ROUNDTRIP_SAMPLE = 300                # documents re-read from the store to prove text integrity
MAX_LISTED = 10                       # identities printed before deferring to the audit CSV
MARKUP_ARTIFACT_RE = re.compile(r"<[a-zA-Z/!]|&nbsp;|&#\d+;|&amp;|&lt;|&gt;|&quot;")
# A lone-letter line signals a word split across inline spans. Roman numerals are
# excluded: front-matter page numbers ("i") are legitimate content, not shattering.
SHATTERED_WORD_RE = re.compile(r"(?m)^(?![ivxIVX]$)[A-Za-z]$")

OUTPUT_COLUMNS = [
    "permno", "cik", "accession", "form_type", "filing_date", "period_of_report",
    "fiscal_year", "item_1a_found", "item_1a_char_count",
    "rest_char_count", "clean_char_count", "cleaning_flags", "source_path",
]


# --------------------------------------------------------------------------- #
# Cleaned-text store                                                          #
# --------------------------------------------------------------------------- #
def text_path(accession: str) -> Path:
    """Path of the gzipped cleaned text for one accession."""
    return CLEAN_TEXT_DIR / f"{accession}.json.gz"


def write_clean_text(accession: str, item_1a: str, rest: str) -> Path:
    """Write one filing's cleaned sections, atomically.

    Written to .part and renamed so a process killed mid-write never leaves a truncated
    document that a later resume would accept as complete - the same guarantee
    edgar_pull.download_primary gives the raw cache. Reuses edgar_pull's replace helper
    because OneDrive briefly locks files it is uploading.
    """
    CLEAN_TEXT_DIR.mkdir(exist_ok=True)
    path = text_path(accession)
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump({"item_1a": item_1a, "rest": rest}, fh)
    edgar_pull._replace_with_retry(tmp, path)
    return path


def read_clean_text(accession: str) -> tuple[str, str]:
    """Return (item_1a_text, rest_text) for one accession from the text store.

    Raises rather than returning empties on a missing file: a silently empty section would
    reach the scorer as a zero denominator and quietly bias the measure.
    """
    path = text_path(accession)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing from {CLEAN_TEXT_DIR.name}/; re-run clean_filings.py "
            f"for accession {accession}.")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc[TEXT_KEYS[0]], doc[TEXT_KEYS[1]]


def stored_accessions() -> set[str]:
    """Accessions whose cleaned text is already on disk."""
    if not CLEAN_TEXT_DIR.exists():
        return set()
    return {p.name[: -len(".json.gz")] for p in CLEAN_TEXT_DIR.glob("*.json.gz")}


def clear_partials() -> int:
    """Delete .part files left by a previous interrupted run."""
    if not CLEAN_TEXT_DIR.exists():
        return 0
    stale = list(CLEAN_TEXT_DIR.glob("*.part"))
    for p in stale:
        p.unlink()
    return len(stale)


# --------------------------------------------------------------------------- #
# Input selection                                                             #
# --------------------------------------------------------------------------- #
def load_filings(log: pd.DataFrame) -> pd.DataFrame:
    """Select successfully-pulled filings whose cached HTML is present, one row per firm-document.

    Takes the already-read log so callers holding it do not re-read the file. Log rows whose
    file was pruned (firm no longer in the universe) are skipped. Files are resolved by name
    under FILINGS_DIR so log rows written on either OS resolve alike.

    Deduplicated on (permno, accession), NOT on the pull row: cleaning is a property of the
    document, not of the reference date that selected it, so across 17 reference dates the
    31,142 successful pull rows cover only 13,983 distinct firm-documents. Keying on the pull
    row would clean the same file up to 13 times. The pair rather than the accession alone is
    the unit because a parent and subsidiary can file one joint 10-K, which legitimately
    carries a score for both firms (see score_filings._check_duplicate_accessions).
    """
    ok = log[log["found_10k"].astype(str).str.strip().str.lower() == "true"].copy()
    ok["source_path"] = ok["local_path"].astype(str).str.replace("\\", "/", regex=False)
    ok["path"] = [FILINGS_DIR / p.rsplit("/", 1)[-1] for p in ok["source_path"]]
    present = ok[[p.exists() for p in ok["path"]]].copy()
    if present.empty:
        raise FileNotFoundError(
            f"No cached filings found on disk from {LOG_CSV.name}; run edgar_pull.py first.")

    # Sorted so the work order - and therefore any partially-complete run - is deterministic.
    return (present.sort_values(["accession", "permno"])
                   .drop_duplicates(["permno", "accession"]))


# --------------------------------------------------------------------------- #
# Markup stripping and text extraction                                        #
# --------------------------------------------------------------------------- #
def _strip_markup(soup: BeautifulSoup) -> int:
    """Drop non-prose markup and every table except bare section headings.

    Financial tables must not leak into the text as pseudo-sentences, so tables are
    decomposed. Some filers lay section headings out in a table, so a table whose
    entire text is a short Item heading is replaced by that heading line instead.
    Returns the number of heading tables retained.
    """
    for tag in soup(DROP_TAGS):
        tag.decompose()

    kept = 0
    for table in soup.find_all("table"):
        if table.decomposed:            # nested inside an already-dropped table
            continue
        text = re.sub(r"\s+", " ", table.get_text(" ")).strip()
        if len(text) <= HEADING_TABLE_MAX_CHARS and HEADING_TABLE_RE.match(text):
            table.replace_with(f"\n{text}\n")
            kept += 1
        else:
            table.decompose()
    return kept


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract text, breaking lines at block tags only.

    get_text() with a separator inserts it between *every* tag, which shatters words
    across the inline spans that inline-XBRL filings are built from. Appending a
    newline to block-level tags and then joining with no separator keeps paragraph
    structure while leaving words intact.
    """
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    return soup.get_text("")


def _normalise(text: str) -> str:
    """Collapse whitespace and unicode spaces while preserving paragraph breaks."""
    text = (text.replace(chr(0xA0), " ")      # non-breaking space
                .replace(chr(0x200B), "")     # zero-width space
                .replace(chr(0x2028), chr(10)))  # unicode line separator
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_boilerplate(text: str) -> tuple[str, int]:
    """Drop repeated page headers/footers, keeping each line's first occurrence.

    Retaining the first occurrence is deliberate: filers reuse a section heading as a
    running page header, so removing every copy would delete the genuine heading.
    """
    lines = text.split("\n")
    counts = Counter(ln for ln in lines if ln and len(ln) <= BOILERPLATE_MAX_CHARS)
    repeated = {ln for ln, n in counts.items() if n >= BOILERPLATE_MIN_REPEATS}

    seen, out = set(), []
    for line in lines:
        if line in repeated:
            if line in seen:
                continue
            seen.add(line)
        out.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cleaned, len(repeated)


def clean_filing(path: Path) -> tuple[str, int, int]:
    """Clean one raw filing to plain text.

    Returns (clean_text, heading tables retained, distinct boilerplate lines removed).
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), PARSER)
    n_heading_tables = _strip_markup(soup)
    text, n_boilerplate = _strip_boilerplate(_normalise(_extract_text(soup)))
    return text, n_heading_tables, n_boilerplate


# --------------------------------------------------------------------------- #
# Section split                                                               #
# --------------------------------------------------------------------------- #
def split_item_1a(text: str) -> tuple[str, str, bool, list[str]]:
    """Split cleaned text into Item 1A and the remaining prose.

    The start pattern is line-anchored and requires the "Risk Factors" caption to
    follow, which rejects the many inline cross-references ("the risks described in
    Item 1A and in Item 7A"). The section ends at the next Item 1B/1C/2 heading.
    Returns (item_1a_text, rest_text, found, flags); when Item 1A cannot be located
    the filing is flagged rather than assigned a guessed boundary.
    """
    starts = list(ITEM_1A_START.finditer(text))
    if not starts:
        return "", text, False, ["item_1a_not_found"]

    flags = []
    if len(starts) > 1:
        flags.append(f"multiple_item1a_candidates={len(starts)}")
    bounded = [m for m in starts if ITEM_1A_END.search(text, m.end())]
    start = (bounded or starts)[-1]

    end_match = ITEM_1A_END.search(text, start.end())
    if end_match:
        end = end_match.start()
    else:
        end = len(text)
        flags.append("item_1a_end_not_found")

    item_1a = text[start.start():end].strip()
    rest = (text[:start.start()] + "\n\n" + text[end:]).strip()
    return item_1a, rest, True, flags


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def open_clean_log():
    """Open clean_filings.csv for append, writing the header if it is new.

    Returns (file handle, DictWriter). Mirrors edgar_pull.open_log: rows are flushed as each
    document finishes, so a run killed after two hours keeps everything it recorded and
    re-running resumes. The end-of-run single write this replaced discarded the whole run,
    and at full-corpus scale it also had to hold ~5 GB of prose in memory to do it.
    """
    CLEAN_DIR.mkdir(exist_ok=True)
    is_new = not CLEAN_OUT.exists() or CLEAN_OUT.stat().st_size == 0
    fh = CLEAN_OUT.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def clean_all(filings: pd.DataFrame, writer, fh) -> pd.DataFrame:
    """Clean and split every document, flushing one metadata row per firm-document.

    Grouped by accession so the HTML is parsed once even when a joint filing is carried by two
    PERMNOs; every PERMNO on that accession then gets its own metadata row sharing the result.
    Text goes to the per-accession store, never into the returned frame, so memory stays flat
    across the whole corpus.

    Progress is periodic rather than per filing: at full-corpus scale a per-filing line is
    ~14,000 lines of noise, while a two-hour job needs an ETA.
    """
    rows, n_found, t0 = [], 0, time.monotonic()
    groups = list(filings.groupby("accession", sort=False))
    total = len(groups)
    for i, (accession, group) in enumerate(groups, 1):
        f = group.iloc[0]
        flags = []
        try:
            text, n_heading_tables, n_boilerplate = clean_filing(f.path)
            item_1a, rest, found, flags = split_item_1a(text)
            if n_heading_tables:
                flags.append(f"heading_tables_kept={n_heading_tables}")
            if n_boilerplate:
                flags.append(f"boilerplate_lines_removed={n_boilerplate}")
            if found and len(item_1a) < SHORT_ITEM_1A_CHARS:
                flags.append(f"short_item_1a={len(item_1a)}")
        except Exception as exc:
            text, item_1a, rest, found = "", "", "", False
            flags = [f"cleaning_failed: {type(exc).__name__}"]

        # Text first: a metadata row must never be flushed for a document whose text is
        # missing, or the resume would skip it and the scorer would fail on the gap.
        write_clean_text(accession, item_1a, rest)

        n_found += found
        for g in group.itertuples(index=False):
            period = str(g.period_of_report)
            row = {
                "permno": g.permno,
                "cik": g.cik,
                "accession": accession,
                "form_type": FORM_TYPE,
                "filing_date": g.filing_date,
                "period_of_report": period,
                "fiscal_year": int(period[:4]) if period[:4].isdigit() else None,
                "item_1a_found": found,
                "item_1a_char_count": len(item_1a),
                "rest_char_count": len(rest),
                "clean_char_count": len(text),
                "cleaning_flags": "; ".join(flags),
                "source_path": g.source_path,
            }
            writer.writerow(row)
            rows.append(row)
        fh.flush()

        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = time.monotonic() - t0
            print(f"[progress] {i:,}/{total:,} ({i / total:.1%}) | Item 1A found "
                  f"{n_found:,} ({n_found / i:.1%}) | {elapsed / 60:.1f} min elapsed | "
                  f"ETA {(total - i) * elapsed / i / 60:.0f} min", flush=True)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _listed(sub: pd.DataFrame) -> str:
    """Format filing identities for the console, capped so a full-corpus run stays readable."""
    if sub.empty:
        return ""
    ids = [f"{r.cik}/{r.accession}" for r in sub.head(MAX_LISTED).itertuples(index=False)]
    more = len(sub) - len(ids)
    return " ".join(ids) + (f" (+{more} more, see {DIAG_OUT.name})" if more else "")


def _scan_text(df: pd.DataFrame) -> tuple[list[int], int]:
    """Scan for residual markup and shattered words one document at a time.

    Reads each document from the text store rather than holding the corpus in memory: the
    full corpus is ~5 GB of prose, so streaming bounds the cost by the largest single
    document. Returns the positional indices of filings holding markup artifacts, and the
    total lone-letter count. Deduplicated on accession so a joint filing is not scanned twice.
    """
    artifact_rows, shattered, seen = [], 0, {}
    for i, r in enumerate(df.itertuples(index=False)):
        if r.accession in seen:
            if seen[r.accession]:
                artifact_rows.append(i)
            continue
        item_1a, rest = read_clean_text(r.accession)
        both = item_1a + "\n" + rest
        hit = bool(MARKUP_ARTIFACT_RE.search(both))
        seen[r.accession] = hit
        if hit:
            artifact_rows.append(i)
        shattered += len(SHATTERED_WORD_RE.findall(both))
    return artifact_rows, shattered


def write_diagnostics(df: pd.DataFrame) -> Path:
    """Write the per-filing audit table: the output schema minus the two text columns.

    At full-corpus scale the console cannot carry one line per dropped or flagged filing, so
    the identities live here and the console reports counts plus the first few.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(DIAG_OUT, index=False)
    print(f"Wrote {DIAG_OUT.relative_to(BASE)} ({len(df):,} rows) - per-filing cleaning audit.")
    return DIAG_OUT


def validate(df: pd.DataFrame) -> dict:
    """Run the required output checks and print the report.

    Covers residual markup, Item 1A detection overall and by fiscal year, short
    Item 1A sections, outright cleaning failures, and a regression guard on the
    inline-span word-shattering that a naive text extraction produces.
    """
    artifact_rows, shattered = _scan_text(df)
    artifacts = df.iloc[artifact_rows]
    failed = df[df["cleaning_flags"].str.startswith("cleaning_failed")]
    missing = df[~df["item_1a_found"]]
    short = df[df["cleaning_flags"].str.contains("short_item_1a")]
    empty_rest = df[df["rest_char_count"] == 0]

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"Filings cleaned            : {len(df):,}")
    print(f"Failed cleaning entirely   : {len(failed):,} {_listed(failed)}")
    print(f"Residual markup artifacts  : {len(artifacts):,} {_listed(artifacts)}")
    print(f"Shattered-word artifacts   : {shattered:,} lone-letter lines")
    print(f"Empty rest_text            : {len(empty_rest):,} {_listed(empty_rest)}")
    print(f"Item 1A located            : {len(df) - len(missing):,}/{len(df):,} "
          f"({(df['item_1a_found'].mean() if len(df) else 0):.1%})")
    if len(missing):
        print(f"  not found                : {_listed(missing)}")
    print(f"Short Item 1A (<{SHORT_ITEM_1A_CHARS:,} ch) : {len(short):,} {_listed(short)}")

    by_year = df.groupby("fiscal_year")["item_1a_found"].agg(["sum", "count"])
    print("\nItem 1A detection by fiscal year:")
    for year, r in by_year.iterrows():
        rate = r["sum"] / r["count"] if r["count"] else 0
        print(f"  FY{int(year)}: {int(r['sum']):>6,}/{int(r['count']):>6,}  {rate:>6.1%}")

    # The research design requires the boundary-detection failure rate to be reported
    # separately per vintage, and states that a materially higher early-vintage rate is a
    # finding to report rather than a bug to silently absorb. The regexes were tuned on
    # 2024-vintage documents; pre-2020 filings largely predate inline XBRL.
    old = df[df["fiscal_year"] < VINTAGE_SPLIT_YEAR]
    new = df[df["fiscal_year"] >= VINTAGE_SPLIT_YEAR]
    print(f"\nItem 1A detection by vintage (split at FY{VINTAGE_SPLIT_YEAR}):")
    for label, sub in ((f"FY<{VINTAGE_SPLIT_YEAR}", old), (f"FY>={VINTAGE_SPLIT_YEAR}", new)):
        if len(sub):
            print(f"  {label:9s}: {int(sub['item_1a_found'].sum()):>6,}/{len(sub):>6,}  "
                  f"{sub['item_1a_found'].mean():>6.1%}")
        else:
            print(f"  {label:9s}: no filings")
    if len(old) and len(new):
        gap = new["item_1a_found"].mean() - old["item_1a_found"].mean()
        verdict = ("REPORT THIS - materially worse on the early vintage"
                   if gap > VINTAGE_GAP_TOLERANCE else "within tolerance")
        print(f"  gap      : {gap:+.1%}  ({verdict}; tolerance "
              f"{VINTAGE_GAP_TOLERANCE:.0%})")

    if len(df):
        print(f"\nItem 1A chars: median "
              f"{df.loc[df['item_1a_found'], 'item_1a_char_count'].median():,.0f}"
              f" | rest chars: median {df['rest_char_count'].median():,.0f}")
    return {
        "n_filings": len(df), "n_failed": len(failed), "n_artifacts": len(artifacts),
        "n_missing_item_1a": len(missing), "n_short": len(short),
        "n_shattered": shattered, "by_fiscal_year": by_year,
    }


def read_clean_filings(path: Path | None = None) -> pd.DataFrame:
    """Read the cleaned-filings CSV with the dtypes Parquet used to carry for free.

    Shared with score_filings so the padding and empty-string rules live once, beside the
    writer. Without the dtype map cik loses its zero padding and every downstream CIK match
    silently fails; without the fillna an empty section reads back as NaN and the sentence
    tokeniser crashes on it.

    ``path`` defaults late rather than in the signature: a default argument binds CLEAN_OUT at
    import, which silently ignores any later reassignment of the constant.
    """
    path = CLEAN_OUT if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; run clean_filings.py first.")
    df = pd.read_csv(path, dtype=READ_DTYPES)
    for col in FILL_EMPTY_COLUMNS:
        df[col] = df[col].fillna("")
    return df


def verify_roundtrip(df: pd.DataFrame) -> dict:
    """Prove the metadata survived CSV quoting and the stored text still matches its counts.

    Two separate guarantees. The metadata table is checked in full - cheap now that the text
    has moved out, and cik silently losing its zero padding would break every downstream CIK
    match. Text integrity is checked on a random sample of ROUNDTRIP_SAMPLE documents rather
    than all of them: each row records its own section lengths, so comparing those against the
    strings the store returns catches a lossy write, and at full-corpus scale reading every
    document back would decompress ~5 GB for a check the migration already ran exhaustively.
    """
    back = read_clean_filings()
    missing = set(df["accession"]) - set(back["accession"])
    if missing:
        raise ValueError(f"{len(missing)} accession(s) written this run are absent from "
                         f"{CLEAN_OUT.name} on re-read, e.g. {sorted(missing)[:5]}")
    widths = set(back["cik"].str.len().unique())
    if widths != {10}:
        raise ValueError(f"cik lost its zero padding on re-read; widths found: {sorted(widths)}")

    sample = back.drop_duplicates("accession")
    sample = sample.sample(min(ROUNDTRIP_SAMPLE, len(sample)), random_state=0)
    bad, chars = [], 0
    for r in sample.itertuples(index=False):
        item_1a, rest = read_clean_text(r.accession)
        if len(item_1a) != r.item_1a_char_count or len(rest) != r.rest_char_count:
            bad.append(r.accession)
        chars += len(item_1a) + len(rest)
    if bad:
        raise ValueError(f"stored text no longer matches its recorded character count for "
                         f"{len(bad)} document(s): {bad[:5]}")
    return {"rows": len(back), "sampled": len(sample), "chars_verified": chars}


def completed_accessions() -> set[str]:
    """Accessions needing no further work: a metadata row AND its text both present.

    The intersection rather than either alone. A row without its document would have the
    resume skip a filing the scorer then cannot read; a document without its row would leave
    the filing absent from every downstream join.
    """
    if not CLEAN_OUT.exists():
        return set()
    logged = set(read_clean_filings()["accession"])
    return logged & stored_accessions()


def main() -> pd.DataFrame | None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true",
                    help="report done/remaining and exit without cleaning anything")
    args = ap.parse_args()

    if not LOG_CSV.exists():
        raise FileNotFoundError(f"{LOG_CSV.name} not found; run edgar_pull.py first.")
    log = pd.read_csv(LOG_CSV, dtype=str).fillna("")
    done = completed_accessions()
    # Resolved once - load_filings stats every cached path, so calling it twice to report a
    # total would double that work over 31,000 rows.
    corpus = load_filings(log)
    filings = corpus[~corpus["accession"].isin(done)]
    n_docs = filings["accession"].nunique()

    print(f"Corpus {corpus['accession'].nunique():,} documents on disk | "
          f"already clean {len(done):,} | to clean {n_docs:,} "
          f"({len(filings):,} firm-document rows)")
    if args.status:
        return None
    if not n_docs:
        print("Nothing to clean.")
        return None

    stale = clear_partials()
    if stale:
        print(f"Cleared {stale} partial file(s) from an interrupted run.")
    edgar_pull._prevent_sleep()
    print()

    fh, writer = open_clean_log()
    try:
        df = clean_all(filings, writer, fh)
    finally:
        fh.close()

    full = read_clean_filings()
    print(f"\nWrote {CLEAN_OUT.relative_to(BASE)} (+{len(df):,} rows this run, "
          f"{len(full):,} total x {len(full.columns)} cols, "
          f"{CLEAN_OUT.stat().st_size / 1e6:,.1f} MB).")
    store_gb = sum(p.stat().st_size for p in CLEAN_TEXT_DIR.glob("*.json.gz")) / 1e9
    print(f"Text store {CLEAN_TEXT_DIR.name}/: "
          f"{full['accession'].nunique():,} documents, {store_gb:,.2f} GB gzipped.")
    rt = verify_roundtrip(df)
    print(f"Round-trip verified: {rt['rows']:,} metadata rows, cik padding preserved, "
          f"{rt['sampled']:,} documents re-read ({rt['chars_verified']:,} chars intact).")
    write_diagnostics(full)
    validate(full)
    return df


if __name__ == "__main__":
    main()
