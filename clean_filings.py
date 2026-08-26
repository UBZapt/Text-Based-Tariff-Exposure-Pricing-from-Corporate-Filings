"""
Step 2a — Clean raw EDGAR 10-K HTML into Item 1A and rest-of-document text blocks.

Reads the cached filings listed in edgar_pull_log.csv, strips HTML/XBRL markup,
financial tables and repeated page furniture, and splits each filing's prose into
Item 1A ("Risk Factors") and everything else. Paragraph boundaries are preserved so
the downstream sentence tokeniser is not corrupted.

Cleaning only: no sentence splitting, tokenisation, lowercasing, stopword removal,
deduplication, bigram or keyword logic (those belong to the tokenisation and scoring
steps). Runs standalone:

    python clean_filings.py
"""

import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL filings carry an XML declaration but are parsed as HTML deliberately:
# the HTML tree builder is what tolerates the malformed markup these documents contain.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
LOG_CSV = BASE / "edgar_pull_log.csv"
FILINGS_DIR = BASE / "filings_raw"     # must match edgar_pull.FILINGS_DIR
CLEAN_OUT = BASE / "clean_filings.parquet"

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
MARKUP_ARTIFACT_RE = re.compile(r"<[a-zA-Z/!]|&nbsp;|&#\d+;|&amp;|&lt;|&gt;|&quot;")
# A lone-letter line signals a word split across inline spans. Roman numerals are
# excluded: front-matter page numbers ("i") are legitimate content, not shattering.
SHATTERED_WORD_RE = re.compile(r"(?m)^(?![ivxIVX]$)[A-Za-z]$")

OUTPUT_COLUMNS = [
    "permno", "cik", "accession", "form_type", "filing_date", "period_of_report",
    "fiscal_year", "item_1a_text", "rest_text", "item_1a_found", "item_1a_char_count",
    "rest_char_count", "clean_char_count", "cleaning_flags", "source_path",
]


# --------------------------------------------------------------------------- #
# Input selection                                                             #
# --------------------------------------------------------------------------- #
def load_filings(log: pd.DataFrame) -> pd.DataFrame:
    """Select successfully-pulled filings whose cached HTML is present on disk.

    Takes the already-read log so callers holding it do not re-read the file. Log
    rows whose file was pruned (firm no longer in the universe) are skipped. Files are
    resolved by name under FILINGS_DIR so log rows written on either OS resolve alike.
    """
    ok = log[log["found_10k"].astype(str).str.strip().str.lower() == "true"].copy()
    ok["source_path"] = ok["local_path"].astype(str).str.replace("\\", "/", regex=False)
    ok["path"] = [FILINGS_DIR / p.rsplit("/", 1)[-1] for p in ok["source_path"]]
    present = ok[[p.exists() for p in ok["path"]]].copy()
    if present.empty:
        raise FileNotFoundError(
            f"No cached filings found on disk from {LOG_CSV.name}; run edgar_pull.py first.")
    return present


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
def clean_all(filings: pd.DataFrame) -> pd.DataFrame:
    """Clean and split every filing, returning one row per filing."""
    rows = []
    for f in filings.itertuples(index=False):
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

        period = str(f.period_of_report)
        rows.append({
            "permno": f.permno,
            "cik": f.cik,
            "accession": f.accession,
            "form_type": FORM_TYPE,
            "filing_date": f.filing_date,
            "period_of_report": period,
            "fiscal_year": int(period[:4]) if period[:4].isdigit() else None,
            "item_1a_text": item_1a,
            "rest_text": rest,
            "item_1a_found": found,
            "item_1a_char_count": len(item_1a),
            "rest_char_count": len(rest),
            "clean_char_count": len(text),
            "cleaning_flags": "; ".join(flags),
            "source_path": f.source_path,
        })
        print(f"  {f.permno:<7} cik={f.cik} "
              f"{'1A ' + format(len(item_1a), ',') + ' ch' if found else 'ITEM 1A NOT FOUND':<18} "
              f"rest={len(rest):,} ch")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame) -> dict:
    """Run the required output checks and print the report.

    Covers residual markup, Item 1A detection overall and by fiscal year, short
    Item 1A sections, outright cleaning failures, and a regression guard on the
    inline-span word-shattering that a naive text extraction produces.
    """
    def identities(sub):
        return [(r.cik, r.accession) for r in sub.itertuples(index=False)]

    both = df["item_1a_text"] + "\n" + df["rest_text"]
    artifacts = df[both.str.contains(MARKUP_ARTIFACT_RE, regex=True)]
    failed = df[df["cleaning_flags"].str.startswith("cleaning_failed")]
    missing = df[~df["item_1a_found"]]
    short = df[df["cleaning_flags"].str.contains("short_item_1a")]
    shattered = int(both.str.count(SHATTERED_WORD_RE).sum())
    empty_rest = df[df["rest_char_count"] == 0]

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"Filings cleaned            : {len(df)}")
    print(f"Failed cleaning entirely   : {len(failed)} {identities(failed) or ''}")
    print(f"Residual markup artifacts  : {len(artifacts)} {identities(artifacts) or ''}")
    print(f"Shattered-word artifacts   : {shattered} lone-letter lines")
    print(f"Empty rest_text            : {len(empty_rest)} {identities(empty_rest) or ''}")
    print(f"Item 1A located            : {len(df) - len(missing)}/{len(df)} "
          f"({(df['item_1a_found'].mean() if len(df) else 0):.1%})")
    if len(missing):
        print("  not found:")
        for cik, acc in identities(missing):
            print(f"    cik={cik} accession={acc}")
    print(f"Short Item 1A (<{SHORT_ITEM_1A_CHARS:,} ch) : {len(short)} {identities(short) or ''}")

    by_year = df.groupby("fiscal_year")["item_1a_found"].agg(["sum", "count"])
    print("\nItem 1A detection by fiscal year:")
    for year, r in by_year.iterrows():
        print(f"  FY{year}: {int(r['sum'])}/{int(r['count'])}")
    print("  (single-vintage sample: the 2018-era EDGAR format is not represented "
          "in this cache, so vintage robustness is untested)")

    if len(df):
        print(f"\nItem 1A chars: median {df.loc[df['item_1a_found'], 'item_1a_char_count'].median():,.0f}"
              f" | rest chars: median {df['rest_char_count'].median():,.0f}")
    return {
        "n_filings": len(df), "n_failed": len(failed), "n_artifacts": len(artifacts),
        "n_missing_item_1a": len(missing), "n_short": len(short),
        "n_shattered": shattered, "by_fiscal_year": by_year,
    }


def main() -> pd.DataFrame:
    if not LOG_CSV.exists():
        raise FileNotFoundError(f"{LOG_CSV.name} not found; run edgar_pull.py first.")
    log = pd.read_csv(LOG_CSV, dtype=str).fillna("")
    filings = load_filings(log)
    print(f"Cleaning {len(filings)} cached filings from {LOG_CSV.name}\n")

    df = clean_all(filings)
    df.to_parquet(CLEAN_OUT, index=False)
    print(f"\nWrote {CLEAN_OUT.name} ({len(df)} rows x {len(df.columns)} cols).")
    validate(df)
    return df


if __name__ == "__main__":
    main()
