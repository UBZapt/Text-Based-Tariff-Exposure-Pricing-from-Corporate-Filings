"""
Step 2b - Sentence tokenisation (Stage 2) and tariff-bigram scoring (Stage 3).

Reads the cleaned filings table, splits each filing's Item 1A and rest-of-document text
into sentences, matches a fixed tariff term list against every sentence, and writes three
raw exposure scores per filing.

Measure formula (section 7.1), with no per-term relevance weight:

    TExp = (sentences containing >= 1 tariff term) / (total sentences in that unit)

This is the sentence-level, length-scaled construction Hassan et al. (2019) apply to political
risk, applied here to a fixed tariff and trade-policy term list; full reference in README.md.

The two stages are deliberately separate: ``tokenise_filing`` holds no matching logic and
``score_section`` holds no sentence-splitting logic. They are driven from one streaming loop
(``score_stream``) so a filing's sentences are discarded once scored - the corpus is ~26
million sentences and cannot be materialised.

Keyed on the document. Scores are raw only; the cross-sectional standardisation belongs to a
reference date, which this table has no column for, so it lives in build_texp_panel.py. Rows
are flushed as each filing finishes, so an interrupted run resumes.

    python src/score_filings.py                  # whole universe
    python src/score_filings.py --n-ciks 50      # first N distinct CIKs by PERMNO
    python src/score_filings.py --status         # done/remaining, no work
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import nltk
import pandas as pd

import clean_filings
import edgar_pull
import run_report
from config import BASE, CLEAN_DIR, INTERMEDIATE_DIR, LEXICON as BIGRAM_JSON, OUTPUT_DIR

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
SCORES_OUT = CLEAN_DIR / "tariff_scores.csv"
# The cleaned-filings path and its read dtypes are owned by clean_filings, which writes it.
DIAG_OUT = INTERMEDIATE_DIR / "scoring_diagnostics.csv"
TERM_HITS_OUT = OUTPUT_DIR / "term_hits.csv"
REPORT_OUT = OUTPUT_DIR / "scoring_validation_report.txt"

# Contemporaneity is NOT enforced here any more. Scoring is a property of the document, so it
# runs once per accession and cannot know which reference date will use the result; the
# corpus now spans FY2016-FY2025 across 17 reference dates. The rule moved to
# build_texp_panel.MAX_PERIOD_STALENESS_MONTHS, where the reference date is known and the
# filter can be applied per cross-section. The Item 1A filter below stays, because it is a
# property of the document rather than of the date.

MIN_SENTENCE_CHARS = 15      # below this a "sentence" is likely a tokenisation artifact
IMPLAUSIBLE_TEXP = 0.5       # tripwire for double-counting or boilerplate artifacts
SPOT_CHECK_FILINGS = 3       # filings whose matched sentences are printed
SPOT_CHECK_SENTENCES = 4     # matched sentences retained per section per filing
SENTENCE_PRINT_CHARS = 260   # truncation when printing a matched sentence
MAX_LISTED = 10              # identities printed before deferring to the audit CSV
PROGRESS_EVERY = 250         # filings between tokenisation progress lines

# Cut from the term list after the pilot, retained here as an audit only: they never enter a
# score. 'excise tax' overwhelmingly matched the Inflation Reduction Act's 1% buyback excise
# tax rather than trade policy; 'regulatory uncertainty' is not tariff-specific.
REMOVED_TERMS = ["excise tax", "regulatory uncertainty"]

# "Tariff" is also the regulated rate schedule of a utility. Counted as a diagnostic so the
# measurement-validity risk is visible; deliberately not filtered out of the score.
UTILITY_CONTEXT_RE = re.compile(
    r"(?i)utilit|interconnect|ratepayer|rate schedule|public service commission|regulated rate")

EN_DASH, EM_DASH = chr(0x2013), chr(0x2014)
DASH_CLASS = "[-" + EN_DASH + EM_DASH + "]"

# Dash variants of the one list term that carries an en dash, counted during the scoring pass
# so the diagnostic costs no extra read of the corpus. Compiled once at import.
EN_DASH_TERM = "U.S." + EN_DASH + "China tariffs"
EN_DASH_VARIANTS = [
    ("exact, U+2013 en dash", r"U\.S\." + EN_DASH + r"China tariffs"),
    ("hyphen variant  U.S.-China tariffs", r"U\.S\.-China tariffs"),
    ("U.S.<any dash>China, any context", r"U\.S\." + DASH_CLASS + r"China"),
    ("US<any dash>China, any context", r"(?<!\w)US" + DASH_CLASS + r"China"),
    ("China tariffs, loose", r"China\s+tariffs"),
]
EN_DASH_COMPILED = [(label, re.compile(p, re.IGNORECASE)) for label, p in EN_DASH_VARIANTS]

TEXP_COLUMNS = ["TExp_item1a", "TExp_rest", "TExp_combined"]
# Raw scores only. The standardised TExp_*_z columns this table used to carry were pooled over
# the whole file, which is meaningful for one reference date and meaningless across 17: the
# cross-section a z-score belongs to is a reference date, which this table does not have a
# column for. They now live in build_texp_panel.py, standardised within each date.
OUTPUT_COLUMNS = [
    "permno", "cik", "accession", "filing_date", "fiscal_year",
    "B_item1a", "B_rest", "bigram_hit_count_item1a", "bigram_hit_count_rest",
    "utility_context_hits",
    "TExp_item1a", "TExp_rest", "TExp_combined", "scoring_flags",
]


class SectionScore(NamedTuple):
    """Result of scoring one section's sentences."""
    n_hit: int                 # sentences containing >= 1 term (binary per sentence)
    per_term: Counter          # term -> sentences it appeared in (terms overlap; see notes)
    examples: list             # up to SPOT_CHECK_SENTENCES (sentence, matched terms) pairs
    n_multi_term: int          # hit sentences matching more than one term
    n_utility: int             # tariff hits sitting in utility-rate context (diagnostic only)


def _printable(text: str) -> str:
    """Make text safe for a legacy Windows console codepage.

    Filing prose carries en dashes and other non-cp1252 characters; printing them raw can
    raise UnicodeEncodeError and kill the validation report mid-run.
    """
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


# --------------------------------------------------------------------------- #
# Setup and input                                                             #
# --------------------------------------------------------------------------- #
def ensure_punkt() -> str:
    """Confirm the NLTK sentence tokenizer data is present, downloading it if missing.

    NLTK >= 3.9 serves sent_tokenize from 'punkt_tab'; earlier versions use 'punkt'. Both
    are probed so the script works on either, and the resource actually used is returned
    so the run record is unambiguous.
    """
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return resource
        except LookupError:
            pass
    for resource in ("punkt_tab", "punkt"):
        if nltk.download(resource, quiet=True):
            print(f"[info] downloaded NLTK '{resource}' tokenizer data.")
            return resource
    raise RuntimeError("Could not obtain NLTK punkt data; check network access.")


def load_bigrams(path: Path) -> list[str]:
    """Load the fixed term list from its standalone config file.

    Kept out of this module so the list can be edited without touching pipeline code.
    Despite the name it is a mixed term list, not strictly bigrams: 'tariff', 'tariffs',
    'levies' and 'protectionism' are single words.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found; the term list is a required input.")
    terms = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(terms, list) or not terms:
        raise ValueError(f"{path.name} must contain a non-empty JSON array of terms.")
    if len(set(terms)) != len(terms):
        dupes = [t for t, n in Counter(terms).items() if n > 1]
        raise ValueError(f"{path.name} contains duplicate terms: {dupes}")
    stale = sorted(set(terms) & set(REMOVED_TERMS))
    if stale:
        raise ValueError(f"{path.name} still contains removed term(s) {stale}; they are "
                         f"audited in validation and must not be scored.")
    return terms


def compile_patterns(terms: list[str]) -> dict[str, re.Pattern]:
    """Compile one case-insensitive, word-bounded regex per term.

    Lookarounds are used rather than \\b so terms ending in a digit or punctuation behave
    uniformly ('Section 232' must not match 'Section 2321'). Internal spaces become \\s+ so
    a phrase split across a line break still matches. Word bounding is what keeps 'tariff'
    from also matching inside 'tariffs' - the substring double-counting to avoid.
    """
    return {
        term: re.compile(
            r"(?<!\w)" + r"\s+".join(re.escape(word) for word in term.split()) + r"(?!\w)",
            re.IGNORECASE)
        for term in terms
    }


def build_gates(terms: list[str]) -> dict[str, tuple[str, ...]]:
    """Pre-compute the literal words each term requires, as a cheap pre-filter for matching.

    ``compile_patterns`` makes only *internal whitespace* flexible; every other character is
    escaped and matches literally. So if a term matches a sentence, each of its lowercased
    words must appear literally in that sentence - a necessary condition a substring test
    checks about six times faster than running the regex, over a corpus where ~97% of
    sentences match nothing. Derived from the term list itself, so it cannot drift out of
    sync the way a hand-maintained keyword list would. Gating changes speed, never results:
    passing empty tuples disables it and must reproduce identical scores.
    """
    return {term: tuple(word.lower() for word in term.split()) for term in terms}


def resolve_scope_ciks(n_ciks: int | None) -> set[str]:
    """Resolve this run's CIK universe, reusing edgar_pull's ordering rather than re-deriving it.

    With ``n_ciks=None`` this is every CIK anywhere in the bridge and the filter acts as a
    sanity check - a cleaned filing should never fall outside the universe the pull was drawn
    from. It must NOT be the set resolved at one reference date: _resolve_cik picks one link
    row per date, so a firm linked in 2018 but not in 2025 would be dropped as
    'cik_outside_universe' purely because of when the scope was evaluated. This is the same
    trap edgar_pull.prune_stale guards against, which is why bridge_ciks() exists.
    """
    if n_ciks is None:
        return edgar_pull.bridge_ciks()
    firms = edgar_pull.resolve_firms(edgar_pull.BRIDGE_CSV, edgar_pull.REFERENCE_DATE)
    scope, _ = edgar_pull.select_gap_firms(firms, set(), n_ciks)
    return {f["cik"] for f in scope if f["cik"]}


def load_scope(filings: pd.DataFrame, scope_ciks: set[str],
               done: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the sample filters in order, keeping every exclusion attributable.

    1. CIK inside the pull universe - a sanity check at full scope, a real filter under
       ``--n-ciks``.
    2. Item 1A located - without it there is no Item 1A denominator, and ``rest_text`` holds
       the whole document rather than the complement of Item 1A, so neither section score is
       comparable with the rest of the panel.

    The fiscal-year filter that used to sit between these two has moved to the panel step; see
    the note where FISCAL_YEARS was defined.

    Returns (kept, dropped); ``dropped`` carries a ``drop_reason`` column and is reported by
    the caller, never silently lost. ``done`` accessions are removed from ``kept`` after the
    filters run, so the reported funnel still describes the whole corpus on a resumed run.
    """
    stages, remaining = [], filings

    in_scope = remaining["cik"].isin(scope_ciks)
    stages.append(remaining[~in_scope].assign(drop_reason="cik_outside_universe"))
    remaining = remaining[in_scope]

    found = remaining["item_1a_found"]
    stages.append(remaining[~found].assign(drop_reason="item_1a_not_found"))
    kept = remaining[found].copy()

    if kept.empty:
        raise ValueError(f"No filings in {clean_filings.CLEAN_OUT.name} survive the scope "
                         f"and Item 1A filters.")
    if done:
        kept = kept[~kept["accession"].isin(done)]
    return kept, pd.concat(stages, ignore_index=True)


# --------------------------------------------------------------------------- #
# Stage 2 - tokenisation (no bigram logic in this section)                    #
# --------------------------------------------------------------------------- #
def tokenise_filing(item_1a_text: str, rest_text: str) -> tuple[list[str], list[str]]:
    """Split one filing's two sections into sentences, independently.

    sent_tokenize is run separately on item_1a_text and rest_text - never on a
    concatenation - so the two sentence lists and their counts B_item1a / B_rest are
    genuinely per section. Because the sections are tokenised apart, B_item1a + B_rest can
    differ by a sentence or two from tokenising the whole document, since the section
    boundary is also a forced sentence boundary.

    Holds no matching logic: this function never sees the term list. It returns one filing at
    a time rather than the whole corpus because the full corpus is ~26 million sentence
    strings, which cannot be held in memory alongside everything else the run needs.
    """
    return nltk.sent_tokenize(item_1a_text), nltk.sent_tokenize(rest_text)


# --------------------------------------------------------------------------- #
# Stage 3 - bigram scoring (no tokenisation logic in this section)            #
# --------------------------------------------------------------------------- #
def score_section(sentences: list[str], patterns: dict[str, re.Pattern],
                  gates: dict[str, tuple[str, ...]],
                  max_examples: int = SPOT_CHECK_SENTENCES) -> SectionScore:
    """Count sentences containing at least one term, and attribute hits per term.

    A sentence contributes exactly one hit however many terms or repetitions it holds
    (sentence-level binary, matching the indicator in the section 7.1 formula). Per-term
    counts are attributed independently, so a sentence reading "retaliatory tariffs"
    increments 'retaliatory tariffs' and 'tariffs' both; those counts therefore overlap by
    construction and do not sum to n_hit. That is deliberate - the per-term table exists to
    catch matching bugs, and longest-match-only attribution would drive 'tariff' to near
    zero and look like one.

    ``gates`` is a speed pre-filter only (see build_gates); it cannot change which sentences
    match. The utility-context count is a diagnostic and does not affect any score.
    """
    n_hit = n_multi_term = n_utility = 0
    per_term: Counter = Counter()
    examples: list = []
    for sentence in sentences:
        low = sentence.lower()
        matched = [term for term, pattern in patterns.items()
                   if all(word in low for word in gates[term]) and pattern.search(sentence)]
        if not matched:
            continue
        n_hit += 1
        per_term.update(matched)
        if len(matched) > 1:
            n_multi_term += 1
        if any("tariff" in term for term in matched) and UTILITY_CONTEXT_RE.search(sentence):
            n_utility += 1
        if len(examples) < max_examples:
            examples.append((sentence, matched))
    return SectionScore(n_hit, per_term, examples, n_multi_term, n_utility)


def open_scores_log():
    """Open tariff_scores.csv for append, writing the header if it is new.

    Mirrors clean_filings.open_clean_log and edgar_pull.open_log: one flush per filing, so an
    interrupted scoring run keeps every row it produced and re-running resumes.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not SCORES_OUT.exists() or SCORES_OUT.stat().st_size == 0
    fh = SCORES_OUT.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def score_stream(kept: pd.DataFrame, patterns: dict[str, re.Pattern],
                 gates: dict[str, tuple[str, ...]], writer, fh) -> tuple[pd.DataFrame, dict]:
    """Tokenise and score each filing in one streaming pass, flushing rows as they finish.

    Stages 2 and 3 remain separate in logic - tokenise_filing never sees the term list and
    score_section never splits sentences - but they are driven from a single loop so a
    filing's sentences are discarded as soon as it is scored. The previous shape materialised
    every sentence of the corpus in a list before scoring any of it, which at 26 million
    sentences cannot fit in memory.

    Three corpus diagnostics that used to require their own full pass over the text are
    accumulated here instead: the removed-term audit, the short-sentence tally and the
    en-dash variant counts.

    Zero denominators are handled explicitly: a section with no sentences gets NaN rather
    than a division, and is flagged. TExp_combined is the pooled document rate, so one
    empty section still yields a combined score from the other; it is NaN only when the
    whole document is empty.
    """
    rows, examples = [], {}
    term_1a: Counter = Counter()
    term_rest: Counter = Counter()
    term_filings: Counter = Counter()
    n_multi_term = n_utility = 0
    b_1a_total = b_rest_total = 0

    removed_patterns = compile_patterns(REMOVED_TERMS)
    removed_gates = build_gates(REMOVED_TERMS)
    removed_tally = {term: {"sentences": 0, "filings": 0} for term in REMOVED_TERMS}
    short = {"item1a": [0, 0], "rest": [0, 0]}      # [n_all, n_short]
    short_examples: list[str] = []
    en_counts = {label: 0 for label, _ in EN_DASH_VARIANTS}
    n_en = n_em = 0

    total, t0 = len(kept), time.monotonic()
    for i, f in enumerate(kept.itertuples(index=False), 1):
        item_1a_text, rest_text = clean_filings.read_clean_text(f.accession)
        sentences_item1a, sentences_rest = tokenise_filing(item_1a_text, rest_text)

        s_1a = score_section(sentences_item1a, patterns, gates)
        s_rest = score_section(sentences_rest, patterns, gates)
        b_1a, b_rest = len(sentences_item1a), len(sentences_rest)
        b_total = b_1a + b_rest
        b_1a_total += b_1a
        b_rest_total += b_rest

        # --- diagnostics folded into this pass ---
        seen: Counter = Counter()
        for sentence in sentences_item1a + sentences_rest:
            low = sentence.lower()
            for term in REMOVED_TERMS:
                if (all(w in low for w in removed_gates[term])
                        and removed_patterns[term].search(sentence)):
                    seen[term] += 1
        for term, n in seen.items():
            removed_tally[term]["sentences"] += n
            removed_tally[term]["filings"] += 1

        for key, sentences in (("item1a", sentences_item1a), ("rest", sentences_rest)):
            short[key][0] += len(sentences)
            brief = [s for s in sentences if len(s.strip()) < MIN_SENTENCE_CHARS]
            short[key][1] += len(brief)
            if len(short_examples) < 8:
                short_examples += brief[:2]

        both = item_1a_text + "\n" + rest_text
        n_en += both.count(EN_DASH)
        n_em += both.count(EM_DASH)
        for label, pattern in EN_DASH_COMPILED:
            en_counts[label] += len(pattern.findall(both))

        flags = []
        if b_1a == 0:
            flags.append("empty_item_1a_section")
        if b_rest == 0:
            flags.append("empty_rest_section")
        if b_total == 0:
            flags.append("empty_document")

        texp_1a = s_1a.n_hit / b_1a if b_1a else float("nan")
        texp_rest = s_rest.n_hit / b_rest if b_rest else float("nan")
        texp_combined = (s_1a.n_hit + s_rest.n_hit) / b_total if b_total else float("nan")
        for name, value in [("TExp_item1a", texp_1a), ("TExp_rest", texp_rest),
                            ("TExp_combined", texp_combined)]:
            if pd.notna(value) and value > IMPLAUSIBLE_TEXP:
                flags.append(f"implausible_{name}={value:.3f}")

        term_1a.update(s_1a.per_term)
        term_rest.update(s_rest.per_term)
        term_filings.update(set(s_1a.per_term) | set(s_rest.per_term))
        n_multi_term += s_1a.n_multi_term + s_rest.n_multi_term
        n_utility += s_1a.n_utility + s_rest.n_utility
        examples[f.accession] = {"item_1a": s_1a.examples, "rest": s_rest.examples}

        row = {
            "permno": f.permno, "cik": f.cik, "accession": f.accession,
            "filing_date": f.filing_date, "fiscal_year": f.fiscal_year,
            "B_item1a": b_1a, "B_rest": b_rest,
            "bigram_hit_count_item1a": s_1a.n_hit,
            "bigram_hit_count_rest": s_rest.n_hit,
            "utility_context_hits": s_1a.n_utility + s_rest.n_utility,
            "TExp_item1a": texp_1a, "TExp_rest": texp_rest,
            "TExp_combined": texp_combined,
            "scoring_flags": "; ".join(flags),
        }
        writer.writerow(row)
        fh.flush()
        rows.append(row)

        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = time.monotonic() - t0
            print(f"[progress] {i:,}/{total:,} ({i / total:.1%}) | "
                  f"{b_1a_total + b_rest_total:,} sentences | "
                  f"{elapsed / 60:.1f} min elapsed | "
                  f"ETA {(total - i) * elapsed / i / 60:.0f} min", flush=True)

    scores = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    total_hits = int(scores["bigram_hit_count_item1a"].sum()
                     + scores["bigram_hit_count_rest"].sum())
    print(f"\nStage 2 - tokenised {len(rows):,} filings into "
          f"{b_1a_total + b_rest_total:,} sentences "
          f"({b_1a_total:,} Item 1A, {b_rest_total:,} rest).")
    print(f"Stage 3 - {total_hits:,} hit sentences across the sample "
          f"({n_multi_term:,} matched more than one term).\n")
    audit = pd.DataFrame([{"term": t, **v} for t, v in removed_tally.items()])
    return scores, {"term_1a": term_1a, "term_rest": term_rest,
                    "term_filings": term_filings, "examples": examples,
                    "n_multi_term": n_multi_term, "n_utility": n_utility,
                    "total_hits": total_hits, "audit": audit, "short": short,
                    "short_examples": short_examples,
                    "en_counts": en_counts, "n_en": n_en, "n_em": n_em}


# --------------------------------------------------------------------------- #
# Outputs                                                                     #
# --------------------------------------------------------------------------- #
def write_diagnostics(scores: pd.DataFrame, dropped: pd.DataFrame) -> Path:
    """Write one row per in-scope filing: every scored filing plus every drop and its reason.

    At full-corpus scale the console cannot carry one line per dropped or flagged filing, so
    the identities live here and the console reports counts plus the first few.
    """
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["permno", "cik", "accession", "filing_date", "fiscal_year", "drop_reason"]
    diag = pd.concat([scores.assign(drop_reason=""), dropped.reindex(columns=keys)],
                     ignore_index=True)
    diag.to_csv(DIAG_OUT, index=False)
    print(f"Wrote {DIAG_OUT.relative_to(BASE)} ({len(diag):,} rows) - per-filing scoring audit.")
    return DIAG_OUT


def write_term_hits(terms: list[str], diag: dict) -> Path:
    """Write the per-term hit table so zero-hit and runaway terms are auditable after the run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "term": t,
        "sentences_item1a": diag["term_1a"][t],
        "sentences_rest": diag["term_rest"][t],
        "sentences_total": diag["term_1a"][t] + diag["term_rest"][t],
        "filings": diag["term_filings"][t],
    } for t in terms]).sort_values("sentences_total", ascending=False)
    df.to_csv(TERM_HITS_OUT, index=False)
    print(f"Wrote {TERM_HITS_OUT.relative_to(BASE)} ({len(df)} terms).")
    return TERM_HITS_OUT


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def _listed(frame: pd.DataFrame) -> str:
    """Format filing identities for the console, capped so a full-corpus run stays readable."""
    if frame.empty:
        return ""
    ids = [f"{r.cik}/{r.accession}" for r in frame.head(MAX_LISTED).itertuples(index=False)]
    more = len(frame) - len(ids)
    return " ".join(ids) + (f" (+{more} more, see {DIAG_OUT.name})" if more else "")


def _pull_coverage() -> dict:
    """Upstream corpus counts, read from the pull log and cache for the funnel check only."""
    log = pd.read_csv(edgar_pull.LOG_CSV, dtype=str).fillna("")
    ok = log["found_10k"].astype(str).str.strip().str.lower() == "true"
    return {"log_success": int(ok.sum()),
            "files_on_disk": len(list(edgar_pull.FILINGS_DIR.glob("*.html")))}


def _check_funnel(filings: pd.DataFrame, kept: pd.DataFrame, dropped: pd.DataFrame,
                  scope_ciks: set[str], done: set[str]) -> None:
    """Reconcile the corpus end to end, asserting the arithmetic rather than only printing it.

    Starts at the pull rather than at the cleaned table so any silent loss between steps is
    visible in one place.

    ``done`` is counted in cleaned-table ROWS, not as the size of the accession set: a joint
    filing carried by two PERMNOs is one accession but two rows, so comparing a set size
    against a row count would leave the identity one short and fail a correct run.
    """
    cov = _pull_coverage()
    n_done = int(filings["accession"].isin(done).sum())
    print("Corpus funnel (every exclusion attributed):")
    for label, n in [("universe CIKs (bridge)", len(scope_ciks)),
                     ("pull log successes", cov["log_success"]),
                     ("cached files on disk", cov["files_on_disk"]),
                     ("cleaned filings", len(filings))]:
        print(f"  {label:34s} {n:>7,}")
    for reason, n in dropped["drop_reason"].value_counts().items():
        print(f"    - dropped {reason:24s} {n:>7,}")
    if n_done:
        print(f"    - already scored{'':18s} {n_done:>7,}")
    print(f"  {'= scored this run':34s} {len(kept):>7,}")

    if len(kept) + len(dropped) + n_done != len(filings):
        raise ValueError(f"funnel does not reconcile: {len(kept):,} scored + "
                         f"{len(dropped):,} dropped + {n_done:,} already scored "
                         f"!= {len(filings):,} cleaned filings")
    print(f"  reconciles: {len(kept):,} + {len(dropped):,} + {n_done:,} = {len(filings):,}")


def _check_drop_reconciliation(dropped: pd.DataFrame) -> None:
    """Cross-check the three fields that independently record a missing Item 1A."""
    no_1a = dropped[dropped["drop_reason"] == "item_1a_not_found"]
    by_flag = set(no_1a.loc[no_1a["cleaning_flags"].str.contains("item_1a_not_found"),
                            "accession"])
    by_found = set(no_1a.loc[~no_1a["item_1a_found"], "accession"])
    by_chars = set(no_1a.loc[no_1a["item_1a_char_count"] == 0, "accession"])
    agree = by_flag == by_found == by_chars

    print(f"\nFilings dropped for missing Item 1A : {len(no_1a):,} {_listed(no_1a)}")
    print(f"  cross-check (item_1a_found / cleaning_flags / item_1a_char_count): "
          f"{'all three agree' if agree else '!! DISAGREEMENT'}")
    if not agree:
        print(f"  !! by_found={len(by_found)} by_flag={len(by_flag)} by_chars={len(by_chars)}; "
              f"symmetric differences {sorted(by_found ^ by_flag)[:5]} "
              f"{sorted(by_found ^ by_chars)[:5]}")


def _check_per_term(terms: list[str], diag: dict, n_filings: int) -> None:
    """Report per-term hit frequency and surface any term that never matched."""
    print("\nPer-term hit frequency (sentences; terms overlap by construction, so the")
    print("column does not sum to the hit total - see the nested-match note in the docstring):")
    print(f"  {'term':36s} {'item1a':>9} {'rest':>9} {'total':>9} {'filings':>8}")
    rows = [(t, diag["term_1a"][t], diag["term_rest"][t]) for t in terms]
    for term, n_1a, n_rest in sorted(rows, key=lambda r: -(r[1] + r[2])):
        print(f"  {_printable(term):36s} {n_1a:>9,} {n_rest:>9,} {n_1a + n_rest:>9,} "
              f"{diag['term_filings'][term]:>8,}")

    zero = [t for t, n_1a, n_rest in rows if n_1a + n_rest == 0]
    print(f"\n  Terms with ZERO hits across all {n_filings:,} filings: {len(zero)}")
    for term in zero:
        print(f"    {_printable(term)}")
    if zero:
        print("    (zero hits may be genuine absence or a matching bug - the en-dash")
        print("     diagnostic below resolves that question for the one term at risk)")


def _check_removed_terms(audit: pd.DataFrame, total_hits: int) -> None:
    """Report the full-corpus footprint of the terms cut from the list."""
    print("\nRemoved-term audit (scored by NO measure; reported so the cut is evidenced):")
    for r in audit.itertuples(index=False):
        print(f"  {_printable(r.term):24s} {r.sentences:>8,} sentences in {r.filings:>6,} filings"
              f"  (= {r.sentences / total_hits:.1%} of the {total_hits:,} retained hits)")


def _check_utility_context(scores: pd.DataFrame, diag: dict) -> None:
    """Quantify the utility-rate meaning of 'tariff', a measurement-validity risk at scale.

    The corpus share understates the risk on its own: contamination concentrates in individual
    utility filers rather than spreading evenly, so the per-filing share among the highest
    scoring filings is what determines whether the long leg is picking up trade policy or
    rate-schedule prose.
    """
    n_util, total = diag["n_utility"], diag["total_hits"]
    print(f"\nUtility-rate context among tariff hits : {n_util:,} of {total:,} hit sentences "
          f"({n_util / total if total else 0:.1%})")
    hits = scores["bigram_hit_count_item1a"] + scores["bigram_hit_count_rest"]
    share = (scores["utility_context_hits"] / hits.where(hits > 0)).fillna(0)
    top = scores.assign(_share=share).nlargest(30, "TExp_combined")
    print(f"  filings >50% utility-context            : {int((share > 0.5).sum()):,} of "
          f"{len(scores):,} ({int((top['_share'] > 0.5).sum())} inside the top-30 by TExp_combined; "
          f"worst in top-30 {top['_share'].max():.0%})")
    print("  ('tariff' also denotes a regulated utility rate schedule; reported, not filtered)")


def _check_en_dash(diag: dict) -> None:
    """Test the en-dash term explicitly, separately from hyphen and spaced variants.

    Counts are accumulated during the scoring pass (see score_stream) rather than recomputed
    here: the text lives in the per-accession store, so a second pass would decompress the
    whole corpus again to answer one diagnostic.
    """
    counts, n_en, n_em = diag["en_counts"], diag["n_en"], diag["n_em"]
    variants = EN_DASH_VARIANTS

    print("\nEn-dash diagnostic for the list term 'U.S.<U+2013>China tariffs':")
    print(f"  sentence hits recorded for the exact list term : "
          f"{diag['term_1a'][EN_DASH_TERM] + diag['term_rest'][EN_DASH_TERM]:,}")
    print(f"  en dashes (U+2013) present in cleaned corpus   : {n_en:,}")
    print(f"  em dashes (U+2014) present in cleaned corpus   : {n_em:,}")
    print("  raw occurrences in the cleaned corpus:")
    for label, _ in variants:
        print(f"    {label:36s} {counts[label]:,}")

    if all(n == 0 for n in counts.values()) and n_en > 0:
        print("  VERDICT: genuine absence, not a matching failure - no dash variant of the")
        print("           phrase occurs at all, while en dashes do survive cleaning, so the")
        print("           matcher had something to match against and correctly found nothing.")
    elif counts["exact, U+2013 en dash"] == 0 and any(n > 0 for n in counts.values()):
        print("  VERDICT: !! the en-dash term misses text the corpus DOES contain in another")
        print("           form - the list term should be widened to the variant(s) above.")
    else:
        print("  VERDICT: the en-dash term matches; no action needed.")


def _check_distributions(scores: pd.DataFrame) -> None:
    """Report distribution stats and flag implausible exposure rates."""
    print("\nDistribution across the sample:")
    print(f"  {'measure':22s} {'n':>6} {'min':>12} {'median':>12} {'mean':>12} {'max':>12}")
    for col in ["B_item1a", "B_rest"] + TEXP_COLUMNS:
        s = scores[col].dropna()
        fmt = ",.0f" if col.startswith("B_") else ".5f"
        print(f"  {col:22s} {len(s):>6,} {s.min():>12{fmt}} {s.median():>12{fmt}} "
              f"{s.mean():>12{fmt}} {s.max():>12{fmt}}")

    flagged = scores[scores["scoring_flags"].str.contains("implausible")]
    print(f"\n  Filings with any TExp > {IMPLAUSIBLE_TEXP} : {len(flagged):,} {_listed(flagged)}")
    if flagged.empty:
        print("    (none - expected: a keyword measure over full 10-K text yields rates of")
        print("     order 1e-3, so an empty list here is the correct result, not a failure)")

    empty = scores[scores["scoring_flags"].str.contains("empty_")]
    print(f"  Filings with an empty section (TExp set to NaN) : {len(empty):,} {_listed(empty)}")


def _check_dispersion(scores: pd.DataFrame) -> None:
    """Report whether the measure can support the section 7.3 decile sort.

    A large mass of exact zeros makes the bottom decile a tie rather than a ranked portfolio,
    so the zero share and the decile breakpoints are stated here rather than being discovered
    at portfolio-construction time.
    """
    print("\nCross-sectional dispersion (section 7.3 decile sort):")
    print(f"  {'measure':16s} {'n':>6} {'zero':>7} {'p10':>10} {'p50':>10} {'p90':>10}")
    for col in TEXP_COLUMNS:
        s = scores[col].dropna()
        print(f"  {col:16s} {len(s):>6,} {(s == 0).mean():>6.1%} {s.quantile(.1):>10.5f} "
              f"{s.median():>10.5f} {s.quantile(.9):>10.5f}")

    deciles = [i / 10 for i in range(1, 10)]
    for col in TEXP_COLUMNS:
        s = scores[col].dropna()
        bps = s.quantile(deciles)
        print(f"\n  {col} decile breakpoints:")
        print(f"    {[round(v, 5) for v in bps]}")
        if bps.iloc[0] == 0:
            print(f"    !! D1 breakpoint is 0.0 - the bottom decile is a tie among "
                  f"{int((s == 0).sum()):,} zero-exposure filings, not a ranked portfolio.")

    print("\n  correlation between measures:")
    corr = scores[TEXP_COLUMNS].corr().round(3).to_string()
    print("    " + corr.replace("\n", "\n    "))
    print("\n  (standardisation is not checked here: a z-score belongs to one cross-section, "
          "and\n   this table pools 17 reference dates. See build_texp_panel.py.)")


def _check_duplicate_accessions(scores: pd.DataFrame) -> None:
    """Flag accessions scored under more than one PERMNO.

    A parent and its subsidiary can file one joint 10-K, so the same document legitimately
    scores for two firms. Reported, not de-duplicated - both firms genuinely carry that
    disclosure - but the tied observations must be visible before any cross-sectional test.
    """
    dup = scores[scores["accession"].duplicated(keep=False)].sort_values("accession")
    print(f"\nAccessions scored under >1 permno : {dup['accession'].nunique()} "
          f"({len(dup)} rows; co-registrant joint filings, reported not de-duplicated)")
    for r in dup.head(MAX_LISTED).itertuples(index=False):
        print(f"  permno={r.permno:<7} cik={r.cik} accession={r.accession} "
              f"TExp_combined={r.TExp_combined:.5f}")


def _check_composition(scores: pd.DataFrame) -> None:
    """Report the scored panel's fiscal-year and filing-date spread."""
    fy = scores["fiscal_year"].value_counts().sort_index()
    fd = scores["filing_date"].astype(str).str[:4].value_counts().sort_index()
    print(f"\nScored panel composition: fiscal_year {fy.to_dict()} | "
          f"filing year {fd.to_dict()}")


def _check_spot_sentences(scores: pd.DataFrame, diag: dict) -> None:
    """Print matched sentences for the highest-hit filings so matches can be eyeballed."""
    top = scores.assign(_hits=scores["bigram_hit_count_item1a"]
                        + scores["bigram_hit_count_rest"]).nlargest(SPOT_CHECK_FILINGS, "_hits")
    print(f"\nSpot-check - matched sentences from the {len(top)} highest-hit filings:")
    for r in top.itertuples(index=False):
        print(f"\n  --- permno={r.permno} cik={r.cik} "
              f"(item1a {r.bigram_hit_count_item1a} hits, rest {r.bigram_hit_count_rest}) ---")
        for section, pairs in diag["examples"].get(r.accession, {}).items():
            for sentence, matched in pairs:
                flat = " ".join(sentence.split())[:SENTENCE_PRINT_CHARS]
                print(f"    [{section}] ({', '.join(matched)})")
                print(f"      {_printable(flat)}")


def _check_short_sentences(diag: dict) -> None:
    """Report implausibly short sentences, which signal tokenisation artifacts.

    Tallied during the scoring pass; see score_stream.
    """
    print(f"\nSentences shorter than {MIN_SENTENCE_CHARS} chars (tokenisation artifacts):")
    for label, (n_all, n_short) in diag["short"].items():
        print(f"  {label:7s} {n_short:>9,} of {n_all:>9,} "
              f"({n_short / n_all if n_all else 0:.2%})")
    print("  examples:")
    for s in diag["short_examples"][:8]:
        print(f"    {_printable(repr(s.strip()))}")


def validate(scores: pd.DataFrame, filings: pd.DataFrame, kept: pd.DataFrame,
             dropped: pd.DataFrame, scope_ciks: set[str], terms: list[str],
             diag: dict, done: set[str]) -> None:
    """Run every required check and print the report."""
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    _check_funnel(filings, kept, dropped, scope_ciks, done)
    _check_drop_reconciliation(dropped)
    _check_composition(scores)
    _check_per_term(terms, diag, len(scores))
    _check_removed_terms(diag["audit"], diag["total_hits"])
    _check_utility_context(scores, diag)
    _check_en_dash(diag)
    _check_distributions(scores)
    _check_dispersion(scores)
    _check_duplicate_accessions(scores)
    _check_spot_sentences(scores, diag)
    _check_short_sentences(diag)


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def scored_accessions() -> set[str]:
    """Accessions already present in tariff_scores.csv."""
    if not SCORES_OUT.exists() or SCORES_OUT.stat().st_size == 0:
        return set()
    return set(pd.read_csv(SCORES_OUT, dtype={"accession": str})["accession"].dropna())


def align_scores_schema() -> list[str]:
    """Drop the pooled TExp_*_z columns from a pre-existing tariff_scores.csv, once.

    New rows are appended with the current fieldnames, so a stale header would silently
    misalign every subsequent row. Only the derived standardised columns are removed - they
    were pooled over a single reference date, are meaningless once the file spans 17, and
    nothing downstream reads them (decile_sort, foreign_sales and run_car_regression all use
    raw TExp_item1a). The raw scores are re-read afterwards and compared element-wise, so a
    run aborts rather than proceeding on a file whose values moved.
    """
    if not SCORES_OUT.exists() or SCORES_OUT.stat().st_size == 0:
        return []
    existing = pd.read_csv(SCORES_OUT, dtype={"cik": str, "accession": str})
    stale = [c for c in existing.columns if c not in OUTPUT_COLUMNS]
    if not stale:
        return []

    trimmed = existing.reindex(columns=OUTPUT_COLUMNS)
    trimmed.to_csv(SCORES_OUT, index=False)
    back = pd.read_csv(SCORES_OUT, dtype={"cik": str, "accession": str})
    if len(back) != len(existing):
        raise ValueError(f"schema alignment changed the row count: {len(existing):,} -> "
                         f"{len(back):,}")
    for col in TEXP_COLUMNS:
        if not back[col].equals(existing[col]):
            raise ValueError(f"schema alignment altered {col}; aborting with the file as read.")
    return stale


def main() -> pd.DataFrame | None:
    ap = argparse.ArgumentParser(description="Tariff-exposure tokenisation and scoring.")
    ap.add_argument("--n-ciks", type=int, default=None,
                    help="restrict to the first N distinct CIKs by PERMNO "
                         "(default: the whole universe)")
    ap.add_argument("--status", action="store_true",
                    help="report done/remaining and exit without scoring anything")
    args = ap.parse_args()

    filings = clean_filings.read_clean_filings()   # raises if the upstream step has not run
    scope_ciks = resolve_scope_ciks(args.n_ciks)
    done = scored_accessions()
    kept, dropped = load_scope(filings, scope_ciks, done)
    print(f"Scope: {'whole universe' if args.n_ciks is None else f'first {args.n_ciks} CIKs'} "
          f"-> {len(scope_ciks):,} CIKs | {len(filings):,} cleaned filings | "
          f"already scored {len(done):,} | to score {len(kept):,} | dropped {len(dropped):,}")
    if args.status:
        print(dropped["drop_reason"].value_counts().to_string())
        return None
    if kept.empty:
        # A fully scored corpus still re-emits its diagnostics and report. Returning here left
        # an idempotent re-run with no record of the stage and stale diagnostic files.
        print("Nothing to score; re-validating the existing scores.")
        full = pd.read_csv(SCORES_OUT, dtype={"cik": str, "accession": str})
        write_diagnostics(full, dropped)
        with run_report.capture(REPORT_OUT, title="STEP 2B - TARIFF SCORING VALIDATION"):
            print(f"Re-validation of {len(full):,} scored filings; nothing new to score.")
            print(dropped["drop_reason"].value_counts().to_string())
        return None

    dropped_cols = align_scores_schema()
    if dropped_cols:
        print(f"Removed stale pooled column(s) from {SCORES_OUT.name}: "
              f"{', '.join(dropped_cols)} (raw scores verified unchanged).")

    resource = ensure_punkt()
    terms = load_bigrams(BIGRAM_JSON)
    patterns, gates = compile_patterns(terms), build_gates(terms)
    print(f"Tokenizer data: NLTK '{resource}' | terms: {len(terms)} from {BIGRAM_JSON.name}\n")
    edgar_pull._prevent_sleep()

    fh, writer = open_scores_log()
    try:
        scores, diag = score_stream(kept, patterns, gates, writer, fh)
    finally:
        fh.close()

    full = pd.read_csv(SCORES_OUT, dtype={"cik": str, "accession": str})
    print(f"Wrote {SCORES_OUT.relative_to(BASE)} (+{len(scores):,} rows this run, "
          f"{len(full):,} total x {len(full.columns)} cols).")
    write_diagnostics(full, dropped)
    write_term_hits(terms, diag)
    print()

    # Validated on this run's rows, not the whole file: the diag counters describe exactly
    # these filings, and pairing them with a corpus-wide row count would misstate every
    # per-term and per-filing rate on a resumed run.
    with run_report.capture(REPORT_OUT, title="STEP 2B - TARIFF SCORING VALIDATION"):
        validate(scores, filings, kept, dropped, scope_ciks, terms, diag, done)
    return scores


if __name__ == "__main__":
    main()
