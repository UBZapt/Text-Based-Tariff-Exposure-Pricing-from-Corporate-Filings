"""
Step 2b - Sentence tokenisation (Stage 2) and tariff-bigram scoring (Stage 3).

Reads the cleaned filings table, splits each filing's Item 1A and rest-of-document text
into sentences, matches a fixed tariff term list against every sentence, and writes three
exposure scores per filing.

Measure formula, from research design section 7.1 with the relevance weight w_b dropped
per instruction (the single deviation):

    TExp = (sentences containing >= 1 tariff term) / (total sentences in that unit)

The two stages are deliberately separate: ``tokenise_sections`` holds no matching logic and
``score_section`` holds no sentence-splitting logic. Pilot scope only - the first
PILOT_N_CIKS distinct CIKs in ascending PERMNO order, matching the original EDGAR pilot.

    python score_filings.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import nltk
import pandas as pd

import edgar_pull

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
FILINGS_PARQUET = BASE / "clean_filings.parquet"
BIGRAM_JSON = BASE / "bigram_list.json"
SCORES_OUT = BASE / "tariff_scores_pilot.parquet"

PILOT_N_CIKS = 50            # first N distinct CIKs by PERMNO; matches the EDGAR pilot
MIN_SENTENCE_CHARS = 15      # below this a "sentence" is likely a tokenisation artifact
IMPLAUSIBLE_TEXP = 0.5       # tripwire for double-counting or boilerplate artifacts
SPOT_CHECK_FILINGS = 3       # filings whose matched sentences are printed
SPOT_CHECK_SENTENCES = 4     # matched sentences retained per section per filing
SENTENCE_PRINT_CHARS = 260   # truncation when printing a matched sentence

EN_DASH, EM_DASH = chr(0x2013), chr(0x2014)
DASH_CLASS = "[-" + EN_DASH + EM_DASH + "]"

OUTPUT_COLUMNS = [
    "permno", "cik", "accession", "filing_date", "fiscal_year",
    "B_item1a", "B_rest", "bigram_hit_count_item1a", "bigram_hit_count_rest",
    "TExp_item1a", "TExp_rest", "TExp_combined", "scoring_flags",
]


class SectionScore(NamedTuple):
    """Result of scoring one section's sentences."""
    n_hit: int                 # sentences containing >= 1 term (binary per sentence)
    per_term: Counter          # term -> sentences it appeared in (terms overlap; see notes)
    examples: list             # up to SPOT_CHECK_SENTENCES (sentence, matched terms) pairs
    n_multi_term: int          # hit sentences matching more than one term


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


def load_pilot(filings: pd.DataFrame, n_ciks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict to the pilot CIK scope, then drop filings with no identified Item 1A.

    Scope reuses edgar_pull's universe ordering rather than re-deriving it, so the sample
    stays exactly the original pilot even after the full pull enlarges the cleaned table.
    Returns (kept, dropped); dropped rows are reported by the caller, never silently lost.
    """
    firms = edgar_pull.resolve_firms(edgar_pull.BRIDGE_CSV, edgar_pull.REFERENCE_DATE)
    pilot_firms, _ = edgar_pull.select_gap_firms(firms, set(), n_ciks)
    pilot_ciks = {f["cik"] for f in pilot_firms if f["cik"]}

    in_scope = filings[filings["cik"].isin(pilot_ciks)].copy()
    if in_scope.empty:
        raise ValueError(f"No filings in {FILINGS_PARQUET.name} fall inside the "
                         f"first-{n_ciks}-CIK pilot scope.")
    kept = in_scope[in_scope["item_1a_found"]].copy()
    dropped = in_scope[~in_scope["item_1a_found"]].copy()

    print(f"Pilot scope: first {n_ciks} distinct CIKs by PERMNO -> {len(pilot_ciks)} CIKs, "
          f"{len(in_scope)} cleaned filings on disk "
          f"({len(pilot_ciks) - len(in_scope)} CIKs have no usable 10-K).")
    print(f"Dropped for missing Item 1A: {len(dropped)}")
    for r in dropped.itertuples(index=False):
        print(f"  permno={r.permno:<7} cik={r.cik} accession={r.accession}")
    print(f"Scoring {len(kept)} filings.\n")
    return kept, dropped


# --------------------------------------------------------------------------- #
# Stage 2 - tokenisation (no bigram logic in this section)                    #
# --------------------------------------------------------------------------- #
def tokenise_sections(kept: pd.DataFrame) -> list[dict]:
    """Split each filing's two sections into sentences, independently.

    sent_tokenize is run separately on item_1a_text and rest_text - never on a
    concatenation - so the two sentence lists and their counts B_item1a / B_rest are
    genuinely per section. Because the sections are tokenised apart, B_item1a + B_rest can
    differ by a sentence or two from tokenising the whole document, since the section
    boundary is also a forced sentence boundary.
    """
    tokenised = []
    for f in kept.itertuples(index=False):
        sentences_item1a = nltk.sent_tokenize(f.item_1a_text)
        sentences_rest = nltk.sent_tokenize(f.rest_text)
        tokenised.append({
            "permno": f.permno, "cik": f.cik, "accession": f.accession,
            "filing_date": f.filing_date, "fiscal_year": f.fiscal_year,
            "sentences_item1a": sentences_item1a,
            "sentences_rest": sentences_rest,
            "B_item1a": len(sentences_item1a),
            "B_rest": len(sentences_rest),
        })
    total = sum(t["B_item1a"] + t["B_rest"] for t in tokenised)
    print(f"Stage 2 - tokenised {len(tokenised)} filings into {total:,} sentences "
          f"({sum(t['B_item1a'] for t in tokenised):,} Item 1A, "
          f"{sum(t['B_rest'] for t in tokenised):,} rest).")
    return tokenised


# --------------------------------------------------------------------------- #
# Stage 3 - bigram scoring (no tokenisation logic in this section)            #
# --------------------------------------------------------------------------- #
def score_section(sentences: list[str], patterns: dict[str, re.Pattern],
                  max_examples: int = SPOT_CHECK_SENTENCES) -> SectionScore:
    """Count sentences containing at least one term, and attribute hits per term.

    A sentence contributes exactly one hit however many terms or repetitions it holds
    (sentence-level binary, matching the indicator in the section 7.1 formula). Per-term
    counts are attributed independently, so a sentence reading "retaliatory tariffs"
    increments 'retaliatory tariffs' and 'tariffs' both; those counts therefore overlap by
    construction and do not sum to n_hit. That is deliberate - the per-term table exists to
    catch matching bugs, and longest-match-only attribution would drive 'tariff' to near
    zero and look like one.
    """
    n_hit = n_multi_term = 0
    per_term: Counter = Counter()
    examples: list = []
    for sentence in sentences:
        matched = [term for term, pattern in patterns.items() if pattern.search(sentence)]
        if not matched:
            continue
        n_hit += 1
        per_term.update(matched)
        if len(matched) > 1:
            n_multi_term += 1
        if len(examples) < max_examples:
            examples.append((sentence, matched))
    return SectionScore(n_hit, per_term, examples, n_multi_term)


def score_filings(tokenised: list[dict],
                  patterns: dict[str, re.Pattern]) -> tuple[pd.DataFrame, dict]:
    """Score every filing and assemble the output table plus corpus diagnostics.

    Zero denominators are handled explicitly: a section with no sentences gets NaN rather
    than a division, and is flagged. TExp_combined is the pooled document rate, so one
    empty section still yields a combined score from the other; it is NaN only when the
    whole document is empty.
    """
    rows, examples = [], {}
    term_1a: Counter = Counter()
    term_rest: Counter = Counter()
    term_filings: Counter = Counter()
    n_multi_term = 0

    for t in tokenised:
        s_1a = score_section(t["sentences_item1a"], patterns)
        s_rest = score_section(t["sentences_rest"], patterns)
        b_1a, b_rest = t["B_item1a"], t["B_rest"]
        b_total = b_1a + b_rest

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
        examples[t["accession"]] = {"item_1a": s_1a.examples, "rest": s_rest.examples}

        rows.append({
            "permno": t["permno"], "cik": t["cik"], "accession": t["accession"],
            "filing_date": t["filing_date"], "fiscal_year": t["fiscal_year"],
            "B_item1a": b_1a, "B_rest": b_rest,
            "bigram_hit_count_item1a": s_1a.n_hit,
            "bigram_hit_count_rest": s_rest.n_hit,
            "TExp_item1a": texp_1a, "TExp_rest": texp_rest,
            "TExp_combined": texp_combined,
            "scoring_flags": "; ".join(flags),
        })

    scores = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    total_hits = int(scores["bigram_hit_count_item1a"].sum()
                     + scores["bigram_hit_count_rest"].sum())
    print(f"Stage 3 - {total_hits:,} hit sentences across the sample "
          f"({n_multi_term:,} matched more than one term).\n")
    return scores, {"term_1a": term_1a, "term_rest": term_rest,
                    "term_filings": term_filings, "examples": examples,
                    "n_multi_term": n_multi_term, "total_hits": total_hits}


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _check_drop_reconciliation(in_scope: pd.DataFrame, dropped: pd.DataFrame) -> None:
    """Cross-check the three parquet fields that independently record a missing Item 1A."""
    by_flag = set(in_scope.loc[
        in_scope["cleaning_flags"].str.contains("item_1a_not_found"), "accession"])
    by_found = set(in_scope.loc[~in_scope["item_1a_found"], "accession"])
    by_chars = set(in_scope.loc[in_scope["item_1a_char_count"] == 0, "accession"])

    print(f"Filings dropped for missing Item 1A : {len(dropped)}")
    for r in dropped.itertuples(index=False):
        print(f"  cik={r.cik} accession={r.accession} (permno {r.permno})")
    agree = by_flag == by_found == by_chars
    print(f"  cross-check (item_1a_found / cleaning_flags / item_1a_char_count): "
          f"{'all three agree' if agree else '!! DISAGREEMENT'}")
    if not agree:
        print(f"  !! by_found={sorted(by_found)} by_flag={sorted(by_flag)} "
              f"by_chars={sorted(by_chars)}")
    print(f"  clean_filings.py reported {len(in_scope) - len(dropped)}/{len(in_scope)} "
          f"located; this run drops {len(dropped)} - "
          f"{'consistent' if len(by_found) == len(dropped) else '!! MISMATCH'}")


def _check_per_term(terms: list[str], diag: dict, n_filings: int) -> None:
    """Report per-term hit frequency and surface any term that never matched."""
    print("\nPer-term hit frequency (sentences; terms overlap by construction, so the")
    print("column does not sum to the hit total - see the nested-match note in the docstring):")
    print(f"  {'term':36s} {'item1a':>7} {'rest':>7} {'total':>7} {'filings':>8}")
    rows = [(t, diag["term_1a"][t], diag["term_rest"][t]) for t in terms]
    for term, n_1a, n_rest in sorted(rows, key=lambda r: -(r[1] + r[2])):
        print(f"  {_printable(term):36s} {n_1a:>7,} {n_rest:>7,} {n_1a + n_rest:>7,} "
              f"{diag['term_filings'][term]:>8,}")

    zero = [t for t, n_1a, n_rest in rows if n_1a + n_rest == 0]
    print(f"\n  Terms with ZERO hits across all {n_filings} filings: {len(zero)}")
    for term in zero:
        print(f"    {_printable(term)}")
    if zero:
        print("    (zero hits may be genuine absence or a matching bug - the en-dash")
        print("     diagnostic below resolves that question for the one term at risk)")


def _check_en_dash(kept: pd.DataFrame, diag: dict) -> None:
    """Test the en-dash term explicitly, separately from hyphen and spaced variants."""
    corpus = "\n".join(kept["item_1a_text"]) + "\n" + "\n".join(kept["rest_text"])
    en_term = "U.S." + EN_DASH + "China tariffs"

    print("\nEn-dash diagnostic for the list term 'U.S.<U+2013>China tariffs':")
    print(f"  sentence hits recorded for the exact list term : "
          f"{diag['term_1a'][en_term] + diag['term_rest'][en_term]}")
    print(f"  en dashes (U+2013) present in cleaned corpus   : {corpus.count(EN_DASH):,}")
    print(f"  em dashes (U+2014) present in cleaned corpus   : {corpus.count(EM_DASH):,}")

    variants = [
        ("exact, U+2013 en dash", r"U\.S\." + EN_DASH + r"China tariffs"),
        ("hyphen variant  U.S.-China tariffs", r"U\.S\.-China tariffs"),
        ("U.S.<any dash>China, any context", r"U\.S\." + DASH_CLASS + r"China"),
        ("US<any dash>China, any context", r"(?<!\w)US" + DASH_CLASS + r"China"),
        ("China tariffs, loose", r"China\s+tariffs"),
    ]
    print("  raw occurrences in the cleaned corpus:")
    counts = {}
    for label, pattern in variants:
        counts[label] = len(re.findall(pattern, corpus, re.IGNORECASE))
        print(f"    {label:36s} {counts[label]:,}")

    if all(n == 0 for n in counts.values()) and corpus.count(EN_DASH) > 0:
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
    print(f"  {'measure':22s} {'n':>4} {'min':>12} {'median':>12} {'mean':>12} {'max':>12}")
    for col in ["B_item1a", "B_rest", "TExp_item1a", "TExp_rest", "TExp_combined"]:
        s = scores[col].dropna()
        fmt = ",.0f" if col.startswith("B_") else ".5f"
        print(f"  {col:22s} {len(s):>4} {s.min():>12{fmt}} {s.median():>12{fmt}} "
              f"{s.mean():>12{fmt}} {s.max():>12{fmt}}")

    flagged = scores[scores["scoring_flags"].str.contains("implausible")]
    print(f"\n  Filings with any TExp > {IMPLAUSIBLE_TEXP} : {len(flagged)}")
    for r in flagged.itertuples(index=False):
        print(f"    cik={r.cik} accession={r.accession} {r.scoring_flags}")
    if flagged.empty:
        print(f"    (none - expected: a keyword measure over full 10-K text yields rates of")
        print(f"     order 1e-3, so an empty list here is the correct result, not a failure)")

    empty = scores[scores["scoring_flags"].str.contains("empty_")]
    print(f"  Filings with an empty section (TExp set to NaN) : {len(empty)}")
    for r in empty.itertuples(index=False):
        print(f"    cik={r.cik} accession={r.accession} {r.scoring_flags}")

    corr = scores[["TExp_rest", "TExp_combined"]].corr().iloc[0, 1]
    print(f"  corr(TExp_rest, TExp_combined) = {corr:.4f}  "
          f"(pooled combined is dominated by the longer rest section, as expected)")


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


def _check_short_sentences(tokenised: list[dict]) -> None:
    """Report implausibly short sentences, which signal tokenisation artifacts."""
    print(f"\nSentences shorter than {MIN_SENTENCE_CHARS} chars (tokenisation artifacts):")
    examples = []
    for key, label in [("sentences_item1a", "item1a"), ("sentences_rest", "rest")]:
        allsent = [s for t in tokenised for s in t[key]]
        short = [s for s in allsent if len(s.strip()) < MIN_SENTENCE_CHARS]
        print(f"  {label:7s} {len(short):>7,} of {len(allsent):>7,} "
              f"({len(short) / len(allsent) if allsent else 0:.2%})")
        examples += short[:4]
    print("  examples:")
    for s in examples[:8]:
        print(f"    {_printable(repr(s.strip()))}")


def validate(scores: pd.DataFrame, in_scope: pd.DataFrame, kept: pd.DataFrame,
             dropped: pd.DataFrame, terms: list[str], diag: dict,
             tokenised: list[dict]) -> None:
    """Run every required check and print the report."""
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    _check_drop_reconciliation(in_scope, dropped)
    _check_per_term(terms, diag, len(scores))
    _check_en_dash(kept, diag)
    _check_distributions(scores)
    _check_spot_sentences(scores, diag)
    _check_short_sentences(tokenised)


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def main() -> pd.DataFrame:
    if not FILINGS_PARQUET.exists():
        raise FileNotFoundError(
            f"{FILINGS_PARQUET.name} not found; run clean_filings.py first.")
    resource = ensure_punkt()
    terms = load_bigrams(BIGRAM_JSON)
    patterns = compile_patterns(terms)
    print(f"Tokenizer data: NLTK '{resource}' | terms: {len(terms)} "
          f"from {BIGRAM_JSON.name}\n")

    filings = pd.read_parquet(FILINGS_PARQUET)
    kept, dropped = load_pilot(filings, PILOT_N_CIKS)
    in_scope = pd.concat([kept, dropped])

    tokenised = tokenise_sections(kept)
    scores, diag = score_filings(tokenised, patterns)

    scores.to_parquet(SCORES_OUT, index=False)
    print(f"Wrote {SCORES_OUT.name} ({len(scores)} rows x {len(scores.columns)} cols).\n")

    validate(scores, in_scope, kept, dropped, terms, diag, tokenised)
    return scores


if __name__ == "__main__":
    main()
