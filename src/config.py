"""Every directory the pipeline reads or writes, defined once. Modules import from here and keep
only their own file names."""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent      # repository root
SRC_DIR = BASE / "src"
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"                         # user-supplied extracts (CRSP, Compustat, ...)
CLEAN_DIR = DATA_DIR / "clean"                     # cleaned panels consumed downstream
INTERMEDIATE_DIR = DATA_DIR / "intermediate"       # derived analytical panels and audit tables
FILINGS_DIR = DATA_DIR / "filings" / "raw"         # cached 10-K HTML
CLEAN_TEXT_DIR = DATA_DIR / "filings" / "clean_text"
SUBMISSIONS_DIR = DATA_DIR / "filings" / "submissions"
EDGAR_LOG = DATA_DIR / "edgar_pull_log.csv"        # resumable pull state, paired with FILINGS_DIR
OUTPUT_DIR = BASE / "output"                       # reported results, figures and reports
ENV_FILE = BASE / ".env"
LEXICON = SRC_DIR / "bigram_list.json"
