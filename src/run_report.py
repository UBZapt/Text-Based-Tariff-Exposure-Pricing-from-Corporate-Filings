"""
Shared helper: persist a stage's console run report beside its outputs.

Seven pipeline stages built their validation output with plain ``print`` and so left no record on
disk, while the six later stages buffer into a ``_REPORT`` list and write a ``.txt``. Rewriting
~150 print calls into a buffer per stage would be a large, risky diff for a filing-order problem,
and it would also change what those stages print. This instead tees stdout: the console output is
byte-identical to before, and the same text lands in a file.

Usage, wrapping the part of main() that produces the report:

    with run_report.capture(REPORT_OUT, title="STEP 2A - FILING CLEANING"):
        validate(full)

Stages that already own a ``_REPORT`` buffer (clean_controls_data, estimate_car,
run_car_regression, decile_sort, foreign_sales, fama_macbeth_pricing) keep it and do NOT use this;
their reports are assembled deliberately rather than captured, and several are re-read by
downstream checks.
"""

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RULE = "=" * 78


class _Tee:
    """Write to several streams at once; only the console stream is flushed eagerly."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def capture(path: Path, title: str = "", header: list[str] | None = None):
    """Tee everything printed inside the block into ``path`` as well as to the console.

    The file is written on exit, including when the block raises - a report of a run that failed
    part way is more use than no report at all, and the traceback still propagates.
    """
    buffer: list[str] = []

    class _Collect:
        def write(self, text: str) -> int:
            buffer.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    original = sys.stdout
    sys.stdout = _Tee(original, _Collect())
    try:
        yield
    finally:
        sys.stdout = original
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [RULE, title or path.stem.replace("_", " ").upper(), RULE,
                 f"Generated     : {stamp}"]
        lines += header or []
        lines += ["", "".join(buffer).rstrip("\n"), ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report written to {path.name}")
