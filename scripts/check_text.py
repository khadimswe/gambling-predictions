#!/usr/bin/env python3
"""Fail if text files contain em dashes or invisible / look-alike characters.

Usage:
  python scripts/check_text.py              # every text file git tracks or would track
  python scripts/check_text.py FILE [FILE]  # just these (pre-commit passes staged files)

What it flags:
  - em dash and horizontal bar (rewrite with a comma, colon, parentheses, or " - ")
  - an en dash (U+2013) with spaces on both sides, i.e. used as an em dash.
    Unspaced en dashes in numeric ranges like page numbers are fine.
  - zero-width and BOM characters, non-breaking spaces (copy-paste debris)

Add "text-check: ignore" anywhere on a line to skip that line.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile("\u2014"), "em dash"),
    (re.compile("\u2015"), "horizontal bar (em dash look-alike)"),
    (re.compile(" \u2013 "), "spaced en dash used as an em dash"),
    (re.compile("[\u200b\u200c\u200d\u2060]"), "zero-width character"),
    (re.compile("\ufeff"), "byte order mark"),
    (re.compile("[\u00a0\u202f]"), "non-breaking space"),
]
IGNORE_MARKER = "text-check: ignore"
SKIP_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".pkl", ".parquet"}
SKIP_DIRS = {"data", ".venv", "node_modules"}


def tracked_files() -> list[Path]:
    """Committed files plus new files git is not ignoring."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def should_skip(path: Path) -> bool:
    return (
        path.suffix.lower() in SKIP_SUFFIXES
        or bool(SKIP_DIRS.intersection(path.parts[:1]))
        or not path.is_file()
    )


def check_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable: not our job
    problems = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if IGNORE_MARKER in line:
            continue
        for pattern, label in RULES:
            for match in pattern.finditer(line):
                problems.append(f"{path}:{lineno}:{match.start() + 1}: {label}")
    return problems


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] if argv else tracked_files()
    problems = [p for path in paths if not should_skip(path) for p in check_file(path)]
    for problem in problems:
        print(problem)
    if problems:
        print(
            f"\n{len(problems)} problem(s). Replace em dashes with a comma, colon, "
            "parentheses, or ' - ', and delete invisible characters.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
