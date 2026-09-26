"""Minimal, dependency-free parser for the subset of Eclipse-deck syntax
needed to pull well definitions, completions, controls and grid-property
arrays out of the Egg Model and SPE9 ``.DATA``/``.INC`` files.

This is deliberately small: it is not a full Eclipse-format parser (it does
not evaluate INCLUDE, does not understand every keyword), it only extracts
the handful of keywords these two benchmarks' well-sequencing adapters need:
WELSPECS, COMPDAT, WCONPROD, WCONINJE, TSTEP/DATES, and flat numeric arrays
(e.g. PERMX) including Eclipse's ``N*value`` repeat-count shorthand.
"""
from __future__ import annotations

import re
from pathlib import Path


def strip_comments(text: str) -> str:
    """Remove Eclipse ``--`` line comments."""
    lines = []
    for line in text.splitlines():
        idx = line.find("--")
        lines.append(line[:idx] if idx >= 0 else line)
    return "\n".join(lines)


def find_keyword_blocks(text: str, keyword: str) -> list[str]:
    """Return the raw text following each standalone occurrence of ``keyword``.

    Each block runs until the next line that is a bare ``/`` terminator is
    reached for array-style keywords, or, for record-style keywords, until a
    line containing only ``/`` closes the keyword (Eclipse convention: the
    keyword's records are terminated by a line with a single ``/``).
    """
    blocks: list[str] = []
    pattern = re.compile(rf"^[ \t]*{re.escape(keyword)}[ \t]*$", re.MULTILINE)
    for match in pattern.finditer(text):
        start = match.end()
        rest = text[start:]
        end_match = re.search(r"^\s*/\s*$", rest, re.MULTILINE)
        if end_match:
            blocks.append(rest[: end_match.start()])
        else:
            blocks.append(rest)
    return blocks


def parse_well_records(text: str, keyword: str) -> list[list[str]]:
    """Parse a record-style keyword (WELSPECS, COMPDAT, WCONPROD, WCONINJE).

    Each record in the block is itself ``/``-terminated; returns the list of
    tokenized records (quotes stripped).
    """
    records: list[list[str]] = []
    for block in find_keyword_blocks(text, keyword):
        token_re = re.compile(r"'[^']*'|[^\s/]+")
        for raw_record in block.split("/"):
            tokens = [tok.strip("'") for tok in token_re.findall(raw_record)]
            if tokens:
                records.append(tokens)
    return records


_REPEAT_RE = re.compile(r"^(\d+)\*(.+)$")


def _find_array_blocks(text: str, keyword: str) -> list[str]:
    """Like ``find_keyword_blocks``, but a single-record array keyword
    (PERMX, DZ, TSTEP, ...) is terminated by the FIRST ``/`` character that
    follows it, not by a later bare ``/`` line (which may belong to an
    unrelated keyword further down the deck, e.g. COPY/MULTIPLY records)."""
    blocks: list[str] = []
    pattern = re.compile(rf"^[ \t]*{re.escape(keyword)}[ \t]*$", re.MULTILINE)
    for match in pattern.finditer(text):
        rest = text[match.end() :]
        end = rest.find("/")
        blocks.append(rest if end == -1 else rest[:end])
    return blocks


def parse_numeric_array(text: str, keyword: str) -> list[float]:
    """Parse a flat numeric array keyword (e.g. PERMX, DZ), expanding
    Eclipse's ``N*value`` repeat shorthand and bare ``N*`` (default) tokens.
    """
    values: list[float] = []
    for block in _find_array_blocks(text, keyword):
        for token in block.split():
            if token == "/":
                continue
            repeat_match = _REPEAT_RE.match(token)
            if repeat_match:
                count = int(repeat_match.group(1))
                value = float(repeat_match.group(2))
                values.extend([value] * count)
            else:
                try:
                    values.append(float(token))
                except ValueError:
                    continue
    return values


def parse_tstep_days(text: str) -> list[float]:
    """Return every timestep length (days), in schedule order, expanding
    ``N*value`` shorthand, from all TSTEP keyword occurrences."""
    return parse_numeric_array(text, "TSTEP")


def read_deck(path: Path) -> str:
    return strip_comments(path.read_text(encoding="utf-8", errors="replace"))
