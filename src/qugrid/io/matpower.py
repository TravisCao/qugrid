"""Read MATPOWER case files (``.m``) without MATLAB.

The parser targets standard MATPOWER *data* files: assignments of scalars,
strings, and numeric matrices to fields of a struct named ``mpc``. It covers
every file shipped with QuGrid and the ordinary case files found in the wild.
It does not evaluate MATLAB code (expressions, concatenation of variables),
and it says so loudly when it meets one.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_STR_RE = re.compile(r"mpc\.(\w+)\s*=\s*'([^']*)'\s*;")
_SCALAR_RE = re.compile(r"mpc\.(\w+)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*;")
_MATRIX_RE = re.compile(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", re.DOTALL)


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cut = line.find("%")
        lines.append(line if cut < 0 else line[:cut])
    return "\n".join(lines)


def _parse_matrix(body: str, name: str) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in re.split(r"[;\n]", body):
        raw = raw.replace(",", " ").strip()
        if not raw:
            continue
        try:
            rows.append([float(tok) for tok in raw.split()])
        except ValueError as err:
            raise ValueError(
                f"mpc.{name} contains a non-numeric row ({raw[:60]!r}). "
                "QuGrid parses plain MATPOWER data files only; evaluate the "
                "file in MATLAB/Octave and use savecase() if it contains code."
            ) from err
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    arr = np.zeros((len(rows), width))
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r  # gencost rows may be ragged; pad with zeros
    return arr


def parse_matpower(text: str) -> dict:
    """Parse MATPOWER case text into a PYPOWER-style ``ppc`` dict."""
    clean = _strip_comments(text)
    ppc: dict = {}
    for m in _STR_RE.finditer(clean):
        ppc[m.group(1)] = m.group(2)
    for m in _SCALAR_RE.finditer(clean):
        ppc[m.group(1)] = float(m.group(2))
    for m in _MATRIX_RE.finditer(clean):
        ppc[m.group(1)] = _parse_matrix(m.group(2), m.group(1))
    for required in ("baseMVA", "bus", "gen", "branch"):
        if required not in ppc:
            raise ValueError(f"not a MATPOWER case: field mpc.{required} not found")
    return ppc


def read_matpower(path: str | Path) -> dict:
    """Read a MATPOWER ``.m`` case file into a ``ppc`` dict."""
    return parse_matpower(Path(path).read_text(encoding="utf-8", errors="replace"))
