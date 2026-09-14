"""Structural validation of exported .deq text against the real deq grammar.

Soft-imports the ``deq``/``deqagram`` packages (Microsoft's qdk-ec toolkit) —
this module works without them installed, falling back to a much weaker
regex/brace-balance check, but a real grammar round-trip is strongly
preferred whenever they're available.
"""
from __future__ import annotations

import importlib.util
import logging

from .pairing import DeqExportError

_log = logging.getLogger(__name__)

_HAS_DEQ = importlib.util.find_spec("deq") is not None and importlib.util.find_spec("deqagram") is not None
if _HAS_DEQ:
    try:
        from deq.circuit import parser as _deq_parser
    except ImportError as exc:  # pragma: no cover - defensive
        _HAS_DEQ = False
        _log.debug("deq installed but import failed: %s", exc)


def deq_available() -> bool:
    """Whether the real `deq`/`deqagram` grammar is available for validation."""
    return _HAS_DEQ


def validate_deq_text(text: str):
    """Round-trip `text` through the real deq grammar.

    Returns the parsed `deq.circuit.model.DeqFile` when `deq`/`deqagram` are
    installed. Falls back to a weak structural (brace-balance) check
    otherwise, returning None.

    Raises:
        DeqExportError: If the text fails to parse (or fails the weak
            fallback check).
    """
    if _HAS_DEQ:
        try:
            return _deq_parser.parse(text)
        except SyntaxError as exc:
            raise DeqExportError(f"generated .deq text failed to parse: {exc}") from exc

    _log.warning("deq/deqagram not installed; falling back to a weak structural check")
    _weak_structural_check(text)
    return None


def _weak_structural_check(text: str) -> None:
    depth = 0
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0]
        depth += line.count("{") - line.count("}")
        if depth < 0:
            raise DeqExportError(f"unbalanced '}}' at line {line_no}")
    if depth != 0:
        raise DeqExportError(f"unbalanced braces: {depth} unclosed '{{' at end of file")
