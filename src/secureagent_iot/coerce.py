"""Total (never-raising) coercions for RAW, pre-validation proposal fields.

A raw proposal is whatever a model emitted: any field can be missing, null, a
string such as "300 seconds", a float, a list, a dict, NaN/Infinity or an
absurdly large number. Scoring and logging code must survive all of it. Nothing
here repairs a proposal for the policy path: `schemas.parse_proposal` stays the
only gate there and it is strict.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

_INT_TEXT = re.compile(r"[+-]?\d{1,18}")
MAX_SANE_DURATION = 10 ** 15   # seconds; keeps simulated-clock float arithmetic and int64 columns safe


def exact_int(x: Any) -> int | None:
    """The integer `x` denotes exactly, else None. Accepts an int, an integral
    finite float (300.0) and a plain digit string ("300"). Rejects bools,
    300.5, "300 seconds", None, NaN/inf, lists, dicts. Never raises."""
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x) if math.isfinite(x) and x == int(x) else None
        if isinstance(x, str):
            s = x.strip()
            return int(s) if _INT_TEXT.fullmatch(s) else None
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def lenient_int(x: Any, default: int = 0) -> int:
    """`int(x or 0)` made total: what a permissive, unvalidated API would do with
    a duration (truncates floats, accepts digit strings), falling back to
    `default` for anything int() refuses and clamping to +/-MAX_SANE_DURATION."""
    try:
        v = int(x or 0)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(-MAX_SANE_DURATION, min(MAX_SANE_DURATION, v))


def scalar(x: Any, max_len: int = 200) -> Any:
    """A CSV/JSON-safe scalar view of a raw field for result rows. Short strings,
    bools, None, int64-range ints and finite floats pass through unchanged;
    everything else becomes a truncated JSON/str rendering."""
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int) and -(2 ** 63) <= x < 2 ** 63:
        return x
    if isinstance(x, float) and math.isfinite(x):
        return x
    if isinstance(x, str):
        return x if len(x) <= max_len else x[:max_len]
    try:
        return json.dumps(x, default=str)[:max_len]
    except (TypeError, ValueError, OverflowError, RecursionError):
        return str(type(x).__name__)
