"""Small, dependency-light helpers shared by the analysis modules.

The helpers here deliberately stay free of Streamlit and of any analysis logic so
that they can be unit-tested and reused from the CLI, the test-suite and the
reporting layer alike.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src import config

_CURRENCY_AND_SEPARATORS = re.compile(r"[,\s\u00a0\u202f]|[$€£¥₹]|^\+")
_DATE_HINT = re.compile(
    r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})"          # 2024-01-31
    r"|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"        # 31/01/2024
    r"|(?:\d{1,2}:\d{2})"                        # 13:45
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Numbers and formatting
# --------------------------------------------------------------------------- #
def tokenize_name(name: str) -> list[str]:
    """Split a column name into lower-case word tokens.

    ``"exam_score"`` -> ``["exam", "score"]``; ``"customerID"`` -> ``["customer", "id"]``;
    ``"final-grade"`` -> ``["final", "grade"]``. Token-based matching is used for all
    name heuristics so that whole words are matched and substrings such as "end"
    inside "gender" are never treated as a signal.
    """
    if name is None:
        return []
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    tokens = re.split(r"[^A-Za-z0-9]+", text)
    return [token.lower() for token in tokens if token]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is zero or invalid."""
    try:
        if denominator is None or numerator is None:
            return default
        denominator = float(denominator)
        numerator = float(numerator)
    except (TypeError, ValueError):
        return default
    if denominator == 0 or not math.isfinite(denominator) or not math.isfinite(numerator):
        return default
    result = numerator / denominator
    return float(result) if math.isfinite(result) else default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Constrain ``value`` to the inclusive ``[low, high]`` interval."""
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return low
    return float(min(max(float(value), low), high))


def rescale(value: float, low: float, high: float) -> float:
    """Map ``value`` from ``[low, high]`` onto ``[0, 1]`` (clamped at both ends)."""
    if high == low:
        return 0.0
    return clamp(safe_divide(value - low, high - low), 0.0, 1.0)


def human_bytes(num_bytes: float | int | None) -> str:
    """Render a byte count using binary units, e.g. ``1.4 MiB``."""
    if num_bytes is None:
        return "n/a"
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(size) or size < 0:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def format_number(value: Any, decimals: int = 2) -> str:
    """Format a number defensively; non-numeric input is returned as ``str``."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    if numeric.is_integer() and abs(numeric) < 1e15 and decimals > 0:
        return f"{int(numeric):,}"
    if abs(numeric) >= 1e12 or (0 < abs(numeric) < 1e-4):
        return f"{numeric:.3e}"
    return f"{numeric:,.{decimals}f}"


def format_percent(fraction: Any, decimals: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string, e.g. ``12.4%``."""
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def ellipsize(text: str, max_chars: int = 60) -> str:
    """Shorten ``text`` for display, appending an ellipsis when truncated."""
    text = "" if text is None else str(text)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "\u2026"


def join_names(names: Sequence[str], limit: int = config.MAX_COLUMNS_IN_TEXT) -> str:
    """Join column names into a readable sentence, truncating long lists."""
    names = [str(n) for n in names]
    if not names:
        return "none"
    if len(names) <= limit:
        return ", ".join(f"`{n}`" for n in names)
    head = ", ".join(f"`{n}`" for n in names[:limit])
    return f"{head} and {len(names) - limit} more"


# --------------------------------------------------------------------------- #
# pandas / numpy conversion helpers
# --------------------------------------------------------------------------- #
def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and containers to plain Python.

    Streamlit, ``json`` and the HTML report template all prefer native types;
    this removes ``np.int64``-style surprises in one place.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_builtin(row) for row in obj.to_dict(orient="records")]
    return str(obj)


def numeric_series(series: pd.Series, drop_non_finite: bool = True) -> pd.Series:
    """Return ``series`` as float, optionally dropping NaN and infinite values."""
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    if drop_non_finite:
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return numeric


def finite_values(series: pd.Series) -> np.ndarray:
    """Return a 1-D float array of the finite values in ``series``."""
    return numeric_series(series).to_numpy()


def is_probably_numeric_text(series: pd.Series, sample: int | None = None) -> bool:
    """Heuristic: does a text column actually contain numbers with decorations?

    Detects values such as ``"1,234.5"`` or ``"$120"`` that were stored as text
    and therefore silently excluded from every numerical analysis.
    """
    sample = sample or config.TYPE_PROBE_SAMPLE
    values = series.dropna().astype(str)
    if values.empty:
        return False
    stripped = values.head(sample).str.strip()
    if stripped.empty or (stripped == "").all():
        return False
    cleaned = stripped.str.replace(_CURRENCY_AND_SEPARATORS, "", regex=True)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123
    cleaned = cleaned.str.replace(r"%$", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    ratio = float(parsed.notna().mean())
    return ratio >= config.NUMERIC_STRING_RATIO


def is_probably_datetime_text(series: pd.Series, sample: int | None = None) -> bool:
    """Heuristic: does a text column contain parseable dates?

    A regex gate runs first so that plain integers or identifiers such as
    ``"20240131"`` are not misread as timestamps.
    """
    sample = sample or config.TYPE_PROBE_SAMPLE
    values = series.dropna().astype(str)
    if values.empty:
        return False
    probe = values.head(sample)
    if not probe.str.contains(_DATE_HINT, regex=True, na=False).mean() >= 0.8:
        return False
    parsed = pd.to_datetime(probe, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= config.DATETIME_STRING_RATIO


def unique_in_order(values: Iterable[Any]) -> list[Any]:
    """Deduplicate while preserving the original order."""
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def memory_usage_bytes(df: pd.DataFrame) -> int:
    """Total in-memory footprint of ``df`` in bytes (deep, excluding the index)."""
    try:
        total = int(df.memory_usage(index=False, deep=True).sum())
    except Exception:  # pragma: no cover - pandas guarantees this works
        return 0
    return max(total, 0)


def hash_frame(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame's contents, used to key caches and versions."""
    import hashlib

    try:
        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


def non_null_columns(df: pd.DataFrame) -> list[str]:
    """Column names that contain at least one non-null value."""
    return [str(c) for c in df.columns if df[c].notna().any()]


def describe_dataframe(df: pd.DataFrame) -> str:
    """Short human sentence describing a table, e.g. ``1,000 rows x 12 columns``."""
    return f"{len(df):,} rows x {df.shape[1]:,} columns"


def is_integer_valued(series: pd.Series) -> bool:
    """True when every finite value of a numeric series is a whole number."""
    values = finite_values(series)
    if values.size == 0:
        return False
    return bool(np.all(np.equal(np.mod(values, 1), 0)))


def truncate_frame(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Return ``(df_head, was_truncated)`` for display purposes."""
    if len(df) <= max_rows:
        return df, False
    return df.head(max_rows), True
