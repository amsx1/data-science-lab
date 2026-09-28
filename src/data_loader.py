"""Robust CSV loading and structural profiling.

Real-world CSV files are messy: mixed encodings, semicolon delimiters, thousands
separators stored as text, blank trailing lines, duplicated headers. This module
loads such files without raising, reports exactly what it had to work around, and
classifies every column so the rest of the analysis knows what it is looking at.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from src import config
from src.schema import ColumnTypes, DatasetOverview, DtypeIssue
from src.utils import (
    is_probably_datetime_text,
    is_probably_numeric_text,
    memory_usage_bytes,
    safe_divide,
)

#: Encodings tried, in order, when reading a CSV.
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

#: Delimiters offered to the sniffer if sniffing fails.
DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t", "|")

_WHITESPACE_ONLY = re.compile(r"^\s*$")


class DatasetLoadError(Exception):
    """Raised when a file cannot be interpreted as a tabular dataset at all."""


@dataclass
class LoadResult:
    """A loaded DataFrame plus a record of every workaround that was applied."""

    frame: pd.DataFrame
    filename: str
    encoding: str
    delimiter: str
    warnings: list[str] = field(default_factory=list)
    n_rows_read: int = 0


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _is_binary_like(data: bytes) -> bool:
    """True when the payload looks like a binary (non-text) file."""
    if not data:
        return False
    sample = data[:4096]
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        # Not fatal (could be cp1252), so only treat clear binary magic as fatal.
        for magic in (b"PK\x03\x04", b"\x89PNG", b"\xff\xd8\xff", b"%PDF", b"\x1f\x8b"):
            if sample.startswith(magic):
                return True
    return False


def _decode_bytes(data: bytes) -> tuple[str, str, list[str]]:
    """Decode raw bytes into text, returning ``(text, encoding, warnings)``."""
    warnings: list[str] = []
    for encoding in ENCODING_CANDIDATES:
        try:
            return data.decode(encoding), encoding, warnings
        except UnicodeDecodeError:
            continue
    text = data.decode("utf-8", errors="replace")
    warnings.append(
        "The file is not valid UTF-8, CP1252 or Latin-1; undecodable bytes were "
        "replaced with U+FFFD. Check for corrupted characters."
    )
    return text, "utf-8 (errors=replace)", warnings


def _sniff_delimiter(text: str, fallback: str = ",") -> tuple[str, bool]:
    """Detect the delimiter of a CSV payload.

    Returns the delimiter and whether sniffing succeeded. The extension-based
    fallback keeps behaviour predictable for single-column files, where sniffing
    is unreliable.
    """
    import csv

    sample_lines = [line for line in text.splitlines()[:50] if line.strip()]
    if not sample_lines:
        return fallback, False
    sample = "\n".join(sample_lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(DELIMITER_CANDIDATES))
        return dialect.delimiter, True
    except csv.Error:
        counts = {d: sample.count(d) for d in DELIMITER_CANDIDATES}
        best = max(counts, key=lambda key: counts[key])
        if counts[best] > 0:
            return best, False
        return fallback, False


def load_dataframe(
    source: str | Path | BinaryIO | bytes,
    filename: str | None = None,
    delimiter: str | None = None,
    max_rows: int | None = None,
) -> LoadResult:
    """Load a CSV payload from a path, a file-like object or raw bytes.

    The loader never raises on malformed content unless the payload is not text
    at all (``DatasetLoadError``). Everything it had to repair is recorded in
    ``LoadResult.warnings`` so the UI can be transparent about it.

    Args:
        source: Path, file-like object or raw bytes of the CSV file.
        filename: Display name; inferred from ``source`` when omitted.
        delimiter: Force a delimiter instead of sniffing one.
        max_rows: Optional row cap for very large uploads.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DatasetLoadError(f"File not found: {path}")
        if path.is_dir():
            raise DatasetLoadError(f"{path} is a directory, not a file.")
        data = path.read_bytes()
        filename = filename or path.name
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        filename = filename or "uploaded.csv"
    else:  # file-like object (e.g. Streamlit's UploadedFile)
        raw = source.read()
        data = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
        filename = filename or getattr(source, "name", "uploaded.csv")

    if not data or _WHITESPACE_ONLY.match(data.decode("utf-8", errors="ignore") or " "):
        raise DatasetLoadError("The file is empty.")

    if _is_binary_like(data):
        raise DatasetLoadError(
            "The file appears to be binary, not text. DATA AUTOPSY reads delimited "
            "text files (CSV/TSV). Export your data to CSV first."
        )

    text, encoding, warnings = _decode_bytes(data)
    if not text.strip():
        raise DatasetLoadError("The file contains no readable characters.")

    delimiter_used = delimiter or ""
    if delimiter is None:
        delimiter_used, sniffed = _sniff_delimiter(text)
        if not sniffed:
            warnings.append(
                f"Delimiter could not be sniffed reliably; '{delimiter_used}' was "
                "used. Verify the resulting column layout."
            )

    frame = _parse_text(text, delimiter_used, warnings)
    frame, extra_warnings = _clean_frame(frame)
    warnings.extend(extra_warnings)

    if max_rows is not None and len(frame) > max_rows:
        warnings.append(
            f"Only the first {max_rows:,} of {len(frame):,} rows were analysed to "
            "keep the investigation responsive."
        )
        frame = frame.head(max_rows)

    return LoadResult(
        frame=frame,
        filename=filename or "uploaded.csv",
        encoding=encoding,
        delimiter=delimiter_used,
        warnings=warnings,
        n_rows_read=len(frame),
    )


def _parse_text(text: str, delimiter: str, warnings: list[str]) -> pd.DataFrame:
    """Parse decoded text into a DataFrame, with progressive fallbacks."""
    attempts: list[dict[str, Any]] = [
        {"sep": delimiter, "engine": "c"},
        {"sep": delimiter, "engine": "python"},
        {"sep": delimiter, "engine": "python", "on_bad_lines": "skip"},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                skip_blank_lines=True,
                skipinitialspace=True,
                **kwargs,
            )
            if kwargs.get("on_bad_lines") == "skip":
                warnings.append(
                    "Some rows had more/fewer fields than the header and were "
                    "skipped. Inspect the raw file for quoting problems."
                )
            return frame
        except pd.errors.EmptyDataError as exc:
            raise DatasetLoadError("The file contains a header but no data rows.") from exc
        except Exception as exc:  # noqa: BLE001 - pandas raises many parser types
            last_error = exc

    # Last resort: let pandas infer the separator from a single column of text.
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise DatasetLoadError(
            f"The file could not be parsed as delimited text: {last_error or exc}"
        ) from exc


_MANGLED_SUFFIX = re.compile(r"^(?P<base>.+)\.(?P<number>\d+)$")


def _recover_mangled_name(name: str, earlier_names: list[str]) -> str:
    """Restore a duplicated header that pandas renamed to ``"<name>.1"``.

    ``read_csv`` silently mangles repeated headers, so ``age, age`` arrives as
    ``age`` and ``age.1``. This function recognises the pattern *only* when the
    stripped base already appeared earlier, which avoids damaging legitimate names
    such as ``v1.1``.
    """
    match = _MANGLED_SUFFIX.match(name)
    if not match:
        return name
    base = match.group("base").strip()
    if base and any(base == earlier.strip() for earlier in earlier_names):
        return f"{base}_duplicate_{match.group('number')}"
    return name


def _clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Tidy headers and drop rows/columns that carry no information."""
    warnings: list[str] = []

    if frame.empty and frame.shape[1] == 0:
        raise DatasetLoadError("No columns were detected in the file.")

    # Normalise column names: strip whitespace, recover pandas' duplicate mangling
    # ("age" + "age " -> ["age", "age .1"]) and then collapse genuine duplicates.
    original = [str(c) for c in frame.columns]
    cleaned = [_recover_mangled_name(c, original[:position]) for position, c in enumerate(original)]
    cleaned = [c.strip() for c in cleaned]
    seen: dict[str, int] = {}
    renamed: list[str] = []
    for name in cleaned:
        if name in seen:
            seen[name] += 1
            renamed.append(f"{name}_duplicate_{seen[name]}")
        else:
            seen[name] = 0
            renamed.append(name if name else "unnamed_column")
    if renamed != original:
        frame.columns = renamed
        warnings.append(
            "Column headers contained whitespace or duplicates and were "
            "normalised (e.g. 'age ' -> 'age', repeated 'id' -> 'id_duplicate_1')."
        )

    first_col = str(frame.columns[0])
    unnamed = bool(re.match(r"^Unnamed: \d+$", first_col.strip()))
    if unnamed and frame.shape[1] > 1:
        warnings.append(
            "The first column looked like a pandas index artefact and was dropped. "
            "If it held real data, re-export the file without an index column."
        )
        frame = frame.drop(columns=[first_col])

    empty_columns = [c for c in frame.columns if frame[c].isna().all()]
    if empty_columns:
        warnings.append(
            "Entirely empty column(s) detected: "
            + ", ".join(f"'{c}'" for c in empty_columns[:5])
            + (" ..." if len(empty_columns) > 5 else "")
        )

    # Object columns sometimes inherit stray whitespace from the source system.
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        series = frame[column]
        stripped = series.astype("string").str.strip()
        changed = (stripped.fillna("") != series.astype("string").fillna("")).sum()
        if changed:
            frame[column] = stripped.where(series.notna())
            if changed > 0:
                warnings.append(
                    f"Leading/trailing whitespace was trimmed in {int(changed):,} "
                    f"value(s) of '{column}'."
                )

    # Drop rows that are completely empty (common in exported spreadsheets).
    before = len(frame)
    frame = frame.dropna(how="all").reset_index(drop=True)
    if len(frame) < before:
        warnings.append(f"{before - len(frame):,} completely empty row(s) were removed.")

    if frame.empty:
        raise DatasetLoadError("No data rows remained after removing empty rows.")

    return frame, warnings


# --------------------------------------------------------------------------- #
# Column classification
# --------------------------------------------------------------------------- #
def _looks_like_datetime_column(series: pd.Series) -> bool:
    """True when an object column should be promoted to datetime."""
    if series.dropna().empty:
        return False
    # Guard against numeric-looking strings: those are handled as numeric text.
    if is_probably_numeric_text(series):
        return False
    return is_probably_datetime_text(series)


def classify_columns(frame: pd.DataFrame) -> ColumnTypes:
    """Group columns into numeric, categorical, datetime and boolean roles."""
    types = ColumnTypes()

    for column in frame.columns:
        name = str(column)
        series = frame[column]

        if pd.api.types.is_bool_dtype(series):
            types.boolean.append(name)
            continue

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            types.numeric.append(name)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            types.datetime.append(name)
            continue

        if isinstance(series.dtype, pd.CategoricalDtype):
            types.categorical.append(name)
            continue

        # Remaining object/string columns: decide by content.
        non_null = series.dropna()
        if non_null.empty:
            types.other.append(name)
            continue

        unique_ratio = safe_divide(non_null.nunique(), max(len(non_null), 1), 0.0)

        if _looks_like_datetime_column(series):
            types.datetime.append(name)
            continue

        # Numeric text is *not* promoted silently: it is surfaced as a dtype issue
        # so the user can decide, and it stays visible as a categorical column.
        if is_probably_numeric_text(series) and unique_ratio > 0.5:
            types.categorical.append(name)
            types.text_like.append(name)
            continue

        # Boolean-like text (yes/no, true/false) is treated as categorical.
        if non_null.astype("string").str.lower().isin(
            {"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"}
        ).mean() > 0.95:
            types.categorical.append(name)
            continue

        types.categorical.append(name)
        if unique_ratio > 0.5 and non_null.nunique() > 20:
            types.text_like.append(name)

    return types


def detect_dtype_issues(frame: pd.DataFrame, column_types: ColumnTypes) -> list[DtypeIssue]:
    """Find columns whose stored dtype hides their real content.

    Detects numeric values stored as text, dates stored as text, mixed-type
    columns and boolean-like text. These are silent analysis hazards: a column of
    ``"1,200"`` strings is excluded from every mean, correlation and outlier test.
    """
    issues: list[DtypeIssue] = []

    for column in column_types.categorical:
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        name = str(column)

        if is_probably_numeric_text(series):
            cleaned = (
                non_null.astype(str)
                .str.replace(r"[,\s$€£¥]", "", regex=True)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            affected = float(parsed.notna().mean())
            issues.append(
                DtypeIssue(
                    column=name,
                    issue="numeric_stored_as_text",
                    stored_dtype=str(series.dtype),
                    suggested_dtype="float64",
                    detail=(
                        f"{affected:.0%} of non-null values in '{name}' parse as numbers "
                        "once separators/currency symbols are removed, but the column is "
                        "stored as text."
                    ),
                    affected_rate=affected,
                    example_values=non_null.astype(str).head(3).tolist(),
                )
            )
            continue

        if _looks_like_datetime_column(series):
            parsed_dates = pd.to_datetime(
                non_null.astype(str).head(config.TYPE_PROBE_SAMPLE),
                errors="coerce",
                format="mixed",
            )
            affected = float(parsed_dates.notna().mean())
            issues.append(
                DtypeIssue(
                    column=name,
                    issue="datetime_stored_as_text",
                    stored_dtype=str(series.dtype),
                    suggested_dtype="datetime64[ns]",
                    detail=(
                        f"{affected:.0%} of sampled values in '{name}' parse as dates, but "
                        "the column is stored as text so time-based analysis is disabled."
                    ),
                    affected_rate=affected,
                    example_values=non_null.astype(str).head(3).tolist(),
                )
            )
            continue

        # Mixed types: some values numeric, some not, inside one text column.
        numeric_like = pd.to_numeric(non_null, errors="coerce").notna().mean()
        if 0.05 < numeric_like < 0.95:
            issues.append(
                DtypeIssue(
                    column=name,
                    issue="mixed_types",
                    stored_dtype=str(series.dtype),
                    suggested_dtype="mixed",
                    detail=(
                        f"'{name}' contains a mixture of value formats "
                        f"({numeric_like:.0%} numeric, rest text). This usually indicates "
                        "inconsistent data entry or merged value domains."
                    ),
                    affected_rate=float(numeric_like),
                    example_values=non_null.astype(str).unique()[:5].tolist(),
                )
            )

    return issues


def build_overview(
    frame: pd.DataFrame,
    filename: str,
    load_warnings: list[str] | None = None,
) -> DatasetOverview:
    """Assemble the structural description of a loaded dataset."""
    column_types = classify_columns(frame)

    dtype_rows = [
        {
            "column": str(column),
            "dtype": str(frame[column].dtype),
            "non_null": int(frame[column].notna().sum()),
            "missing": int(frame[column].isna().sum()),
            "missing_pct": float(frame[column].isna().mean() * 100),
            "unique": int(frame[column].nunique(dropna=True)),
            "example": next(
                (str(v) for v in frame[column].dropna().head(1)), ""
            ),
        }
        for column in frame.columns
    ]
    dtype_table = pd.DataFrame(dtype_rows)
    if not dtype_table.empty:
        dtype_table["role"] = dtype_table["column"].map(_role_lookup(column_types))

    duplicate_rows = int(frame.duplicated().sum())

    return DatasetOverview(
        filename=filename,
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        memory_bytes=memory_usage_bytes(frame),
        total_cells=int(frame.shape[0] * frame.shape[1]),
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_rows=duplicate_rows,
        column_types=column_types,
        dtype_table=dtype_table,
        load_warnings=list(load_warnings or []),
    )


def _role_lookup(column_types: ColumnTypes) -> dict[str, str]:
    """Map each column name to its assigned analysis role."""
    roles: dict[str, str] = {}
    for name in column_types.numeric:
        roles[name] = "numeric"
    for name in column_types.boolean:
        roles[name] = "boolean"
    for name in column_types.datetime:
        roles[name] = "datetime"
    for name in column_types.categorical:
        roles[name] = "text-like" if name in column_types.text_like else "categorical"
    for name in column_types.other:
        roles[name] = "empty"
    return roles
