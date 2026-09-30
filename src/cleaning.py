"""Opt-in dataset cleaning that preserves the uploaded source frame.

Cleaning is deliberately explicit: each operation is selected by the user,
applied to a copy, summarized, and exported as a separate CSV. The source frame
used by the investigation is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from src import config
from src.utils import is_probably_numeric_text, parse_numeric_text, tokenize_name

COMMON_MISSING_MARKERS = frozenset({"", "na", "n/a", "nan", "null", "none"})
IDENTIFIER_TOKENS = set(config.ID_NAME_TOKENS) | {
    "ean", "isbn", "postal", "postcode", "sku", "upc", "zip",
}


@dataclass(frozen=True)
class CleaningChange:
    """One applied transformation, with its impact for the audit table."""

    operation: str
    cells_changed: int = 0
    rows_removed: int = 0
    detail: str = ""


@dataclass
class CleaningResult:
    """A cleaned copy and a compact record of the transformations applied."""

    frame: pd.DataFrame
    changes: list[CleaningChange] = field(default_factory=list)

    @property
    def cells_changed(self) -> int:
        """Number of values converted, filled, or normalized."""
        return sum(change.cells_changed for change in self.changes)

    @property
    def rows_removed(self) -> int:
        """Number of rows removed by selected operations."""
        return sum(change.rows_removed for change in self.changes)


def numeric_text_columns(frame: pd.DataFrame) -> list[str]:
    """Return text columns that appear to contain mostly numeric values."""
    candidates: list[str] = []
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        if set(tokenize_name(str(column))) & IDENTIFIER_TOKENS:
            continue
        if is_probably_numeric_text(frame[column]):
            candidates.append(str(column))
    return candidates


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    normalize_missing_markers: bool = False,
    numeric_columns: Sequence[str] = (),
    remove_duplicate_rows: bool = False,
    missing_strategy: str = "keep",
) -> CleaningResult:
    """Apply selected, auditable transformations to a copy of ``frame``.

    ``missing_strategy`` is one of ``keep``, ``drop_rows``, or ``fill``. Filling
    uses each numeric column's median and each other column's most common value.
    """
    strategies = {"keep", "drop_rows", "fill"}
    if missing_strategy not in strategies:
        raise ValueError(f"missing_strategy must be one of {sorted(strategies)}")

    selected_columns = list(numeric_columns)
    unknown_columns = [column for column in selected_columns if column not in frame.columns]
    if unknown_columns:
        raise ValueError(f"Unknown numeric-text column(s): {unknown_columns}")

    cleaned = frame.copy(deep=True)
    changes: list[CleaningChange] = []

    if normalize_missing_markers:
        normalized_count = 0
        for column in cleaned.select_dtypes(include=["object", "string"]).columns:
            values = cleaned[column]
            markers = (
                values.astype("string")
                .str.strip()
                .str.casefold()
                .isin(COMMON_MISSING_MARKERS)
            )
            markers &= values.notna()
            count = int(markers.sum())
            if count:
                cleaned[column] = values.mask(markers, pd.NA)
                normalized_count += count
        if normalized_count:
            changes.append(
                CleaningChange(
                    operation="Treat common text markers as missing",
                    cells_changed=normalized_count,
                    detail=(
                        f"Converted {normalized_count:,} value(s) such as NA, N/A, "
                        "NULL, None, or NaN to missing."
                    ),
                )
            )

    for column in selected_columns:
        values = cleaned[column]
        parsed = parse_numeric_text(values)
        converted = int((values.notna() & parsed.notna()).sum())
        unparsed = int((values.notna() & parsed.isna()).sum())
        if converted or unparsed:
            cleaned[column] = parsed
            details = f"Converted {converted:,} text value(s) to numbers."
            if unparsed:
                details += f" {unparsed:,} non-empty value(s) could not be parsed and became missing."
            changes.append(
                CleaningChange(
                    operation=f"Convert '{column}' to numeric",
                    cells_changed=converted + unparsed,
                    detail=details,
                )
            )

    if remove_duplicate_rows:
        duplicate_count = int(cleaned.duplicated(keep="first").sum())
        if duplicate_count:
            cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
            changes.append(
                CleaningChange(
                    operation="Remove exact duplicate rows",
                    rows_removed=duplicate_count,
                    detail=f"Removed {duplicate_count:,} repeated row(s); kept the first copy.",
                )
            )

    if missing_strategy == "drop_rows":
        before = len(cleaned)
        cleaned = cleaned.dropna(how="any").reset_index(drop=True)
        removed = before - len(cleaned)
        if removed:
            changes.append(
                CleaningChange(
                    operation="Drop rows with missing values",
                    rows_removed=removed,
                    detail=f"Removed {removed:,} row(s) containing at least one missing value.",
                )
            )
    elif missing_strategy == "fill":
        filled = 0
        filled_columns: list[str] = []
        for column in cleaned.columns:
            if not cleaned[column].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(cleaned[column].dtype):
                fill_value = cleaned[column].median(skipna=True)
                if (
                    pd.api.types.is_integer_dtype(cleaned[column].dtype)
                    and pd.notna(fill_value)
                    and not float(fill_value).is_integer()
                ):
                    cleaned[column] = cleaned[column].astype("Float64")
            else:
                modes = cleaned[column].mode(dropna=True)
                if modes.empty:
                    continue
                fill_value = modes.iloc[0]
            if pd.isna(fill_value):
                continue
            before = int(cleaned[column].isna().sum())
            cleaned[column] = cleaned[column].fillna(fill_value)
            count = before - int(cleaned[column].isna().sum())
            if count:
                filled += count
                filled_columns.append(str(column))
        if filled:
            changes.append(
                CleaningChange(
                    operation="Fill missing values",
                    cells_changed=filled,
                    detail=(
                        f"Filled {filled:,} value(s) using column medians for numeric "
                        "columns and most-common values for other columns. Columns: "
                        + ", ".join(f"'{name}'" for name in filled_columns)
                        + "."
                    ),
                )
            )

    return CleaningResult(frame=cleaned, changes=changes)


def cleaned_filename(filename: str) -> str:
    """Build a simple download name without carrying any uploaded path through."""
    basename = Path(str(filename).replace("\\", "/")).name
    stem = Path(basename).stem or "dataset"
    return f"{stem}_cleaned.csv"

