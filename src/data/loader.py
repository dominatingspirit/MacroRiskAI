"""Dataset loading with format auto-detection and schema normalization.

Responsibilities
----------------
* Locate each configured dataset by *stem* (extension-agnostic) in the
  project root.
* Auto-detect whether the file is CSV or Excel and read it accordingly.
* Normalize column headers to the canonical schema by stripping unit
  suffixes (e.g. ``"Sales (INR Cr)"`` -> ``"Sales"``).
* Coerce numeric columns to floats and add a monotone ``time_index`` derived
  from ``Year`` + ``Quarter`` so downstream code can order the panel without
  re-parsing strings.

The loader makes **no assumptions** beyond what the config declares and what
it can verify from the file itself. Anything unexpected (missing stem,
missing canonical columns) raises loudly rather than being silently patched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.io import PROJECT_ROOT, resolve_path


@dataclass
class LoadedDataset:
    """Container for a loaded, schema-normalized dataset."""

    name: str
    path: Path
    file_format: str            # "csv" | "excel"
    raw_columns: list[str]      # headers exactly as found in the file
    frame: pd.DataFrame         # normalized frame (canonical columns + time_index)
    renamed_columns: dict[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.frame)


_QUARTER_TO_NUM = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _detect_and_read(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a file as CSV or Excel based on its extension.

    Returns the raw DataFrame and a format label. Raises for unsupported
    extensions so we never guess about binary content.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), "csv"
    if suffix in {".xlsx", ".xls"}:
        # Requires openpyxl/xlrd; surfaced via requirements.txt.
        return pd.read_excel(path), "excel"
    raise ValueError(
        f"Unsupported file extension '{suffix}' for {path.name}. "
        "Supported: .csv, .xlsx, .xls"
    )


def _find_dataset_file(stem: str, extensions: list[str], root: Path) -> Path:
    """Find exactly one file matching ``stem`` + a supported extension."""
    matches = [root / f"{stem}{ext}" for ext in extensions if (root / f"{stem}{ext}").exists()]
    if not matches:
        raise FileNotFoundError(
            f"No file found for dataset stem '{stem}' in {root} "
            f"with extensions {extensions}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous dataset stem '{stem}': multiple files found {[m.name for m in matches]}."
        )
    return matches[0]


def _normalize_columns(columns: list[str], unit_pattern: str) -> dict[str, str]:
    """Build a rename map that strips unit suffixes and trims whitespace."""
    pattern = re.compile(unit_pattern)
    rename: dict[str, str] = {}
    for col in columns:
        canonical = pattern.sub("", col).strip()
        if canonical != col:
            rename[col] = canonical
    return rename


def _add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``quarter_num`` and a monotone integer ``time_index``.

    ``time_index = Year * 4 + (quarter_num - 1)`` yields a globally
    increasing integer that orders quarters correctly across years and is
    directly comparable across companies.
    """
    if "Quarter" not in df.columns or "Year" not in df.columns:
        raise KeyError("Expected 'Year' and 'Quarter' columns for time indexing.")

    unknown = set(df["Quarter"].unique()) - set(_QUARTER_TO_NUM)
    if unknown:
        raise ValueError(f"Unrecognized Quarter labels: {sorted(unknown)}")

    df = df.copy()
    df["quarter_num"] = df["Quarter"].map(_QUARTER_TO_NUM).astype(int)
    df["Year"] = df["Year"].astype(int)
    df["time_index"] = df["Year"] * 4 + (df["quarter_num"] - 1)
    return df


def load_dataset(
    name: str,
    stem: str,
    *,
    schema: dict[str, Any],
    extensions: list[str],
    root: Path | None = None,
) -> LoadedDataset:
    """Load and normalize a single dataset described by (name, stem)."""
    root = root or PROJECT_ROOT
    path = _find_dataset_file(stem, extensions, root)
    raw_df, fmt = _detect_and_read(path)
    raw_columns = list(raw_df.columns)

    rename = _normalize_columns(raw_columns, schema["unit_suffix_pattern"])
    df = raw_df.rename(columns=rename)

    # Verify the canonical schema is present before proceeding.
    expected = set(schema["identifier_columns"]) | set(schema["numeric_columns"])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{name}' is missing expected canonical columns after "
            f"normalization: {sorted(missing)}"
        )

    # Coerce numeric columns; non-parseable values become NaN and are caught
    # by the audit's missing-value checks rather than crashing the load.
    for col in schema["numeric_columns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _add_time_index(df)

    return LoadedDataset(
        name=name,
        path=path,
        file_format=fmt,
        raw_columns=raw_columns,
        frame=df,
        renamed_columns=rename,
    )


def load_all(config: dict[str, Any]) -> dict[str, LoadedDataset]:
    """Load every dataset declared in ``config['data']['datasets']``."""
    root = resolve_path(config["paths"]["project_root"])
    schema = config["schema"]
    extensions = config["data"]["supported_extensions"]
    datasets: dict[str, LoadedDataset] = {}
    for spec in config["data"]["datasets"]:
        datasets[spec["name"]] = load_dataset(
            spec["name"],
            spec["stem"],
            schema=schema,
            extensions=extensions,
            root=root,
        )
    return datasets
