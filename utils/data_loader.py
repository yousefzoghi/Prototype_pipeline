"""
data_loader.py — Unified loader for OpenBCI, BrainFlow, and survey data.

Supports:
  - OpenBCI raw CSV/TXT  (comma-delimited, %-prefixed header)
  - BrainFlow raw CSV     (tab-delimited, headerless, 32 columns)
  - Participant survey    (.xlsx with Likert-scale responses)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EEGRecording:
    """Container for a loaded EEG recording."""
    data: pd.DataFrame               # Columns = channels, rows = samples
    sfreq: float                     # Sampling frequency in Hz
    ch_names: list[str]              # EEG channel names
    metadata: dict                   # Arbitrary metadata (board, file, etc.)
    raw: Optional[mne.io.RawArray] = None  # MNE Raw object (lazy-built)

    def build_mne_raw(self) -> mne.io.RawArray:
        """Build (or return cached) MNE RawArray from the EEG channels."""
        if self.raw is not None:
            return self.raw
        
        # Clean data: Fill NaNs/Infs to prevent pipeline crashes
        clean_df = self.data[self.ch_names].copy()
        if clean_df.isna().any().any() or np.isinf(clean_df.values).any():
            # Replace inf with nan, interpolate missing values, and fill remaining with 0
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan).interpolate(method="linear").fillna(0.0)
            
        eeg_data = clean_df.values.T  # (n_channels, n_samples)
        # Convert from µV to V for MNE
        eeg_data = eeg_data * 1e-6
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types="eeg",
        )
        self.raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return self.raw


@dataclass
class SurveyData:
    """Container for parsed participant survey responses."""
    participant_id: str
    responses: dict[str, float]      # question → numeric response
    raw_df: pd.DataFrame             # Original parsed table
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(filepath: str | Path) -> str:
    """Return 'openbci', 'brainflow', or 'survey' based on file content."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return "survey"

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    # OpenBCI lines may be quote-wrapped: "%OpenBCI..."
    stripped = first_line.strip().strip('"')
    if stripped.startswith("%"):
        return "openbci"

    # Tab-delimited numeric data → BrainFlow
    parts = first_line.strip().split("\t")
    if len(parts) >= 20:
        try:
            float(parts[0])
            return "brainflow"
        except ValueError:
            pass

    raise ValueError(
        f"Cannot detect format of '{filepath.name}'. "
        "Expected OpenBCI (%-header), BrainFlow (tab-delimited numeric), "
        "or survey (.xlsx)."
    )


# ---------------------------------------------------------------------------
# OpenBCI loader
# ---------------------------------------------------------------------------

_OPENBCI_META_RE = re.compile(r'^"?%\s*(.+?)"?$')


def _parse_openbci_header(filepath: Path) -> tuple[dict, int]:
    """Parse %-prefixed metadata lines; return metadata dict and data-start line."""
    metadata: dict = {}
    skip_rows = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip().strip('"')
            if stripped.startswith("%"):
                skip_rows += 1
                m = _OPENBCI_META_RE.match(stripped)
                if m:
                    content = m.group(1).strip()
                    if "Number of channels" in content:
                        try:
                            metadata["n_channels"] = int(content.split("=")[1].strip())
                        except (IndexError, ValueError):
                            pass
                    elif "Sample Rate" in content:
                        try:
                            metadata["sfreq"] = float(content.split("=")[1].strip().split()[0])
                        except (IndexError, ValueError):
                            pass
                    elif "Board" in content:
                        metadata["board"] = content
            else:
                break

    return metadata, skip_rows


def load_openbci(filepath: str | Path) -> EEGRecording:
    """Load an OpenBCI raw CSV or TXT file.

    Handles the OpenBCI GUI export format where each line (including
    the column header) may be wrapped in double quotes.
    """
    import io

    filepath = Path(filepath)
    metadata, skip_rows = _parse_openbci_header(filepath)

    # Read all lines, strip outer quotes, skip metadata rows
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    cleaned_lines = []
    for line in all_lines[skip_rows:]:
        stripped = line.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            stripped = stripped[1:-1]
        cleaned_lines.append(stripped)

    csv_text = "\n".join(cleaned_lines)
    df = pd.read_csv(io.StringIO(csv_text), sep=",", engine="python", on_bad_lines="skip")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Identify EXG channels
    exg_cols = [c for c in df.columns if c.startswith("EXG Channel")]
    if not exg_cols:
        # Fallback: columns 1..16
        exg_cols = [df.columns[i] for i in range(1, min(17, len(df.columns)))]

    sfreq = metadata.get("sfreq", 125.0)

    metadata.update({
        "format": "openbci",
        "file": filepath.name,
        "total_samples": len(df),
        "duration_s": len(df) / sfreq,
    })

    return EEGRecording(
        data=df,
        sfreq=sfreq,
        ch_names=exg_cols,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# BrainFlow loader
# ---------------------------------------------------------------------------

# Standard BrainFlow column mapping for Cyton+Daisy (32 cols)
_BRAINFLOW_COLS_32 = (
    ["Sample Index"]
    + [f"EXG Channel {i}" for i in range(16)]
    + [f"Accel Channel {i}" for i in range(3)]
    + [f"Other {i}" for i in range(10)]
    + ["Timestamp", "Marker"]
)


def load_brainflow(filepath: str | Path) -> EEGRecording:
    """Load a BrainFlow raw CSV (tab-delimited, no header)."""
    filepath = Path(filepath)

    df = pd.read_csv(filepath, sep="\t", header=None, engine="python")

    # Drop any fully-NaN trailing columns (from trailing tabs)
    df = df.dropna(axis=1, how="all")

    n_cols = df.shape[1]

    # Build column names dynamically
    if n_cols >= 32:
        col_names = _BRAINFLOW_COLS_32[:n_cols]
        df.columns = col_names
    elif n_cols >= 22:
        # At least Sample Index + 16 EXG + 3 Accel + extras
        col_names = (
            ["Sample Index"]
            + [f"EXG Channel {i}" for i in range(16)]
            + [f"Accel Channel {i}" for i in range(3)]
            + [f"Other {i}" for i in range(n_cols - 20)]
        )
        df.columns = col_names
    else:
        df.columns = [f"Col_{i}" for i in range(n_cols)]

    exg_cols = [c for c in df.columns if c.startswith("EXG Channel")]

    # Try to infer sfreq from the data
    sfreq = 125.0
    if "Other 2" in df.columns:
        mode_val = df["Other 2"].mode()
        if len(mode_val) > 0 and 50 <= mode_val.iloc[0] <= 1000:
            sfreq = float(mode_val.iloc[0])

    metadata = {
        "format": "brainflow",
        "file": filepath.name,
        "n_channels": len(exg_cols),
        "sfreq": sfreq,
        "total_samples": len(df),
        "duration_s": len(df) / sfreq,
    }

    return EEGRecording(
        data=df,
        sfreq=sfreq,
        ch_names=exg_cols,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Survey loader
# ---------------------------------------------------------------------------

def load_survey(filepath: str | Path) -> SurveyData:
    """Load a participant survey XLSX file."""
    filepath = Path(filepath)
    df = pd.read_excel(filepath, header=None, engine="openpyxl")

    responses: dict[str, float] = {}
    raw_rows = []

    for _, row in df.iterrows():
        q = row.iloc[0]
        if pd.isna(q):
            continue
        q_str = str(q).strip()
        if q_str.startswith("•") or q_str.startswith("-"):
            q_clean = q_str.lstrip("•-– ").strip()
            val = row.iloc[2] if len(row) > 2 else np.nan
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan
            responses[q_clean] = val
            raw_rows.append({"question": q_clean, "response": val})

    participant_id = filepath.stem  # e.g. "Participant1"

    return SurveyData(
        participant_id=participant_id,
        responses=responses,
        raw_df=pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame(),
        metadata={"file": filepath.name},
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def load_file(filepath: str | Path) -> EEGRecording | SurveyData:
    """Auto-detect format and load the file."""
    fmt = detect_format(filepath)
    if fmt == "openbci":
        return load_openbci(filepath)
    elif fmt == "brainflow":
        return load_brainflow(filepath)
    elif fmt == "survey":
        return load_survey(filepath)
    else:
        raise ValueError(f"Unknown format: {fmt}")
