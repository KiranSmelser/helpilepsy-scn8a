from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List
import re
import hashlib
import shutil

import pandas as pd
from tqdm import tqdm

from . import config


_LOCKFILE_PREFIXES = ("~$", "._")


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield non-temp files under *root* matching configured patterns."""
    if not root.exists():
        return []
    for pat in config.MOOD_SLEEP_GLOB:
        for p in root.glob(pat):
            name = p.name
            if name.startswith(_LOCKFILE_PREFIXES):
                continue
            try:
                if p.stat().st_size == 0:
                    continue
            except OSError:
                continue
            yield p


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single Excel/CSV file."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            xls = pd.ExcelFile(path)
            sheets: List[pd.DataFrame] = []
            for sheet in xls.sheet_names:
                try:
                    sdf = pd.read_excel(path, sheet_name=sheet)
                    if sdf is None or sdf.empty:
                        continue
                    sdf = sdf.copy()
                    sdf["source_sheet"] = sheet
                    sheets.append(sdf)
                except Exception:
                    continue
            if not sheets:
                raise ValueError("No readable sheets found")
            df = pd.concat(sheets, ignore_index=True, sort=False)
        except Exception as exc:  # noqa: BLE001
            try:
                df = pd.read_excel(path)
            except Exception as inner:  # noqa: BLE001
                raise inner from exc
    else:
        df = pd.read_csv(path)
    df = df.copy()
    df["source_file"] = path.name
    if "source_sheet" not in df.columns:
        df["source_sheet"] = None
    return df


def _normalize_names(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\s*\(.*?\)\s*", "", regex=True)
    return s.str.strip()


def _to_long(df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    """Standardize df into the canonical long format."""
    out = df.rename(
        columns={
            "Patient Email": "patient_email_raw",
            "Patient Name": "patient_name_raw",
            "Patient Surname": "patient_surname_raw",
            "Date": "datetime_raw",
            "Event Type": "type",
            "Value": "value",
        }
    ).copy()
    # Parse timestamps as day-first to match vendor files
    out["datetime"] = pd.to_datetime(
        out["datetime_raw"], dayfirst=True, errors="coerce"
    )
    # Normalized keys for matching
    out["email_key"] = out["patient_email_raw"].astype(str).str.lower().str.strip()
    out["name_key"] = (
        out["patient_name_raw"].astype(str).str.strip().str.lower()
        + " "
        + _normalize_names(out["patient_surname_raw"]).str.lower()
    )
    out["type"] = out["type"].astype(str).str.strip().str.lower()
    out = out[out["type"].isin(["mood", "sleep"])].copy()

    # Deterministic id
    def _hash_row(r) -> str:
        base = f"{r.get('email_key')}|{r.get('name_key')}|{r.get('datetime')}|{r.get('type')}|{r.get('value')}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    out["ms_id"] = out.apply(_hash_row, axis=1)
    out["run_timestamp"] = run_timestamp
    return out


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows by ms_id, keeping earliest by datetime."""
    if df.empty:
        return df
    return df.sort_values("datetime").drop_duplicates(subset=["ms_id"], keep="first")


def _load_patients_index(patients_csv: Path) -> Tuple[pd.DataFrame, dict, dict]:
    """Build lookup indices for email and name -> patient_id."""
    p = pd.read_csv(patients_csv)
    # Require standard columns
    if "patient_id" not in p.columns or "email" not in p.columns:
        return p, {}, {}

    email_index = {
        str(v).lower().strip(): pid
        for v, pid in zip(p["email"], p["patient_id"])
        if pd.notna(v)
    }

    name_index: dict[str, str] = {}
    first = p["first_name"].astype(str).str.strip().str.lower()
    last = p["last_name"].astype(str).str.strip().str.lower()
    nk = first + " " + last
    name_index.update(
        {k: pid for k, pid in zip(nk, p["patient_id"]) if k and k != "nan nan"}
    )

    return p, email_index, name_index


def _attach_patient_id(df: pd.DataFrame, patients_csv: Path) -> pd.DataFrame:
    """Attach patient_id using email first, then name."""
    if df.empty:
        return df.copy()

    _, email_index, name_index = _load_patients_index(patients_csv)
    out = df.copy()

    out["patient_id"] = out["email_key"].map(email_index)
    out["match_method"] = (
        out["patient_id"].notna().map(lambda b: "email" if b else None)
    )

    missing = out["patient_id"].isna()
    if missing.any():
        out.loc[missing, "patient_id"] = out.loc[missing, "name_key"].map(name_index)
        resolved_by_name = missing & out["patient_id"].notna()
        out.loc[resolved_by_name, "match_method"] = "name"

    matched = out[out["patient_id"].notna()].copy()
    return matched


def ingest_local(
    local_dir: Path,
    patients_csv: Path,
    raw_snapshot_dir: Path,
    run_timestamp: str,
) -> tuple[pd.DataFrame, Path]:
    """Ingest local files."""
    local_dir = Path(local_dir)
    raw_snapshot_dir = Path(raw_snapshot_dir)
    raw_snapshot_dir.mkdir(parents=True, exist_ok=True)

    files = list(_iter_files(local_dir))
    if not files:
        tqdm.write(f"[mood_sleep] No files found under {local_dir}.")
        return pd.DataFrame(), raw_snapshot_dir

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df_raw = _read_one(f)
            try:
                shutil.copy2(f, raw_snapshot_dir / f.name)
            except Exception:
                pass
            df_long = _to_long(df_raw, run_timestamp)
            if not df_long.empty:
                frames.append(df_long)
            else:
                tqdm.write(
                    f"[mood_sleep] No usable rows after normalization in {f.name}."
                )
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[mood_sleep] Skipping {f.name}: {exc}")

    if not frames:
        return pd.DataFrame(), raw_snapshot_dir

    ms = _dedup(pd.concat(frames, ignore_index=True))
    matched = _attach_patient_id(ms, patients_csv)
    return matched, raw_snapshot_dir


def write_outputs(ms_df: pd.DataFrame, processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)

    if ms_df.empty:
        out = pd.DataFrame(columns=["ms_id", "patient_id", "date", "type", "value"])
        out.to_csv(processed_dir / "mood_sleep.csv", index=False)
        return

    out = ms_df.rename(columns={"datetime": "date"})[
        [
            "ms_id",
            "patient_id",
            "date",
            "type",
            "value",
        ]
    ].copy()

    out.to_csv(processed_dir / "mood_sleep.csv", index=False)
