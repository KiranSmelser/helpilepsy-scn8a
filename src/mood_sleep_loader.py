"""Ingest mood and sleep data from Box."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import re
import hashlib

import pandas as pd
from tqdm import tqdm

from . import config, box_utils


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
    """Standardize df into the canonical long format.

    Accepts multiple vendor schemas by loosely matching column names.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "patient_email_raw",
                "patient_name_raw",
                "patient_surname_raw",
                "datetime",
                "type",
                "value",
                "email_key",
                "name_key",
                "ms_id",
                "run_timestamp",
            ]
        )

    # Normalize column names
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    colmap = {_norm(c): c for c in df.columns}

    def _find(*cands: str) -> str | None:
        for c in cands:
            if _norm(c) in colmap:
                return colmap[_norm(c)]
        for c in cands:
            if c in df.columns:
                return c
        return None

    # Core fields
    email_col = _find(
        "Patient Email", "Email", "Email Address", "E-mail", "PatientEmail"
    )
    first_col = _find(
        "Patient Name", "First Name", "Firstname", "Given Name", "GivenName"
    )
    last_col = _find(
        "Patient Surname",
        "Last Name",
        "Lastname",
        "Surname",
        "Family Name",
        "FamilyName",
    )
    full_name_col = _find(
        "Patient", "Patient Full Name", "Name", "Full Name", "PatientName"
    )
    date_col = _find(
        "Date",
        "Datetime",
        "Date/Time",
        "Timestamp",
        "Recorded At",
        "Created At",
        "Event Date",
    )
    type_col = _find("Event Type", "Type", "Event", "Parameter", "Metric", "Category")
    value_col = _find(
        "Value", "Data Value", "Score", "Amount", "Number", "Measurement", "Quantity"
    )

    # Wide layout hints
    mood_wide_col = _find("Mood", "Mood Score", "Mood Rating", "MoodValue")
    sleep_wide_col = _find(
        "Sleep",
        "Sleep Hours",
        "Sleep Duration",
        "Sleep Time",
        "SleepValue",
        "Sleep Score",
    )

    work = df.copy()

    rows: list[pd.DataFrame] = []

    if type_col is not None and (
        value_col is not None or (mood_wide_col or sleep_wide_col)
    ):
        # Long schema
        out = work.copy()
        out = out.rename(
            columns={
                email_col or "patient_email_raw": "patient_email_raw",
                first_col or "patient_name_raw": "patient_name_raw",
                last_col or "patient_surname_raw": "patient_surname_raw",
                date_col or "datetime_raw": "datetime_raw",
                type_col: "type",
                **({value_col: "value"} if value_col else {}),
            }
        )

        if "value" not in out.columns:

            def _extract_value(r):
                t = str(r.get("type", "")).strip().lower()
                if t == "mood" and mood_wide_col:
                    return r.get(mood_wide_col)
                if t == "sleep" and sleep_wide_col:
                    return r.get(sleep_wide_col)
                return None

            out["value"] = out.apply(_extract_value, axis=1)

        rows.append(out)
    elif mood_wide_col or sleep_wide_col:
        # Wide schema
        base_cols = {}
        if email_col:
            base_cols[email_col] = "patient_email_raw"
        if first_col:
            base_cols[first_col] = "patient_name_raw"
        if last_col:
            base_cols[last_col] = "patient_surname_raw"
        if date_col:
            base_cols[date_col] = "datetime_raw"

        out = work.rename(columns=base_cols)

        long_frames: list[pd.DataFrame] = []
        if mood_wide_col:
            mm = out.copy()
            mm["type"] = "mood"
            mm["value"] = mm[mood_wide_col]
            long_frames.append(mm)
        if sleep_wide_col:
            ss = out.copy()
            ss["type"] = "sleep"
            ss["value"] = ss[sleep_wide_col]
            long_frames.append(ss)

        out = pd.concat(long_frames, ignore_index=True, sort=False)
        rows.append(out)
    else:
        out = work.rename(
            columns={
                "Patient Email": "patient_email_raw",
                "Patient Name": "patient_name_raw",
                "Patient Surname": "patient_surname_raw",
                "Date": "datetime_raw",
                "Event Type": "type",
                "Value": "value",
            }
        )
        rows.append(out)

    out = pd.concat(rows, ignore_index=True, sort=False)

    # Ensure required columns exist
    for col in [
        "patient_email_raw",
        "patient_name_raw",
        "patient_surname_raw",
        "datetime_raw",
        "type",
        "value",
    ]:
        if col not in out.columns:
            out[col] = None

    # Build first/last names from full name if needed
    if (
        out["patient_name_raw"].isna()
        | (out["patient_name_raw"].astype(str).str.strip() == "")
    ).all() and full_name_col:
        full_series = work[full_name_col].astype(str)
        firsts = (
            full_series.str.strip()
            .str.split()
            .apply(
                lambda parts: (
                    " ".join(parts[:-1])
                    if len(parts) > 1
                    else (parts[0] if parts else "")
                )
            )
        )
        lasts = (
            full_series.str.strip()
            .str.split()
            .apply(lambda parts: parts[-1] if len(parts) > 1 else "")
        )
        out["patient_name_raw"] = firsts
        out["patient_surname_raw"] = lasts

    # Parse timestamps
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


def ingest_box(
    folder_id: str,
    access_token: str,
    patients_csv: Path,
    raw_snapshot_dir: Path,
    run_timestamp: str,
) -> tuple[pd.DataFrame, Path]:
    """Ingest files from a Box folder using a primary access token."""
    # List items in the folder
    try:
        files = box_utils.list_files(
            folder_id,
            access_token,
            patterns=config.MOOD_SLEEP_GLOB,
        )
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[mood_sleep/box] Failed to list folder items: {exc}")
        return pd.DataFrame(), raw_snapshot_dir

    if not files:
        tqdm.write(f"[mood_sleep/box] No matching files found in folder {folder_id}.")
        return pd.DataFrame(), raw_snapshot_dir

    frames: list[pd.DataFrame] = []
    with tqdm(total=len(files), desc="Mood/Sleep (Box)", unit="file") as pbar:
        for it in files:
            file_id = str(it.get("id"))
            name = str(it.get("name"))
            try:
                # Download into raw snapshot dir for traceability
                local_path = box_utils.download_file(
                    file_id, name, access_token, raw_snapshot_dir
                )
                df_raw = _read_one(local_path)
                df_long = _to_long(df_raw, run_timestamp)
                if not df_long.empty:
                    frames.append(df_long)
                else:
                    tqdm.write(
                        f"[mood_sleep/box] No usable rows after normalization in {name}."
                    )
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[mood_sleep/box] Skipping {name}: {exc}")
            finally:
                pbar.update(1)

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
