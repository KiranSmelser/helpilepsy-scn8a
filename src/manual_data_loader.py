"""Ingest manual caregiver medication/seizure CSV data from Box."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
import hashlib
import re
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

from . import box_utils, config

_MEDS_REQUIRED_COLS = {"patient_id", "date", "medication_name"}
_SEIZURE_REQUIRED_COLS = {"patient_id", "seizure_date", "seizure_time"}
_TRUE_TOKENS = {"true", "1", "yes", "y"}
_FALSE_TOKENS = {"false", "0", "no", "n"}
_NULL_TOKENS = {"", "n/a", "na", "none", "null", "nan"}


@dataclass
class ManualDataTables:
    """Container for normalized manual caregiver tables."""

    patient_map: pd.DataFrame
    patients: pd.DataFrame
    medications: pd.DataFrame
    med_intakes: pd.DataFrame
    events: pd.DataFrame

    @classmethod
    def empty(cls) -> "ManualDataTables":
        return cls(
            patient_map=pd.DataFrame(
                columns=[
                    "source_patient_id",
                    "patient_id",
                    "mapping_method",
                    "createdAt",
                    "run_timestamp",
                ]
            ),
            patients=pd.DataFrame(
                columns=[
                    "patient_id",
                    "first_name",
                    "last_name",
                    "email",
                    "timezone",
                    "createdAt",
                    "updatedAt",
                    "type",
                    "run_timestamp",
                ]
            ),
            medications=pd.DataFrame(
                columns=[
                    "medication_id",
                    "patient_id",
                    "name",
                    "reason",
                    "treatment_type",
                    "intake_type",
                    "is_deleted",
                    "createdAt",
                    "updatedAt",
                ]
            ),
            med_intakes=pd.DataFrame(
                columns=[
                    "intake_event_id",
                    "patient_id",
                    "medication_id",
                    "intake_id",
                    "dosage_index",
                    "date",
                    "time",
                    "real_time",
                    "moment",
                    "dose",
                    "unit",
                    "taken",
                    "taken_date",
                    "deleted",
                    "createdAt",
                    "updatedAt",
                    "intake_from",
                    "intake_to",
                    "day_1",
                    "day_2",
                    "day_3",
                    "day_4",
                    "day_5",
                    "day_6",
                    "day_7",
                ]
            ),
            events=pd.DataFrame(
                columns=[
                    "event_id",
                    "patient_id",
                    "type",
                    "date",
                    "duration",
                    "seizure_type",
                    "triggers",
                    "felt",
                    "during_sleep",
                    "remark",
                    "createdAt",
                    "updatedAt",
                ]
            ),
        )


def _normalize_col(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_moment(value: object) -> str:
    token = _normalize_text(value).lower()
    mapping = {
        "morning": "morning",
        "noon": "noon",
        "afternoon": "afternoon",
        "evening": "evening",
        "night": "night",
        "notapplicable": "prn",
        "not_applicable": "prn",
        "na": "prn",
        "n_a": "prn",
    }
    return mapping.get(token, token or "unknown")


def _normalize_clock(value: object) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        match = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", text)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
    else:
        hour = int(parsed.hour)
        minute = int(parsed.minute)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _default_clock(moment: str) -> str:
    defaults = {
        "morning": "08:00",
        "noon": "12:00",
        "afternoon": "15:00",
        "evening": "19:00",
        "night": "22:00",
        "prn": "00:00",
        "unknown": "00:00",
    }
    return defaults.get(moment, "00:00")


def _normalize_timezone(value: object) -> str:
    token = _normalize_text(value)
    if not token:
        return "UTC"
    try:
        ZoneInfo(token)
        return token
    except Exception:
        return "UTC"


def _to_bool(value: object, *, default: bool = False) -> bool:
    token = _normalize_text(value).lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return default


def _null_if_empty(value: object) -> str | None:
    token = _normalize_text(value)
    if token.lower() in _NULL_TOKENS:
        return None
    return token


def _to_utc_iso(
    date_value: object,
    time_value: object = None,
    timezone_value: object = None,
) -> str | None:
    date_part = _normalize_text(date_value)
    if not date_part:
        return None

    clock = _normalize_clock(time_value) or "00:00"
    parsed = pd.to_datetime(f"{date_part} {clock}", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(date_part, errors="coerce")
    if pd.isna(parsed):
        return None

    tz_name = _normalize_timezone(timezone_value)
    dt = parsed.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz_name))
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _stable_object_id(*parts: object) -> str:
    payload = "|".join(_normalize_text(p).lower() for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def _manual_patient_id(source_patient_id: str) -> str:
    return _stable_object_id("manual_patient", source_patient_id)


def _detect_kind(df: pd.DataFrame, filename: str) -> str | None:
    cols = {_normalize_col(c) for c in df.columns}
    if _MEDS_REQUIRED_COLS.issubset(cols):
        return "meds"
    if _SEIZURE_REQUIRED_COLS.issubset(cols):
        return "seizures"

    lower_name = filename.lower()
    if "seizure" in lower_name:
        return "seizures"
    if "med" in lower_name:
        return "meds"
    return None


def _series_or_blank(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([""] * len(df), index=df.index, dtype=object)


def _append_dedup(
    path: Path,
    incoming: pd.DataFrame,
    *,
    key_cols: Iterable[str],
) -> tuple[int, int]:
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame()

    if existing.empty and incoming.empty:
        if not path.exists():
            pd.DataFrame().to_csv(path, index=False)
        return (0, 0)

    ordered_cols: List[str] = list(existing.columns)
    for col in incoming.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    if not ordered_cols:
        ordered_cols = list(incoming.columns)

    merged = pd.concat(
        [
            existing.reindex(columns=ordered_cols),
            incoming.reindex(columns=ordered_cols),
        ],
        ignore_index=True,
    )

    subset = [c for c in key_cols if c in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="last")

    merged.to_csv(path, index=False)
    return (len(incoming), len(merged))


def _build_tables(
    meds_raw: pd.DataFrame,
    seizures_raw: pd.DataFrame,
    run_timestamp: str,
) -> ManualDataTables:
    if meds_raw.empty and seizures_raw.empty:
        return ManualDataTables.empty()

    meds = meds_raw.copy()
    seizures = seizures_raw.copy()

    if not meds.empty:
        meds.columns = [_normalize_col(c) for c in meds.columns]
        meds["source_patient_id"] = _series_or_blank(meds, "patient_id").map(
            _normalize_text
        )
        meds = meds[meds["source_patient_id"].ne("")].copy()
    if not seizures.empty:
        seizures.columns = [_normalize_col(c) for c in seizures.columns]
        seizures["source_patient_id"] = _series_or_blank(seizures, "patient_id").map(
            _normalize_text
        )
        seizures = seizures[seizures["source_patient_id"].ne("")].copy()

    source_ids = set()
    if not meds.empty:
        source_ids.update(meds["source_patient_id"].dropna().tolist())
    if not seizures.empty:
        source_ids.update(seizures["source_patient_id"].dropna().tolist())
    source_ids = {sid for sid in source_ids if _normalize_text(sid)}
    if not source_ids:
        return ManualDataTables.empty()

    patient_id_map: Dict[str, str] = {
        sid: _manual_patient_id(sid) for sid in sorted(source_ids)
    }

    timezone_by_source: Dict[str, str] = {}
    if not seizures.empty and "timezone" in seizures.columns:
        sz_tz = seizures[["source_patient_id", "timezone"]].copy()
        sz_tz["timezone"] = sz_tz["timezone"].map(_normalize_timezone)
        tz_mode = (
            sz_tz.groupby("source_patient_id")["timezone"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UTC")
            .to_dict()
        )
        timezone_by_source.update(tz_mode)

    if not meds.empty:
        meds["patient_id"] = meds["source_patient_id"].map(patient_id_map)
        meds["moment"] = _series_or_blank(meds, "time_of_day").map(_normalize_moment)
        meds["clock"] = _series_or_blank(meds, "taken_time").map(_normalize_clock)
        meds["clock"] = meds.apply(
            lambda r: r["clock"] or _default_clock(r["moment"]),
            axis=1,
        )
        meds["timezone"] = meds["source_patient_id"].map(timezone_by_source).fillna(
            "UTC"
        )
        meds["date_iso"] = meds.apply(
            lambda r: _to_utc_iso(r.get("date"), r.get("clock"), r.get("timezone")),
            axis=1,
        )
        meds["taken_flag"] = _series_or_blank(meds, "taken").map(
            lambda v: _to_bool(v, default=False)
        )
        meds["taken_date"] = meds.apply(
            lambda r: r["date_iso"] if bool(r["taken_flag"]) else None, axis=1
        )
        meds["medication_name"] = _series_or_blank(meds, "medication_name").map(
            _normalize_text
        )
        meds["indication"] = _series_or_blank(meds, "indication").map(_normalize_text)
        meds["form"] = _series_or_blank(meds, "form").map(_normalize_text)
        meds["dose"] = _series_or_blank(meds, "dose").map(_null_if_empty)
        meds["dose_unit"] = _series_or_blank(meds, "dose_unit").map(_null_if_empty)
        meds = meds[meds["medication_name"].ne("")].copy()
        meds["medication_id"] = meds.apply(
            lambda r: _stable_object_id(
                "manual_medication",
                r["patient_id"],
                r["medication_name"],
                r["form"],
                r["indication"],
            ),
            axis=1,
        )
        meds["intake_id"] = meds.apply(
            lambda r: _stable_object_id(
                "manual_intake",
                r["patient_id"],
                r["medication_id"],
                r["moment"],
                _normalize_text(r.get("prn")),
            ),
            axis=1,
        )
        meds["intake_event_id"] = meds.apply(
            lambda r: _stable_object_id(
                "manual_intake_event",
                r["patient_id"],
                r.get("date"),
                r.get("clock"),
                r["medication_id"],
                r["dose"],
                r["dose_unit"],
                r["moment"],
                _normalize_text(r.get("notes")),
            ),
            axis=1,
        )
    else:
        meds = pd.DataFrame()

    if not seizures.empty:
        seizures["patient_id"] = seizures["source_patient_id"].map(patient_id_map)
        seizures["timezone"] = _series_or_blank(seizures, "timezone").map(
            _normalize_timezone
        )
        seizures["date_iso"] = seizures.apply(
            lambda r: _to_utc_iso(
                r.get("seizure_date"),
                r.get("seizure_time"),
                r.get("timezone"),
            ),
            axis=1,
        )
        seizures["duration"] = pd.to_numeric(
            _series_or_blank(seizures, "duration_seconds"), errors="coerce"
        )
        seizures["seizure_type"] = _series_or_blank(seizures, "seizure_type").map(
            _normalize_text
        )
        seizures["triggers"] = _series_or_blank(seizures, "triggers").map(
            _null_if_empty
        )
        seizures["felt"] = _series_or_blank(seizures, "felt_by_patient").map(
            _null_if_empty
        )
        sleep_col = "occured_during_sleep"
        if sleep_col not in seizures.columns:
            sleep_col = "occurred_during_sleep"
        seizures["during_sleep"] = _series_or_blank(seizures, sleep_col).map(
            _null_if_empty
        )
        seizures["remark"] = _series_or_blank(seizures, "remark").map(_null_if_empty)
        seizures["event_id"] = seizures.apply(
            lambda r: _stable_object_id(
                "manual_seizure_event",
                r["patient_id"],
                r.get("seizure_date"),
                r.get("seizure_time"),
                r.get("seizure_type"),
                r.get("duration_seconds"),
            ),
            axis=1,
        )
    else:
        seizures = pd.DataFrame()

    timestamp_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    patient_map = pd.DataFrame(
        [
            {
                "source_patient_id": sid,
                "patient_id": pid,
                "mapping_method": "sha1_24hex_manual_patient_namespace",
                "createdAt": timestamp_now,
                "run_timestamp": run_timestamp,
            }
            for sid, pid in patient_id_map.items()
        ]
    )

    patient_bounds: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    for sid, pid in patient_id_map.items():
        dates: list[pd.Timestamp] = []
        if not meds.empty:
            m_dates = pd.to_datetime(
                meds.loc[meds["patient_id"] == pid, "date_iso"],
                utc=True,
                errors="coerce",
            )
            dates.extend([d for d in m_dates if pd.notna(d)])
        if not seizures.empty:
            s_dates = pd.to_datetime(
                seizures.loc[seizures["patient_id"] == pid, "date_iso"],
                utc=True,
                errors="coerce",
            )
            dates.extend([d for d in s_dates if pd.notna(d)])
        if dates:
            patient_bounds[sid] = (min(dates), max(dates))
        else:
            patient_bounds[sid] = (None, None)

    patients = []
    for source_id, patient_id in patient_id_map.items():
        first_dt, last_dt = patient_bounds.get(source_id, (None, None))
        created_at = (
            first_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            if first_dt is not None
            else timestamp_now
        )
        updated_at = (
            last_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            if last_dt is not None
            else created_at
        )
        patients.append(
            {
                "patient_id": patient_id,
                "first_name": source_id,
                "last_name": "",
                "email": "",
                "timezone": timezone_by_source.get(source_id, "UTC"),
                "createdAt": created_at,
                "updatedAt": updated_at,
                "type": "manual_data",
                "run_timestamp": run_timestamp,
            }
        )
    patients_df = pd.DataFrame(patients)

    if meds.empty:
        medications_df = ManualDataTables.empty().medications
        med_intakes_df = ManualDataTables.empty().med_intakes
    else:
        meds_grouped = meds.groupby(["patient_id", "medication_id"], as_index=False)
        medications_df = meds_grouped.agg(
            name=("medication_name", "first"),
            reason=("indication", "first"),
            intake_type=("form", "first"),
            createdAt=("date_iso", "min"),
            updatedAt=("date_iso", "max"),
        )
        medications_df["treatment_type"] = "drug"
        medications_df["is_deleted"] = False
        medications_df = medications_df[
            [
                "medication_id",
                "patient_id",
                "name",
                "reason",
                "treatment_type",
                "intake_type",
                "is_deleted",
                "createdAt",
                "updatedAt",
            ]
        ]

        med_intakes_df = meds[
            [
                "intake_event_id",
                "patient_id",
                "medication_id",
                "intake_id",
                "date_iso",
                "clock",
                "moment",
                "dose",
                "dose_unit",
                "taken_flag",
                "taken_date",
                "date",
            ]
        ].copy()
        med_intakes_df = med_intakes_df.rename(
            columns={
                "date_iso": "date",
                "clock": "time",
                "dose_unit": "unit",
                "taken_flag": "taken",
                "date": "intake_from",
            }
        )
        med_intakes_df["dosage_index"] = 0
        med_intakes_df["real_time"] = med_intakes_df["time"]
        med_intakes_df["deleted"] = False
        med_intakes_df["createdAt"] = med_intakes_df["date"]
        med_intakes_df["updatedAt"] = med_intakes_df["date"]
        med_intakes_df["intake_to"] = pd.NA
        for day_col in ("day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7"):
            med_intakes_df[day_col] = 1
        med_intakes_df = med_intakes_df[
            [
                "intake_event_id",
                "patient_id",
                "medication_id",
                "intake_id",
                "dosage_index",
                "date",
                "time",
                "real_time",
                "moment",
                "dose",
                "unit",
                "taken",
                "taken_date",
                "deleted",
                "createdAt",
                "updatedAt",
                "intake_from",
                "intake_to",
                "day_1",
                "day_2",
                "day_3",
                "day_4",
                "day_5",
                "day_6",
                "day_7",
            ]
        ]

    if seizures.empty:
        events_df = ManualDataTables.empty().events
    else:
        events_df = seizures[
            [
                "event_id",
                "patient_id",
                "date_iso",
                "duration",
                "seizure_type",
                "triggers",
                "felt",
                "during_sleep",
                "remark",
            ]
        ].copy()
        events_df = events_df.rename(columns={"date_iso": "date"})
        events_df["type"] = "seizure"
        events_df["createdAt"] = events_df["date"]
        events_df["updatedAt"] = events_df["date"]
        events_df = events_df[
            [
                "event_id",
                "patient_id",
                "type",
                "date",
                "duration",
                "seizure_type",
                "triggers",
                "felt",
                "during_sleep",
                "remark",
                "createdAt",
                "updatedAt",
            ]
        ]

    return ManualDataTables(
        patient_map=patient_map,
        patients=patients_df,
        medications=medications_df,
        med_intakes=med_intakes_df,
        events=events_df,
    )


def ingest_box(
    folder_id: str,
    access_token: str,
    raw_snapshot_dir: Path,
    run_timestamp: str,
) -> tuple[ManualDataTables, Path]:
    """Download manual caregiver CSV files from Box and normalize them."""
    try:
        files = box_utils.list_files(
            folder_id,
            access_token,
            patterns=config.MANUAL_DATA_GLOB,
        )
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[manual_data/box] Failed to list folder items: {exc}")
        return ManualDataTables.empty(), raw_snapshot_dir

    if not files:
        tqdm.write(f"[manual_data/box] No matching files found in folder {folder_id}.")
        return ManualDataTables.empty(), raw_snapshot_dir

    meds_frames: List[pd.DataFrame] = []
    seizure_frames: List[pd.DataFrame] = []

    with tqdm(total=len(files), desc="Manual Data (Box)", unit="file") as pbar:
        for entry in files:
            file_id = str(entry.get("id"))
            name = str(entry.get("name"))
            try:
                local_path = box_utils.download_file(
                    file_id=file_id,
                    filename=name,
                    token=access_token,
                    dest_dir=Path(raw_snapshot_dir),
                )
                df = pd.read_csv(local_path, dtype=str, encoding="utf-8-sig")
                df.columns = [_normalize_col(c) for c in df.columns]
                kind = _detect_kind(df, name)
                if kind is None:
                    tqdm.write(
                        f"[manual_data/box] Skipping {name}: unsupported columns."
                    )
                    continue
                df["source_file"] = name
                df["run_timestamp"] = run_timestamp
                if kind == "meds":
                    meds_frames.append(df)
                else:
                    seizure_frames.append(df)
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[manual_data/box] Skipping {name}: {exc}")
            finally:
                pbar.update(1)

    meds = (
        pd.concat(meds_frames, ignore_index=True)
        if meds_frames
        else pd.DataFrame(columns=list(_MEDS_REQUIRED_COLS))
    )
    seizures = (
        pd.concat(seizure_frames, ignore_index=True)
        if seizure_frames
        else pd.DataFrame(columns=list(_SEIZURE_REQUIRED_COLS))
    )

    tables = _build_tables(meds, seizures, run_timestamp)
    return tables, raw_snapshot_dir


def write_outputs(
    tables: ManualDataTables | None,
    processed_dir: Path,
) -> dict[str, int]:
    """Append normalized manual data tables into processed outputs."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if tables is None:
        return {
            "patient_map_rows": 0,
            "patients_rows": 0,
            "medications_rows": 0,
            "med_intakes_rows": 0,
            "events_rows": 0,
        }

    if (
        tables.patient_map.empty
        and tables.patients.empty
        and tables.medications.empty
        and tables.med_intakes.empty
        and tables.events.empty
    ):
        return {
            "patient_map_rows": 0,
            "patients_rows": 0,
            "medications_rows": 0,
            "med_intakes_rows": 0,
            "events_rows": 0,
        }

    appended, _ = _append_dedup(
        processed_dir / "manual_patient_map.csv",
        tables.patient_map,
        key_cols=["patient_id"],
    )
    patients_appended, _ = _append_dedup(
        processed_dir / "patients.csv",
        tables.patients,
        key_cols=["patient_id"],
    )
    medications_appended, _ = _append_dedup(
        processed_dir / "medications.csv",
        tables.medications,
        key_cols=["medication_id"],
    )
    med_intakes_appended, _ = _append_dedup(
        processed_dir / "med_intakes.csv",
        tables.med_intakes,
        key_cols=["intake_event_id"],
    )
    events_appended, _ = _append_dedup(
        processed_dir / "events.csv",
        tables.events,
        key_cols=["event_id"],
    )

    return {
        "patient_map_rows": appended,
        "patients_rows": patients_appended,
        "medications_rows": medications_appended,
        "med_intakes_rows": med_intakes_appended,
        "events_rows": events_appended,
    }
