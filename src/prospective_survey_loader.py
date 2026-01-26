"""Ingest and normalize prospective survey data from Box."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict
import hashlib
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

from . import box_utils

_FILENAME_PATTERN = re.compile(
    r"(?P<prefix>.+?)_DATA_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4})\.csv$",
    re.IGNORECASE,
)

_SEIZURE_TYPE_COLS: Tuple[str, ...] = (
    "absense",
    "myoclonic",
    "focal",
    "gtc",
    "clonic",
    "atonic",
    "szother",
)

_DEVELOPMENT_COLS: Tuple[str, ...] = (
    "eye_ps",
    "grasp_ps",
    "reach_ps",
    "pincer_ps",
    "blocks_ps",
    "circle_ps",
    "hc_ps",
    "roll_ps",
    "sit_ps",
    "stand_ps",
    "walk_ps",
    "run_ps",
    "smile_ps",
    "wave_ps",
    "cup_ps",
    "fork_ps",
    "wh_ps",
    "bt_ps",
    "vocalize_ps",
    "laugh_ps",
    "babble_ps",
    "namecolors_ps",
    "words_ps",
    "phrase_ps",
    "reade_ps",
)

_CAREGIVER_QOL_COLS: Tuple[str, ...] = (
    "h_ps",
    "r_ps",
    "g_ps",
    "i_ps",
    "m_ps",
    "rp_ps",
    "pe_ps",
    "gm_ps",
    "sbm_ps",
    "bl_ps",
    "ccm_ps",
    "epr_ps",
    "gh_ps",
    "sw_ps",
    "be_ps",
    "bu_ps",
    "sa_ps",
    "aa_ps",
    "bwlw_ps",
    "dh_ps",
    "edcr_ps",
    "ssoba_ps",
    "emtb_ps",
    "efspa_ps",
    "epa_ps",
    "ego_ps",
    "esto_ps",
    "etn_ps",
    "mtc_ps",
    "hcr_ps",
    "emtwh_ps",
    "eut_ps",
    "cgs_ps",
    "cgd_ps",
    "cgm_ps",
    "cgst_ps",
    "cgt_ps",
    "cge_ps",
    "cgexercise_ps",
    "cgstress_ps",
    "stress_resources_ps",
    "cgbreak_ps",
)

_CHECKBOX_GROUPS: Dict[str, Tuple[str, ...]] = {
    "is_dev2": ("is_dev2_ps___1", "is_dev2_ps___2", "is_dev2_ps___3"),
    "whyhospital": (
        "whyhospital_ps___1",
        "whyhospital_ps___2",
        "whyhospital_ps___3",
        "whyhospital_ps___4",
        "whyhospital_ps___5",
        "whyhospital_ps___6",
        "whyhospital_ps___7",
        "whyhospital_ps___8",
        "whyhospital_ps___9",
        "whyhospital_ps___10",
        "whyhospital_ps___11",
        "whyhospital_ps___12",
        "whyhospital_ps___13",
    ),
    "cgsocial": (
        "cgsocial_ps___8",
        "cgsocial_ps___9",
        "cgsocial_ps___10",
        "cgsocial_ps___11",
        "cgsocial_ps___12",
    ),
}

_HOUR_KEYWORDS = {
    "half": 0.5,
    "quarter": 0.25,
}

_FULL_DAY_PATTERNS = (
    "all day",
    "all the time",
    "whole day",
    "entire day",
    "around the clock",
    "24/7",
)

_ZERO_HOUR_PATTERNS = (
    "none yet",
    "none at the moment",
    "none right now",
    "have not",
    "haven't",
    "not currently",
    "not yet",
    "no hours",
    "zero hours",
)

_INVALID_HOUR_RESPONSES = (
    "same as above",
    "n/a",
)


@dataclass
class ProspectiveSurveyTables:
    """Container for normalized prospective survey tables."""

    surveys: pd.DataFrame
    seizure_scores: pd.DataFrame
    development_milestones: pd.DataFrame
    caregiver_qol: pd.DataFrame
    checkbox_responses: pd.DataFrame

    @classmethod
    def empty(cls) -> "ProspectiveSurveyTables":
        return cls(
            surveys=pd.DataFrame(columns=["survey_instance_id"]),
            seizure_scores=pd.DataFrame(
                columns=["survey_instance_id", "seizure_type", "score"]
            ),
            development_milestones=pd.DataFrame(
                columns=["survey_instance_id", "milestone", "status"]
            ),
            caregiver_qol=pd.DataFrame(
                columns=["survey_instance_id", "scale", "score"]
            ),
            checkbox_responses=pd.DataFrame(
                columns=["survey_instance_id", "checkbox_group", "option_code"]
            ),
        )


def _parse_filename_timestamp(filename: str) -> datetime | None:
    match = _FILENAME_PATTERN.search(filename)
    if not match:
        return None
    date_part = match.group("date")
    time_part = match.group("time")
    try:
        dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H%M")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _entry_sort_key(entry: dict) -> tuple[datetime, str]:
    name = str(entry.get("name", ""))
    parsed = _parse_filename_timestamp(name)
    if parsed is None:
        modified = entry.get("modified_at")
        try:
            parsed = datetime.fromisoformat(str(modified).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            parsed = datetime.min.replace(tzinfo=timezone.utc)
    return parsed, name


def _select_latest_entry(entries: List[dict]) -> dict | None:
    if not entries:
        return None
    best: dict | None = None
    best_key: tuple[datetime, str] | None = None
    with tqdm(total=len(entries), desc="Prospective Survey (Box)", unit="file") as pbar:
        for entry in entries:
            key = _entry_sort_key(entry)
            if best_key is None or key > best_key:
                best = entry
                best_key = key
            pbar.update(1)
    return best


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_name(raw: str | float | None) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw)
    text = _strip_diacritics(text)
    text = text.lower().strip()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(part for part in text.split())


@dataclass
class _PatientLookup:
    full: Dict[str, set]
    first_exact: Dict[str, set]
    last_exact: Dict[str, set]
    first_token_unique: Dict[str, str]
    last_token_unique: Dict[str, str]

    def resolve(self, key: str) -> tuple[str, str | None, str | None]:
        """Resolve key to (status, patient_id, strategy)."""
        if not key:
            return "unmatched", None, None

        ids = self.full.get(key)
        if ids:
            if len(ids) == 1:
                return "matched", next(iter(ids)), "full"
            return "ambiguous", None, "full"

        tokens = [tok for tok in key.split() if tok]

        ids = self.first_exact.get(key)
        if ids:
            if len(ids) == 1:
                return "matched", next(iter(ids)), "first_exact"
            return "ambiguous", None, "first_exact"

        ids = self.last_exact.get(key)
        if ids:
            if len(ids) == 1:
                return "matched", next(iter(ids)), "last_exact"
            return "ambiguous", None, "last_exact"

        for tok in tokens:
            pid = self.first_token_unique.get(tok)
            if pid:
                return "matched", pid, "first_token"

        for tok in tokens:
            pid = self.last_token_unique.get(tok)
            if pid:
                return "matched", pid, "last_token"

        return "unmatched", None, None


def _build_patient_lookup(patients_csv: Path) -> _PatientLookup | None:
    try:
        patients = pd.read_csv(
            patients_csv, usecols=["patient_id", "first_name", "last_name"]
        )
    except Exception:
        return None

    if patients.empty:
        return None

    patients = patients.dropna(subset=["patient_id"])
    if patients.empty:
        return None

    patients = patients.fillna({"first_name": "", "last_name": ""})

    full: Dict[str, set] = defaultdict(set)
    first_exact: Dict[str, set] = defaultdict(set)
    last_exact: Dict[str, set] = defaultdict(set)
    first_token_map: Dict[str, set] = defaultdict(set)
    last_token_map: Dict[str, set] = defaultdict(set)

    for _, row in patients.iterrows():
        pid = str(row["patient_id"])
        first_norm = _normalize_name(row.get("first_name"))
        last_norm = _normalize_name(row.get("last_name"))

        full_key = f"{first_norm} {last_norm}".strip()
        if full_key:
            full[full_key].add(pid)

        if first_norm:
            first_exact[first_norm].add(pid)
        if last_norm:
            last_exact[last_norm].add(pid)

        for token in first_norm.split():
            if token:
                first_token_map[token].add(pid)
        for token in last_norm.split():
            if token:
                last_token_map[token].add(pid)

    first_token_unique = {
        token: next(iter(ids))
        for token, ids in first_token_map.items()
        if len(ids) == 1
    }
    last_token_unique = {
        token: next(iter(ids)) for token, ids in last_token_map.items() if len(ids) == 1
    }

    return _PatientLookup(
        full=dict(full),
        first_exact=dict(first_exact),
        last_exact=dict(last_exact),
        first_token_unique=first_token_unique,
        last_token_unique=last_token_unique,
    )


def _attach_patient_ids(df: pd.DataFrame, patients_csv: Path) -> pd.DataFrame:
    lookup = _build_patient_lookup(patients_csv)
    if lookup is None or df.empty:
        out = df.copy()
        out["patient_id"] = None
        out["match_field"] = None
        out["match_status"] = "unmatched"
        out["match_strategy"] = None
        return out

    def _match_row(row: pd.Series) -> Tuple[str | None, str | None, str, str | None]:
        ambiguous_field: str | None = None
        ambiguous_strategy: str | None = None
        candidates = [
            ("is_pname_ps", _normalize_name(row.get("is_pname_ps"))),
            ("is_cgname_ps", _normalize_name(row.get("is_cgname_ps"))),
        ]
        for field, key in candidates:
            if not key:
                continue
            status, pid, strategy = lookup.resolve(key)
            if status == "matched" and pid is not None:
                return pid, field, "matched", strategy
            if status == "ambiguous" and ambiguous_field is None:
                ambiguous_field = field
                ambiguous_strategy = strategy
        if ambiguous_field is not None:
            return None, ambiguous_field, "ambiguous", ambiguous_strategy
        return None, None, "unmatched", None

    matches = df.apply(_match_row, axis=1, result_type="expand")
    matches.columns = ["patient_id", "match_field", "match_status", "match_strategy"]
    out = df.copy()
    out[["patient_id", "match_field", "match_status", "match_strategy"]] = matches
    return out


def _parse_hours(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"nan", "", "-"}:
        return None
    normalized = re.sub(r"\s+", " ", text)
    for invalid in _INVALID_HOUR_RESPONSES:
        if invalid in normalized:
            return None
    for zero in _ZERO_HOUR_PATTERNS:
        if zero in normalized:
            return 0.0
    for phrase in _FULL_DAY_PATTERNS:
        if phrase in normalized:
            return 24.0
    for key, val in _HOUR_KEYWORDS.items():
        if key in normalized:
            if "minute" in text or "min" in text:
                return round(val / 60.0, 3)
            return val
    # Replace commas and extract numeric token
    cleaned = normalized.replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None
    hours = float(match.group(1))
    if any(tok in normalized for tok in ["minute", "min"]):
        return round(hours / 60.0, 3)
    return hours


def _normalize_timestamp_value(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none"}:
        return None
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def _normalize_timestamp_columns(df: pd.DataFrame) -> None:
    timestamp_cols = [col for col in df.columns if str(col).endswith("_timestamp")]
    if not timestamp_cols:
        return
    for col in timestamp_cols:
        target = f"{col}_utc"
        if target in df.columns:
            continue
        normalized = df[col].apply(_normalize_timestamp_value)
        df[target] = normalized


def _compute_survey_ids(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    skip_cols = {
        "survey_instance_id",
        "ingested_at",
        "run_timestamp",
        "source_file",
        "source_exported_at",
    }
    cols = [c for c in sorted(df.columns) if c not in skip_cols]

    def _hash_row(row: pd.Series) -> str:
        signature = "|".join(str(row.get(col, "")) for col in cols)
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()

    return df.apply(_hash_row, axis=1)


def _melt_numeric(
    df: pd.DataFrame,
    columns: Iterable[str],
    value_name: str,
    numeric: bool = True,
) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame(columns=["survey_instance_id", value_name, "score"])

    subset = df[["survey_instance_id"] + available].copy()
    melted = subset.melt(
        id_vars="survey_instance_id",
        value_vars=available,
        var_name=value_name,
        value_name="score",
    )
    if numeric:
        melted["score"] = pd.to_numeric(melted["score"], errors="coerce")
    melted = melted.dropna(subset=["score"])
    return melted.reset_index(drop=True)


def _melt_development(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in _DEVELOPMENT_COLS if c in df.columns]
    if not available:
        return pd.DataFrame(columns=["survey_instance_id", "milestone", "status"])
    subset = df[["survey_instance_id"] + available].copy()
    melted = subset.melt(
        id_vars="survey_instance_id",
        value_vars=available,
        var_name="milestone",
        value_name="status",
    )
    melted["status"] = pd.to_numeric(melted["status"], errors="coerce")
    melted = melted.dropna(subset=["status"])
    return melted.reset_index(drop=True)


def _melt_checkboxes(df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for group, columns in _CHECKBOX_GROUPS.items():
        available = [c for c in columns if c in df.columns]
        if not available:
            continue
        subset = df[["survey_instance_id"] + available].copy()
        melted = subset.melt(
            id_vars="survey_instance_id",
            value_vars=available,
            var_name="option_code",
            value_name="value",
        )
        if melted.empty:
            continue
        norm = melted["value"].astype(str).str.strip().str.lower()
        positive = norm.isin({"1", "true", "yes", "y"})
        melted = melted[positive]
        if melted.empty:
            continue
        melted = melted[["survey_instance_id", "option_code"]]
        melted["checkbox_group"] = group
        frames.append(melted)
    if not frames:
        return pd.DataFrame(
            columns=["survey_instance_id", "checkbox_group", "option_code"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[["survey_instance_id", "checkbox_group", "option_code"]]


def _prepare_tables(
    df: pd.DataFrame,
    patients_csv: Path,
) -> ProspectiveSurveyTables:
    if df.empty:
        return ProspectiveSurveyTables.empty()

    df = _attach_patient_ids(df, patients_csv)
    _normalize_timestamp_columns(df)

    if "observationhrsweek_ps" in df.columns:
        df["observationhrsweek_ps_hours"] = df["observationhrsweek_ps"].apply(
            _parse_hours
        )
    else:
        df["observationhrsweek_ps_hours"] = None

    if "observationhrsweekend_ps" in df.columns:
        df["observationhrsweekend_ps_hours"] = df["observationhrsweekend_ps"].apply(
            _parse_hours
        )
    else:
        df["observationhrsweekend_ps_hours"] = None

    seizure_scores = _melt_numeric(df, _SEIZURE_TYPE_COLS, "seizure_type")
    caregiver_qol = _melt_numeric(df, _CAREGIVER_QOL_COLS, "scale")
    development = _melt_development(df)
    checkbox = _melt_checkboxes(df)

    return ProspectiveSurveyTables(
        surveys=df.reset_index(drop=True),
        seizure_scores=seizure_scores,
        development_milestones=development,
        caregiver_qol=caregiver_qol,
        checkbox_responses=checkbox,
    )


def ingest_box(
    folder_id: str,
    access_token: str,
    *,
    patients_csv: Path,
    raw_snapshot_dir: Path,
    run_timestamp: str,
    filename_patterns: Iterable[str],
) -> Tuple[ProspectiveSurveyTables, Path]:
    """Ingest the most recent survey export from Box."""
    raw_snapshot_dir = Path(raw_snapshot_dir)
    try:
        items = box_utils.list_files(
            folder_id,
            access_token,
            patterns=list(filename_patterns),
        )
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[prospective/box] Failed to list folder items: {exc}")
        return ProspectiveSurveyTables.empty(), raw_snapshot_dir

    if not items:
        tqdm.write(f"[prospective/box] No matching files found in folder {folder_id}.")
        return ProspectiveSurveyTables.empty(), raw_snapshot_dir

    latest = _select_latest_entry(items)
    if latest is None:
        return ProspectiveSurveyTables.empty(), raw_snapshot_dir
    filename = str(latest.get("name"))
    file_id = str(latest.get("id"))
    exported_at = _parse_filename_timestamp(filename)

    try:
        local_path = box_utils.download_file(
            file_id,
            filename,
            access_token,
            raw_snapshot_dir,
        )
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[prospective/box] Failed to download {filename}: {exc}")
        return ProspectiveSurveyTables.empty(), raw_snapshot_dir

    try:
        df = pd.read_csv(local_path, dtype=object)
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[prospective/box] Could not read {filename}: {exc}")
        return ProspectiveSurveyTables.empty(), raw_snapshot_dir

    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    df["source_file"] = filename
    df["run_timestamp"] = run_timestamp
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["source_exported_at"] = (
        exported_at.isoformat() if exported_at is not None else None
    )
    df["survey_instance_id"] = _compute_survey_ids(df)
    df = df.drop_duplicates(subset=["survey_instance_id"]).reset_index(drop=True)

    tables = _prepare_tables(df, patients_csv)
    return tables, raw_snapshot_dir


def write_outputs(tables: ProspectiveSurveyTables | None, processed_dir: Path) -> None:
    if tables is None:
        tables = ProspectiveSurveyTables.empty()

    processed_dir = Path(processed_dir)

    tables.surveys.to_csv(processed_dir / "prospective_surveys.csv", index=False)
    tables.seizure_scores.to_csv(
        processed_dir / "prospective_seizure_scores.csv", index=False
    )
    tables.development_milestones.to_csv(
        processed_dir / "prospective_development_milestones.csv", index=False
    )
    tables.caregiver_qol.to_csv(
        processed_dir / "prospective_caregiver_qol.csv", index=False
    )
    tables.checkbox_responses.to_csv(
        processed_dir / "prospective_checkbox_responses.csv", index=False
    )
