"""Responsible for writing cleaned/merged outputs to disk."""

from __future__ import annotations

import csv
import re
import pandas as pd
import traceback
from pathlib import Path
from typing import Any
import textwrap

from . import config

from tqdm import tqdm


def _flatten_events(raw_event_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Turn nested *raw_event_json* into a list of individual events."""
    if not (raw_event_json and raw_event_json.get("success")):
        return []

    # Map container keys to canonical singular type names used downstream
    _TYPE_FROM_CONTAINER = {
        "seizures": "seizure",
        "side_effects": "side_effect",
        "appointments": "appointment",
        "reminders": "reminder",
        "headaches": "headache",
        "others": "other",
        "forms": "form",
        "nightwatch_reports": "nightwatch_report",
        "nightwatch_seizures": "nightwatch_seizure",
    }

    flattened: list[dict[str, Any]] = []
    for daily in raw_event_json.get("result", []):
        if not isinstance(daily, dict):
            continue
        for key in config.EVENT_CONTAINER_KEYS:
            events = daily.get(key) or []
            for evt in events:
                if not isinstance(evt, dict):
                    continue
                evt_type = (
                    evt.get("type") or _TYPE_FROM_CONTAINER.get(key) or key.rstrip("s")
                )
                evt["type"] = evt_type
                flattened.append(evt)
    return flattened


_LANG_PREFERENCE: tuple[str, ...] = ("en", "nl", "fr", "de", "es", "it")
_RESCUE_TEXT_TOKENS: tuple[str, ...] = (
    "rescue",
    "emergency",
    "as needed",
    "as_needed",
    "prn",
)
_RESCUE_STRONG_NAME_TOKENS: tuple[str, ...] = (
    "midazolam",
    "diazepam",
    "valtoco",
    "nayzilam",
    "diastat",
    "buccolam",
    "versed",
)
_RESCUE_CONDITIONAL_NAME_TOKENS: tuple[str, ...] = (
    "clonazepam",
    "klonopin",
    "lorazepam",
    "ativan",
)


def _coerce_text(value: Any) -> str:
    """Return a human-readable string for value."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_localized_text(value: Any) -> str:
    """Extract a localized string from value."""
    if isinstance(value, dict):
        for lang in _LANG_PREFERENCE:
            text = value.get(lang)
            if text:
                return _coerce_text(text)
        for text in value.values():
            if text:
                return _coerce_text(text)
        return ""
    return _coerce_text(value)


def _slugify(text: str) -> str:
    """Generate a slug suitable for question codes."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _contains_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    haystack = _coerce_text(text).lower()
    return any(tok in haystack for tok in tokens)


def _is_prn_like_intake(intake_obj: dict[str, Any]) -> bool:
    days = intake_obj.get("days")
    if isinstance(days, list) and len(days) > 0 and not any(bool(x) for x in days):
        return True

    specific = intake_obj.get("specific_dosage")
    if isinstance(specific, dict):
        moment = _coerce_text(specific.get("moment")).lower()
        if moment == "prn":
            return True
        # API often stores rescue meds as unscheduled specific-dosage intakes.
        if intake_obj.get("dosage") in (None, [], ""):
            return True

    moment = _coerce_text(intake_obj.get("moment")).lower()
    if moment == "prn":
        return True
    return False


def _is_rescue_medication(medication_obj: dict[str, Any] | None) -> bool:
    if not isinstance(medication_obj, dict):
        return False

    text_parts = [
        medication_obj.get("name"),
        medication_obj.get("reason"),
        medication_obj.get("intake_type"),
        medication_obj.get("treatment_type"),
        medication_obj.get("remark"),
        medication_obj.get("notes"),
    ]
    med_text = " ".join(_coerce_text(part) for part in text_parts if _coerce_text(part))

    if _contains_any_token(med_text, _RESCUE_TEXT_TOKENS):
        return True

    for key in ("rescue_medications", "rescues", "emergency_treatments"):
        value = medication_obj.get(key)
        if isinstance(value, list):
            if any(_coerce_text(v).strip() for v in value):
                return True
        elif _coerce_text(value).strip():
            return True

    if _contains_any_token(med_text, _RESCUE_STRONG_NAME_TOKENS):
        return True

    prn_like = any(
        _is_prn_like_intake(intake)
        for intake in (medication_obj.get("intakes") or [])
        if isinstance(intake, dict)
    )
    if prn_like and _contains_any_token(med_text, _RESCUE_CONDITIONAL_NAME_TOKENS):
        return True

    return False


def _iter_form_questions(evt: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten questions from legacy questions."""
    records: list[dict[str, Any]] = []

    questions = evt.get("questions") or []
    for q_idx, question in enumerate(questions):
        records.append(
            {
                "question": question,
                "section_index": None,
                "section_label": None,
                "question_position": q_idx,
            }
        )

    sections = evt.get("sections") or []
    for sec_idx, section in enumerate(sections):
        section_label = _resolve_localized_text(section.get("name"))
        for q_idx, question in enumerate(section.get("questions") or []):
            records.append(
                {
                    "question": question,
                    "section_index": sec_idx,
                    "section_label": section_label,
                    "question_position": q_idx,
                }
            )

    return records


# HELPERS
def _write_table(df: pd.DataFrame, root: Path, name: str) -> None:  # noqa: D401
    """Write *df* to a CSV file named <name>.csv inside *root*."""
    csv_path = root / f"{name}.csv"
    # Ensure consistent column order
    if df is None:
        df = pd.DataFrame()
    df.to_csv(csv_path, index=False)
    return len(df)


# DATA‑DICTIONARY GENERATOR
def _dtype_to_string(dtype: "pd.api.extensions.ExtensionDtype") -> str:
    """Return a concise readable dtype."""
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "string"


def _generate_data_dictionary(root: Path) -> Path:
    """
    Build a Markdown data dictionary for every CSV table in *root* and write it
    to *root/data_dictionary.md*.
    """
    md_lines: list[str] = ["# Data Dictionary", ""]

    for csv_path in sorted(root.glob("*.csv")):
        table = csv_path.stem
        # Read a sample
        try:
            df = pd.read_csv(csv_path, nrows=10_000)
        except Exception:  # noqa: BLE001
            tqdm.write(f"[warn] Could not read {csv_path}; skipping.")
            continue

        md_lines.append(f"## {table}")
        md_lines.append("")
        md_lines.append("| Column | Type | Non‑null % | Example |")
        md_lines.append("|--------|------|-----------|---------|")

        total = len(df)
        for col in df.columns:
            series = df[col]
            dtype_str = _dtype_to_string(series.dtype)
            non_null_pct = f"{series.notna().mean() * 100:0.1f}%" if total else "0%"
            ex_val = ""
            if series.notna().any():
                ex_val = str(series.dropna().iloc[0])
                # truncate long examples
                if len(ex_val) > 30:
                    ex_val = ex_val[:27] + "…"
                ex_val = ex_val.replace("\n", " ").replace("|", " ")
            md_lines.append(f"| {col} | {dtype_str} | {non_null_pct} | {ex_val} |")
        md_lines.append("")  # blank line between tables

    out_path = root / "data_dictionary.md"
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_lines))

    return out_path


# TIDY TABLE PIPELINE
def generate_processed_tables(  # noqa: C901, PLR0915
    patients: list[dict[str, Any]],
    meds_by_patient: dict[str, Any],
    events_by_patient: dict[str, Any],
    processed_dir: Path,
) -> None:
    """
    Explode *patients*, *meds_by_patient*, and *events_by_patient* into tidy tables
    and write each to CSV inside *processed_dir*.

    Tables: patients, medications, med_dosages, med_intakes, events, forms, form_answers
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # patients
    patients_df = (
        pd.json_normalize(patients)
        .rename(columns={"_id": "patient_id"})
        .assign(run_timestamp=processed_dir.name)  # provenance
    )

    # medications
    med_rows = []
    med_keys_seen: set[tuple[str, str]] = set()
    dosage_rows = []
    # Map to resolve dosage_index for reminder events later on
    dosage_index_map: dict[tuple, int] = {}

    def _is_deleted_from_deleted(deleted_value: Any) -> bool:
        """Normalize deleted flag to a strict boolean."""
        if isinstance(deleted_value, bool):
            return deleted_value
        if isinstance(deleted_value, (int, float)):
            return deleted_value != 0
        if isinstance(deleted_value, str):
            return deleted_value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    def _register_medication_row(
        patient_id: str, medication_obj: dict[str, Any] | None
    ) -> str | None:
        """Append a medication row once per (patient_id, medication_id)."""
        if not isinstance(medication_obj, dict):
            return None
        med_id = medication_obj.get("_id") or medication_obj.get("id")
        if not med_id:
            return None
        med_key = (patient_id, med_id)
        if med_key in med_keys_seen:
            return med_id
        med_rows.append(
            {
                "medication_id": med_id,
                "patient_id": patient_id,
                "name": medication_obj.get("name"),
                "reason": medication_obj.get("reason"),
                "treatment_type": medication_obj.get("treatment_type"),
                "intake_type": medication_obj.get("intake_type"),
                "is_rescue_med": _is_rescue_medication(medication_obj),
                "is_deleted": _is_deleted_from_deleted(
                    medication_obj.get("deleted")
                ),
                "createdAt": medication_obj.get("createdAt"),
                "updatedAt": medication_obj.get("updatedAt"),
            }
        )
        med_keys_seen.add(med_key)
        return med_id

    for pid, med_json in meds_by_patient.items():
        if not (isinstance(med_json, dict) and med_json.get("result")):
            continue
        for med in med_json["result"]:
            med_id = _register_medication_row(pid, med)
            if not med_id:
                continue
            # Flatten intake schedules and dosage details
            for intake in med.get("intakes") or []:
                if not isinstance(intake, dict):
                    continue
                intake_id = intake.get("_id") or intake.get("id")
                # Standardize days into 7 binary columns
                days_val = intake.get("days")
                if isinstance(days_val, list):
                    days_flags = [1 if bool(x) else 0 for x in days_val[:7]]
                    # pad to 7 items if shorter
                    if len(days_flags) < 7:
                        days_flags += [None] * (7 - len(days_flags))
                else:
                    days_flags = [None] * 7

                dosage_list = intake.get("dosage") or []
                if not isinstance(dosage_list, list):
                    dosage_list = []

                for d_idx, d in enumerate(dosage_list):
                    if not isinstance(d, dict):
                        continue
                    # normalized lookup key for later joins with reminder events
                    try:
                        key = (
                            pid,
                            med_id,
                            intake_id,
                            (d.get("moment") or "").strip(),
                            float(d.get("dose")) if d.get("dose") is not None else None,
                            (d.get("unit") or "").strip(),
                        )
                        if key not in dosage_index_map:
                            dosage_index_map[key] = d_idx
                    except Exception:
                        pass
                    dosage_rows.append(
                        {
                            # provenance fields
                            "patient_id": pid,
                            "medication_id": med_id,
                            "intake_id": intake_id,
                            "dosage_index": d_idx,
                            # intake‑level fields
                            "from": intake.get("from"),
                            "to": intake.get("to"),
                            "day_1": days_flags[0],
                            "day_2": days_flags[1],
                            "day_3": days_flags[2],
                            "day_4": days_flags[3],
                            "day_5": days_flags[4],
                            "day_6": days_flags[5],
                            "day_7": days_flags[6],
                            "intake_createdAt": intake.get("createdAt"),
                            "intake_updatedAt": intake.get("updatedAt"),
                            # dosage‑level fields
                            "moment": d.get("moment"),
                            "dose": d.get("dose"),
                            "unit": d.get("unit"),
                        }
                    )

    evt_rows, form_rows, ans_rows = [], [], []
    med_intake_rows = []
    # Map container key -> canonical type if evt lacks one
    _TYPE_FROM_CONTAINER = {
        "seizures": "seizure",
        "side_effects": "side_effect",
        "appointments": "appointment",
        "reminders": "reminder",
        "headaches": "headache",
        "others": "other",
        "forms": "form",
        "nightwatch_reports": "nightwatch_report",
        "nightwatch_seizures": "nightwatch_seizure",
    }

    for pid, evt_json in events_by_patient.items():
        if not (isinstance(evt_json, dict) and evt_json.get("result")):
            continue
        for daily in evt_json["result"]:
            if not isinstance(daily, dict):
                continue
            for key in config.EVENT_CONTAINER_KEYS:
                events = daily.get(key) or []
                for evt in events:
                    if not isinstance(evt, dict):
                        continue
                    evt_type = (
                        evt.get("type")
                        or _TYPE_FROM_CONTAINER.get(key)
                        or key.rstrip("s")
                    )
                    # Normalize in-place so downstream logic sees a canonical type
                    evt["type"] = evt_type

                    if evt_type == "form":
                        form_id = evt.get("_id")
                        meta = evt.get("form") or {}
                        form_rows.append(
                            {
                                "form_id": form_id,
                                "patient_id": pid,
                                "form_name": meta.get("name"),
                                "label": meta.get("label"),
                                "date": evt.get("date"),
                                "end_date": evt.get("end_date"),
                                "createdAt": evt.get("createdAt"),
                                "updatedAt": evt.get("updatedAt"),
                            }
                        )
                        seen_codes: set[str] = set()
                        for order, record in enumerate(_iter_form_questions(evt)):
                            question = record.get("question") or {}
                            raw_code = question.get("question")
                            raw_index = question.get("index")
                            question_index = _coerce_text(raw_index) if raw_index is not None else ""
                            question_text = _resolve_localized_text(
                                question.get("questionText")
                            )
                            section_label = record.get("section_label")
                            section_index = record.get("section_index")

                            if raw_code:
                                question_code = _coerce_text(raw_code)
                            else:
                                parts: list[str] = []
                                if section_label:
                                    section_slug = _slugify(section_label)
                                    if section_slug:
                                        parts.append(section_slug)
                                if question_index:
                                    parts.append(f"q{question_index}")
                                if question_text:
                                    parts.append(_slugify(question_text))
                                fallback = f"q{order + 1:03d}"
                                question_code = "_".join(part for part in parts if part)
                                if not question_code:
                                    question_code = fallback

                            canonical = question_code
                            suffix = 2
                            while canonical in seen_codes:
                                canonical = f"{question_code}_{suffix}"
                                suffix += 1
                            seen_codes.add(canonical)

                            if raw_code:
                                if raw_index is not None:
                                    answer_id = f"{form_id}_{_coerce_text(raw_index)}"
                                else:
                                    answer_id = f"{form_id}_None"
                            else:
                                answer_id = f"{form_id}_{canonical}"

                            ans_rows.append(
                                {
                                    "answer_id": answer_id,
                                    "form_id": form_id,
                                    "patient_id": pid,
                                    "question_code": canonical,
                                    "answer_type": question.get("answer_type"),
                                    "answer": question.get("answer"),
                                    "question_index": question_index or None,
                                    "question_text": question_text or None,
                                    "section_index": section_index,
                                    "section_label": section_label or None,
                                    "is_displayed": question.get("isDisplay"),
                                }
                            )
                    else:
                        if evt_type != "reminder":
                            evt_rows.append(
                                {
                                    "event_id": evt.get("_id"),
                                    "patient_id": pid,
                                    "type": evt_type,
                                    "date": evt.get("date"),
                                    "duration": evt.get("duration"),
                                    "seizure_type": evt.get("seizure_type"),
                                    "triggers": "|".join(
                                        map(str, evt.get("triggers") or [])
                                    ),
                                    "felt": evt.get("felt"),
                                    "during_sleep": evt.get("during_sleep"),
                                    "remark": evt.get("remark"),
                                    "createdAt": evt.get("createdAt"),
                                    "updatedAt": evt.get("updatedAt"),
                                }
                            )
                        # Record medication reminder instances as med_intakes
                        if evt_type == "reminder":
                            intake_obj = evt.get("intake") or {}
                            med_obj = evt.get("medication") or {}
                            reminder_meta = evt.get("reminder") or {}

                            # Resolve identifiers
                            intake_id = intake_obj.get("_id") or intake_obj.get("id")
                            med_id = (
                                med_obj.get("_id")
                                or intake_obj.get("medication")
                                or med_obj.get("id")
                            )
                            # Some reminders reference deleted medications that are
                            # omitted from /medications/all. Backfill these so
                            # med_intakes always has a matching medications row.
                            if med_id and (pid, med_id) not in med_keys_seen:
                                med_payload = med_obj if isinstance(med_obj, dict) else {}
                                if not (med_payload.get("_id") or med_payload.get("id")):
                                    med_payload = {**med_payload, "_id": med_id}
                                _register_medication_row(pid, med_payload)

                            # Determine dose/unit and moment
                            moment = (evt.get("moment") or "").strip()
                            dose_val = reminder_meta.get("dose")
                            unit_val = reminder_meta.get("unit")
                            if (
                                dose_val is None or unit_val in (None, "")
                            ) and isinstance(intake_obj.get("dosage"), list):
                                for d in intake_obj.get("dosage"):
                                    if not isinstance(d, dict):
                                        continue
                                    if (d.get("moment") or "").strip() == moment:
                                        dose_val = (
                                            dose_val
                                            if dose_val is not None
                                            else d.get("dose")
                                        )
                                        unit_val = unit_val or d.get("unit")
                                        break

                            # Normalize for dosage_index lookup
                            try:
                                dose_num = (
                                    float(dose_val) if dose_val is not None else None
                                )
                            except Exception:
                                dose_num = None
                            unit_norm = (unit_val or "").strip()

                            dosage_index = None
                            try:
                                key_tuple = (
                                    pid,
                                    med_id,
                                    intake_id,
                                    moment,
                                    dose_num,
                                    unit_norm,
                                )
                                dosage_index = dosage_index_map.get(key_tuple)
                            except Exception:
                                pass

                            # Intake schedule bounds and weekday flags
                            days_val = intake_obj.get("days")
                            if isinstance(days_val, list):
                                day_flags = [1 if bool(x) else 0 for x in days_val[:7]]
                                if len(day_flags) < 7:
                                    day_flags += [None] * (7 - len(day_flags))
                            else:
                                day_flags = [None] * 7

                            med_intake_rows.append(
                                {
                                    # primary identifiers
                                    "intake_event_id": evt.get("_id") or evt.get("id"),
                                    "patient_id": pid,
                                    "medication_id": med_id,
                                    "intake_id": intake_id,
                                    "dosage_index": dosage_index,
                                    # timing
                                    "date": evt.get("date"),
                                    "time": evt.get("time"),
                                    "real_time": evt.get("real_time"),
                                    "moment": moment,
                                    # dose details
                                    "dose": dose_val,
                                    "unit": unit_norm,
                                    # adherence flags
                                    "taken": evt.get("taken"),
                                    "taken_date": evt.get("taken_date"),
                                    # soft‑deletes + provenance
                                    "deleted": evt.get("deleted"),
                                    "createdAt": evt.get("createdAt"),
                                    "updatedAt": evt.get("updatedAt"),
                                    # schedule context from intake
                                    "intake_from": intake_obj.get("from"),
                                    "intake_to": intake_obj.get("to"),
                                    "day_1": day_flags[0],
                                    "day_2": day_flags[1],
                                    "day_3": day_flags[2],
                                    "day_4": day_flags[3],
                                    "day_5": day_flags[4],
                                    "day_6": day_flags[5],
                                    "day_7": day_flags[6],
                                }
                            )

    # tables to write
    med_intake_cols = [
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
    tables_to_write = [
        ("patients", patients_df),
        ("medications", pd.DataFrame(med_rows)),
        ("med_dosages", pd.DataFrame(dosage_rows)),
        ("med_intakes", pd.DataFrame(med_intake_rows, columns=med_intake_cols)),
        ("events", pd.DataFrame(evt_rows)),
        ("forms", pd.DataFrame(form_rows)),
        ("form_answers", pd.DataFrame(ans_rows)),
    ]
    with tqdm(total=len(tables_to_write), desc="Writing tables", unit="table") as pbar:
        for name, df in tables_to_write:
            _write_table(df, processed_dir, name)
            pbar.update(1)

    # data dictionary
    _generate_data_dictionary(processed_dir)
