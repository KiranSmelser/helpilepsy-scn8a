"""Responsible for writing cleaned/merged outputs to disk."""

from __future__ import annotations

import csv
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

    flattened: list[dict[str, Any]] = []
    for daily in raw_event_json.get("result", []):
        if not isinstance(daily, dict):
            continue
        for key in config.EVENT_CONTAINER_KEYS:
            for evt in daily.get(key, []):
                if isinstance(evt, dict):
                    evt.setdefault("type", key)
                    flattened.append(evt)
    return flattened


# HELPERS
def _write_table(df: pd.DataFrame, root: Path, name: str) -> None:  # noqa: D401
    """Write *df* to a CSV file named <name>.csv inside *root*. Returns row‑count."""
    if df.empty:
        return 0

    csv_path = root / f"{name}.csv"
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

    The dictionary records: column name, logical dtype, % non‑null, example value.
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
    Explode *patients*, *meds_by_patient*, and *events_by_patient* into six tables
    and write each to CSV inside *processed_dir*.

    Tables: patients, medications, med_intakes, med_dosages,
            events, forms, form_answers
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # patients
    patients_df = (
        pd.json_normalize(patients)
        .rename(columns={"_id": "patient_id"})
        .assign(run_timestamp=processed_dir.name)  # provenance
    )

    # medications / intakes / dosages
    med_rows, intake_rows, dose_rows = [], [], []
    for pid, med_json in meds_by_patient.items():
        if not (isinstance(med_json, dict) and med_json.get("result")):
            continue
        for med in med_json["result"]:
            med_id = med.get("_id")
            med_rows.append(
                {
                    "medication_id": med_id,
                    "patient_id": pid,
                    "name": med.get("name"),
                    "reason": med.get("reason"),
                    "treatment_type": med.get("treatment_type"),
                    "intake_type": med.get("intake_type"),
                    "createdAt": med.get("createdAt"),
                    "updatedAt": med.get("updatedAt"),
                }
            )
            for intake in med.get("intakes", []):
                intake_id = intake.get("_id")
                intake_rows.append(
                    {
                        "intake_id": intake_id,
                        "medication_id": med_id,
                        "patient_id": pid,
                        "from": intake.get("from"),
                        "to": intake.get("to"),
                        **{
                            f"day_{i+1}": d
                            for i, d in enumerate(intake.get("days", []))
                        },
                        "createdAt": intake.get("createdAt"),
                        "updatedAt": intake.get("updatedAt"),
                    }
                )
                for dosage in intake.get("dosage") or []:
                    dose_rows.append(
                        {
                            "dosage_id": f"{intake_id}_{dosage.get('moment')}",
                            "intake_id": intake_id,
                            "patient_id": pid,
                            "moment": dosage.get("moment"),
                            "dose": dosage.get("dose"),
                            "unit": dosage.get("unit"),
                        }
                    )

    # events / forms / answers
    evt_rows, form_rows, ans_rows = [], [], []
    for pid, evt_json in events_by_patient.items():
        if not (isinstance(evt_json, dict) and evt_json.get("result")):
            continue
        for daily in evt_json["result"]:
            if not isinstance(daily, dict):
                continue
            for key in config.EVENT_CONTAINER_KEYS:
                for evt in daily.get(key, []):
                    if not isinstance(evt, dict):
                        continue
                    evt_type = evt.get("type")
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
                        for q in evt.get("questions") or []:
                            ans_rows.append(
                                {
                                    "answer_id": f"{form_id}_{q.get('index')}",
                                    "form_id": form_id,
                                    "patient_id": pid,
                                    "question_code": q.get("question"),
                                    "answer_type": q.get("answer_type"),
                                    "answer": q.get("answer"),
                                }
                            )
                    else:
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

    # tables to write
    tables_to_write = [
        ("patients", patients_df),
        ("medications", pd.DataFrame(med_rows)),
        ("med_intakes", pd.DataFrame(intake_rows)),
        ("med_dosages", pd.DataFrame(dose_rows)),
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
