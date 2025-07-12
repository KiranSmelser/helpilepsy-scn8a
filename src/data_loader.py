"""API‑facing functions: login + data download."""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from . import config, utils
from tqdm import tqdm


# AUTHENTICATION
def login(email: str, password: str) -> str | None:
    """Authenticate and return the JWT token, or None on failure."""
    payload = {
        "email": email,
        "password": password,
        "dashboard": True,
        "language": "en",
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(config.API_URL, headers=headers, json=payload, timeout=30)
        data = resp.json()
        resp.raise_for_status()

        if data.get("success") and (token := data.get("result", {}).get("token")):
            return str(token)

        tqdm.write(f"Login failed: {data.get('errorMessage', 'No token in response')}")
        return None
    except (json.JSONDecodeError, requests.RequestException) as exc:
        tqdm.write(f"Request error during login: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"Unexpected error during login: {exc}")
        traceback.print_exc()
        return None


# PATIENT LIST
def _fetch_patient_data_page(
    token: str, limit: int, skip: int
) -> tuple[list[Any] | None, int | None]:
    """Retrieve a single pagination page."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    params = {"limit": limit, "skip": skip}

    try:
        resp = requests.get(
            config.PATIENT_DATA_URL, headers=headers, params=params, timeout=30
        )
        data = resp.json()
        resp.raise_for_status()

        if not data.get("success"):
            tqdm.write(
                f"Patient page skip={skip} unsuccessful: {data.get('errorMessage')}"
            )
            return None, None

        result = data["result"]
        patients = result.get("patients", {}).get("asDoctor", [])
        total_pages = int(result.get("totalPages", 0)) or None
        return patients, total_pages
    except (json.JSONDecodeError, requests.RequestException) as exc:
        tqdm.write(f"Error fetching patient page skip={skip}: {exc}")
        return None, None
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"Unexpected error fetching patient page skip={skip}: {exc}")
        traceback.print_exc()
        return None, None


def get_all_patient_data(token: str) -> list[Any]:
    """Download all patient records, transparently handling pagination."""
    all_patients: list[Any] = []

    patients, total_pages = _fetch_patient_data_page(token, config.LIMIT_PER_PAGE, 0)
    if patients is None:
        return all_patients

    all_patients.extend(patients)
    if not total_pages or total_pages == 1:
        return all_patients

    from tqdm import tqdm  # local import safe even if global

    pbar = tqdm(total=total_pages - 1, desc="Patient pages", unit="page")
    for page in range(1, total_pages):
        skip = page * config.LIMIT_PER_PAGE
        patients, _ = _fetch_patient_data_page(token, config.LIMIT_PER_PAGE, skip)
        if patients is None:
            pbar.close()
            break
        all_patients.extend(patients)
        pbar.update(1)
    pbar.close()
    return all_patients


# PER‑PATIENT HISTORY (medications + events)
def _get_medications(token: str, patient_id: str) -> dict[str, Any] | None:
    url = f"{config.BASE_PATIENT_URL}/{patient_id}/medications/all"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        data = resp.json()
        resp.raise_for_status()
        if isinstance(data, dict) and data.get("success"):
            return data
        if isinstance(data, list):
            return {"success": True, "result": data}
        tqdm.write(f"Medications request failed for {patient_id}: {data}")
        return None
    except (json.JSONDecodeError, requests.RequestException) as exc:
        tqdm.write(f"Error fetching meds for {patient_id}: {exc}")
        return None


def _get_events(token: str, patient_id: str) -> dict[str, Any] | None:
    url = f"{config.BASE_PATIENT_URL}/{patient_id}/events"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    now_utc = datetime.now(timezone.utc)
    to_str = (now_utc + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    params = {
        "from": "1970-01-01T00:00:00.000Z",
        "to": to_str,
        "type": "seizure,headache,appointment,form,side_effect,other,nightwatch_report,nightwatch_seizure",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        data = resp.json()
        resp.raise_for_status()
        if isinstance(data, dict) and data.get("success"):
            return data
        if isinstance(data, list):
            return {"success": True, "result": data}
        tqdm.write(f"Events request failed for {patient_id}: {data}")
        return None
    except (json.JSONDecodeError, requests.RequestException) as exc:
        tqdm.write(f"Error fetching events for {patient_id}: {exc}")
        return None


def _fetch_single_history(
    token: str, patient: dict[str, Any], out_dir: Path
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    pid = patient.get("_id")
    if not pid:
        return None, None, None

    med_json = _get_medications(token, pid)
    evt_json = _get_events(token, pid)

    if med_json is not None:
        utils.save_json(med_json, out_dir / f"patient_{pid}_medications.json")
    if evt_json is not None:
        utils.save_json(evt_json, out_dir / f"patient_{pid}_events.json")

    return pid, med_json, evt_json


def fetch_all_patient_history(
    token: str,
    patients: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_workers: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download meds/events for every patient concurrently."""
    med_map: dict[str, Any] = {}
    evt_map: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_history, token, p, out_dir): p.get("_id")
            for p in patients
            if p.get("_id")
        }

        with tqdm(total=len(futures), desc="Patient histories", unit="patient") as pbar:
            for idx, fut in enumerate(as_completed(futures), start=1):
                pid = futures[fut]
                try:
                    patient_id, meds, evts = fut.result()
                    if patient_id is not None:
                        med_map[patient_id] = meds
                        evt_map[patient_id] = evts
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"Error in patient task {pid}: {exc}")
                    med_map[pid] = None
                    evt_map[pid] = None
                pbar.update(1)

    return med_map, evt_map
