"""Centralized configuration constants used across the project."""

from pathlib import Path
import json
from typing import Any

# API ENDPOINTS
API_URL = "https://api.epione.care/api/v1/dashboard/auth/generate"
PATIENT_DATA_URL = "https://api.epione.care/api/v1/dashboard/me/dash"
BASE_PATIENT_URL = "https://api.epione.care/api/v1/dashboard/patients"

# FILESYSTEM PATHS
CONFIG_FILE = "config.json"

OUTPUT_BASE_DIR = "data"  # root output directory

# sub‑directories for raw snapshots and processed tables
RAW_DIR_NAME: str = "raw"
PROCESSED_DIR_NAME: str = "processed"

# OTHER CONSTANTS
LIMIT_PER_PAGE = 20
EVENT_CONTAINER_KEYS = [
    "seizures",
    "side_effects",
    "appointments",
    "reminders",
    "headaches",
    "others",
    "forms",
    "nightwatch_reports",
    "nightwatch_seizures",
]

# MOOD/SLEEP INGESTION
# File patterns to consider when ingesting mood/sleep files
MOOD_SLEEP_GLOB = ("*.xlsx", "*.xls", "*.csv")

# Prospective survey ingestion
PROSPECTIVE_SURVEY_GLOB = (
    "InternationalSCN8ARe-ProspectiveStudy_DATA_*.csv",
)

# Lookback window for computing summary metrics
METRICS_LOOKBACK_DAYS: int = 90

# Box integration
_BOX_ACCESS_TOKEN: str | None = None
_BOX_FOLDER_ID: str | None = None
_BOX_PROSPECTIVE_FOLDER_ID: str | None = None

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as _fp:
        _cfg: dict[str, Any] = json.load(_fp)
    _BOX_ACCESS_TOKEN = _cfg.get("box_access_token") or _cfg.get(
        "box_primary_access_token"
    )
    _BOX_FOLDER_ID = _cfg.get("box_mood_sleep_folder_id") or _cfg.get("box_folder_id")
    _BOX_PROSPECTIVE_FOLDER_ID = _cfg.get("box_prospective_survey_folder_id")
except Exception:
    pass

BOX_ACCESS_TOKEN: str | None = _BOX_ACCESS_TOKEN
BOX_MOOD_SLEEP_FOLDER_ID: str | None = _BOX_FOLDER_ID
BOX_PROSPECTIVE_SURVEY_FOLDER_ID: str | None = _BOX_PROSPECTIVE_FOLDER_ID
