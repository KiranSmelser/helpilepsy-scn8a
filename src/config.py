"""Centralised configuration constants used across the project."""

from pathlib import Path

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
from pathlib import Path as _PathAlias 

MOOD_SLEEP_LOCAL_DIR = _PathAlias(OUTPUT_BASE_DIR) / RAW_DIR_NAME / "mood_sleep" 
MOOD_SLEEP_GLOB = ("*.xlsx", "*.xls", "*.csv")  

# Placeholder for Box integration
BOX_MOOD_SLEEP_FOLDER_ID: str | None = None
