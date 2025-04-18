import os
from src import config
import hashlib
import json

def get_patient_dir(patient_id):
    """Returns the directory path for a given patient."""
    return os.path.join(config.PROCESSED_DATA_DIR, patient_id)

def ensure_directory_exists(path):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)

def hash_patient_id(patient_identifier: str) -> str:
    """Generates a salted hash of a patient's email."""
    salt = config.HASH_SALT
    # hash patient ID
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        patient_identifier.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    )
    return hash_bytes.hex()

def load_id_mapping() -> dict:
    """Loads raw patient ID to hashed ID mapping from JSON."""
    mapping = {}
    if os.path.exists(config.MAPPING_FILE):
        with open(config.MAPPING_FILE, "r") as f:
            raw_map = json.load(f)
        for raw_id, entry in raw_map.items():
            if isinstance(entry, str):
                mapping[raw_id] = {"hashed_id": entry}
            elif isinstance(entry, dict):
                mapping[raw_id] = entry.copy()
    return mapping

def save_id_mapping(mapping: dict):
    """Save raw patient ID to hashed ID mapping to JSON."""
    ensure_directory_exists(os.path.dirname(config.MAPPING_FILE))
    with open(config.MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)