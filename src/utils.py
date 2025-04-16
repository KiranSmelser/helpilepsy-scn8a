import os
from src import config

def get_patient_dir(patient_id):
    """Returns the directory path for a given patient."""
    return os.path.join(config.PROCESSED_DATA_DIR, patient_id)

def ensure_directory_exists(path):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)