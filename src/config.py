import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Salt for hashing patient IDs
HASH_SALT = os.environ.get("HASH_SALT", "CHANGE_THIS_TO_SECURE_RANDOM_SALT")

# Path for raw-to-hash mapping file
MAPPING_FILE = os.path.join(PROCESSED_DATA_DIR, "id_mapping.json")