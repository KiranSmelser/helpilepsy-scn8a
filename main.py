"""Entry point for Helpilepsy‑SCN8A data extraction.

Authenticates with the Epione Dashboard API, downloads the latest
patient/medication/event data, and stores the raw JSON as well as a
consolidated CSV inside the `data/` directory.
"""

from datetime import datetime
from pathlib import Path

from src import config, utils, data_loader, data_writer
from tqdm import tqdm


def orchestrate() -> None:
    """High‑level orchestration of a single extraction run."""
    # Credentials
    email, password = utils.load_credentials()
    if not (email and password):
        return

    # Login
    token = data_loader.login(email, password)
    if not token:
        tqdm.write("Login failed.")
        return
    tqdm.write("Login successfull.")

    # Patient list
    patients = data_loader.get_all_patient_data(token)
    if not patients:
        tqdm.write("No patient records retrieved. Exiting.")
        return

    # Output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # raw JSON snapshot for this scrape
    raw_dir = Path(config.OUTPUT_BASE_DIR) / config.RAW_DIR_NAME / f"run_{timestamp}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # tidy tables produced from the snapshot
    processed_dir = (
        Path(config.OUTPUT_BASE_DIR) / config.PROCESSED_DIR_NAME / f"run_{timestamp}"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    utils.save_json(patients, raw_dir / "patient_list.json")

    # Meds & events
    meds, events = data_loader.fetch_all_patient_history(token, patients, raw_dir)

    # Consolidated CSV/tables
    data_writer.generate_processed_tables(patients, meds, events, processed_dir)

    tqdm.write("Extraction finished successfully.")


if __name__ == "__main__":
    orchestrate()
