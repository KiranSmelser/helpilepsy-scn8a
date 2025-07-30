"""Entry point for Helpilepsy‑SCN8A data extraction.

Authenticates with the Epione Dashboard API, downloads the latest
patient/medication/event data, and stores the raw JSON as well as a
consolidated CSV inside the `data/` directory.
"""

from datetime import datetime
from pathlib import Path
import subprocess
import shutil

from src import config, utils, data_loader, data_writer, mood_sleep_loader
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

    # Mood & Sleep ingestion 
    try:
        patients_csv = processed_dir / "patients.csv"
        ms_df, _ = mood_sleep_loader.ingest_local(
            local_dir=config.MOOD_SLEEP_LOCAL_DIR,
            patients_csv=patients_csv,
            raw_snapshot_dir=(raw_dir / "mood_sleep"),
            run_timestamp=timestamp,
        )
        mood_sleep_loader.write_outputs(
            ms_df=ms_df,
            processed_dir=processed_dir,
        )
        tqdm.write("Mood/Sleep ingestion finished.")
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[warn] Mood/Sleep ingestion skipped due to error: {exc}")

    # Emit run metadata & update 'latest' pointer
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        commit_hash = None

    metadata = {
        "timestamp": timestamp,
        "git_commit": commit_hash,
        "raw_dir": str(raw_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
    }
    utils.save_json(metadata, processed_dir / "metadata.json")

    # Refresh the 'latest' symlink
    latest_link = Path(config.OUTPUT_BASE_DIR) / config.PROCESSED_DIR_NAME / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()  # remove previous link/dir
        latest_link.symlink_to(processed_dir.resolve())
    except Exception:  # noqa: BLE001
        # Fall back to copying the directory when symlinks aren't supported
        if latest_link.exists():
            shutil.rmtree(latest_link)
        shutil.copytree(processed_dir, latest_link)

    tqdm.write("Extraction finished successfully.")


if __name__ == "__main__":
    orchestrate()
