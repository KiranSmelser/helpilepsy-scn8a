"""Ingest WhatsApp group membership data from Box."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from . import box_utils, config


def ingest_box(
    folder_id: str,
    access_token: str,
    raw_snapshot_dir: Path,
    run_timestamp: str,
) -> tuple[pd.DataFrame, Path]:
    """Download WhatsApp membership CSVs from Box and return combined data."""
    try:
        files = box_utils.list_files(
            folder_id,
            access_token,
            patterns=config.WHATSAPP_GROUP_GLOB,
        )
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[whatsapp/box] Failed to list folder items: {exc}")
        return pd.DataFrame(), raw_snapshot_dir

    if not files:
        tqdm.write(f"[whatsapp/box] No matching files found in folder {folder_id}.")
        return pd.DataFrame(), raw_snapshot_dir

    frames: List[pd.DataFrame] = []
    with tqdm(total=len(files), desc="WhatsApp Group (Box)", unit="file") as pbar:
        for entry in files:
            file_id = str(entry.get("id"))
            name = str(entry.get("name"))
            try:
                local_path = box_utils.download_file(
                    file_id=file_id,
                    filename=name,
                    token=access_token,
                    dest_dir=Path(raw_snapshot_dir),
                )
                df = pd.read_csv(local_path, dtype=str, encoding="utf-8-sig")
                # Normalize WhatsApp membership to binary flags.
                if "whatsapp" in df.columns:
                    normalized = (
                        df["whatsapp"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                    df["whatsapp"] = (normalized == "yes").astype(int)
                else:
                    df["whatsapp"] = 0
                df["source_file"] = name
                df["run_timestamp"] = run_timestamp
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[whatsapp/box] Skipping {name}: {exc}")
            finally:
                pbar.update(1)

    if not frames:
        return pd.DataFrame(), raw_snapshot_dir

    combined = pd.concat(frames, ignore_index=True)
    return combined, raw_snapshot_dir


def write_outputs(df: pd.DataFrame, processed_dir: Path) -> None:
    """Persist combined WhatsApp membership data to processed outputs."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "patient_id",
        "first_name",
        "last_name",
        "whatsapp",
        "source_file",
        "run_timestamp",
    ]

    if df.empty:
        pd.DataFrame(columns=required_cols).to_csv(
            processed_dir / "whatsapp_status.csv", index=False
        )
        return

    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[required_cols]
    out.to_csv(processed_dir / "whatsapp_status.csv", index=False)
