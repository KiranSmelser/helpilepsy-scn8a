"""Light‑weight utility helpers shared by multiple modules."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from . import config

from tqdm import tqdm


def save_json(data: dict | list, file_path: str | Path) -> None:
    """Persist *data* to *file_path* in UTF‑8 encoded JSON format."""
    try:
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"Error saving JSON to {file_path}: {exc}")
        traceback.print_exc()


def load_credentials(
    file_path: str | Path | None = None,
) -> tuple[str | None, str | None]:
    """Return (email, password) loaded from *file_path* or (None, None)."""
    path = Path(file_path or config.CONFIG_FILE)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data: dict[str, Any] = json.load(fp)
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            tqdm.write(f"'email' or 'password' missing in {path}.")
            return None, None
        return str(email), str(password)
    except FileNotFoundError:
        tqdm.write(
            f"Config file '{path}' not found. "
            "Create it (e.g. copy config.template.json) and try again."
        )
        return None, None
    except json.JSONDecodeError:
        tqdm.write(f"Could not decode JSON in {path}.")
        return None, None
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"Unexpected error while loading credentials: {exc}")
        traceback.print_exc()
        return None, None
