"""Shared helpers for interacting with Box."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import fnmatch

import requests

_LOCKFILE_PREFIXES: tuple[str, ...] = ("~$", "._")


def list_files(
    folder_id: str,
    token: str,
    *,
    patterns: Iterable[str] | None = None,
    fields: str = "id,name,size,modified_at",
    limit: int = 1000,
) -> List[dict]:
    """Return file entries in folder_id."""
    base_url = f"https://api.box.com/2.0/folders/{folder_id}/items"
    headers = {"Authorization": f"Bearer {token}"}

    items: list[dict] = []
    offset = 0
    while True:
        params = {"limit": limit, "offset": offset, "fields": fields}
        resp = requests.get(base_url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Box list error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        entries = data.get("entries", [])
        if not entries:
            break
        items.extend(entries)
        if len(entries) < limit:
            break
        offset += len(entries)

    filtered: list[dict] = []
    for entry in items:
        if entry.get("type") != "file":
            continue
        name = str(entry.get("name") or "")
        if not name or name.startswith(_LOCKFILE_PREFIXES):
            continue
        if patterns is None:
            filtered.append(entry)
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            filtered.append(entry)
    return filtered


def download_file(
    file_id: str,
    filename: str,
    token: str,
    dest_dir: Path,
) -> Path:
    """Download file_id into dest_dir and return the local file path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / filename

    url = f"https://api.box.com/2.0/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code not in (200, 302):
            raise RuntimeError(
                f"Box download error {resp.status_code} for file {file_id}"
            )
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    return out_path


__all__ = ["list_files", "download_file"]
