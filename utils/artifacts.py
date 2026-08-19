"""Filesystem helpers for reproducible experiment artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_run_directory(root: str | Path, kind: str) -> Path:
    """Create a timestamped run directory without overwriting an existing run."""
    base = Path(root) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / f"{stamp}_{kind}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{kind}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(path: str | Path, payload: Any) -> Path:
    """Write UTF-8 JSON with stable human-readable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return output
