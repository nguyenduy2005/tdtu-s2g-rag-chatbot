"""Small, product-local persistence helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    """Write UTF-8 JSON atomically so interrupted runs do not leave partial files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
