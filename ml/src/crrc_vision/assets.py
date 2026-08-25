"""Private asset workspace resolution."""

from __future__ import annotations

import os
from pathlib import Path


def asset_root() -> Path:
    """Return the existing private asset root configured for this run."""
    value = os.environ.get("CRRC_VISION_DATA_ROOT")
    if not value:
        raise RuntimeError("CRRC_VISION_DATA_ROOT is not set")

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"CRRC_VISION_DATA_ROOT does not exist: {root}")
    return root
