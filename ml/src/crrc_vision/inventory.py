"""Deterministic image inventory for the private CRRC data workspace."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
_CAPTURE_TIME = re.compile(r"IMG_(\d{8})_(\d{6})", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    sha256: str
    width: int
    height: int
    captured_at: datetime
    phash: str
    focus_score: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["captured_at"] = self.captured_at.isoformat()
        return row


def _capture_time(path: Path) -> datetime:
    match = _CAPTURE_TIME.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer capture time from image name: {path.name}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_images(root: Path) -> list[ImageRecord]:
    """Scan supported images below *root* in stable relative-path order."""
    root = root.resolve()
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )

    records: list[ImageRecord] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            phash = str(imagehash.phash(rgb))
            gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
            focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        records.append(
            ImageRecord(
                relative_path=path.relative_to(root).as_posix(),
                sha256=_sha256(path),
                width=width,
                height=height,
                captured_at=_capture_time(path),
                phash=phash,
                focus_score=focus_score,
            )
        )
    return records
