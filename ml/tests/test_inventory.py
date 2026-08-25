from pathlib import Path

from PIL import Image

from crrc_vision.inventory import scan_images


def test_scan_images_records_hash_dimensions_and_capture_time(tmp_path: Path) -> None:
    image = tmp_path / "IMG_20240529_111456.jpg"
    Image.new("RGB", (20, 10), "red").save(image)

    row = scan_images(tmp_path)[0]

    assert row.relative_path == image.name
    assert row.width == 20
    assert row.height == 10
    assert len(row.sha256) == 64
    assert len(row.phash) == 16
    assert row.captured_at.isoformat() == "2024-05-29T11:14:56"
    assert row.focus_score >= 0.0


def test_scan_images_ignores_non_images(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")

    assert scan_images(tmp_path) == []
