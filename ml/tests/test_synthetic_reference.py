from pathlib import Path

import cv2
import numpy as np

from crrc_vision.synthetic_reference import select_reference_candidates


def test_reference_selection_rejects_dark_and_blurry_crops(tmp_path: Path) -> None:
    source_dir = tmp_path / "中文路径"
    source_dir.mkdir()
    dark = np.full((120, 120, 3), 12, dtype=np.uint8)
    blurry = np.full((120, 120, 3), 150, dtype=np.uint8)
    sharp = np.indices((120, 120)).sum(axis=0) % 2 * 180 + 40
    sharp = np.repeat(sharp[..., None], 3, axis=2).astype(np.uint8)
    for name, image in (("dark.png", dark), ("blurry.png", blurry), ("sharp.png", sharp)):
        success, encoded = cv2.imencode(".png", image)
        assert success
        encoded.tofile(source_dir / name)
    coco = {
        "images": [
            {"id": 1, "file_name": "dark.png", "scene_group": "a", "width": 120, "height": 120},
            {"id": 2, "file_name": "blurry.png", "scene_group": "b", "width": 120, "height": 120},
            {"id": 3, "file_name": "sharp.png", "scene_group": "c", "width": 120, "height": 120},
        ],
        "annotations": [
            {"image_id": image_id, "bbox": [40, 40, 40, 40]}
            for image_id in (1, 2, 3)
        ],
    }
    selected = select_reference_candidates(coco, source_dir, count=1, minimum_brightness=50, minimum_sharpness=30)
    assert selected[0].image["file_name"] == "sharp.png"
    assert selected[0].brightness >= 50
    assert selected[0].sharpness >= 30
