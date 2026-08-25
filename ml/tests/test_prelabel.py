import cv2
import numpy as np
from PIL import Image

from crrc_vision.prelabel import find_marked_fasteners, read_bgr_image


def test_red_mark_produces_one_candidate_box():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.line(image, (130, 100), (170, 100), (0, 0, 255), 6)

    candidates = find_marked_fasteners(image, min_mark_area=20)

    assert len(candidates) == 1
    assert candidates[0].bbox.contains(150, 100)
    assert candidates[0].line.confidence > 0.5
    assert candidates[0].mark_color == "red"


def test_yellow_mark_is_detected_but_gray_object_is_not():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.line(image, (40, 50), (80, 70), (0, 255, 255), 7)
    cv2.rectangle(image, (180, 40), (240, 100), (160, 160, 160), -1)

    candidates = find_marked_fasteners(image, min_mark_area=20)

    assert len(candidates) == 1
    assert candidates[0].mark_color == "yellow"


def test_tiny_color_noise_is_ignored():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[20:22, 20:22] = (0, 0, 255)

    assert find_marked_fasteners(image, min_mark_area=20) == []


def test_read_bgr_image_supports_unicode_windows_path(tmp_path):
    path = tmp_path / "中车现场.jpg"
    Image.new("RGB", (12, 8), "red").save(path)

    image = read_bgr_image(path)

    assert image.shape == (8, 12, 3)
    assert image[0, 0, 2] > 200


def test_large_red_structure_is_not_a_mark():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (350, 250), (0, 0, 220), -1)

    assert find_marked_fasteners(image, min_mark_area=20) == []


def test_brown_rust_colored_line_is_not_a_mark():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.line(image, (80, 100), (160, 100), (35, 70, 125), 8)

    assert find_marked_fasteners(image, min_mark_area=20) == []
