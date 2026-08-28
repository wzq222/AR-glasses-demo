import cv2
import numpy as np

from crrc_vision.mark_proposals import find_color_mark_proposals


def test_red_and_yellow_marks_are_kept_with_geometry():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.line(image, (50, 100), (95, 100), (0, 0, 210), 5)
    cv2.line(image, (250, 200), (295, 210), (0, 210, 210), 5)
    proposals = find_color_mark_proposals(image, minimum_area=8)
    assert {row.color for row in proposals} == {"red", "yellow"}
    assert all(row.mark_xyxy[2] > row.mark_xyxy[0] for row in proposals)
    assert all(row.roi_xyxy[2] > row.roi_xyxy[0] for row in proposals)
    assert all(len(row.line_xyxy) == 4 for row in proposals)


def test_nearby_mark_fragments_are_not_dropped_by_roi_overlap():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(image, (70, 90), (95, 90), (0, 0, 220), 4)
    cv2.line(image, (105, 90), (130, 90), (0, 0, 220), 4)
    proposals = find_color_mark_proposals(image, minimum_area=5)
    assert len(proposals) == 2
    assert all(row.color == "red" for row in proposals)
    assert proposals[0].roi_xyxy[2] > proposals[1].roi_xyxy[0]


def test_one_orange_mark_is_not_split_into_red_and_yellow_duplicates():
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.line(image, (50, 80), (100, 80), (0, 90, 220), 5)
    cv2.line(image, (101, 80), (150, 80), (0, 150, 220), 5)
    proposals = find_color_mark_proposals(image, minimum_area=8)
    assert len(proposals) == 1


def test_dark_red_lab_fallback_is_detected():
    image = np.full((160, 200, 3), 25, dtype=np.uint8)
    cv2.line(image, (60, 80), (130, 80), (15, 15, 85), 6)
    proposals = find_color_mark_proposals(image, minimum_area=8)
    assert any(row.color == "red" for row in proposals)


def test_achromatic_white_and_gray_are_not_marks():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    cv2.line(image, (20, 40), (170, 40), (220, 220, 220), 8)
    cv2.line(image, (20, 100), (170, 100), (80, 80, 80), 8)
    assert find_color_mark_proposals(image, minimum_area=5) == []


def test_roi_is_clipped_to_image_bounds():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.line(image, (0, 0), (25, 10), (0, 0, 230), 5)
    proposal = find_color_mark_proposals(image, minimum_area=5)[0]
    assert proposal.roi_xyxy[0] == 0
    assert proposal.roi_xyxy[1] == 0
    assert proposal.roi_xyxy[2] <= 160
    assert proposal.roi_xyxy[3] <= 120
