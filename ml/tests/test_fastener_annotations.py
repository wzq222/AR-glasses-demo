from crrc_vision.fastener_annotations import build_fastener_document, evaluate_fastener_truth


def sample_document(image_status: str = "accept", annotation_status: str = "accept") -> dict:
    return {
        "images": [
            {
                "id": 1,
                "file_name": "a.jpg",
                "width": 100,
                "height": 80,
                "scene_group": "g1",
                "split": "train",
                "image_review_status": image_status,
            },
            {
                "id": 2,
                "file_name": "b.jpg",
                "width": 100,
                "height": 80,
                "scene_group": "g2",
                "split": "val",
                "image_review_status": image_status,
            },
        ],
        "categories": [
            {"id": 1, "name": "fastener"},
            {"id": 2, "name": "pipe_joint"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 2, 30, 40],
                "review_status": annotation_status,
            }
        ],
    }


def test_truth_gate_rejects_unreviewed_images_and_annotations():
    report = evaluate_fastener_truth(
        sample_document("unreviewed", "unreviewed"), minimum_groups=2
    )

    assert report.can_train is False
    assert "UNREVIEWED_IMAGE" in report.error_codes
    assert "UNREVIEWED_ANNOTATION" in report.error_codes
    assert report.reviewed_groups == 0


def test_truth_gate_accepts_reviewed_train_and_val_groups():
    report = evaluate_fastener_truth(sample_document(), minimum_groups=2)

    assert report.can_train is True
    assert report.accepted_boxes == 1
    assert report.reviewed_groups == 2


def test_truth_gate_rejects_scene_group_leakage():
    document = sample_document()
    document["images"][1]["scene_group"] = "g1"

    report = evaluate_fastener_truth(document, minimum_groups=1)

    assert "SCENE_GROUP_LEAKAGE" in report.error_codes


def test_truth_gate_rejects_box_on_accepted_empty_image():
    document = sample_document()
    document["images"][0]["image_review_status"] = "accept_empty"

    report = evaluate_fastener_truth(document, minimum_groups=2)

    assert "BOX_ON_ACCEPTED_EMPTY_IMAGE" in report.error_codes


def test_truth_gate_rejects_invalid_box_and_unknown_image_reference():
    document = sample_document()
    document["annotations"][0]["bbox"] = [95, 70, 20, 20]
    document["annotations"].append(
        {
            "id": 2,
            "image_id": 999,
            "category_id": 1,
            "bbox": [1, 1, 5, 5],
            "review_status": "reject",
        }
    )

    report = evaluate_fastener_truth(document, minimum_groups=2)

    assert "BOX_OUTSIDE_IMAGE" in report.error_codes
    assert "UNKNOWN_IMAGE_REFERENCE" in report.error_codes


def test_label_document_starts_unreviewed_without_proposal_boxes():
    document = build_fastener_document(
        [
            {
                "relative_path": "a.jpg",
                "scene_group": "g1",
                "split": "train",
                "width": 2000,
                "height": 1500,
            }
        ]
    )

    assert document["images"][0]["image_review_status"] == "unreviewed"
    assert document["categories"] == [
        {"id": 1, "name": "fastener"},
        {"id": 2, "name": "pipe_joint"},
    ]
    assert document["annotations"] == []
