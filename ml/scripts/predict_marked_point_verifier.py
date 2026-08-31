"""Run a semantic MobileNet verifier on every E1 validation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_verifier import (
    select_dual_pipeline_thresholds,
    select_pipeline_threshold,
    suppress_overlapping_candidates,
)


FORMAL_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class CandidateDataset:
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(str(self.rows[index]["crop_path"])) as image:
            return self.transform(image.convert("RGB")), index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="runs/marked-point-verifier-e3/dataset-v2/manifest.json"
    )
    parser.add_argument(
        "--checkpoint", default="runs/marked-point-verifier-e4/mobilenetv3-small-semantic/best.pt"
    )
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument(
        "--output", default="runs/marked-point-verifier-e4/e1-candidate-predictions.json"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    root = asset_root().resolve()
    dataset_path = (root / args.dataset).resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    formal_truth = (root / args.formal_truth).resolve()
    output = (root / args.output).resolve()
    if _sha256(formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    if manifest.get("sealed_test_opened") is not False:
        raise RuntimeError("SEALED_TEST_STATE_INVALID")
    rows = [row for row in manifest["examples"] if row["split"] == "val"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = list(checkpoint["classes"])
    marked_index = classes.index("marked_point")
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(
        CandidateDataset(rows, weights.transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predictions = [None] * len(rows)
    with torch.inference_mode():
        for images, indices in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            for index, scores in zip(indices.tolist(), probabilities.tolist(), strict=True):
                predictions[index] = {
                    **rows[index],
                    "proposal_score": rows[index]["score"],
                    "score": scores[marked_index],
                    "class_scores": dict(zip(classes, scores, strict=True)),
                    "predicted_class": classes[max(range(len(scores)), key=scores.__getitem__)],
                }
    image_count = len({row["image_id"] for row in predictions})
    report = select_pipeline_threshold(
        predictions, image_count=image_count, minimum_truth_recall=1.0
    )
    dual_report = select_dual_pipeline_thresholds(
        [
            {
                **row,
                "verifier_score": row["score"],
            }
            for row in predictions
        ],
        image_count=image_count,
    )
    post_nms = suppress_overlapping_candidates(
        predictions,
        verifier_threshold=dual_report.verifier_threshold,
        proposal_threshold=dual_report.proposal_threshold,
        iou_threshold=0.3,
    )
    truth_ids = {truth_id for row in predictions for truth_id in row["truth_ids"]}
    covered_truth = {truth_id for row in post_nms for truth_id in row["truth_ids"]}
    per_image = [
        sum(row["image_id"] == image_id for row in post_nms)
        for image_id in sorted({row["image_id"] for row in predictions})
    ]
    post_nms_report = {
        "iou_threshold": 0.3,
        "truth_recall": len(covered_truth) / len(truth_ids),
        "covered_truth": len(covered_truth),
        "total_truth": len(truth_ids),
        "selected": len(post_nms),
        "relevant_selected": sum(bool(row["truth_ids"]) for row in post_nms),
        "irrelevant_selected": sum(not row["truth_ids"] for row in post_nms),
        "candidate_relevance": sum(bool(row["truth_ids"]) for row in post_nms)
        / len(post_nms),
        "candidates_per_image": len(post_nms) / image_count,
        "median_candidates_per_image": statistics.median(per_image),
        "maximum_candidates_in_one_image": max(per_image),
        "passed_recall": covered_truth == truth_ids,
        "passed_burden": len(post_nms) / image_count <= 20.0,
        "passed": covered_truth == truth_ids and len(post_nms) / image_count <= 20.0,
    }
    result = {
        "schema_version": "marked-point-semantic-verifier-predictions-v1",
        "classes": classes,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_sha256": _sha256(dataset_path),
        "formal_truth_sha256": _sha256(formal_truth),
        "sealed_test_opened": False,
        "report": asdict(report),
        "dual_report": asdict(dual_report),
        "post_nms_report": post_nms_report,
        "predicted_class_counts": {
            label: sum(row["predicted_class"] == label for row in predictions)
            for label in classes
        },
        "predictions": predictions,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
