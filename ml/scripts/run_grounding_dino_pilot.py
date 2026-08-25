"""Run a guarded GroundingDINO proposal pilot without mutating truth labels."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from PIL import Image, ImageDraw

from crrc_vision.assets import asset_root
from crrc_vision.proposals import (
    MODEL_ID,
    MODEL_REVISION,
    TEXT_PROMPT,
    TRANSFORMERS_VERSION,
    PilotAudit,
    Proposal,
    pilot_can_expand,
    select_pilot_items,
    validate_loading_info,
    validate_transformers_version,
)

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_audit(path: Path) -> PilotAudit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PilotAudit(
        accepted=int(payload["accepted"]),
        rejected=int(payload["rejected"]),
        images_with_missed_targets=int(payload["images_with_missed_targets"]),
        images=int(payload["images"]),
    )


def _category(label: str) -> str:
    return "pipe_joint" if "pipe" in label.lower() else "fastener"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="selections/selection-v2.json")
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--output", default="review-packs/fastener-v2/pilot")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--expand", action="store_true")
    args = parser.parse_args()

    root = asset_root()
    output = (root / args.output).resolve()
    if root.resolve() not in output.parents:
        raise ValueError("output must stay below CRRC_VISION_DATA_ROOT")
    output.mkdir(parents=True, exist_ok=True)

    selection = json.loads((root / args.selection).read_text(encoding="utf-8"))
    if args.expand:
        audit_path = output / "pilot-audit.json"
        if not audit_path.is_file() or not pilot_can_expand(_load_audit(audit_path)):
            print("pilot expansion refused: audited quality gate has not passed", file=sys.stderr)
            return 2
        items = selection["items"]
    else:
        items = select_pilot_items(selection["items"], count=args.count)

    try:
        import torch
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as error:
        _write_json(
            output / "dependency-error.json",
            {"error_code": "MODEL_DEPENDENCY_UNAVAILABLE", "detail": str(error)},
        )
        print("MODEL_DEPENDENCY_UNAVAILABLE", file=sys.stderr)
        return 3
    version_errors = validate_transformers_version(transformers.__version__)
    if version_errors:
        _write_json(
            output / "dependency-error.json",
            {
                "error_code": version_errors[0],
                "expected": TRANSFORMERS_VERSION,
                "actual": transformers.__version__,
            },
        )
        print(version_errors[0], file=sys.stderr)
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model, loading_info = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_loading_info=True,
    )
    loading_errors = validate_loading_info(loading_info)
    if loading_errors:
        _write_json(
            output / "model-loading-error.json",
            {
                "error_codes": loading_errors,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "loading_info": loading_info,
            },
        )
        print("model checkpoint did not load completely", file=sys.stderr)
        return 4
    model = model.to(device)
    model.eval()

    proposals: list[Proposal] = []
    overlay_root = output / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        image_path = root / args.source / item["relative_path"]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), precision_context:
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            left, top, right, bottom = (float(value) for value in box.tolist())
            proposal = Proposal(
                relative_path=str(item["relative_path"]),
                category=_category(str(label)),
                bbox=(left, top, right - left, bottom - top),
                score=float(score.item()),
                source=f"{MODEL_ID}@{MODEL_REVISION}",
            )
            proposals.append(proposal)
            draw.rectangle((left, top, right, bottom), outline=(0, 255, 0), width=4)
            draw.text((left, max(0, top - 18)), f"{proposal.category} {proposal.score:.2f}", fill=(0, 255, 0))
        overlay.save(overlay_root / f"{index:02d}_{Path(str(item['relative_path'])).name}", quality=90)
        print(f"processed {index}/{len(items)}: {item['relative_path']} ({len(result['boxes'])} proposals)")

    _write_json(
        output / ("expanded-proposals.json" if args.expand else "pilot-proposals.json"),
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": device,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "images": items,
            "proposals": [proposal.to_dict() for proposal in proposals],
        },
    )
    print(json.dumps({"images": len(items), "proposals": len(proposals), "device": device}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
