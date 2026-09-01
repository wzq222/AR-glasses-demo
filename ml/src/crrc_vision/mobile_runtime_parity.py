"""Shared preprocessing and postprocessing for desktop mobile-runtime parity."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def resolve_below_preserving_alias(root: Path, relative: str) -> Path:
    """Validate through real paths while retaining an ASCII junction or symlink."""

    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(lexical_root / relative))
    real_root = lexical_root.resolve()
    real_path = lexical_path.resolve()
    if real_path == real_root or real_root not in real_path.parents:
        raise ValueError(f"path escapes runtime root: {relative}")
    return lexical_path


@dataclass(frozen=True)
class Letterbox:
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def _java_round(value: float) -> int:
    return int(np.floor(value + 0.5))


def letterbox_rgb(
    image: np.ndarray,
    *,
    target_size: int = 640,
    fill: int = 114,
) -> tuple[np.ndarray, Letterbox]:
    """Return RGB NCHW input using the Android detector's geometry contract."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB_IMAGE_REQUIRED")
    original_height, original_width = image.shape[:2]
    if original_width <= 0 or original_height <= 0 or target_size <= 0:
        raise ValueError("POSITIVE_IMAGE_DIMENSIONS_REQUIRED")
    scale = min(target_size / original_width, target_size / original_height)
    resized_width = _java_round(original_width * scale)
    resized_height = _java_round(original_height * scale)
    horizontal_padding = target_size - resized_width
    vertical_padding = target_size - resized_height
    pad_left = horizontal_padding // 2
    pad_top = vertical_padding // 2
    transform = Letterbox(
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=horizontal_padding - pad_left,
        pad_bottom=vertical_padding - pad_top,
    )
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), transform


def _iou_xywh(first: list[float], second: list[float]) -> float:
    first_right = first[0] + first[2]
    first_bottom = first[1] + first[3]
    second_right = second[0] + second[2]
    second_bottom = second[1] + second[3]
    intersection_width = max(0.0, min(first_right, second_right) - max(first[0], second[0]))
    intersection_height = max(
        0.0, min(first_bottom, second_bottom) - max(first[1], second[1])
    )
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def decode_yolo_predictions(
    prediction: np.ndarray,
    *,
    image_id: int,
    original_width: int,
    original_height: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    confidence_threshold: float = 0.20,
    nms_iou_threshold: float = 0.45,
    pre_nms_top_k: int = 1_000,
    max_detections: int = 100,
    category_id_offset: int = 0,
) -> list[dict[str, object]]:
    """Mirror the Android YOLO decoder, including class-agnostic NMS."""

    values = np.asarray(prediction, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] < 5:
        raise ValueError(f"YOLO_OUTPUT_SHAPE_MISMATCH:{values.shape}")
    if original_width <= 0 or original_height <= 0 or scale <= 0.0:
        raise ValueError("INVALID_IMAGE_GEOMETRY")
    candidates: list[dict[str, object]] = []
    for index in range(values.shape[1]):
        class_id = int(np.argmax(values[4:, index]))
        score = float(values[4 + class_id, index])
        if not np.isfinite(score) or score < confidence_threshold:
            continue
        center_x, center_y, width, height = (
            float(values[row, index]) for row in range(4)
        )
        if not all(np.isfinite(value) for value in (center_x, center_y, width, height)):
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        left = min(max((center_x - width / 2.0 - pad_x) / scale, 0.0), original_width)
        top = min(max((center_y - height / 2.0 - pad_y) / scale, 0.0), original_height)
        right = min(max((center_x + width / 2.0 - pad_x) / scale, 0.0), original_width)
        bottom = min(max((center_y + height / 2.0 - pad_y) / scale, 0.0), original_height)
        if right <= left or bottom <= top:
            continue
        candidates.append(
            {
                "image_id": image_id,
                "category_id": class_id + category_id_offset,
                "bbox": [left, top, right - left, bottom - top],
                "score": score,
            }
        )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    selected: list[dict[str, object]] = []
    for proposal in candidates[:pre_nms_top_k]:
        if len(selected) >= max_detections:
            break
        proposal_box = proposal["bbox"]
        if any(
            _iou_xywh(proposal_box, accepted["bbox"]) > nms_iou_threshold  # type: ignore[arg-type]
            for accepted in selected
        ):
            continue
        selected.append(proposal)
    return selected


def predict_image(
    infer: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    *,
    image_id: int,
    input_size: int = 640,
    output_channels: int = 6,
    output_candidates: int = 34_000,
    confidence_threshold: float = 0.20,
    nms_iou_threshold: float = 0.45,
    pre_nms_top_k: int = 1_000,
    max_detections: int = 100,
    category_id_offset: int = 0,
) -> list[dict[str, object]]:
    """Run one runtime adapter and enforce the frozen mobile tensor contract."""

    tensor, transform = letterbox_rgb(image, target_size=input_size)
    output = np.asarray(infer(tensor), dtype=np.float32)
    if output.shape == (output_channels, output_candidates):
        output = output[None]
    if output.shape != (1, output_channels, output_candidates):
        raise ValueError(f"YOLO_OUTPUT_SHAPE_MISMATCH:{output.shape}")
    return decode_yolo_predictions(
        output,
        image_id=image_id,
        original_width=image.shape[1],
        original_height=image.shape[0],
        scale=transform.scale,
        pad_x=float(transform.pad_left),
        pad_y=float(transform.pad_top),
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        pre_nms_top_k=pre_nms_top_k,
        max_detections=max_detections,
        category_id_offset=category_id_offset,
    )


def make_onnx_infer(model_path: Path, *, threads: int = 4) -> Callable[[np.ndarray], np.ndarray]:
    """Create a CPU ONNX Runtime adapter for the frozen single-input model."""

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(model_path.resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("SINGLE_INPUT_OUTPUT_MODEL_REQUIRED")
    input_name = inputs[0].name
    output_name = outputs[0].name

    def infer(tensor: np.ndarray) -> np.ndarray:
        return np.asarray(session.run([output_name], {input_name: tensor})[0])

    return infer


def make_ncnn_infer(
    param_path: Path,
    bin_path: Path,
    *,
    threads: int = 4,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a CPU ncnn adapter for pnnx's stable in0/out0 contract."""

    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = threads
    if net.load_param(str(param_path.resolve())) != 0:
        raise RuntimeError("NCNN_PARAM_LOAD_FAILED")
    if net.load_model(str(bin_path.resolve())) != 0:
        raise RuntimeError("NCNN_MODEL_LOAD_FAILED")

    def infer(tensor: np.ndarray) -> np.ndarray:
        with net.create_extractor() as extractor:
            if extractor.input("in0", ncnn.Mat(tensor[0]).clone()) != 0:
                raise RuntimeError("NCNN_INPUT_FAILED")
            status, output = extractor.extract("out0")
            if status != 0:
                raise RuntimeError("NCNN_EXTRACT_FAILED")
            return np.asarray(output)

    return infer


def make_mnn_infer(model_path: Path, *, threads: int = 4) -> Callable[[np.ndarray], np.ndarray]:
    """Create a CPU MNN adapter using NCHW host tensors."""

    import MNN

    interpreter = MNN.Interpreter(str(model_path.absolute()))
    session = interpreter.createSession(
        {"backend": "CPU", "numThread": threads, "precision": "high"}
    )
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (1, 3, 640, 640))
    interpreter.resizeSession(session)

    def infer(tensor: np.ndarray) -> np.ndarray:
        host = MNN.Tensor(
            (1, 3, 640, 640),
            MNN.Halide_Type_Float,
            tensor,
            MNN.Tensor_DimensionType_Caffe,
        )
        input_tensor.copyFrom(host)
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session)
        shape = tuple(int(value) for value in output.getShape())
        return np.asarray(output.getData(), dtype=np.float32).reshape(shape)

    return infer
