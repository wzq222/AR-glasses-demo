import pytest


torch = pytest.importorskip("torch")

from crrc_vision.witness_roi_model import (  # noqa: E402
    MobileNetV3SmallWitnessRoi,
    validate_witness_roi_outputs,
)


def test_model_emits_segmentation_keypoints_and_quality_without_state_classifier() -> None:
    model = MobileNetV3SmallWitnessRoi(pretrained=False).eval()
    images = torch.zeros((2, 3, 320, 320), dtype=torch.float32)

    with torch.inference_mode():
        segmentation, keypoints, quality = model(images)

    assert segmentation.shape == (2, 4, 320, 320)
    assert keypoints.shape == (2, 4, 320, 320)
    assert quality.shape == (2, 4)
    assert not hasattr(model, "state_classifier")
    assert sum(parameter.numel() for parameter in model.parameters()) < 4_000_000
    validate_witness_roi_outputs(segmentation, keypoints, quality, batch_size=2)


def test_model_rejects_wrong_input_shape() -> None:
    model = MobileNetV3SmallWitnessRoi(pretrained=False).eval()

    with pytest.raises(ValueError, match="NCHW RGB"):
        model(torch.zeros((1, 1, 320, 320)))


def test_keypoint_head_uses_stride_four_detail_features() -> None:
    model = MobileNetV3SmallWitnessRoi(pretrained=False).eval()
    captured: list[tuple[int, ...]] = []
    handle = model.keypoint_head.register_forward_pre_hook(
        lambda _module, inputs: captured.append(tuple(inputs[0].shape))
    )

    with torch.inference_mode():
        model(torch.zeros((1, 3, 320, 320)))
    handle.remove()

    assert captured == [(1, 48, 80, 80)]


def test_model_rejects_non_finite_input() -> None:
    model = MobileNetV3SmallWitnessRoi(pretrained=False).eval()
    images = torch.zeros((1, 3, 320, 320))
    images[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        model(images)


def test_output_validator_rejects_wrong_shape_or_non_finite_values() -> None:
    segmentation = torch.zeros((1, 4, 320, 320))
    keypoints = torch.zeros((1, 4, 320, 320))
    quality = torch.zeros((1, 4))
    quality[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        validate_witness_roi_outputs(segmentation, keypoints, quality, batch_size=1)
    with pytest.raises(ValueError, match="segmentation"):
        validate_witness_roi_outputs(
            torch.zeros((1, 3, 320, 320)),
            keypoints,
            torch.zeros((1, 4)),
            batch_size=1,
        )
