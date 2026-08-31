import pytest


torch = pytest.importorskip("torch")

from crrc_vision.witness_roi_losses import (  # noqa: E402
    keypoint_geometry_loss,
    keypoint_distribution_loss,
    spatial_soft_argmax,
    witness_mask_loss,
)


def test_sparse_mask_loss_gives_positive_pixel_stronger_gradient() -> None:
    logits = torch.zeros((1, 16, 16), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 8, 8] = 1.0

    loss = witness_mask_loss(logits, target)
    loss.backward()

    positive_gradient = abs(float(logits.grad[0, 8, 8]))
    negative_mean = float(logits.grad[target == 0].abs().mean())
    assert positive_gradient > negative_mean * 20.0


def test_keypoint_distribution_loss_rewards_probability_at_target_peak() -> None:
    target = torch.zeros((1, 1, 8, 8))
    target[0, 0, 3, 5] = 1.0
    wrong = torch.zeros_like(target)
    correct = torch.zeros_like(target)
    correct[0, 0, 3, 5] = 8.0

    assert keypoint_distribution_loss(correct, target) < keypoint_distribution_loss(
        wrong, target
    )


def test_keypoint_distribution_loss_rejects_empty_or_wrong_shape_target() -> None:
    logits = torch.zeros((1, 4, 8, 8))

    with pytest.raises(ValueError, match="shape"):
        keypoint_distribution_loss(logits, torch.zeros((1, 3, 8, 8)))
    with pytest.raises(ValueError, match="positive mass"):
        keypoint_distribution_loss(logits, torch.zeros_like(logits))


def test_keypoint_geometry_loss_penalizes_coupled_angle_error() -> None:
    target = torch.zeros((1, 4, 16, 16))
    target[0, 0, 8, 2] = 1.0
    target[0, 1, 8, 8] = 1.0
    target[0, 2, 8, 8] = 1.0
    target[0, 3, 8, 14] = 1.0
    correct = target * 12.0
    wrong = correct.clone()
    wrong[0, 3] = 0.0
    wrong[0, 3, 14, 8] = 12.0

    correct_loss = keypoint_geometry_loss(correct, target)
    wrong_loss = keypoint_geometry_loss(wrong, target)

    assert correct_loss < wrong_loss
    assert torch.isfinite(wrong_loss)


def test_spatial_soft_argmax_returns_normalized_xy_coordinates() -> None:
    logits = torch.full((1, 1, 5, 5), -20.0)
    logits[0, 0, 3, 4] = 20.0

    point = spatial_soft_argmax(logits)

    assert point[0, 0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert point[0, 0, 1].item() == pytest.approx(0.75, abs=1e-4)
