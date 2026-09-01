from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crrc_vision.verifier_model_soup import average_state_dicts, shared_verifier_contract


def test_shared_verifier_contract_keeps_input_size() -> None:
    checkpoints = [
        {
            "architecture": "mobilenet_v3_small",
            "classes": ["marked_point", "not_marked_point"],
            "dataset_sha256": "A" * 64,
            "input_size": 128,
        },
        {
            "architecture": "mobilenet_v3_small",
            "classes": ["marked_point", "not_marked_point"],
            "dataset_sha256": "A" * 64,
            "input_size": 128,
        },
    ]

    assert shared_verifier_contract(checkpoints) == (
        "mobilenet_v3_small",
        ("marked_point", "not_marked_point"),
        "A" * 64,
        128,
    )

    checkpoints[1]["input_size"] = 224
    with pytest.raises(ValueError, match="SOUP_CHECKPOINT_CONTRACT_MISMATCH"):
        shared_verifier_contract(checkpoints)


def test_model_soup_averages_floats_and_uses_max_batch_counter() -> None:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "bn.num_batches_tracked": torch.tensor(4, dtype=torch.long),
    }
    second = {
        "weight": torch.tensor([3.0, 5.0]),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.long),
    }

    result = average_state_dicts([first, second])

    assert torch.equal(result["weight"], torch.tensor([2.0, 4.0]))
    assert result["bn.num_batches_tracked"].item() == 7


def test_model_soup_rejects_other_nonfloat_mismatch() -> None:
    first = {"index": torch.tensor([1], dtype=torch.long)}
    second = {"index": torch.tensor([2], dtype=torch.long)}

    with pytest.raises(ValueError, match="SOUP_NONFLOAT_MISMATCH:index"):
        average_state_dicts([first, second])
