from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crrc_vision.verifier_model_soup import average_state_dicts


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
