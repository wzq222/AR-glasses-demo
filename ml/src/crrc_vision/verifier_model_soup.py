from __future__ import annotations

from collections.abc import Mapping, Sequence


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Average aligned floating tensors with narrow handling of BN counters."""
    import torch

    if len(state_dicts) < 2:
        raise ValueError("SOUP_REQUIRES_MULTIPLE_STATE_DICTS")
    keys = tuple(state_dicts[0])
    if any(tuple(state_dict) != keys for state_dict in state_dicts[1:]):
        raise ValueError("SOUP_STATE_KEYS_MISMATCH")
    averaged: dict[str, object] = {}
    for key in keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError(f"SOUP_VALUE_NOT_TENSOR:{key}")
        first = tensors[0]
        if any(
            tensor.shape != first.shape or tensor.dtype != first.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(f"SOUP_TENSOR_CONTRACT_MISMATCH:{key}")
        if first.is_floating_point():
            averaged[key] = torch.stack(
                [tensor.detach().cpu().to(torch.float64) for tensor in tensors]
            ).mean(0).to(first.dtype)
        elif key.endswith("num_batches_tracked"):
            averaged[key] = torch.stack(
                [tensor.detach().cpu() for tensor in tensors]
            ).max(0).values
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(f"SOUP_NONFLOAT_MISMATCH:{key}")
            averaged[key] = first.detach().cpu().clone()
    return averaged
