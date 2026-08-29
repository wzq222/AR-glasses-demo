"""Pinned Ultralytics adapter for capped synthetic batches."""

from __future__ import annotations

import os
from pathlib import Path

from .synthetic_ablation import SYNTHETIC_IMAGE_ID_OFFSET, build_capped_batches


class CappedSyntheticBatchSampler:
    def __init__(
        self,
        real_indices: list[int],
        synthetic_indices: list[int],
        *,
        batch_size: int,
        maximum_synthetic_fraction: float,
        seed: int,
    ) -> None:
        self.real_indices = real_indices
        self.synthetic_indices = synthetic_indices
        self.batch_size = batch_size
        self.maximum_synthetic_fraction = maximum_synthetic_fraction
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.real_indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = build_capped_batches(
            self.real_indices,
            self.synthetic_indices,
            batch_size=self.batch_size,
            maximum_synthetic_fraction=self.maximum_synthetic_fraction,
            seed=self.seed,
            epoch=self.epoch,
        )
        self.epoch += 1
        yield from batches


def _materialized_image_id(path: str) -> int:
    return int(Path(path).stem.split("_", 1)[0])


def make_synthetic_cap_trainer(*, maximum_synthetic_fraction: float, seed: int):
    import torch
    from ultralytics.data.build import PIN_MEMORY, InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import torch_distributed_zero_first

    class SyntheticCapDetectionTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank not in (-1, 0):
                raise RuntimeError("SYNTHETIC_CAP_SINGLE_GPU_ONLY")
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            real_indices = []
            synthetic_indices = []
            for index, image_path in enumerate(dataset.im_files):
                target = (
                    synthetic_indices
                    if _materialized_image_id(image_path) >= SYNTHETIC_IMAGE_ID_OFFSET
                    else real_indices
                )
                target.append(index)
            sampler = CappedSyntheticBatchSampler(
                real_indices,
                synthetic_indices,
                batch_size=batch_size,
                maximum_synthetic_fraction=maximum_synthetic_fraction,
                seed=seed,
            )
            workers = min(os.cpu_count() or 1, self.args.workers)
            generator = torch.Generator()
            generator.manual_seed(6148914691236517205)
            return InfiniteDataLoader(
                dataset=dataset,
                batch_sampler=sampler,
                num_workers=workers,
                pin_memory=PIN_MEMORY,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=generator,
            )

    return SyntheticCapDetectionTrainer
