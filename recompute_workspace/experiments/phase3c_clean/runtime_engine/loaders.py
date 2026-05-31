#!/usr/bin/env python3
"""Shared loader helpers for clean phase 3C runners."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from torch.utils.data import DataLoader, Dataset


def loader_runtime_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": int(max(0, num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if int(prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


class IndexedSubset(Dataset):
    """Subset that also returns original sample index for replay weighting."""

    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int):
        orig_idx = int(self.indices[idx])
        q, a = self.dataset[orig_idx]
        return q, a, orig_idx


def build_loader(
    dataset,
    indices,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
):
    return DataLoader(
        IndexedSubset(dataset, np.asarray(indices, dtype=np.int64)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=bool(shuffle),
        **loader_runtime_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )


def build_eval_loaders(
    test_dataset,
    eval_pools,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    max_samples_per_pool=0,
):
    cap = int(max(0, max_samples_per_pool or 0))
    return {
        lv: build_loader(
            test_dataset,
            np.asarray(pool, dtype=np.int64)[:cap] if cap > 0 else pool,
            batch_size,
            False,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )
        for lv, pool in eval_pools.items()
        if len(pool) > 0
    }
