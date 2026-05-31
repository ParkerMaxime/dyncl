#!/usr/bin/env python3
"""Two-level preindexed sampler for clean dense/dynamic runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .types import RuntimeIndexSelection, SamplerPlan


def _normalize(mix: Mapping[str, float]) -> Dict[str, float]:
    clean = {str(k): float(v) for k, v in mix.items() if float(v) > 0.0}
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("mix vide")
    return {k: v / total for k, v in clean.items()}


def _quota_counts(mix: Mapping[str, float], total_samples: int) -> Dict[str, int]:
    mix_norm = _normalize(mix)
    keys = list(mix_norm.keys())
    expected = np.array([mix_norm[k] * total_samples for k in keys], dtype=np.float64)
    counts = np.floor(expected).astype(np.int64)
    remainder = int(total_samples - int(counts.sum()))
    if remainder > 0:
        fractions = expected - counts
        order = np.argsort(-fractions)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return {k: int(c) for k, c in zip(keys, counts.tolist()) if int(c) > 0}


@dataclass
class PreindexedTwoLevelSampler:
    base_pools: Dict[str, np.ndarray]
    nested_pools: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    seed: int = 42
    current_epoch: int = 0
    current_plan: SamplerPlan | None = None

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def set_plan(self, plan: SamplerPlan) -> None:
        self.current_plan = plan

    def pool_sizes(self) -> Dict[str, int]:
        return {
            str(key): int(np.asarray(values, dtype=np.int64).size)
            for key, values in self.base_pools.items()
        }

    def nested_pool_sizes(self) -> Dict[str, Dict[str, int]]:
        return {
            str(key): {
                str(inner_key): int(np.asarray(inner_values, dtype=np.int64).size)
                for inner_key, inner_values in inner.items()
            }
            for key, inner in self.nested_pools.items()
        }

    def sample_pool_counts(self) -> Dict[str, object]:
        if self.current_plan is None:
            raise RuntimeError("SamplerPlan absent")
        top_counts = _quota_counts(self.current_plan.top_level_mix, self.current_plan.total_samples)
        summary: Dict[str, object] = {}
        for key, count in top_counts.items():
            nested = self.nested_pools.get(key)
            nested_mix = self.current_plan.nested_mix.get(key, {})
            if nested and nested_mix:
                nested_mix = self.current_plan.nested_mix.get(key, {})
                summary[str(key)] = _quota_counts(nested_mix, count)
            else:
                summary[str(key)] = int(count)
        return summary

    def sample_epoch_indices(self) -> np.ndarray:
        if self.current_plan is None:
            raise RuntimeError("SamplerPlan absent")

        rng = np.random.default_rng(self.seed + int(self.current_epoch))
        top_counts = _quota_counts(self.current_plan.top_level_mix, self.current_plan.total_samples)
        chunks = []

        for key, count in top_counts.items():
            nested = self.nested_pools.get(key)
            nested_mix = self.current_plan.nested_mix.get(key, {})
            if nested and nested_mix:
                nested_mix = self.current_plan.nested_mix.get(key, {})
                nested_counts = _quota_counts(nested_mix, count)
                for nested_key, nested_count in nested_counts.items():
                    pool = np.asarray(nested.get(nested_key, np.empty(0, dtype=np.int64)), dtype=np.int64)
                    if pool.size == 0:
                        continue
                    replace = nested_count > int(pool.size)
                    sampled = rng.choice(pool, size=nested_count, replace=replace)
                    chunks.append(np.asarray(sampled, dtype=np.int64))
                continue

            pool = np.asarray(self.base_pools.get(key, np.empty(0, dtype=np.int64)), dtype=np.int64)
            if pool.size == 0:
                continue
            replace = count > int(pool.size)
            sampled = rng.choice(pool, size=count, replace=replace)
            chunks.append(np.asarray(sampled, dtype=np.int64))

        if not chunks:
            raise RuntimeError("Aucun indice echantillonne")
        indices = np.concatenate(chunks)
        rng.shuffle(indices)
        return indices

    def iter_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        indices = self.sample_epoch_indices()
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size doit etre > 0")
        for start in range(0, int(indices.size), bs):
            yield indices[start:start + bs]


class IndexedDatasetAdapter(Dataset):
    """Dataset wrapper that preserves the original sample index."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample_idx = int(idx)
        q, a = self.dataset[sample_idx]
        return q, a, sample_idx


class PrecomputedIndexBatchSampler:
    """Batch sampler over a precomputed index selection."""

    def __init__(
        self,
        indices: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        shuffle_seed: int | None = None,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.shuffle_seed = None if shuffle_seed is None else int(shuffle_seed)

    def __iter__(self) -> Iterator[list[int]]:
        if self.batch_size <= 0:
            raise ValueError("batch_size doit etre > 0")
        ordered = np.asarray(self.indices, dtype=np.int64).copy()
        if self.shuffle and ordered.size > 1:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(ordered)
        limit = int(ordered.size)
        if self.drop_last:
            limit = (limit // self.batch_size) * self.batch_size
        for start in range(0, limit, self.batch_size):
            batch = ordered[start:start + self.batch_size]
            if batch.size < self.batch_size and self.drop_last:
                continue
            yield [int(idx) for idx in batch.tolist()]

    def __len__(self) -> int:
        if self.batch_size <= 0:
            raise ValueError("batch_size doit etre > 0")
        total = int(self.indices.size)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


class MutableRuntimeIndexBatchSampler:
    """Mutable batch sampler so a DataLoader can survive across epochs."""

    def __init__(
        self,
        selection: RuntimeIndexSelection,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.selection = selection

    def set_selection(self, selection: RuntimeIndexSelection) -> None:
        self.selection = selection

    def __iter__(self) -> Iterator[list[int]]:
        if self.batch_size <= 0:
            raise ValueError("batch_size doit etre > 0")
        ordered = np.asarray(self.selection.indices, dtype=np.int64).copy()
        if self.shuffle and ordered.size > 1:
            rng = np.random.default_rng(self.selection.shuffle_seed)
            rng.shuffle(ordered)
        limit = int(ordered.size)
        if self.drop_last:
            limit = (limit // self.batch_size) * self.batch_size
        for start in range(0, limit, self.batch_size):
            batch = ordered[start:start + self.batch_size]
            if batch.size < self.batch_size and self.drop_last:
                continue
            yield [int(idx) for idx in batch.tolist()]

    def __len__(self) -> int:
        if self.batch_size <= 0:
            raise ValueError("batch_size doit etre > 0")
        total = int(np.asarray(self.selection.indices, dtype=np.int64).size)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def _loader_runtime_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": int(max(0, num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if int(kwargs["num_workers"]) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if int(prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def build_preindexed_loader(
    dataset: Dataset,
    selection: RuntimeIndexSelection,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    wrapped_dataset = IndexedDatasetAdapter(dataset)
    batch_sampler = PrecomputedIndexBatchSampler(
        np.asarray(selection.indices, dtype=np.int64),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(shuffle),
        shuffle_seed=selection.shuffle_seed,
    )
    return DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        **_loader_runtime_kwargs(
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        ),
    )


def build_mutable_preindexed_loader(
    dataset: Dataset,
    selection: RuntimeIndexSelection,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> tuple[DataLoader, MutableRuntimeIndexBatchSampler]:
    wrapped_dataset = IndexedDatasetAdapter(dataset)
    batch_sampler = MutableRuntimeIndexBatchSampler(
        selection,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(shuffle),
    )
    loader = DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        **_loader_runtime_kwargs(
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        ),
    )
    return loader, batch_sampler
