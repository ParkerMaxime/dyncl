#!/usr/bin/env python3
"""Explicit runtime backend for train-time dataset sampling/loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional

import numpy as np
from torch.utils.data import DataLoader

from .pool_registry import PoolRegistry
from .preindexed_two_level_sampler import (
    MutableRuntimeIndexBatchSampler,
    PreindexedTwoLevelSampler,
    build_mutable_preindexed_loader,
    build_preindexed_loader,
)
from .types import RuntimeIndexSelection, SamplerPlan


def _normalize_mix(mix: Mapping[str, float]) -> Dict[str, float]:
    clean = {
        str(key): float(value)
        for key, value in dict(mix).items()
        if float(value) > 0.0
    }
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {}
    return {
        str(key): float(value) / total
        for key, value in clean.items()
    }


@dataclass
class PreindexedRuntimeDatasetBackend:
    tiered_modules: tuple[str, ...]
    tier_key_fn: Callable[[str, str], str]
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    pool_registry: PoolRegistry = field(default_factory=PoolRegistry)
    seed_hint: int = 42
    current_epoch: int = 0
    runtime_top_level_mix: Dict[str, float] = field(default_factory=dict)
    runtime_nested_mix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    module_mix_active: bool = False
    train_sampler: Optional[PreindexedTwoLevelSampler] = None
    train_loader: DataLoader | None = None
    train_batch_sampler: MutableRuntimeIndexBatchSampler | None = None
    train_loader_key: object = None
    managed_train_flat_keys: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.current_epoch = 0
        self.runtime_top_level_mix = {}
        self.runtime_nested_mix = {}
        self.module_mix_active = False
        self.train_sampler = None
        self.train_loader = None
        self.train_batch_sampler = None
        self.train_loader_key = None
        self.managed_train_flat_keys = set()

    def set_seed_hint(self, seed_hint: int) -> None:
        self.seed_hint = int(seed_hint)
        self.train_sampler = None

    def set_pool_registry(self, pool_registry: PoolRegistry) -> None:
        self.pool_registry = pool_registry
        self.managed_train_flat_keys = pool_registry.managed_train_flat_keys(
            self.tiered_modules,
            self.tier_key_fn,
        )
        self.train_sampler = None
        self.train_loader = None
        self.train_batch_sampler = None
        self.train_loader_key = None

    def set_runtime_mix(
        self,
        *,
        top_level_mix: Mapping[str, float],
        nested_mix: Mapping[str, Mapping[str, float]],
        epoch: int,
        module_mix_active: bool,
    ) -> None:
        self.current_epoch = int(epoch)
        self.module_mix_active = bool(module_mix_active)
        self.runtime_top_level_mix = _normalize_mix(top_level_mix)
        self.runtime_nested_mix = {
            str(module_name): _normalize_mix(module_mix)
            for module_name, module_mix in dict(nested_mix).items()
            if _normalize_mix(module_mix)
        }

    def _build_sampler_plan(self, total_samples: int) -> SamplerPlan | None:
        if not self.pool_registry.train_module_pools:
            return None
        if not self.runtime_top_level_mix:
            return None

        top_level_mix: Dict[str, float] = {}
        nested_mix: Dict[str, Dict[str, float]] = {}
        for module_name, weight in self.runtime_top_level_mix.items():
            pool = self.pool_registry.train_module_pools.get(module_name)
            if pool is None or len(np.asarray(pool, dtype=np.int64)) <= 0:
                continue
            top_level_mix[str(module_name)] = float(weight)
            if module_name not in self.tiered_modules:
                continue
            module_tiers = self.pool_registry.train_tier_pools.get(module_name, {})
            module_nested_mix = self.runtime_nested_mix.get(module_name, {})
            filtered_nested = {
                str(tier_name): float(tier_weight)
                for tier_name, tier_weight in module_nested_mix.items()
                if tier_name in module_tiers
                and len(np.asarray(module_tiers[tier_name], dtype=np.int64)) > 0
                and float(tier_weight) > 0.0
            }
            if filtered_nested:
                nested_mix[str(module_name)] = _normalize_mix(filtered_nested)

        top_level_mix = _normalize_mix(top_level_mix)
        if not top_level_mix:
            return None
        return SamplerPlan(
            top_level_mix=top_level_mix,
            nested_mix=nested_mix,
            total_samples=int(total_samples),
        )

    def _get_sampler(self) -> PreindexedTwoLevelSampler | None:
        if isinstance(self.train_sampler, PreindexedTwoLevelSampler):
            return self.train_sampler

        if not self.pool_registry.train_module_pools:
            return None

        nested_pools = {
            str(module_name): {
                str(tier_name): np.asarray(pool, dtype=np.int64)
                for tier_name, pool in dict(module_tiers).items()
                if len(np.asarray(pool, dtype=np.int64)) > 0
            }
            for module_name, module_tiers in self.pool_registry.train_tier_pools.items()
            if module_name in self.tiered_modules
        }
        base_pools = {
            str(module_name): np.asarray(pool, dtype=np.int64)
            for module_name, pool in self.pool_registry.train_module_pools.items()
            if len(np.asarray(pool, dtype=np.int64)) > 0
        }
        sampler = PreindexedTwoLevelSampler(
            base_pools=base_pools,
            nested_pools=nested_pools,
            seed=int(self.seed_hint),
        )
        self.train_sampler = sampler
        return sampler

    def sample_indices(
        self,
        *,
        pools: Mapping[str, np.ndarray],
        mix: Mapping[str, float],
        total_samples: int,
        rng,
    ) -> RuntimeIndexSelection | None:
        mix_keys = {
            str(key)
            for key, value in dict(mix).items()
            if float(value) > 0.0
        }
        if (
            not self.module_mix_active
            or not self.managed_train_flat_keys
            or not mix_keys
            or not mix_keys.issubset(self.managed_train_flat_keys)
        ):
            return None

        plan = self._build_sampler_plan(int(total_samples))
        if plan is None:
            return None

        sampler = self._get_sampler()
        if sampler is None:
            return None
        shuffle_seed = (
            int(rng.integers(0, np.iinfo(np.uint32).max))
            if isinstance(rng, np.random.Generator)
            else None
        )
        sampler.set_epoch(int(self.current_epoch))
        sampler.set_plan(plan)
        indices = sampler.sample_epoch_indices()
        return RuntimeIndexSelection(
            indices=np.asarray(indices, dtype=np.int64),
            shuffle_seed=shuffle_seed,
            source="gestion_dataset_preindexed",
        )

    def build_loader(
        self,
        dataset,
        indices,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
    ) -> DataLoader | None:
        if isinstance(indices, RuntimeIndexSelection):
            selection = indices
        else:
            shuffle_seed = None
            if bool(shuffle):
                shuffle_seed = int(self.seed_hint) * 100003 + int(self.current_epoch)
            selection = RuntimeIndexSelection(
                indices=np.asarray(indices, dtype=np.int64),
                shuffle_seed=shuffle_seed,
                source="precomputed_indices",
            )

        loader_key = (
            id(dataset),
            int(batch_size),
            bool(shuffle),
            int(num_workers),
            bool(pin_memory),
            bool(persistent_workers),
            int(prefetch_factor),
        )
        if (
            bool(shuffle)
            and isinstance(indices, RuntimeIndexSelection)
            and str(selection.source) == "gestion_dataset_preindexed"
        ):
            if (
                self.train_loader is not None
                and self.train_batch_sampler is not None
                and self.train_loader_key == loader_key
            ):
                self.train_batch_sampler.set_selection(selection)
                return self.train_loader
            loader, batch_sampler = build_mutable_preindexed_loader(
                dataset,
                selection,
                batch_size=int(batch_size),
                shuffle=bool(shuffle),
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                persistent_workers=bool(persistent_workers),
                prefetch_factor=int(prefetch_factor),
            )
            self.train_loader = loader
            self.train_batch_sampler = batch_sampler
            self.train_loader_key = loader_key
            return loader
        return build_preindexed_loader(
            dataset,
            selection,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        )
