#!/usr/bin/env python3
"""Registry helpers for preindexed module/tier pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np


@dataclass
class PoolRegistry:
    train_module_pools: Dict[str, np.ndarray] = field(default_factory=dict)
    eval_module_pools: Dict[str, np.ndarray] = field(default_factory=dict)
    train_tier_pools: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    eval_tier_pools: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    def module_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.train_module_pools.keys()))

    def tier_keys(self, module_name: str) -> tuple[str, ...]:
        pools = self.train_tier_pools.get(module_name, {})
        return tuple(sorted(pools.keys()))

    def reset_train(self) -> None:
        self.train_module_pools = {}
        self.train_tier_pools = {}

    def reset_eval(self) -> None:
        self.eval_module_pools = {}
        self.eval_tier_pools = {}

    def managed_train_flat_keys(
        self,
        tiered_modules: tuple[str, ...],
        tier_key_fn: Callable[[str, str], str],
    ) -> set[str]:
        keys = set(str(module_name) for module_name in self.train_module_pools.keys())
        for module_name in tiered_modules:
            module_tiers = self.train_tier_pools.get(module_name, {})
            for tier_name, pool in module_tiers.items():
                if len(np.asarray(pool, dtype=np.int64)) <= 0:
                    continue
                keys.add(str(tier_key_fn(str(module_name), str(tier_name))))
        return keys
