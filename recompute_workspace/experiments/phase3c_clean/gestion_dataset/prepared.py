#!/usr/bin/env python3
"""Prepared dataset/backend artifact returned to runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


@dataclass
class PreparedDatasetBundle:
    spec: object
    bundle: object
    pool_registry: object
    train_pools: Mapping[str, object]
    eval_pools: Mapping[str, object]
    extra_eval_loaders: Mapping[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def train_dataset(self):
        return self.bundle.train_dataset

    @property
    def test_dataset(self):
        return self.bundle.test_dataset

    def summary(self) -> dict[str, object]:
        return {
            "spec": self.metadata.get("spec_summary", {}),
            "bundle_source": str(getattr(self.bundle, "metadata", {}).get("source", "")),
            "bundle_root": str(getattr(self.bundle, "metadata", {}).get("bundle_root", "")),
            "train_n": int(len(self.bundle.train_dataset)),
            "test_n": int(len(self.bundle.test_dataset)),
            "train_pool_keys": int(len(self.train_pools)),
            "eval_levels": sorted(str(k) for k in self.eval_pools.keys()),
            "extra_eval_splits": sorted(str(k) for k in self.extra_eval_loaders.keys()),
        }

    def validate_levels(
        self,
        *,
        modules_by_level: Mapping[str, Sequence[str]],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        selected_levels = set(str(level) for level in getattr(self.spec, "levels", ()))
        allowed_modules = {
            str(module_name)
            for level_name, modules in dict(modules_by_level).items()
            if str(level_name) in selected_levels
            for module_name in modules
        }
        train_modules = set(str(name) for name in getattr(self.pool_registry, "train_module_pools", {}).keys())
        eval_modules = set(str(name) for name in getattr(self.pool_registry, "eval_module_pools", {}).keys())
        leaked_train = sorted(train_modules - allowed_modules)
        leaked_eval = sorted(eval_modules - allowed_modules)
        if leaked_train or leaked_eval:
            message = (
                "Prepared dataset bundle leaked modules outside selected levels: "
                f"train={leaked_train} eval={leaked_eval} "
                f"selected_levels={sorted(selected_levels)}"
            )
            if on_error is not None:
                on_error(message)
                return
            raise RuntimeError(message)
