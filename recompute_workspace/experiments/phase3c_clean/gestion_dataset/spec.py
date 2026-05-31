#!/usr/bin/env python3
"""Declarative dataset/backend spec for clean runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    levels: tuple[str, ...]
    start_level: str
    bootstrap_completed_levels: tuple[str, ...]
    # Descriptive run label only. Runtime mix policy is still implemented by the
    # family backend/profile, not by this field directly.
    mix_variant: str
    scratchpad_version: str
    subset_cache_dir: str
    derived_cache_root: str
    derived_cache_name: str
    raw_max_samples_per_module: int
    raw_max_test_samples_per_module: int
    runtime_train_cap_per_module: int
    runtime_eval_cap_per_module: int
    retention_train_cap_per_module: int
    extra_eval_splits: tuple[str, ...]
    college_tier_curriculum_enabled: bool = False

    @classmethod
    def from_run_config(
        cls,
        cfg,
        *,
        scratchpad_version: str = "",
    ) -> "DatasetSpec":
        return cls(
            levels=tuple(str(level) for level in getattr(cfg, "levels", ())),
            start_level=str(getattr(cfg, "start_level", "")),
            bootstrap_completed_levels=tuple(str(level) for level in getattr(cfg, "bootstrap_completed_levels", ())),
            mix_variant=str(getattr(cfg, "mix_variant", "")),
            scratchpad_version=str(scratchpad_version or getattr(cfg, "derived_cache_version", "")),
            subset_cache_dir=str(getattr(cfg, "subset_cache_dir", "")),
            derived_cache_root=str(getattr(cfg, "derived_cache_root", "")),
            derived_cache_name=str(getattr(cfg, "derived_cache_name", "")),
            raw_max_samples_per_module=int(getattr(cfg, "raw_max_samples_per_module", 0) or 0),
            raw_max_test_samples_per_module=int(getattr(cfg, "raw_max_test_samples_per_module", 0) or 0),
            runtime_train_cap_per_module=int(getattr(cfg, "runtime_train_cap_per_module", 0) or 0),
            runtime_eval_cap_per_module=int(getattr(cfg, "runtime_eval_cap_per_module", 0) or 0),
            retention_train_cap_per_module=int(getattr(cfg, "retention_train_cap_per_module", 0) or 0),
            extra_eval_splits=tuple(str(split) for split in getattr(cfg, "extra_eval_splits", ()) or ()),
            college_tier_curriculum_enabled=bool(getattr(cfg, "college_tier_curriculum_enabled", False)),
        )

    def summary(self) -> Mapping[str, object]:
        return {
            "levels": list(self.levels),
            "start_level": self.start_level,
            "bootstrap_completed_levels": list(self.bootstrap_completed_levels),
            "mix_variant_label": self.mix_variant,
            "scratchpad_version": self.scratchpad_version,
            "subset_cache_dir": self.subset_cache_dir,
            "derived_cache_root": self.derived_cache_root,
            "derived_cache_name": self.derived_cache_name,
            "raw_max_samples_per_module": self.raw_max_samples_per_module,
            "raw_max_test_samples_per_module": self.raw_max_test_samples_per_module,
            "runtime_train_cap_per_module": self.runtime_train_cap_per_module,
            "runtime_eval_cap_per_module": self.runtime_eval_cap_per_module,
            "retention_train_cap_per_module": self.retention_train_cap_per_module,
            "extra_eval_splits": list(self.extra_eval_splits),
            "college_tier_curriculum_enabled": self.college_tier_curriculum_enabled,
        }


def normalize_level_modules(modules_by_level: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {
        str(level): tuple(str(module) for module in modules)
        for level, modules in dict(modules_by_level).items()
    }
