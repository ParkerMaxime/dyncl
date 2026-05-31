#!/usr/bin/env python3
"""
Primary-stage dataset pipeline used by the released paper runs.

This module assembles the four-module primary curriculum, applies the
scratchpad post-processing used in the release, and builds the flat sampler
state consumed by the primary training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from gestion_dataset.dataset_pipeline import build_or_load_bundle
from gestion_dataset.pool_registry import PoolRegistry
from gestion_dataset.preindexed_two_level_sampler import PreindexedTwoLevelSampler
from gestion_dataset.types import DerivedCacheSpec

from config_3c5p21_clean_dense import RunConfig
from scratchpad_pipeline import apply_scratchpad_postprocess, run_answer_audit


PRIMAIRE_INIT_MIX: Dict[str, float] = {
    "arithmetic__mul": 0.40,
    "arithmetic__div": 0.30,
    "arithmetic__add_sub_multiple": 0.20,
    "arithmetic__add_or_sub": 0.10,
}


@dataclass(frozen=True)
class LocalMathLevelSpec:
    name: str
    modules: tuple[str, ...]
    new_modules: tuple[str, ...]
    all_modules: tuple[str, ...]
    description: str


def normalize_mix(mix: Dict[str, float]) -> Dict[str, float]:
    clean = {str(k): float(v) for k, v in dict(mix).items() if float(v) > 0.0}
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {}
    return {str(k): float(v) / total for k, v in clean.items()}


def level_specs(curriculum_variant: str | None = None):
    del curriculum_variant
    return [
        LocalMathLevelSpec(
            name="primaire",
            modules=(
                "arithmetic__add_or_sub",
                "arithmetic__add_sub_multiple",
                "arithmetic__mul",
                "arithmetic__div",
            ),
            new_modules=(
                "arithmetic__add_or_sub",
                "arithmetic__add_sub_multiple",
                "arithmetic__mul",
                "arithmetic__div",
            ),
            all_modules=(
                "arithmetic__add_or_sub",
                "arithmetic__add_sub_multiple",
                "arithmetic__mul",
                "arithmetic__div",
            ),
            description="Operations arithmetiques de base",
        ),
        LocalMathLevelSpec(name="college", modules=(), new_modules=(), all_modules=(), description=""),
        LocalMathLevelSpec(name="lycee", modules=(), new_modules=(), all_modules=(), description=""),
        LocalMathLevelSpec(name="superieur", modules=(), new_modules=(), all_modules=(), description=""),
    ]


def derived_cache_spec(cfg: RunConfig) -> DerivedCacheSpec:
    return DerivedCacheSpec(
        cache_root=Path(cfg.derived_cache_root),
        cache_name=cfg.derived_cache_name,
        version=cfg.derived_cache_version,
        max_q_len=int(cfg.max_q_len),
        max_a_len=int(cfg.max_a_len),
        raw_max_samples_per_module=int(cfg.raw_max_samples_per_module),
        raw_max_test_samples_per_module=int(cfg.raw_max_test_samples_per_module),
        runtime_train_cap_per_module=int(cfg.runtime_train_cap_per_module),
        runtime_eval_cap_per_module=int(cfg.runtime_eval_cap_per_module),
        scratchpad_policy="all4_primary_plus_explicit_answer_audit_v2",
        tiering_policy="module_only_no_tiers_v1",
        extra_eval_splits=tuple(cfg.extra_eval_splits),
    )


def build_bundle(cfg: RunConfig, *, progress_callback=None):
    run_answer_audit()
    spec = derived_cache_spec(cfg)
    specs = level_specs()

    def _postprocess(train_dataset, test_dataset, extra_eval_datasets):
        if progress_callback is not None:
            progress_callback({"stage": "scratchpad_start"})
        stats = apply_scratchpad_postprocess(
            train_dataset,
            test_dataset,
            extra_eval_datasets,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback({"stage": "scratchpad_done"})
        return stats

    return build_or_load_bundle(
        spec,
        level_specs=specs,
        dataset_postprocess_fn=_postprocess,
        modules_by_level={
            "primaire": [
                "arithmetic__add_or_sub",
                "arithmetic__add_sub_multiple",
                "arithmetic__mul",
                "arithmetic__div",
            ],
            "college": [],
            "lycee": [],
            "superieur": [],
        },
        use_subset_cache=True,
        subset_cache_dir=cfg.subset_cache_dir,
        vocab=None,
        max_q_len=int(cfg.max_q_len),
        max_a_len=int(cfg.max_a_len),
    )


def build_pool_registry(bundle) -> PoolRegistry:
    return PoolRegistry(
        train_module_pools={
            str(module_name): np.asarray(pool, dtype=np.int64)
            for module_name, pool in dict(bundle.train_module_pools).items()
            if len(np.asarray(pool, dtype=np.int64)) > 0
        },
        eval_module_pools={
            str(module_name): np.asarray(pool, dtype=np.int64)
            for module_name, pool in dict(bundle.eval_module_pools).items()
            if len(np.asarray(pool, dtype=np.int64)) > 0
        },
        train_tier_pools={},
        eval_tier_pools={},
    )


def build_flat_train_pools(bundle, pool_registry: PoolRegistry, tier_key_fn=None) -> Dict[str, np.ndarray]:
    del tier_key_fn
    combined: Dict[str, np.ndarray] = {
        str(module_name): np.asarray(pool, dtype=np.int64)
        for module_name, pool in pool_registry.train_module_pools.items()
    }
    if getattr(bundle, "train_level_pools", None):
        for level_name, pool in dict(bundle.train_level_pools).items():
            combined[str(level_name)] = np.asarray(pool, dtype=np.int64)
    return combined


def build_level_eval_pools(bundle) -> Dict[str, np.ndarray]:
    return {
        str(level_name): np.asarray(pool, dtype=np.int64)
        for level_name, pool in dict(bundle.eval_level_pools).items()
        if len(np.asarray(pool, dtype=np.int64)) > 0
    }


def build_sampler(cfg: RunConfig, bundle) -> PreindexedTwoLevelSampler:
    sampler = PreindexedTwoLevelSampler(
        base_pools=dict(bundle.train_module_pools),
        nested_pools={},
        seed=int(cfg.seeds[0]) if cfg.seeds else 42,
    )
    sampler.set_plan(
        type("SamplerPlanLike", (), {
            "top_level_mix": dict(PRIMAIRE_INIT_MIX),
            "nested_mix": {},
            "total_samples": int(cfg.samples_per_epoch),
        })()
    )
    sampler.set_epoch(1)
    return sampler
