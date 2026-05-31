#!/usr/bin/env python3
"""
Sequential curriculum dataset pipeline used by the released paper runs.

This module assembles the primary, middle-school, and high-school module
bundles, applies scratchpad post-processing, prepares retention-aware train
views, and builds the sampler state used by the continual-learning recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import time
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from gestion_dataset.dataset_pipeline import build_or_load_bundle
from gestion_dataset.cache_bundle import (
    compatible_cache_specs,
    load_dataset,
    load_nested_pools,
    load_pools,
    save_nested_pools,
    save_pools,
)
from gestion_dataset.pool_registry import PoolRegistry
from gestion_dataset.preindexed_two_level_sampler import PreindexedTwoLevelSampler
from gestion_dataset.types import CompositeMathSeq2SeqDataset, DatasetBundle, DerivedCacheSpec

from config_3c5dync1_dynamic import RunConfig
from scratchpad_pipeline import apply_scratchpad_postprocess, run_answer_audit


LOGGER = logging.getLogger(__name__)


PRIMAIRE_INIT_MIX: Dict[str, float] = {
    "arithmetic__mul": 0.40,
    "arithmetic__div": 0.30,
    "arithmetic__add_sub_multiple": 0.20,
    "arithmetic__add_or_sub": 0.10,
}

COLLEGE_INIT_MIX: Dict[str, float] = {
    "arithmetic__mixed": 0.24,
    "numbers__gcd": 0.20,
    "numbers__lcm": 0.20,
    "numbers__div_remainder": 0.14,
    "numbers__is_factor": 0.10,
    "numbers__place_value": 0.06,
    "numbers__round_number": 0.06,
}

COLLEGE_RETENTION_MIX: Dict[str, float] = {
    "arithmetic__mixed": 0.10,
    "numbers__gcd": 0.14,
    "numbers__lcm": 0.14,
    "numbers__div_remainder": 0.14,
    "numbers__is_factor": 0.14,
    "numbers__place_value": 0.17,
    "numbers__round_number": 0.17,
}

LYCEE_INIT_MIX: Dict[str, float] = {
    "algebra__sequence_next_term": 0.35,
    "algebra__linear_1d": 0.30,
    "polynomials__evaluate": 0.20,
    "algebra__sequence_nth_term": 0.15,
}

MODULES_BY_LEVEL: Dict[str, tuple[str, ...]] = {
    "primaire": (
        "arithmetic__add_or_sub",
        "arithmetic__add_sub_multiple",
        "arithmetic__mul",
        "arithmetic__div",
    ),
    "college": (
        "arithmetic__mixed",
        "numbers__gcd",
        "numbers__lcm",
        "numbers__div_remainder",
        "numbers__is_factor",
        "numbers__place_value",
        "numbers__round_number",
    ),
    "lycee": (
        "algebra__linear_1d",
        "algebra__sequence_next_term",
        "algebra__sequence_nth_term",
        "polynomials__evaluate",
    ),
    "superieur": (),
}

HARD_TIERED_MODULES: tuple[str, ...] = (
    "numbers__gcd",
    "numbers__lcm",
    "numbers__div_remainder",
)

MODULE_ALLOWED_TIERS: Dict[str, tuple[int, ...]] = {
    "numbers__gcd": (1, 2, 3, 4),
    "numbers__lcm": (2, 3, 4),
    "numbers__div_remainder": (1, 2, 3, 4),
}


@dataclass(frozen=True)
class LocalMathLevelSpec:
    name: str
    modules: tuple[str, ...]
    new_modules: tuple[str, ...]
    all_modules: tuple[str, ...]
    description: str


class _IndexedDatasetView(torch.utils.data.Dataset):
    """Small train-only view used to cap retention levels before assembly."""

    def __init__(self, base_dataset: object, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.vocab = dict(base_dataset.vocab)
        self.max_q_len = int(base_dataset.max_q_len)
        self.max_a_len = int(base_dataset.max_a_len)
        self.pad_id = int(base_dataset.pad_id)
        self.unk_id = int(base_dataset.unk_id)
        self.sos_id = int(base_dataset.sos_id)
        self.eos_id = int(base_dataset.eos_id)
        self.level_ids = np.asarray(base_dataset.level_ids, dtype=np.int64)[self.indices].astype(np.int64, copy=False)
        self.targets = self.level_ids
        # Retention views are capped, so materializing these slices keeps
        # downstream token inference simple without changing behavior.
        self.encoded_questions = np.asarray(base_dataset.encoded_questions)[self.indices].astype(np.int64, copy=False)
        self.encoded_answers = np.asarray(base_dataset.encoded_answers)[self.indices].astype(np.int64, copy=False)
        self.level_names = [str(base_dataset.level_names[int(i)]) for i in self.indices.tolist()]
        self.module_names = [str(base_dataset.module_names[int(i)]) for i in self.indices.tolist()]
        self.questions = None
        self.answers = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int):
        return self.base_dataset[int(self.indices[int(idx)])]

    def get_raw_item(self, idx: int) -> Dict[str, object]:
        base_idx = int(self.indices[int(idx)])
        if hasattr(self.base_dataset, "get_raw_item"):
            return self.base_dataset.get_raw_item(base_idx)
        q, a = self.base_dataset[base_idx]
        return {
            "question": q if isinstance(q, str) else "",
            "answer": a if isinstance(a, str) else "",
            "level_name": str(self.base_dataset.level_names[base_idx]),
            "module_name": str(self.base_dataset.module_names[base_idx]),
            "level_id": int(self.base_dataset.level_ids[base_idx]),
        }


def normalize_mix(mix: Dict[str, float]) -> Dict[str, float]:
    clean = {str(k): float(v) for k, v in dict(mix).items() if float(v) > 0.0}
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {}
    return {str(k): float(v) / total for k, v in clean.items()}


def _modules_from_levels(modules_by_level: Mapping[str, Sequence[str]]) -> set[str]:
    modules: set[str] = set()
    for values in dict(modules_by_level).values():
        modules.update(str(module_name) for module_name in values)
    return modules


def _tiering_policy_for_modules(modules_by_level: Mapping[str, Sequence[str]]) -> str:
    modules = _modules_from_levels(modules_by_level)
    if modules.intersection(HARD_TIERED_MODULES):
        return "college_hard_tiers_run6_floor_v1_no_fallback_train"
    return "module_only_no_tiers_v1"


def _module_cache_name(module_name: str) -> str:
    module_key = str(module_name)
    if module_key in {"numbers__gcd", "numbers__lcm"}:
        suffix = "_scratchpad_v3_tiers_run5"
    elif module_key == "numbers__div_remainder":
        suffix = "_scratchpad_v3_tiers_run6"
    else:
        suffix = "_scratchpad_v3"
    return f"module_bundle_{_safe_cache_name(module_name)}{suffix}"


def _extract_first_two_ints(question: str) -> tuple[int, int]:
    nums = re.findall(r"-?\d+", str(question))
    if len(nums) < 2:
        raise ValueError(f"Unable to extract two integers from question={question!r}")
    return int(nums[0]), int(nums[1])


def _count_euclid_steps(a: int, b: int) -> int:
    x = abs(int(a))
    y = abs(int(b))
    steps = 0
    while y != 0:
        x, y = y, x % y
        steps += 1
    return steps


def _gcd_tier(question: str) -> int:
    a, b = _extract_first_two_ints(question)
    steps = _count_euclid_steps(a, b)
    if steps <= 1:
        return 1
    if steps == 2:
        return 2
    if steps == 3:
        return 3
    return 4


def _lcm_tier(question: str) -> int:
    a, b = _extract_first_two_ints(question)
    steps = _count_euclid_steps(a, b)
    if steps <= 1:
        return 2
    if steps == 2:
        return 3
    return 4


def _div_remainder_tier(question: str) -> int:
    target, factor = _extract_first_two_ints(question)
    q, _r = divmod(abs(int(target)), abs(int(factor)))
    factor_abs = abs(int(factor))
    if q < 100 and factor_abs < 100:
        return 1
    if q < 1000 and factor_abs < 1000:
        return 2
    if q < 10000 and factor_abs < 10000:
        return 3
    return 4


def classify_module_tier(module_name: str, question: str) -> str:
    module_key = str(module_name)
    if module_key == "numbers__gcd":
        tier_id = _gcd_tier(question)
    elif module_key == "numbers__lcm":
        tier_id = _lcm_tier(question)
    elif module_key == "numbers__div_remainder":
        tier_id = _div_remainder_tier(question)
    else:
        raise KeyError(module_key)
    return f"tier{int(tier_id)}"


def _is_fallback_result(dataset: object, idx: int) -> bool:
    answers = getattr(dataset, "answers", None)
    if answers is None:
        return False
    return str(answers[int(idx)]).startswith("result=")


def _partition_hard_pool_by_tier(
    dataset: object,
    pool: Sequence[int],
    module_name: str,
    *,
    exclude_fallback_result: bool,
) -> Dict[str, np.ndarray]:
    module_key = str(module_name)
    tier_indices: Dict[str, list[int]] = {
        f"tier{tier_id}": []
        for tier_id in MODULE_ALLOWED_TIERS[module_key]
    }
    for raw_idx in np.asarray(pool, dtype=np.int64).tolist():
        if bool(exclude_fallback_result) and _is_fallback_result(dataset, int(raw_idx)):
            continue
        item = dataset.get_raw_item(int(raw_idx))
        tier_name = classify_module_tier(module_key, str(item["question"]))
        if tier_name in tier_indices:
            tier_indices[tier_name].append(int(raw_idx))
    return {
        tier_name: np.asarray(values, dtype=np.int64)
        for tier_name, values in tier_indices.items()
        if values
    }


def _finalize_hard_module_tiers(bundle: DatasetBundle, *, module_name: str, spec: DerivedCacheSpec) -> DatasetBundle:
    module_key = str(module_name)
    if module_key not in HARD_TIERED_MODULES:
        return bundle
    if module_key not in dict(bundle.train_module_pools):
        return bundle
    if dict(bundle.train_tier_pools).get(module_key):
        return bundle

    train_pool = np.asarray(bundle.train_module_pools[module_key], dtype=np.int64)
    eval_pool = np.asarray(bundle.eval_module_pools.get(module_key, np.empty(0, dtype=np.int64)), dtype=np.int64)
    train_tiers = _partition_hard_pool_by_tier(
        bundle.train_dataset,
        train_pool,
        module_key,
        exclude_fallback_result=True,
    )
    eval_tiers = _partition_hard_pool_by_tier(
        bundle.test_dataset,
        eval_pool,
        module_key,
        exclude_fallback_result=False,
    )
    if train_tiers:
        filtered_train_pool = np.concatenate(list(train_tiers.values())).astype(np.int64, copy=False)
        filtered_train_pool.sort()
        bundle.train_module_pools = dict(bundle.train_module_pools)
        bundle.train_module_pools[module_key] = filtered_train_pool
        bundle.train_level_pools = {
            str(level_name): filtered_train_pool
            for level_name, modules in MODULES_BY_LEVEL.items()
            if module_key in set(str(m) for m in modules)
        }

    bundle.train_tier_pools = dict(bundle.train_tier_pools)
    bundle.eval_tier_pools = dict(bundle.eval_tier_pools)
    bundle.train_tier_pools[module_key] = train_tiers
    bundle.eval_tier_pools[module_key] = eval_tiers

    fallback_count = int(train_pool.size - np.asarray(bundle.train_module_pools[module_key], dtype=np.int64).size)
    metadata = dict(bundle.metadata)
    tiering_stats = dict(metadata.get("tiering", {}))
    tiering_stats[module_key] = {
        "policy": "run6_floor_hard_tiers",
        "train_total_before_filter": int(train_pool.size),
        "train_total_after_filter": int(np.asarray(bundle.train_module_pools[module_key], dtype=np.int64).size),
        "train_fallback_result_excluded": fallback_count,
        "train_tier_counts": {tier: int(len(pool)) for tier, pool in train_tiers.items()},
        "eval_tier_counts": {tier: int(len(pool)) for tier, pool in eval_tiers.items()},
    }
    metadata["tiering"] = tiering_stats
    bundle.metadata = metadata

    # Persist the corrected pools after module-level tier finalization so the
    # training run can reload them directly.
    save_pools(spec, "train_level_pools", bundle.train_level_pools)
    save_pools(spec, "train_module_pools", bundle.train_module_pools)
    save_nested_pools(spec, "train_tier_pools", bundle.train_tier_pools)
    save_nested_pools(spec, "eval_tier_pools", bundle.eval_tier_pools)
    return bundle


def level_specs(curriculum_variant: str | None = None):
    del curriculum_variant
    active = []
    specs = []
    descriptions = {
        "primaire": "Operations arithmetiques de base",
        "college": "College core scratchpad",
        "lycee": "Lycee core scratchpad",
        "superieur": "",
    }
    for level_name in ("primaire", "college", "lycee", "superieur"):
        new_modules = tuple(MODULES_BY_LEVEL[level_name])
        active.extend(new_modules)
        specs.append(
            LocalMathLevelSpec(
                name=level_name,
                modules=new_modules,
                new_modules=new_modules,
                all_modules=tuple(active),
                description=descriptions.get(level_name, ""),
            )
        )
    return specs


def derived_cache_spec(cfg: RunConfig) -> DerivedCacheSpec:
    return _derived_cache_spec_for_modules(
        cfg,
        cache_name=cfg.derived_cache_name,
        modules_by_level=MODULES_BY_LEVEL,
    )


def _derived_cache_spec_for_modules(
    cfg: RunConfig,
    *,
    cache_name: str,
    modules_by_level: Dict[str, tuple[str, ...]],
) -> DerivedCacheSpec:
    return DerivedCacheSpec(
        cache_root=Path(cfg.derived_cache_root),
        cache_name=cache_name,
        version=cfg.derived_cache_version,
        max_q_len=int(cfg.max_q_len),
        max_a_len=int(cfg.max_a_len),
        raw_max_samples_per_module=int(cfg.raw_max_samples_per_module),
        raw_max_test_samples_per_module=int(cfg.raw_max_test_samples_per_module),
        runtime_train_cap_per_module=int(cfg.runtime_train_cap_per_module),
        runtime_eval_cap_per_module=int(cfg.runtime_eval_cap_per_module),
        scratchpad_policy="primary_college_lycee_core_scratchpad_a128_v3",
        tiering_policy=_tiering_policy_for_modules(modules_by_level),
        extra_eval_splits=tuple(cfg.extra_eval_splits),
        modules_by_level={str(k): tuple(v) for k, v in modules_by_level.items()},
    )


def _specs_for_modules(modules_by_level: Dict[str, tuple[str, ...]]):
    active = []
    specs = []
    descriptions = {
        "primaire": "Operations arithmetiques de base",
        "college": "College full7 scratchpad",
        "lycee": "Lycee core scratchpad",
        "superieur": "",
    }
    for level_name in ("primaire", "college", "lycee", "superieur"):
        new_modules = tuple(modules_by_level[level_name])
        active.extend(new_modules)
        specs.append(
            LocalMathLevelSpec(
                name=level_name,
                modules=new_modules,
                new_modules=new_modules,
                all_modules=tuple(active),
                description=descriptions.get(level_name, ""),
            )
        )
    return specs


def _offset_pools(pools: Dict[str, np.ndarray], offset: int) -> Dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=np.int64) + int(offset)
        for key, value in pools.items()
        if len(np.asarray(value, dtype=np.int64)) > 0
    }


def _assemble_split_datasets(datasets: list[object]) -> CompositeMathSeq2SeqDataset:
    return CompositeMathSeq2SeqDataset(datasets)


def _cap_train_bundle_for_retention_level(
    bundle: DatasetBundle,
    *,
    level_name: str,
    current_level: str,
    cap_per_module: int,
) -> DatasetBundle:
    if str(level_name) == str(current_level) or int(cap_per_module) <= 0:
        return bundle

    selected_chunks: list[np.ndarray] = []
    capped_module_pools: Dict[str, np.ndarray] = {}
    local_cursor = 0
    for module_name, pool in sorted(dict(bundle.train_module_pools).items()):
        arr = np.asarray(pool, dtype=np.int64)
        if arr.size <= 0:
            continue
        capped = arr[: int(cap_per_module)].astype(np.int64, copy=False)
        selected_chunks.append(capped)
        capped_module_pools[str(module_name)] = np.arange(local_cursor, local_cursor + int(capped.size), dtype=np.int64)
        local_cursor += int(capped.size)

    if not selected_chunks:
        return bundle

    selected_indices = np.concatenate(selected_chunks).astype(np.int64, copy=False)
    train_dataset = _IndexedDatasetView(bundle.train_dataset, selected_indices)
    train_level_pools = {
        str(level_name): np.arange(0, len(train_dataset), dtype=np.int64)
    }
    metadata = dict(bundle.metadata)
    metadata["retention_cap"] = {
        "level": str(level_name),
        "current_level": str(current_level),
        "cap_per_module": int(cap_per_module),
        "train_total_after_cap": int(len(train_dataset)),
        "module_counts_after_cap": {module: int(len(pool)) for module, pool in capped_module_pools.items()},
    }
    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=bundle.test_dataset,
        extra_eval_datasets=bundle.extra_eval_datasets,
        train_level_pools=train_level_pools,
        eval_level_pools=bundle.eval_level_pools,
        extra_eval_level_pools=bundle.extra_eval_level_pools,
        train_module_pools=capped_module_pools,
        eval_module_pools=bundle.eval_module_pools,
        extra_eval_module_pools=bundle.extra_eval_module_pools,
        train_tier_pools={},
        eval_tier_pools=bundle.eval_tier_pools,
        extra_eval_tier_pools=bundle.extra_eval_tier_pools,
        metadata=metadata,
    )


def _merge_many_pools(bundle_pools: list[Dict[str, np.ndarray]], offsets: list[int]) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for pools, offset in zip(bundle_pools, offsets):
        for key, values in _offset_pools(pools, int(offset)).items():
            if key in merged:
                merged[key] = np.concatenate([merged[key], values]).astype(np.int64, copy=False)
            else:
                merged[key] = values
    return merged


def _merge_many_nested_pools(
    bundle_pools: list[Dict[str, Dict[str, np.ndarray]]],
    offsets: list[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for pools, offset in zip(bundle_pools, offsets):
        for module_name, module_tiers in dict(pools).items():
            target = merged.setdefault(str(module_name), {})
            for tier_name, values in dict(module_tiers).items():
                arr = np.asarray(values, dtype=np.int64)
                if arr.size <= 0:
                    continue
                shifted = arr + int(offset)
                if str(tier_name) in target:
                    target[str(tier_name)] = np.concatenate([target[str(tier_name)], shifted]).astype(np.int64, copy=False)
                else:
                    target[str(tier_name)] = shifted.astype(np.int64, copy=False)
    return {
        module_name: {
            tier_name: tier_pool
            for tier_name, tier_pool in module_tiers.items()
            if np.asarray(tier_pool, dtype=np.int64).size > 0
        }
        for module_name, module_tiers in merged.items()
        if module_tiers
    }


def _offsets_for_datasets(datasets: list[object]) -> list[int]:
    offsets = []
    total = 0
    for dataset in datasets:
        offsets.append(int(total))
        total += len(dataset)
    return offsets


def _assemble_bundles(bundles: list[DatasetBundle]) -> DatasetBundle:
    if not bundles:
        raise RuntimeError("Aucun bundle a assembler.")

    train_datasets = [bundle.train_dataset for bundle in bundles]
    test_datasets = [bundle.test_dataset for bundle in bundles]
    train_offsets = _offsets_for_datasets(train_datasets)
    test_offsets = _offsets_for_datasets(test_datasets)

    extra_eval_datasets = {}
    extra_eval_level_pools = {}
    extra_eval_module_pools = {}
    split_names = sorted(set.intersection(*(set(bundle.extra_eval_datasets) for bundle in bundles)))
    for split_name in split_names:
        split_datasets = [bundle.extra_eval_datasets[split_name] for bundle in bundles]
        split_offsets = _offsets_for_datasets(split_datasets)
        extra_eval_datasets[split_name] = _assemble_split_datasets(split_datasets)
        extra_eval_level_pools[split_name] = _merge_many_pools(
            [bundle.extra_eval_level_pools.get(split_name, {}) for bundle in bundles],
            split_offsets,
        )
        extra_eval_module_pools[split_name] = _merge_many_pools(
            [bundle.extra_eval_module_pools.get(split_name, {}) for bundle in bundles],
            split_offsets,
        )

    return DatasetBundle(
        train_dataset=_assemble_split_datasets(train_datasets),
        test_dataset=_assemble_split_datasets(test_datasets),
        extra_eval_datasets=extra_eval_datasets,
        train_level_pools=_merge_many_pools([bundle.train_level_pools for bundle in bundles], train_offsets),
        eval_level_pools=_merge_many_pools([bundle.eval_level_pools for bundle in bundles], test_offsets),
        extra_eval_level_pools=extra_eval_level_pools,
        train_module_pools=_merge_many_pools([bundle.train_module_pools for bundle in bundles], train_offsets),
        eval_module_pools=_merge_many_pools([bundle.eval_module_pools for bundle in bundles], test_offsets),
        extra_eval_module_pools=extra_eval_module_pools,
        train_tier_pools=_merge_many_nested_pools([bundle.train_tier_pools for bundle in bundles], train_offsets),
        eval_tier_pools=_merge_many_nested_pools([bundle.eval_tier_pools for bundle in bundles], test_offsets),
        extra_eval_tier_pools={},
        metadata={
            "source": "assembled_modules_lazy",
            "bundle_root": "assembled:module_bundles_lazy",
            "component_count": int(len(bundles)),
            "components": [dict(bundle.metadata) for bundle in bundles],
        },
    )


def _load_eval_only_bundle_from_cache(spec: DerivedCacheSpec, vocab: dict[str, int]) -> DatasetBundle:
    effective_vocab = dict(vocab) if vocab else {}
    test_dataset = load_dataset(spec, "test", effective_vocab if effective_vocab else None)
    available_extra_eval_splits: list[str] = []
    extra_eval_datasets = {}
    extra_eval_level_pools = {}
    extra_eval_module_pools = {}
    for split_name in spec.extra_eval_splits:
        split_token = f"eval_{split_name}"
        try:
            extra_eval_datasets[split_name] = load_dataset(
                spec,
                split_token,
                effective_vocab if effective_vocab else None,
            )
        except FileNotFoundError:
            continue
        available_extra_eval_splits.append(split_name)
        extra_eval_level_pools[split_name] = load_pools(spec, f"eval_level_pools__{split_name}")
        extra_eval_module_pools[split_name] = load_pools(spec, f"eval_module_pools__{split_name}")
    return DatasetBundle(
        train_dataset=test_dataset,
        test_dataset=test_dataset,
        extra_eval_datasets=extra_eval_datasets,
        train_level_pools={},
        eval_level_pools=load_pools(spec, "eval_level_pools"),
        extra_eval_level_pools=extra_eval_level_pools,
        train_module_pools={},
        eval_module_pools=load_pools(spec, "eval_module_pools"),
        extra_eval_module_pools=extra_eval_module_pools,
        train_tier_pools={},
        eval_tier_pools=load_nested_pools(spec, "eval_tier_pools"),
        extra_eval_tier_pools={},
        metadata={
            "source": "cache_eval_only",
            "bundle_root": str(spec.cache_root / f"{spec.cache_name}_{spec.version}"),
            "available_extra_eval_splits": list(available_extra_eval_splits),
        },
    )


def _assemble_eval_only_bundles(bundles: list[DatasetBundle]) -> DatasetBundle:
    if not bundles:
        raise RuntimeError("Aucun bundle eval-only a assembler.")
    test_datasets = [bundle.test_dataset for bundle in bundles]
    test_offsets = _offsets_for_datasets(test_datasets)

    extra_eval_datasets = {}
    extra_eval_level_pools = {}
    extra_eval_module_pools = {}
    split_names = sorted(set.intersection(*(set(bundle.extra_eval_datasets) for bundle in bundles)))
    for split_name in split_names:
        split_datasets = [bundle.extra_eval_datasets[split_name] for bundle in bundles]
        split_offsets = _offsets_for_datasets(split_datasets)
        extra_eval_datasets[split_name] = _assemble_split_datasets(split_datasets)
        extra_eval_level_pools[split_name] = _merge_many_pools(
            [bundle.extra_eval_level_pools.get(split_name, {}) for bundle in bundles],
            split_offsets,
        )
        extra_eval_module_pools[split_name] = _merge_many_pools(
            [bundle.extra_eval_module_pools.get(split_name, {}) for bundle in bundles],
            split_offsets,
        )

    assembled_test = _assemble_split_datasets(test_datasets)
    return DatasetBundle(
        train_dataset=assembled_test,
        test_dataset=assembled_test,
        extra_eval_datasets=extra_eval_datasets,
        train_level_pools={},
        eval_level_pools=_merge_many_pools([bundle.eval_level_pools for bundle in bundles], test_offsets),
        extra_eval_level_pools=extra_eval_level_pools,
        train_module_pools={},
        eval_module_pools=_merge_many_pools([bundle.eval_module_pools for bundle in bundles], test_offsets),
        extra_eval_module_pools=extra_eval_module_pools,
        train_tier_pools={},
        eval_tier_pools=_merge_many_nested_pools([bundle.eval_tier_pools for bundle in bundles], test_offsets),
        extra_eval_tier_pools={},
        metadata={
            "source": "assembled_modules_eval_only",
            "bundle_root": "assembled:module_bundles_eval_only",
            "component_count": int(len(bundles)),
            "components": [dict(bundle.metadata) for bundle in bundles],
        },
    )


def _filter_bundle_to_levels(bundle: DatasetBundle, *, selected_levels: set[str]) -> DatasetBundle:
    allowed_levels = {str(level_name) for level_name in selected_levels}
    allowed_modules = {
        str(module_name)
        for level_name in allowed_levels
        for module_name in MODULES_BY_LEVEL.get(str(level_name), ())
    }
    return DatasetBundle(
        train_dataset=bundle.train_dataset,
        test_dataset=bundle.test_dataset,
        extra_eval_datasets=dict(bundle.extra_eval_datasets),
        train_level_pools={
            str(level_name): np.asarray(pool, dtype=np.int64)
            for level_name, pool in dict(bundle.train_level_pools).items()
            if str(level_name) in allowed_levels
        },
        eval_level_pools={
            str(level_name): np.asarray(pool, dtype=np.int64)
            for level_name, pool in dict(bundle.eval_level_pools).items()
            if str(level_name) in allowed_levels
        },
        extra_eval_level_pools={
            str(split_name): {
                str(level_name): np.asarray(pool, dtype=np.int64)
                for level_name, pool in dict(level_pools).items()
                if str(level_name) in allowed_levels
            }
            for split_name, level_pools in dict(bundle.extra_eval_level_pools).items()
        },
        train_module_pools={
            str(module_name): np.asarray(pool, dtype=np.int64)
            for module_name, pool in dict(bundle.train_module_pools).items()
            if str(module_name) in allowed_modules
        },
        eval_module_pools={
            str(module_name): np.asarray(pool, dtype=np.int64)
            for module_name, pool in dict(bundle.eval_module_pools).items()
            if str(module_name) in allowed_modules
        },
        extra_eval_module_pools={
            str(split_name): {
                str(module_name): np.asarray(pool, dtype=np.int64)
                for module_name, pool in dict(module_pools).items()
                if str(module_name) in allowed_modules
            }
            for split_name, module_pools in dict(bundle.extra_eval_module_pools).items()
        },
        train_tier_pools={
            str(module_name): {
                str(tier_name): np.asarray(pool, dtype=np.int64)
                for tier_name, pool in dict(module_tiers).items()
                if len(np.asarray(pool, dtype=np.int64)) > 0
            }
            for module_name, module_tiers in dict(bundle.train_tier_pools).items()
            if str(module_name) in allowed_modules
        },
        eval_tier_pools={
            str(module_name): {
                str(tier_name): np.asarray(pool, dtype=np.int64)
                for tier_name, pool in dict(module_tiers).items()
                if len(np.asarray(pool, dtype=np.int64)) > 0
            }
            for module_name, module_tiers in dict(bundle.eval_tier_pools).items()
            if str(module_name) in allowed_modules
        },
        extra_eval_tier_pools={
            str(split_name): {
                str(module_name): {
                    str(tier_name): np.asarray(pool, dtype=np.int64)
                    for tier_name, pool in dict(module_tiers).items()
                    if len(np.asarray(pool, dtype=np.int64)) > 0
                }
                for module_name, module_tiers in dict(split_tiers).items()
                if str(module_name) in allowed_modules
            }
            for split_name, split_tiers in dict(bundle.extra_eval_tier_pools).items()
        },
        metadata={
            **dict(bundle.metadata),
            "filtered_levels": sorted(allowed_levels),
            "filtered_modules": sorted(allowed_modules),
        },
    )


def _single_module_levels(level_name: str, module_name: str) -> Dict[str, tuple[str, ...]]:
    return {
        "primaire": (str(module_name),) if str(level_name) == "primaire" else (),
        "college": (str(module_name),) if str(level_name) == "college" else (),
        "lycee": (str(module_name),) if str(level_name) == "lycee" else (),
        "superieur": (),
    }


def _safe_cache_name(module_name: str) -> str:
    return str(module_name).replace("__", "_").replace("/", "_").replace("-", "_")


def _fallback_module_cache_specs(module_name: str, spec: DerivedCacheSpec) -> list[DerivedCacheSpec]:
    root = Path(spec.cache_root)
    if not root.exists():
        return []
    prefix = f"module_bundle_{_safe_cache_name(module_name)}"
    suffix = f"_{spec.version}"
    matches: list[DerivedCacheSpec] = []
    for cache_root in sorted(root.glob(f"{prefix}*{suffix}")):
        if not cache_root.is_dir():
            continue
        cache_dir_name = cache_root.name
        if not cache_dir_name.endswith(suffix):
            continue
        cache_name = cache_dir_name[: -len(suffix)]
        if not cache_name:
            continue
        candidate = DerivedCacheSpec(
            cache_root=spec.cache_root,
            cache_name=cache_name,
            version=spec.version,
            max_q_len=spec.max_q_len,
            max_a_len=spec.max_a_len,
            raw_max_samples_per_module=spec.raw_max_samples_per_module,
            raw_max_test_samples_per_module=spec.raw_max_test_samples_per_module,
            runtime_train_cap_per_module=spec.runtime_train_cap_per_module,
            runtime_eval_cap_per_module=spec.runtime_eval_cap_per_module,
            scratchpad_policy=spec.scratchpad_policy,
            tiering_policy=spec.tiering_policy,
            extra_eval_splits=spec.extra_eval_splits,
            modules_by_level=spec.modules_by_level,
        )
        meta_path = cache_root / "meta.json"
        test_dir = cache_root / "test_dataset"
        eval_pools = cache_root / "eval_module_pools.npz"
        if meta_path.exists() and test_dir.exists() and eval_pools.exists():
            matches.append(candidate)
    return matches


def build_bundle(
    cfg: RunConfig,
    *,
    progress_callback=None,
    modules_by_level_override: Dict[str, tuple[str, ...]] | None = None,
    selected_levels_override: Sequence[str] | None = None,
    current_level_override: str | None = None,
    eval_only: bool = False,
):
    run_answer_audit()

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

    active_modules_by_level = {
        str(level_name): tuple(str(module_name) for module_name in modules)
        for level_name, modules in dict(modules_by_level_override or MODULES_BY_LEVEL).items()
    }
    selected_levels = (
        {str(level_name) for level_name in selected_levels_override}
        if selected_levels_override is not None
        else {str(level_name) for level_name in getattr(cfg, "levels", ("primaire", "college", "lycee"))}
    )
    current_level = str(current_level_override or getattr(cfg, "start_level", "lycee") or "lycee")
    component_bundles: list[DatasetBundle] = []
    shared_vocab = None
    module_sequence = []
    if "primaire" in selected_levels:
        module_sequence.extend(
            ("primaire", module_name)
            for module_name in active_modules_by_level.get("primaire", ())
        )
    if "college" in selected_levels:
        module_sequence.extend(
            ("college", module_name)
            for module_name in active_modules_by_level.get("college", ())
        )
    if "lycee" in selected_levels:
        module_sequence.extend(
            ("lycee", module_name)
            for module_name in active_modules_by_level.get("lycee", ())
        )
    for level_name, module_name in module_sequence:
        module_levels = _single_module_levels(str(level_name), str(module_name))
        started = time.monotonic()
        LOGGER.info("[dataset-load] start module=%s level=%s", module_name, level_name)
        module_spec = _derived_cache_spec_for_modules(
            cfg,
            cache_name=_module_cache_name(module_name),
            modules_by_level=module_levels,
        )
        if bool(eval_only):
            compatible_specs = compatible_cache_specs(module_spec)
            fallback_specs = _fallback_module_cache_specs(str(module_name), module_spec)
            selected_spec = compatible_specs[0] if compatible_specs else (fallback_specs[0] if fallback_specs else None)
            if selected_spec is not None:
                if compatible_specs:
                    LOGGER.info("[dataset-load] cache hit module=%s cache=%s mode=compatible", module_name, selected_spec.cache_name)
                else:
                    LOGGER.warning("[dataset-load] cache fallback module=%s cache=%s mode=prefix_only", module_name, selected_spec.cache_name)
                bundle = _load_eval_only_bundle_from_cache(
                    selected_spec,
                    dict(shared_vocab) if shared_vocab is not None else {},
                )
            else:
                raise RuntimeError(
                    "E64 eval_only cache miss for module="
                    f"{module_name} level={level_name} cache_name={module_spec.cache_name} "
                    f"cache_root={module_spec.cache_root}. Refusing full rebuild in eval_only mode."
                )
        else:
            bundle = build_or_load_bundle(
                module_spec,
                level_specs=_specs_for_modules(module_levels),
                dataset_postprocess_fn=_postprocess,
                modules_by_level={level: list(modules) for level, modules in module_levels.items()},
                use_subset_cache=True,
                subset_cache_dir=cfg.subset_cache_dir,
                vocab=shared_vocab,
                max_q_len=int(cfg.max_q_len),
                max_a_len=int(cfg.max_a_len),
            )
            bundle = _finalize_hard_module_tiers(
                bundle,
                module_name=str(module_name),
                spec=module_spec,
            )
            bundle = _cap_train_bundle_for_retention_level(
                bundle,
                level_name=str(level_name),
                current_level=current_level,
                cap_per_module=int(getattr(cfg, "retention_train_cap_per_module", 0) or 0),
            )
        if shared_vocab is None:
            shared_vocab = dict(bundle.test_dataset.vocab if bool(eval_only) else bundle.train_dataset.vocab)
        component_bundles.append(bundle)
        LOGGER.info(
            "[dataset-load] done module=%s level=%s train=%d test=%d retention_cap=%d current_level=%s elapsed=%.1fs",
            module_name,
            level_name,
            len(bundle.train_dataset),
            len(bundle.test_dataset),
            int(getattr(cfg, "retention_train_cap_per_module", 0) or 0) if str(level_name) != current_level else 0,
            current_level,
            time.monotonic() - started,
        )
    LOGGER.info("[dataset-load] assembling %d module bundles", len(component_bundles))
    bundle = _assemble_eval_only_bundles(component_bundles) if bool(eval_only) else _assemble_bundles(component_bundles)
    bundle = _filter_bundle_to_levels(bundle, selected_levels=selected_levels)
    return bundle


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
        train_tier_pools={
            str(module_name): {
                str(tier_name): np.asarray(pool, dtype=np.int64)
                for tier_name, pool in dict(module_tiers).items()
                if len(np.asarray(pool, dtype=np.int64)) > 0
            }
            for module_name, module_tiers in dict(bundle.train_tier_pools).items()
        },
        eval_tier_pools={
            str(module_name): {
                str(tier_name): np.asarray(pool, dtype=np.int64)
                for tier_name, pool in dict(module_tiers).items()
                if len(np.asarray(pool, dtype=np.int64)) > 0
            }
            for module_name, module_tiers in dict(bundle.eval_tier_pools).items()
        },
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
    tiered_modules = tuple(HARD_TIERED_MODULES) if bool(getattr(cfg, "college_tier_curriculum_enabled", False)) else ()
    sampler = PreindexedTwoLevelSampler(
        base_pools=dict(bundle.train_module_pools),
        nested_pools={
            str(module_name): dict(module_tiers)
            for module_name, module_tiers in dict(bundle.train_tier_pools).items()
            if str(module_name) in tiered_modules
        },
        seed=int(cfg.seeds[0]) if cfg.seeds else 42,
    )
    sampler.set_plan(
        type("SamplerPlanLike", (), {
            "top_level_mix": normalize_mix({**PRIMAIRE_INIT_MIX, **COLLEGE_INIT_MIX, **LYCEE_INIT_MIX}),
            "nested_mix": {},
            "total_samples": int(cfg.samples_per_epoch),
        })()
    )
    sampler.set_epoch(1)
    return sampler
