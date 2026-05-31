#!/usr/bin/env python3
"""Common dataset-bundle builder/loader for clean dense/dynamic runners."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import numpy as np

from curriculum.math_curriculum import (
    MathLevelSpec,
    build_eval_index_pools,
    build_eval_index_pools_per_module,
    build_math_datasets,
    build_math_datasets_with_eval_splits,
    build_train_index_pools,
    build_train_index_pools_per_module,
)

from .cache_bundle import (
    bundle_is_compatible,
    bundle_files_present,
    cache_dir,
    compatible_cache_specs,
    load_dataset,
    load_metadata,
    load_nested_pools,
    load_pools,
    save_dataset,
    save_metadata,
    save_nested_pools,
    save_pools,
)
from .types import DatasetBundle, DerivedCacheSpec


def _normalize_pools(pools: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=np.int64)
        for key, value in pools.items()
        if len(np.asarray(value, dtype=np.int64)) > 0
    }


def _merge_module_pools_from_specs(
    dataset: object,
    specs: Sequence[MathLevelSpec],
    *,
    eval_pools: bool,
) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for spec in specs:
        if eval_pools:
            pools = build_eval_index_pools_per_module(dataset, spec)
        else:
            pools = build_train_index_pools_per_module(dataset, spec)
        for module_name, pool in pools.items():
            arr = np.asarray(pool, dtype=np.int64)
            if arr.size <= 0:
                continue
            merged[str(module_name)] = arr
    return _normalize_pools(merged)


def _cap_pool(pool: np.ndarray, cap: int) -> np.ndarray:
    arr = np.asarray(pool, dtype=np.int64)
    if cap <= 0:
        return arr
    if arr.size <= cap:
        return arr
    return np.asarray(arr[:cap], dtype=np.int64)


def _cap_module_pools(
    pools: dict[str, np.ndarray],
    cap: int,
) -> dict[str, np.ndarray]:
    return {
        str(module_name): _cap_pool(pool, cap)
        for module_name, pool in pools.items()
        if np.asarray(pool, dtype=np.int64).size > 0
    }


def _cap_tier_pools(
    module_pools: dict[str, np.ndarray],
    tier_pools: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    capped: dict[str, dict[str, np.ndarray]] = {}
    for module_name, module_pool in module_pools.items():
        module_tiers = tier_pools.get(str(module_name), {})
        if not module_tiers:
            continue
        allowed = np.asarray(module_pool, dtype=np.int64)
        capped_module_tiers = {
            str(tier_name): np.asarray(
                np.asarray(tier_pool, dtype=np.int64)[
                    np.isin(np.asarray(tier_pool, dtype=np.int64), allowed)
                ],
                dtype=np.int64,
            )
            for tier_name, tier_pool in module_tiers.items()
        }
        capped_module_tiers = {
            tier_name: tier_pool
            for tier_name, tier_pool in capped_module_tiers.items()
            if tier_pool.size > 0
        }
        if capped_module_tiers:
            capped[str(module_name)] = capped_module_tiers
    return capped


def _rebuild_level_pools_from_modules(
    dataset: object,
    module_pools: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    level_names = getattr(dataset, "level_names", None)
    if level_names is None:
        return {}

    level_to_chunks: dict[str, list[np.ndarray]] = {}
    for module_pool in module_pools.values():
        arr = np.asarray(module_pool, dtype=np.int64)
        if arr.size <= 0:
            continue
        level_name = str(level_names[int(arr[0])])
        level_to_chunks.setdefault(level_name, []).append(arr)

    rebuilt: dict[str, np.ndarray] = {}
    for level_name, chunks in level_to_chunks.items():
        if not chunks:
            continue
        rebuilt[str(level_name)] = np.concatenate(chunks).astype(np.int64, copy=False)
    return rebuilt


def _apply_runtime_caps(bundle: DatasetBundle, spec: DerivedCacheSpec) -> DatasetBundle:
    bundle.train_module_pools = _cap_module_pools(
        dict(bundle.train_module_pools),
        int(spec.runtime_train_cap_per_module),
    )
    bundle.eval_module_pools = _cap_module_pools(
        dict(bundle.eval_module_pools),
        int(spec.runtime_eval_cap_per_module),
    )
    bundle.train_tier_pools = _cap_tier_pools(
        bundle.train_module_pools,
        dict(bundle.train_tier_pools),
    )
    bundle.eval_tier_pools = _cap_tier_pools(
        bundle.eval_module_pools,
        dict(bundle.eval_tier_pools),
    )

    rebuilt_train_levels = _rebuild_level_pools_from_modules(bundle.train_dataset, bundle.train_module_pools)
    if rebuilt_train_levels:
        bundle.train_level_pools = rebuilt_train_levels

    rebuilt_eval_levels = _rebuild_level_pools_from_modules(bundle.test_dataset, bundle.eval_module_pools)
    if rebuilt_eval_levels:
        bundle.eval_level_pools = rebuilt_eval_levels

    if bundle.extra_eval_module_pools and bundle.extra_eval_datasets:
        capped_extra_eval_module_pools: dict[str, dict[str, np.ndarray]] = {}
        capped_extra_eval_level_pools: dict[str, dict[str, np.ndarray]] = {}
        for split_name, module_pools in bundle.extra_eval_module_pools.items():
            capped_module_pools = _cap_module_pools(
                dict(module_pools),
                int(spec.runtime_eval_cap_per_module),
            )
            capped_extra_eval_module_pools[str(split_name)] = capped_module_pools
            dataset = bundle.extra_eval_datasets.get(str(split_name))
            if dataset is None:
                continue
            rebuilt_levels = _rebuild_level_pools_from_modules(dataset, capped_module_pools)
            capped_extra_eval_level_pools[str(split_name)] = rebuilt_levels
        bundle.extra_eval_module_pools = capped_extra_eval_module_pools
        if capped_extra_eval_level_pools:
            bundle.extra_eval_level_pools = capped_extra_eval_level_pools

    if bundle.extra_eval_tier_pools and bundle.extra_eval_module_pools:
        capped_extra_eval_tier_pools: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for split_name, split_tiers in bundle.extra_eval_tier_pools.items():
            module_pools = bundle.extra_eval_module_pools.get(str(split_name), {})
            capped_extra_eval_tier_pools[str(split_name)] = _cap_tier_pools(
                dict(module_pools),
                dict(split_tiers),
            )
        bundle.extra_eval_tier_pools = capped_extra_eval_tier_pools

    bundle.metadata = dict(bundle.metadata)
    bundle.metadata["runtime_caps"] = {
        "train_per_module": int(spec.runtime_train_cap_per_module),
        "eval_per_module": int(spec.runtime_eval_cap_per_module),
    }
    return bundle


def _load_bundle_from_cache(spec: DerivedCacheSpec, vocab: dict[str, int]) -> DatasetBundle:
    metadata = load_metadata(spec) or {}
    train_dataset = load_dataset(spec, "train", vocab if vocab else None)
    test_dataset = load_dataset(spec, "test", vocab if vocab else None)
    extra_eval_datasets = {
        split_name: load_dataset(spec, f"eval_{split_name}", vocab if vocab else None)
        for split_name in spec.extra_eval_splits
    }
    bundle = DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        extra_eval_datasets=extra_eval_datasets,
        train_level_pools=load_pools(spec, "train_level_pools"),
        eval_level_pools=load_pools(spec, "eval_level_pools"),
        extra_eval_level_pools={
            split_name: load_pools(spec, f"eval_level_pools__{split_name}")
            for split_name in spec.extra_eval_splits
        },
        train_module_pools=load_pools(spec, "train_module_pools"),
        eval_module_pools=load_pools(spec, "eval_module_pools"),
        extra_eval_module_pools={
            split_name: load_pools(spec, f"eval_module_pools__{split_name}")
            for split_name in spec.extra_eval_splits
        },
        train_tier_pools=load_nested_pools(spec, "train_tier_pools"),
        eval_tier_pools=load_nested_pools(spec, "eval_tier_pools"),
        extra_eval_tier_pools={},
        metadata={
            "bundle_root": str(cache_dir(spec)),
            "meta": metadata,
            "source": "cache",
            "postprocess": metadata.get("postprocess", {}),
        },
    )
    return _apply_runtime_caps(bundle, spec)


def _save_bundle_to_cache(spec: DerivedCacheSpec, bundle: DatasetBundle) -> DatasetBundle:
    root = cache_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    metadata = save_metadata(spec)
    if bundle.metadata.get("postprocess"):
        metadata = dict(metadata)
        metadata["postprocess"] = dict(bundle.metadata.get("postprocess", {}))
        save_metadata_path = root / "meta.json"
        save_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    save_dataset(spec, "train", bundle.train_dataset)
    save_dataset(spec, "test", bundle.test_dataset)
    for split_name, dataset in bundle.extra_eval_datasets.items():
        save_dataset(spec, f"eval_{split_name}", dataset)
    save_pools(spec, "train_level_pools", bundle.train_level_pools)
    save_pools(spec, "eval_level_pools", bundle.eval_level_pools)
    for split_name, pools in bundle.extra_eval_level_pools.items():
        save_pools(spec, f"eval_level_pools__{split_name}", pools)
    save_pools(spec, "train_module_pools", bundle.train_module_pools)
    save_pools(spec, "eval_module_pools", bundle.eval_module_pools)
    for split_name, pools in bundle.extra_eval_module_pools.items():
        save_pools(spec, f"eval_module_pools__{split_name}", pools)
    save_nested_pools(spec, "train_tier_pools", bundle.train_tier_pools)
    save_nested_pools(spec, "eval_tier_pools", bundle.eval_tier_pools)
    bundle.metadata = {
        "bundle_root": str(root),
        "meta": metadata,
        "source": "rebuilt",
        "postprocess": bundle.metadata.get("postprocess", {}),
        "runtime_caps": bundle.metadata.get("runtime_caps", {}),
    }
    return bundle


def build_or_load_bundle(
    spec: DerivedCacheSpec,
    *,
    level_specs: Sequence[MathLevelSpec],
    train_tier_pools: dict[str, dict[str, np.ndarray]] | None = None,
    eval_tier_pools: dict[str, dict[str, np.ndarray]] | None = None,
    dataset_postprocess_fn: Callable[[object, object, dict[str, object]], Any] | None = None,
    **build_dataset_kwargs,
) -> DatasetBundle:
    """
    Build or load a derived dataset bundle versioned by `DerivedCacheSpec`.

    The caller remains responsible for computing any tier pools not handled by
    the shared pipeline yet, then can pass them here for persistence.
    """
    build_kwargs = dict(build_dataset_kwargs)
    build_kwargs.pop("level_specs", None)
    build_kwargs.pop("max_samples_per_module", None)
    build_kwargs.pop("max_test_samples_per_module", None)
    build_kwargs.pop("max_eval_samples_per_module", None)
    build_kwargs.pop("extra_eval_splits", None)
    build_kwargs.pop("dataset_postprocess_fn", None)
    vocab = build_kwargs.get("vocab")

    compatible_specs = compatible_cache_specs(spec)
    if compatible_specs:
        return _load_bundle_from_cache(compatible_specs[0], dict(vocab) if vocab is not None else {})

    extra_eval_splits = tuple(spec.extra_eval_splits)
    if extra_eval_splits:
        train_dataset, eval_datasets = build_math_datasets_with_eval_splits(
            **build_kwargs,
            max_samples_per_module=int(spec.raw_max_samples_per_module),
            max_test_samples_per_module=int(spec.raw_max_test_samples_per_module),
            max_eval_samples_per_module=int(spec.raw_max_test_samples_per_module),
            extra_eval_splits=extra_eval_splits,
        )
        test_dataset = eval_datasets["test"]
        extra_eval_datasets = {
            split_name: dataset
            for split_name, dataset in eval_datasets.items()
            if split_name != "test"
        }
    else:
        train_dataset, test_dataset = build_math_datasets(
            **build_kwargs,
            max_samples_per_module=int(spec.raw_max_samples_per_module),
            max_test_samples_per_module=int(spec.raw_max_test_samples_per_module),
        )
        extra_eval_datasets = {}

    postprocess_stats: dict[str, object] = {}
    if dataset_postprocess_fn is not None:
        result = dataset_postprocess_fn(train_dataset, test_dataset, extra_eval_datasets)
        if isinstance(result, dict):
            postprocess_stats = dict(result)

    train_level_pools = _normalize_pools(build_train_index_pools(train_dataset, level_specs))
    eval_level_pools = _normalize_pools(build_eval_index_pools(test_dataset, level_specs))
    train_module_pools = _merge_module_pools_from_specs(train_dataset, level_specs, eval_pools=False)
    eval_module_pools = _merge_module_pools_from_specs(test_dataset, level_specs, eval_pools=True)
    extra_eval_level_pools = {
        split_name: _normalize_pools(build_eval_index_pools(dataset, level_specs))
        for split_name, dataset in extra_eval_datasets.items()
    }
    extra_eval_module_pools = {
        split_name: _merge_module_pools_from_specs(dataset, level_specs, eval_pools=True)
        for split_name, dataset in extra_eval_datasets.items()
    }

    bundle = DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        extra_eval_datasets=extra_eval_datasets,
        train_level_pools=train_level_pools,
        eval_level_pools=eval_level_pools,
        extra_eval_level_pools=extra_eval_level_pools,
        train_module_pools=train_module_pools,
        eval_module_pools=eval_module_pools,
        extra_eval_module_pools=extra_eval_module_pools,
        train_tier_pools=train_tier_pools or {},
        eval_tier_pools=eval_tier_pools or {},
        extra_eval_tier_pools={},
        metadata={"postprocess": postprocess_stats},
    )
    bundle = _apply_runtime_caps(bundle, spec)
    return _save_bundle_to_cache(spec, bundle)
