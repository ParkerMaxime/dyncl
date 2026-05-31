#!/usr/bin/env python3
"""Minimal hub that centralizes dataset/backend preparation for runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .prepared import PreparedDatasetBundle
from .spec import DatasetSpec, normalize_level_modules


@dataclass(frozen=True)
class DatasetHubAdapters:
    build_bundle: Callable[..., object]
    finalize_bundle: Callable[[object], object] | None
    build_pool_registry: Callable[[object], object]
    build_flat_train_pools: Callable[[object, object, object], Mapping[str, object]]
    build_level_eval_pools: Callable[[object], Mapping[str, object]]
    build_extra_eval_loaders: Callable[..., Mapping[str, object]]
    modules_by_level: Mapping[str, Sequence[str]]


def _filter_bundle_to_spec_levels(bundle, *, spec: DatasetSpec, modules_by_level: Mapping[str, Sequence[str]]):
    allowed_levels = {str(level_name) for level_name in spec.levels}
    allowed_modules = {
        str(module_name)
        for level_name, modules in dict(modules_by_level).items()
        if str(level_name) in allowed_levels
        for module_name in modules
    }
    bundle.train_level_pools = {
        str(level_name): pool
        for level_name, pool in dict(getattr(bundle, "train_level_pools", {})).items()
        if str(level_name) in allowed_levels
    }
    bundle.eval_level_pools = {
        str(level_name): pool
        for level_name, pool in dict(getattr(bundle, "eval_level_pools", {})).items()
        if str(level_name) in allowed_levels
    }
    bundle.extra_eval_level_pools = {
        str(split_name): {
            str(level_name): pool
            for level_name, pool in dict(level_pools).items()
            if str(level_name) in allowed_levels
        }
        for split_name, level_pools in dict(getattr(bundle, "extra_eval_level_pools", {})).items()
    }
    bundle.train_module_pools = {
        str(module_name): pool
        for module_name, pool in dict(getattr(bundle, "train_module_pools", {})).items()
        if str(module_name) in allowed_modules
    }
    bundle.eval_module_pools = {
        str(module_name): pool
        for module_name, pool in dict(getattr(bundle, "eval_module_pools", {})).items()
        if str(module_name) in allowed_modules
    }
    bundle.extra_eval_module_pools = {
        str(split_name): {
            str(module_name): pool
            for module_name, pool in dict(module_pools).items()
            if str(module_name) in allowed_modules
        }
        for split_name, module_pools in dict(getattr(bundle, "extra_eval_module_pools", {})).items()
    }
    bundle.train_tier_pools = {
        str(module_name): dict(module_tiers)
        for module_name, module_tiers in dict(getattr(bundle, "train_tier_pools", {})).items()
        if str(module_name) in allowed_modules
    }
    bundle.eval_tier_pools = {
        str(module_name): dict(module_tiers)
        for module_name, module_tiers in dict(getattr(bundle, "eval_tier_pools", {})).items()
        if str(module_name) in allowed_modules
    }
    bundle.extra_eval_tier_pools = {
        str(split_name): {
            str(module_name): dict(module_tiers)
            for module_name, module_tiers in dict(split_tiers).items()
            if str(module_name) in allowed_modules
        }
        for split_name, split_tiers in dict(getattr(bundle, "extra_eval_tier_pools", {})).items()
    }
    metadata = dict(getattr(bundle, "metadata", {}))
    metadata["hub_filtered_levels"] = sorted(allowed_levels)
    metadata["hub_filtered_modules"] = sorted(allowed_modules)
    bundle.metadata = metadata
    return bundle


def prepare_dataset_bundle(
    *,
    cfg,
    spec: DatasetSpec,
    adapters: DatasetHubAdapters,
) -> PreparedDatasetBundle:
    bundle = adapters.build_bundle(cfg)
    if adapters.finalize_bundle is not None:
        bundle = adapters.finalize_bundle(bundle)
    bundle = _filter_bundle_to_spec_levels(bundle, spec=spec, modules_by_level=adapters.modules_by_level)
    pool_registry = adapters.build_pool_registry(bundle)
    train_pools = adapters.build_flat_train_pools(bundle, pool_registry, None)
    eval_pools = adapters.build_level_eval_pools(bundle)
    extra_eval_loaders = adapters.build_extra_eval_loaders(bundle=bundle, cfg=cfg)
    prepared = PreparedDatasetBundle(
        spec=spec,
        bundle=bundle,
        pool_registry=pool_registry,
        train_pools=train_pools,
        eval_pools=eval_pools,
        extra_eval_loaders=extra_eval_loaders,
        metadata={
            "spec_summary": dict(spec.summary()),
            "modules_by_level": normalize_level_modules(adapters.modules_by_level),
        },
    )
    prepared.validate_levels(modules_by_level=adapters.modules_by_level)
    return prepared
