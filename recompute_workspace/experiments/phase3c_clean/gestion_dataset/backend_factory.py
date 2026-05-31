#!/usr/bin/env python3
"""Backend factory registry for clean runners."""

from __future__ import annotations

from dataclasses import dataclass

from .prepared import PreparedDatasetBundle
from .profiles import _load_family_module, get_dataset_profile
from .runtime_backend import PreindexedRuntimeDatasetBackend


@dataclass(frozen=True)
class PreparedBackends:
    runtime_backend: PreindexedRuntimeDatasetBackend
    experiment_backend: object


def _build_3c5_family_backends(
    profile_name: str,
    *,
    cfg,
    prepared: PreparedDatasetBundle,
    logger,
    results_dir: str,
) -> PreparedBackends:
    experiment_backend_mod = _load_family_module(
        family_dir="3c5dync1_dynamic",
        module_name="experiment_backend",
    )
    profile = get_dataset_profile(profile_name)
    runtime_backend = PreindexedRuntimeDatasetBackend(
        tiered_modules=tuple(profile.hard_tiered_modules) if bool(getattr(cfg, "college_tier_curriculum_enabled", False)) else (),
        tier_key_fn=lambda module_name, tier_name: f"{module_name}__{tier_name}",
        logger=logger,
    )
    runtime_backend.set_pool_registry(prepared.pool_registry)
    experiment_backend = experiment_backend_mod.Clean3C5c1aExperimentBackend(
        logger=logger,
        cfg=cfg,
        bundle=prepared.bundle,
        pool_registry=prepared.pool_registry,
        runtime_backend=runtime_backend,
        results_dir=str(results_dir),
    )
    return PreparedBackends(
        runtime_backend=runtime_backend,
        experiment_backend=experiment_backend,
    )


def _build_3c5dynp_backends(
    *,
    cfg,
    prepared: PreparedDatasetBundle,
    logger,
    results_dir: str,
) -> PreparedBackends:
    experiment_backend_mod = _load_family_module(
        family_dir="3c5p21_clean_dense",
        module_name="experiment_backend",
    )
    runtime_backend = PreindexedRuntimeDatasetBackend(
        tiered_modules=(),
        tier_key_fn=lambda module_name, tier_name: f"{module_name}__{tier_name}",
        logger=logger,
    )
    runtime_backend.set_pool_registry(prepared.pool_registry)
    experiment_backend = experiment_backend_mod.Clean3C5p21ExperimentBackend(
        logger=logger,
        cfg=cfg,
        bundle=prepared.bundle,
        pool_registry=prepared.pool_registry,
        runtime_backend=runtime_backend,
        results_dir=str(results_dir),
    )
    return PreparedBackends(
        runtime_backend=runtime_backend,
        experiment_backend=experiment_backend,
    )


def build_backends_for_profile(
    profile_name: str,
    *,
    cfg,
    prepared: PreparedDatasetBundle,
    logger,
    results_dir: str,
) -> PreparedBackends:
    normalized = str(profile_name).strip()
    if normalized in {"3c5dynl", "3c5dynl1", "3c5dync", "3c5dync1"}:
        return _build_3c5_family_backends(
            normalized,
            cfg=cfg,
            prepared=prepared,
            logger=logger,
            results_dir=results_dir,
        )
    if normalized in {"3c5dynp", "3c5dynp1"}:
        return _build_3c5dynp_backends(
            cfg=cfg,
            prepared=prepared,
            logger=logger,
            results_dir=results_dir,
        )
    raise KeyError(f"Unknown backend profile: {normalized}")
