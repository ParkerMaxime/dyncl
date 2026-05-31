#!/usr/bin/env python3
"""Profile registry for dataset/backend preparation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, Sequence

from .hub import DatasetHubAdapters, prepare_dataset_bundle
from .spec import DatasetSpec


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    expected_levels: tuple[str, ...]
    expected_start_level: str
    adapters: DatasetHubAdapters
    level_specs_fn: Callable[[], Sequence[object]]
    evaluate_extra_splits_fn: Callable[..., object]
    hard_tiered_modules: tuple[str, ...] = ()
    default_scratchpad_version: str = ""
    preflight_fn: Callable[[], None] | None = None


_PHASE3C_CLEAN_ROOT = Path(__file__).resolve().parents[1]


def _load_family_module(*, family_dir: str, module_name: str) -> ModuleType:
    module_path = _PHASE3C_CLEAN_ROOT / str(family_dir) / f"{module_name}.py"
    cache_key = f"gestion_dataset_family::{family_dir}::{module_name}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]
    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load family module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


def _build_3c5dynl_profile() -> DatasetProfile:
    dataset_pipeline_mod = _load_family_module(family_dir="3c5dync1_dynamic", module_name="dataset_pipeline")
    eval_pipeline_mod = _load_family_module(family_dir="3c5dync1_dynamic", module_name="eval_pipeline")
    scratchpad_pipeline_mod = _load_family_module(family_dir="3c5dync1_dynamic", module_name="scratchpad_pipeline")

    return DatasetProfile(
        name="3c5dynl",
        expected_levels=("primaire", "college", "lycee"),
        expected_start_level="lycee",
        adapters=DatasetHubAdapters(
            build_bundle=lambda cfg: dataset_pipeline_mod.build_bundle(
                cfg,
                modules_by_level_override=dataset_pipeline_mod.MODULES_BY_LEVEL,
                selected_levels_override=("primaire", "college", "lycee"),
                current_level_override="lycee",
            ),
            finalize_bundle=None,
            build_pool_registry=dataset_pipeline_mod.build_pool_registry,
            build_flat_train_pools=dataset_pipeline_mod.build_flat_train_pools,
            build_level_eval_pools=dataset_pipeline_mod.build_level_eval_pools,
            build_extra_eval_loaders=eval_pipeline_mod.build_extra_eval_loaders,
            modules_by_level=dataset_pipeline_mod.MODULES_BY_LEVEL,
        ),
        level_specs_fn=dataset_pipeline_mod.level_specs,
        evaluate_extra_splits_fn=eval_pipeline_mod.evaluate_extra_splits,
        hard_tiered_modules=tuple(str(module_name) for module_name in dataset_pipeline_mod.HARD_TIERED_MODULES),
        default_scratchpad_version="3c5_scratchpad_v3",
        preflight_fn=scratchpad_pipeline_mod.run_answer_audit,
    )


def _wrap_level_specs(level_specs_fn: Callable[..., Sequence[object]]) -> Callable[..., Sequence[object]]:
    def _wrapped(*args, **kwargs) -> Sequence[object]:
        return level_specs_fn(*args, **kwargs)

    return _wrapped


def _build_3c5dync_profile() -> DatasetProfile:
    dataset_pipeline_mod = _load_family_module(family_dir="3c5dync1_dynamic", module_name="dataset_pipeline")
    eval_pipeline_mod = _load_family_module(family_dir="3c5dync1_dynamic", module_name="eval_pipeline")
    scratchpad_pipeline_mod = _load_family_module(family_dir="3c5dync1_dynamic", module_name="scratchpad_pipeline")

    dync_modules_by_level = {
        "primaire": tuple(dataset_pipeline_mod.MODULES_BY_LEVEL["primaire"]),
        "college": tuple(dataset_pipeline_mod.MODULES_BY_LEVEL["college"]),
        "lycee": (),
        "superieur": (),
    }
    return DatasetProfile(
        name="3c5dync",
        expected_levels=("primaire", "college"),
        expected_start_level="college",
        adapters=DatasetHubAdapters(
            build_bundle=lambda cfg: dataset_pipeline_mod.build_bundle(
                cfg,
                modules_by_level_override=dync_modules_by_level,
                selected_levels_override=("primaire", "college"),
                current_level_override="college",
            ),
            finalize_bundle=None,
            build_pool_registry=dataset_pipeline_mod.build_pool_registry,
            build_flat_train_pools=dataset_pipeline_mod.build_flat_train_pools,
            build_level_eval_pools=dataset_pipeline_mod.build_level_eval_pools,
            build_extra_eval_loaders=eval_pipeline_mod.build_extra_eval_loaders,
            modules_by_level=dync_modules_by_level,
        ),
        level_specs_fn=lambda *args, **kwargs: dataset_pipeline_mod._specs_for_modules(dync_modules_by_level),
        evaluate_extra_splits_fn=eval_pipeline_mod.evaluate_extra_splits,
        hard_tiered_modules=tuple(str(module_name) for module_name in dataset_pipeline_mod.HARD_TIERED_MODULES),
        default_scratchpad_version="3c5_scratchpad_v3",
        preflight_fn=scratchpad_pipeline_mod.run_answer_audit,
    )


def _build_3c5dynp_profile() -> DatasetProfile:
    dataset_pipeline_mod = _load_family_module(family_dir="3c5p21_clean_dense", module_name="dataset_pipeline")
    eval_pipeline_mod = _load_family_module(family_dir="3c5p21_clean_dense", module_name="eval_pipeline")
    scratchpad_pipeline_mod = _load_family_module(family_dir="3c5p21_clean_dense", module_name="scratchpad_pipeline")

    dynp_modules_by_level = {
        "primaire": (
            "arithmetic__add_or_sub",
            "arithmetic__add_sub_multiple",
            "arithmetic__mul",
            "arithmetic__div",
        ),
        "college": (),
        "lycee": (),
        "superieur": (),
    }
    return DatasetProfile(
        name="3c5dynp",
        expected_levels=("primaire",),
        expected_start_level="primaire",
        adapters=DatasetHubAdapters(
            build_bundle=lambda cfg: dataset_pipeline_mod.build_bundle(cfg),
            finalize_bundle=None,
            build_pool_registry=dataset_pipeline_mod.build_pool_registry,
            build_flat_train_pools=dataset_pipeline_mod.build_flat_train_pools,
            build_level_eval_pools=dataset_pipeline_mod.build_level_eval_pools,
            build_extra_eval_loaders=eval_pipeline_mod.build_extra_eval_loaders,
            modules_by_level=dynp_modules_by_level,
        ),
        level_specs_fn=dataset_pipeline_mod.level_specs,
        evaluate_extra_splits_fn=eval_pipeline_mod.evaluate_extra_splits,
        hard_tiered_modules=(),
        default_scratchpad_version="all4_primary_plus_explicit_answer_audit_v2",
        preflight_fn=scratchpad_pipeline_mod.run_answer_audit,
    )


def get_dataset_profile(profile_name: str) -> DatasetProfile:
    normalized = str(profile_name).strip()
    if normalized in {"3c5dynl", "3c5dynl1"}:
        return _build_3c5dynl_profile()
    if normalized in {"3c5dync", "3c5dync1"}:
        return _build_3c5dync_profile()
    if normalized in {"3c5dynp", "3c5dynp1"}:
        return _build_3c5dynp_profile()
    raise KeyError(f"Unknown dataset profile: {normalized}")


def validate_profile_config(*, profile: DatasetProfile, spec: DatasetSpec) -> None:
    actual_levels = tuple(str(level_name) for level_name in spec.levels)
    expected_levels = tuple(str(level_name) for level_name in profile.expected_levels)
    if actual_levels != expected_levels:
        raise RuntimeError(
            f"Dataset profile {profile.name} expects levels={list(expected_levels)} "
            f"but config provided levels={list(actual_levels)}"
        )
    actual_start_level = str(spec.start_level)
    expected_start_level = str(profile.expected_start_level)
    if actual_start_level != expected_start_level:
        raise RuntimeError(
            f"Dataset profile {profile.name} expects start_level={expected_start_level!r} "
            f"but config provided start_level={actual_start_level!r}"
        )


def prepare_dataset_bundle_for_profile(
    profile_name: str,
    *,
    cfg,
    spec: DatasetSpec | None = None,
    scratchpad_version: str = "",
):
    profile = get_dataset_profile(profile_name)
    dataset_spec = spec or DatasetSpec.from_run_config(
        cfg,
        scratchpad_version=str(scratchpad_version or profile.default_scratchpad_version),
    )
    validate_profile_config(profile=profile, spec=dataset_spec)
    return prepare_dataset_bundle(
        cfg=cfg,
        spec=dataset_spec,
        adapters=profile.adapters,
    )
