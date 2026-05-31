#!/usr/bin/env python3
"""Configuration centralisee du runner dynamique baseline 3C5dynl lycee."""

from __future__ import annotations

from dataclasses import dataclass

from config_3c5dync_baseline_dynamic import (
    RunConfig as BaseRunConfig,
    apply_experiment_overrides as _base_apply,
    default_config as _base_default_config,
    to_engine_argv as _base_to_engine_argv,
)


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5dynl_baseline_lycee750_from_dync_baseline_core4_retention50"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5dynl_baseline_lycee750_from_dync_baseline_core4_retention50_scratchpadv3_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire", "college", "lycee")
    mix_variant: str = "lycee_soft_transfer_retention50"
    derived_cache_name: str = "3c5dynl_baseline_lycee_core4_retention50_scratchpad_v3"
    retention_train_cap_per_module: int = 100000
    college_tier_curriculum_enabled: bool = False
    init_checkpoint: str = (
        "results/phase3/3c/3c5/"
        "3c5dync_baseline_college750_validate_42_43_44_from_dynp1b_tiers_gcd_lcm_divr_floor_mixed_flat_scratchpadv3_dm256_l2_ff1024_spe16k"
        "/seed_42/dynamic/checkpoints/seed_42_college_best.pt"
    )
    bootstrap_completed_levels: tuple[str, ...] = ("primaire", "college")
    start_level: str = "lycee"


def default_config() -> RunConfig:
    base = _base_default_config()
    defaults = RunConfig()
    return RunConfig(
        **{
            **base.__dict__,
            **defaults.__dict__,
        }
    )


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dynamic_lycee_from_dync_baseline_core4_retention50 | "
        "runtime_engine=clean_shared | "
        "dataset=3c5_scratchpad_v3_primary_college_lycee_core4_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return _base_to_engine_argv(cfg)


def apply_experiment_overrides(cfg, defaults: RunConfig):
    return _base_apply(cfg, defaults)
