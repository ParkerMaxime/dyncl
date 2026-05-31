#!/usr/bin/env python3
"""Dense joint-training extension over primaire+college+lycee from densejoint1 last."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config_3c5densejoint0_dynamic import RunConfig as BaseRunConfig
    from .config_3c5densejoint0_dynamic import apply_experiment_overrides as _base_apply
    from .config_3c5densejoint0_dynamic import to_engine_argv
except ImportError:  # pragma: no cover
    from config_3c5densejoint0_dynamic import RunConfig as BaseRunConfig
    from config_3c5densejoint0_dynamic import apply_experiment_overrides as _base_apply
    from config_3c5densejoint0_dynamic import to_engine_argv


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5densejoint2_full33_from_densejoint1last900"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5densejoint2_full33_from_densejoint1last900_cosinefloor03_noamp_dm256_l2_ff1024_spe16k"
    )
    derived_cache_name: str = "3c5densejoint2_full33_scratchpad_v3"
    init_checkpoint: str = (
        "results/phase3/3c/3c5/"
        "3c5densejoint1_full33_from_densejoint0last600_cosinefloor03_noamp_dm256_l2_ff1024_spe16k/"
        "dynamic/checkpoints/seed_42_joint_last.pt"
    )
    resume_optimizer: bool = True
    epochs_per_level: int = 900
    last_level_epochs: int = 900


def default_config() -> RunConfig:
    return RunConfig()


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dense_joint_full33_from_densejoint1last900 | "
        "runtime_engine=shared_epoch_loop_no_cl_resume | "
        "dataset=3c5_scratchpad_v3_primary_college_lycee_full15_module_bundles"
    )


def apply_experiment_overrides(cfg, defaults: RunConfig):
    return _base_apply(cfg, defaults)
