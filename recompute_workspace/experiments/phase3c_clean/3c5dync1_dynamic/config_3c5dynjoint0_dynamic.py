#!/usr/bin/env python3
"""Dynamic joint-training control over primaire+college+lycee, based on dynp1 infra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
_PHASE3C_CLEAN_ROOT = _SCRIPT_DIR.parent
_DYNP1_ROOT = _PHASE3C_CLEAN_ROOT / "3c5dynp1_dynamic"


def _prepend_sys_path(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


for _path in (_SCRIPT_DIR, _PHASE3C_CLEAN_ROOT, _DYNP1_ROOT):
    if _path.exists():
        _prepend_sys_path(_path)

try:
    from config_3c5dynp1_dynamic import RunConfig as BaseRunConfig
    from config_3c5dynp1_dynamic import apply_experiment_overrides as _base_apply
    from config_3c5dynp1_dynamic import to_engine_argv
except ImportError:  # pragma: no cover
    from config_3c5dynp1_dynamic import RunConfig as BaseRunConfig
    from config_3c5dynp1_dynamic import apply_experiment_overrides as _base_apply
    from config_3c5dynp1_dynamic import to_engine_argv


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5dynjoint0_full33_900ep"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5dynjoint0_full33_900ep_cosinefloor03_noamp_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire", "college", "lycee")
    mix_variant: str = "fixed_dyn_joint33"
    epochs_per_level: int = 900
    last_level_epochs: int = 900
    eval_per_module_interval: int = 100
    module_eval_frequency: int = 100
    samples_per_epoch: int = 16384
    lr_schedule_min_ratio: float = 0.30
    raw_max_samples_per_module: int = 500000
    raw_max_test_samples_per_module: int = 1000
    retention_train_cap_per_module: int = 500000
    num_workers: int = 2
    derived_cache_name: str = "3c5dynjoint0_full33_scratchpad_v3"
    init_checkpoint: str = ""
    resume_optimizer: bool = False
    bootstrap_completed_levels: tuple[str, ...] = ()
    start_level: str = "joint"
    amp_dense_enabled: bool = False

    phase1_to_phase2_min_epoch: int = 150
    phase1_to_phase2_force_epoch: int = 300
    handoff_duration: int = 15
    phase2_micro_budget_target: int = 96
    phase2_micro_budget_scale_by_synapses: bool = False
    phase2_pruning_start_delay: int = 10
    phase2_to_phase3_min_phase2_epochs: int = 150
    phase2_to_phase3_force_epoch: int = 450

    enable_replay_tag_protection: bool = False
    replay_tag_quantile: float = 0.70
    replay_tag_min: float = 0.0


def default_config() -> RunConfig:
    return RunConfig()


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dynamic_joint_full33_900ep_from_dynp1 | "
        "runtime_engine=shared_epoch_loop_joint_triphasic | "
        "dataset=3c5_scratchpad_v3_primary_college_lycee_full15_module_bundles"
    )


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg = _base_apply(cfg, defaults)
    cfg.exp_id = str(defaults.exp_id)
    cfg.results_dir = str(defaults.results_dir)
    cfg.levels = tuple(defaults.levels)
    cfg.mix_variant = str(defaults.mix_variant)
    cfg.epochs_per_level = int(defaults.epochs_per_level)
    cfg.last_level_epochs = int(defaults.last_level_epochs)
    cfg.eval_per_module_interval = int(defaults.eval_per_module_interval)
    cfg.module_eval_frequency = int(defaults.module_eval_frequency)
    cfg.samples_per_epoch = int(defaults.samples_per_epoch)
    cfg.lr_schedule_min_ratio = float(defaults.lr_schedule_min_ratio)
    cfg.raw_max_samples_per_module = int(defaults.raw_max_samples_per_module)
    cfg.raw_max_test_samples_per_module = int(defaults.raw_max_test_samples_per_module)
    cfg.retention_train_cap_per_module = int(defaults.retention_train_cap_per_module)
    cfg.num_workers = int(defaults.num_workers)
    cfg.derived_cache_name = str(defaults.derived_cache_name)
    cfg.init_checkpoint = str(defaults.init_checkpoint)
    cfg.resume_optimizer = bool(defaults.resume_optimizer)
    cfg.bootstrap_completed_levels = tuple(defaults.bootstrap_completed_levels)
    cfg.start_level = str(defaults.start_level)
    cfg.amp_dense_enabled = bool(defaults.amp_dense_enabled)
    cfg.phase1_to_phase2_min_epoch = int(defaults.phase1_to_phase2_min_epoch)
    cfg.phase1_to_phase2_force_epoch = int(defaults.phase1_to_phase2_force_epoch)
    cfg.handoff_duration = int(defaults.handoff_duration)
    cfg.phase2_micro_budget_target = int(defaults.phase2_micro_budget_target)
    cfg.phase2_micro_budget_scale_by_synapses = bool(defaults.phase2_micro_budget_scale_by_synapses)
    cfg.phase2_pruning_start_delay = int(defaults.phase2_pruning_start_delay)
    cfg.phase2_to_phase3_min_phase2_epochs = int(defaults.phase2_to_phase3_min_phase2_epochs)
    cfg.phase2_to_phase3_force_epoch = int(defaults.phase2_to_phase3_force_epoch)
    cfg.enable_replay_tag_protection = bool(defaults.enable_replay_tag_protection)
    cfg.replay_tag_quantile = float(defaults.replay_tag_quantile)
    cfg.replay_tag_min = float(defaults.replay_tag_min)
    return cfg
