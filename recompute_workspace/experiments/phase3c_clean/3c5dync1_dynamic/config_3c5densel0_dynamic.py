#!/usr/bin/env python3
"""Dense lycee control baseline using the calibrated 3C5dync1 CL plumbing."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config_3c5dync1_dynamic import RunConfig as BaseRunConfig
    from .config_3c5dync1_dynamic import apply_experiment_overrides as _base_apply
    from .config_3c5dync1_dynamic import to_engine_argv
except ImportError:  # pragma: no cover
    from config_3c5dync1_dynamic import RunConfig as BaseRunConfig
    from config_3c5dync1_dynamic import apply_experiment_overrides as _base_apply
    from config_3c5dync1_dynamic import to_engine_argv


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5densel0_lycee200_from_densec1best_softtransfer"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5densel0_lycee200_from_densec1best_softtransfer_noamp_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire", "college", "lycee")
    mix_variant: str = "fixed_dense_ewc"
    epochs_per_level: int = 200
    last_level_epochs: int = 200
    lr: float = 1e-3
    optimizer_name: str = "adamw"
    adam_weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    lr_warmup_epochs: int = 10
    lr_schedule_min_ratio: float = 0.30
    lr_sgdr_prev_floor: float = 0.0
    lr_sgdr_cycles: tuple[tuple[int, int, float, float, int], ...] = ()
    phase0_fixed_epochs: int = 0
    phase0_unfreeze_epochs: int = 0
    phase0_lr_start: float = 1e-3
    phase0_lr_end: float = 3e-4
    phase12_old_lr_target: float = 1e-3
    phase12_new_lr_target: float = 1e-3
    phase12_warm_restart_epochs: int = 0
    phase12_dual_lr_enabled: bool = False
    phase3_transfer_cosine_enabled: bool = False
    phase3_transfer_lr_min: float = 3e-4
    adaptive_module_mix_enabled: bool = False
    retention_train_cap_per_module: int = 200000
    raw_max_test_samples_per_module: int = 1000
    num_workers: int = 2
    derived_cache_name: str = "3c5densel0_lycee_scratchpad_v3"
    init_checkpoint: str = (
        "results/phase3/3c/3c5/"
        "3c5densec1_college200_from_denseconv0best_fixedmix_adamw_cosine_noamp_dm256_l2_ff1024_spe16k"
        "/dynamic/checkpoints/seed_42_college_best.pt"
    )
    resume_optimizer: bool = True
    bootstrap_completed_levels: tuple[str, ...] = ("primaire", "college")
    start_level: str = "lycee"

    use_ewc: bool = False
    ewc_lambda: float = 0.0
    ewc_fisher_samples: int = 8192
    ewc_update_frequency: int = 0
    amp_dense_enabled: bool = False

    initial_connectivity_ffn: float = 1.0
    enable_newsyn: bool = False
    triphasic_controller_enabled: bool = False
    pruning_frequency: int = 100000
    max_new_synapses: int = 0
    max_new_synapses_cap: int = 0
    channel_creation_enabled: bool = False
    phase1_registry_enabled: bool = False
    phase1_channel_creation_enabled: bool = False
    phase1_pruning_enabled: bool = False
    handoff_channel_creation: bool = False
    handoff_pruning: bool = False
    phase2_enabled: bool = False
    phase2_micro_creation_enabled: bool = False
    phase2_pruning_enabled: bool = False
    phase3_micro_creation_enabled: bool = False
    phase3_pruning_enabled: bool = False
    local_reinforce_enabled: bool = False


def default_config() -> RunConfig:
    return RunConfig()


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dense_lycee_control_from_densec1best | "
        "runtime_engine=clean_shared_dense_control | "
        "dataset=3c5_scratchpad_v3_primary_college_lycee_core4_module_bundles"
    )


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg = _base_apply(cfg, defaults)
    cfg.optimizer_name = str(defaults.optimizer_name)
    cfg.adam_weight_decay = float(defaults.adam_weight_decay)
    cfg.lr_schedule = str(defaults.lr_schedule)
    cfg.lr_warmup_epochs = int(defaults.lr_warmup_epochs)
    cfg.lr_schedule_min_ratio = float(defaults.lr_schedule_min_ratio)
    cfg.lr_sgdr_prev_floor = float(defaults.lr_sgdr_prev_floor)
    cfg.lr_sgdr_cycles = [tuple(cycle) for cycle in defaults.lr_sgdr_cycles]
    cfg.phase0_fixed_epochs = int(defaults.phase0_fixed_epochs)
    cfg.phase0_unfreeze_epochs = int(defaults.phase0_unfreeze_epochs)
    cfg.phase0_lr_start = float(defaults.phase0_lr_start)
    cfg.phase0_lr_end = float(defaults.phase0_lr_end)
    cfg.phase12_old_lr_target = float(defaults.phase12_old_lr_target)
    cfg.phase12_new_lr_target = float(defaults.phase12_new_lr_target)
    cfg.phase12_warm_restart_epochs = int(defaults.phase12_warm_restart_epochs)
    cfg.phase12_dual_lr_enabled = bool(defaults.phase12_dual_lr_enabled)
    cfg.phase3_transfer_cosine_enabled = bool(defaults.phase3_transfer_cosine_enabled)
    cfg.phase3_transfer_lr_min = float(defaults.phase3_transfer_lr_min)
    cfg.adaptive_module_mix_enabled = bool(defaults.adaptive_module_mix_enabled)
    cfg.retention_train_cap_per_module = int(defaults.retention_train_cap_per_module)
    cfg.num_workers = int(defaults.num_workers)
    cfg.use_ewc = bool(defaults.use_ewc)
    cfg.ewc_lambda = float(defaults.ewc_lambda)
    cfg.ewc_fisher_samples = int(defaults.ewc_fisher_samples)
    cfg.ewc_update_frequency = int(defaults.ewc_update_frequency)
    cfg.triphasic_controller_enabled = bool(defaults.triphasic_controller_enabled)
    cfg.amp_dense_enabled = bool(defaults.amp_dense_enabled)
    return cfg
