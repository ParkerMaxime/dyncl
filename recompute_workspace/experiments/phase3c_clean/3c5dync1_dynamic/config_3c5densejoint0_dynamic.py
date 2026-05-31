#!/usr/bin/env python3
"""Dense joint-training baseline over primaire+college+lycee."""

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
    exp_id: str = "3C5densejoint0_full33_fromscratch400"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5densejoint0_full33_fromscratch400_cosinefloor03_noamp_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire", "college", "lycee")
    mix_variant: str = "fixed_dense_joint33"
    epochs_per_level: int = 400
    last_level_epochs: int = 400
    lr: float = 1e-3
    optimizer_name: str = "adamw"
    adam_weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    lr_warmup_epochs: int = 10
    lr_schedule_min_ratio: float = 0.30
    lr_sgdr_prev_floor: float = 0.0
    lr_sgdr_cycles: tuple[tuple[int, int, float, float, int], ...] = ()
    adaptive_module_mix_enabled: bool = False
    raw_max_samples_per_module: int = 500000
    retention_train_cap_per_module: int = 500000
    raw_max_test_samples_per_module: int = 1000
    num_workers: int = 2
    derived_cache_name: str = "3c5densejoint0_full33_scratchpad_v3"
    init_checkpoint: str = ""
    resume_optimizer: bool = False
    bootstrap_completed_levels: tuple[str, ...] = ()
    start_level: str = "joint"
    enable_transition_buffer: bool = False
    transition_epochs_default: int = 0
    transition_epochs_lycee: int = 0
    transition_epochs_last: int = 0

    use_ewc: bool = False
    ewc_lambda: float = 0.0
    ewc_fisher_samples: int = 0
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
        "run_config=3c5_dense_joint_full33_fromscratch400 | "
        "runtime_engine=shared_epoch_loop_no_cl | "
        "dataset=3c5_scratchpad_v3_primary_college_lycee_full15_module_bundles"
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
    cfg.adaptive_module_mix_enabled = bool(defaults.adaptive_module_mix_enabled)
    cfg.retention_train_cap_per_module = int(defaults.retention_train_cap_per_module)
    cfg.num_workers = int(defaults.num_workers)
    cfg.use_ewc = bool(defaults.use_ewc)
    cfg.ewc_lambda = float(defaults.ewc_lambda)
    cfg.ewc_fisher_samples = int(defaults.ewc_fisher_samples)
    cfg.ewc_update_frequency = int(defaults.ewc_update_frequency)
    cfg.triphasic_controller_enabled = bool(defaults.triphasic_controller_enabled)
    cfg.amp_dense_enabled = bool(defaults.amp_dense_enabled)
    cfg.enable_transition_buffer = bool(defaults.enable_transition_buffer)
    cfg.transition_epochs_default = int(defaults.transition_epochs_default)
    cfg.transition_epochs_lycee = int(defaults.transition_epochs_lycee)
    cfg.transition_epochs_last = int(defaults.transition_epochs_last)
    return cfg
