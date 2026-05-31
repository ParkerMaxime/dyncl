#!/usr/bin/env python3
"""Configuration centralisee du runner dynamique baseline 3C5dync college."""

from __future__ import annotations

from dataclasses import dataclass

from config_3c5dync1_dynamic import (
    RunConfig as BaseRunConfig,
    apply_experiment_overrides as _base_apply,
    default_config as _base_default_config,
    to_engine_argv as _base_to_engine_argv,
)


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5dync_baseline_college750_from_dynp1b_tiers_gcd_lcm_divr_floor_mixed_flat"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5dync_baseline_college750_from_dynp1b_tiers_gcd_lcm_divr_floor_mixed_flat_scratchpadv3_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire", "college")
    mix_variant: str = "college_soft_transfer"
    extra_eval_splits: tuple[str, ...] = ()
    derived_cache_name: str = "3c5dync_baseline_college_scratchpad_v3"
    retention_train_cap_per_module: int = 0
    college_tier_curriculum_enabled: bool = True
    init_checkpoint: str = (
        "results/phase3/_unsorted/runs/cloud_pulls/"
        "3c5dynp1b_primaire750_scratchpad_v2_triphasic_phase2fix_ffn30_budget400_dm256_l2_ff1024_spe16k"
        "/dynamic/checkpoints/seed_42_primaire_last.pt"
    )
    resume_optimizer: bool = True
    bootstrap_completed_levels: tuple[str, ...] = ("primaire",)
    start_level: str = "college"
    # Primary-current dynamic creation/pruning baseline.
    phase2_pruning_enabled: bool = True
    max_new_synapses: int = 400
    max_new_synapses_cap: int = 4000
    pruning_frequency: int = 5
    pruning_keep_ratio: float = 0.98
    creation_floor_per_connection: int = 16
    creation_group_min_share: float = 0.40
    creation_group_mix: float = 0.50
    creation_group_prior_enc_share: float = 0.45
    creation_group_share_ema_alpha: float = 0.20
    creation_selection_weight: str = "sqrt_score_gap"
    creation_auto_budget_from_train_seq: bool = False
    creation_train_seq_ema_alpha: float = 0.20
    creation_train_seq_budget_gamma: float = 0.50
    creation_train_seq_budget_min_ratio: float = 0.25
    creation_train_seq_budget_trigger: int = 0
    creation_train_seq_fill_ratio_threshold: float = 0.90
    creation_train_seq_fill_ratio_streak: int = 5
    creation_train_seq_budget_min_epoch: int = 120
    pruning_couple_creation_need: bool = True
    pruning_creation_need_ema_alpha: float = 0.20
    pruning_creation_coupling_alpha: float = 1.0
    enable_late_pruning_relaxation: bool = True
    late_pruning_min_epoch: int = 500
    late_pruning_fill_ratio_threshold: float = 0.50
    late_pruning_growth_need_threshold: float = 0.35
    late_pruning_keep_ratio: float = 0.999
    late_pruning_freeze_on_zero_growth: bool = False
    collect_ffn_hidden_stats_each_epoch: bool = False
    channel_fan_in: int = 3
    channel_fan_out: int = 3
    channel_bundle_size: int = 6
    channel_candidate_pool: int = 256
    channel_bundle_init_scale: float = 0.01
    channel_underused_max_quantile: float = 0.70
    channel_require_full_bundle: bool = False
    block_density_cap: float = 0.60
    channel_budget_fraction: float = 1.0
    channel_probe_threshold_ema_alpha: float = 0.20
    max_probationary_channels_per_prefix: int = 144
    max_probationary_channels_global: int = 448
    max_new_probationary_channels_per_epoch: int = 32
    phase1_registry_enabled: bool = True
    phase1_reselection_decay: float = 0.98
    phase1_unexplored_bonus: float = 0.20
    phase1_reexplored_bonus: float = 0.05
    phase1_channel_creation_enabled: bool = True
    phase1_pruning_enabled: bool = False
    phase1_base_channel_budget: int = 32
    phase1_min_channel_budget: int = 4
    phase1_budget_coverage_onset: float = 0.55
    phase1_budget_min_factor: float = 0.15
    phase1_tenure_rate_ema_alpha: float = 0.10
    phase1_to_phase2_trigger_enabled: bool = True
    phase1_to_phase2_min_epoch: int = 120
    phase1_to_phase2_coverage_global: float = 0.70
    phase1_to_phase2_min_prefix_coverage: float = 0.60
    phase1_to_phase2_force_epoch: int = 240
    phase1_min_evaluated_channels: int = 256
    handoff_duration: int = 15
    handoff_channel_creation: bool = False
    handoff_pruning: bool = False
    phase2_enabled: bool = True
    phase2_micro_creation_enabled: bool = True
    phase2_micro_creation_policy: str = "need_quality_inverse_coverage"
    phase2_micro_budget_target: int = 192
    phase2_micro_budget_scale_by_synapses: bool = True
    phase2_micro_budget_reference_active_synapses: int = 0
    phase2_micro_budget_max: int = 0
    phase2_pruning_start_delay: int = 10
    prune_cap_from_created_window_enabled: bool = True
    prune_cap_from_created_window_ratio: float = 1.20
    prune_cap_from_created_window_floor: int = 200
    prune_cap_from_created_window_ceiling: int = 1200
    phase2_to_phase3_min_phase2_epochs: int = 120
    phase2_to_phase3_force_epoch: int = 350
    phase3_micro_creation_enabled: bool = False
    phase3_pruning_enabled: bool = False
    tenured_channel_grace_period: int = 30
    local_reinforce_enabled: bool = True
    local_reinforce_epochs: int = 3
    local_reinforce_max_edges_per_epoch: int = 192
    local_reinforce_max_edges_per_channel_total: int = 6
    local_reinforce_edges_per_channel_per_epoch: int = 2
    local_reinforce_top_pool: int = 32
    creation_quality_mad_factor: float = 0.50
    creation_coherence_threshold: float = 0.85
    creation_need_ema_alpha: float = 0.20


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
        "run_config=3c5_dynamic_college_from_dynp1b_tiers_run6_floor_mixed_flat_baseline_primary_creation_pruning | "
        "runtime_engine=clean_shared | "
        "dataset=3c5c1a_scratchpad_v3_gcd_lcm_divr_tiers_mixed_flat_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return _base_to_engine_argv(cfg)


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cli_results_dir = str(getattr(cfg, "results_dir", "") or "")
    cli_init_checkpoint = str(getattr(cfg, "init_checkpoint", "") or "")
    patched = _base_apply(cfg, defaults)
    if cli_results_dir:
        patched.results_dir = cli_results_dir
    if cli_init_checkpoint:
        patched.init_checkpoint = cli_init_checkpoint
    return patched
