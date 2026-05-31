#!/usr/bin/env python3
"""Configuration du premier primaire dynamique clean 3C5 scratchpad."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple


_SCRIPT_DIR = Path(__file__).resolve().parent
_PHASE3C_CLEAN_ROOT = _SCRIPT_DIR.parent
_REFERENCE_RUNNER_ROOT = _PHASE3C_CLEAN_ROOT / "3c5p21_clean_dense"


def _prepend_sys_path(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


for _path in (_SCRIPT_DIR, _PHASE3C_CLEAN_ROOT, _REFERENCE_RUNNER_ROOT):
    if _path.exists():
        _prepend_sys_path(_path)

from config_3c5p21_clean_dense import (
    RunConfig as BaseRunConfig,
    apply_experiment_overrides as _base_apply_experiment_overrides,
    default_config as _base_default_config,
    to_engine_argv as _base_to_engine_argv,
)


CONFIG_SOURCE_OF_TRUTH = {
    "run_config": "3C5 dynamic primary scratchpad defaults, with 3C4 triphasic controller imported.",
    "runtime_engine": "Generic clean runtime; dataset/model/eval/checkpoint plumbing reused from 3C5p21 clean.",
    "batch_config": "Remote transport, target selection, sync and explicit CLI overrides only.",
}


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5dynp1b_primaire750_scratchpad_v2_triphasic_phase2fix_ffn30_budget400"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5dynp1b_primaire750_scratchpad_v2_triphasic_phase2fix_ffn30_budget400_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire",)
    start_level: str = "primaire"
    epochs_per_level: int = 750
    last_level_epochs: int = 750
    eval_per_module_interval: int = 50
    module_eval_frequency: int = 50
    runtime_eval_cap_per_module: int = 500
    checkpoint_eval_frequency: int = 250
    snapshot_eval_frequency: int = 0
    snapshot_eval_mode: str = "ar"
    checkpoint_eval_mode: str = "ar_tf"
    final_eval_mode: str = "ar_tf"
    lr_schedule_min_ratio: float = 0.05
    lr_schedule_maturity_epoch_override: int = 200
    initial_connectivity_ffn: float = 0.30
    max_new_synapses: int = 400
    max_new_synapses_cap: int = 4000
    pruning_frequency: int = 5
    pruning_keep_ratio: float = 0.98
    creation_policy: str = "paired_ffn_channel_probe"
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
    growth_target_primaire: float = 0.80
    hebbian_threshold: float = 0.002
    hebbian_threshold_min: float = 1e-5
    # Primary-only run: newsyn/loss masking matter on level transitions, not on the first level.
    enable_newsyn: bool = False
    enable_loss_masking: bool = False
    enable_replay_boost: bool = False
    enable_output_head_guard: bool = False
    # 3C4 triphasic controller, scaled from d_ff=512 to d_ff=1024.
    triphasic_controller_enabled: bool = True
    channel_creation_enabled: bool = True
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
    phase2_pruning_enabled: bool = True
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
    return RunConfig(
        **{
            **base.__dict__,
            **{
                "exp_id": RunConfig.exp_id,
                "results_dir": RunConfig.results_dir,
                "levels": RunConfig.levels,
                "epochs_per_level": RunConfig.epochs_per_level,
                "last_level_epochs": RunConfig.last_level_epochs,
                "eval_per_module_interval": RunConfig.eval_per_module_interval,
                "module_eval_frequency": RunConfig.module_eval_frequency,
                "runtime_eval_cap_per_module": RunConfig.runtime_eval_cap_per_module,
                "checkpoint_eval_frequency": RunConfig.checkpoint_eval_frequency,
                "snapshot_eval_frequency": RunConfig.snapshot_eval_frequency,
                "snapshot_eval_mode": RunConfig.snapshot_eval_mode,
                "checkpoint_eval_mode": RunConfig.checkpoint_eval_mode,
                "final_eval_mode": RunConfig.final_eval_mode,
                "lr_schedule_min_ratio": RunConfig.lr_schedule_min_ratio,
                "lr_schedule_maturity_epoch_override": RunConfig.lr_schedule_maturity_epoch_override,
                "initial_connectivity_ffn": RunConfig.initial_connectivity_ffn,
                "max_new_synapses": RunConfig.max_new_synapses,
                "max_new_synapses_cap": RunConfig.max_new_synapses_cap,
                "pruning_frequency": RunConfig.pruning_frequency,
                "pruning_keep_ratio": RunConfig.pruning_keep_ratio,
                "creation_policy": RunConfig.creation_policy,
                "creation_auto_budget_from_train_seq": RunConfig.creation_auto_budget_from_train_seq,
                "collect_ffn_hidden_stats_each_epoch": RunConfig.collect_ffn_hidden_stats_each_epoch,
            },
        }
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dynamic_primary_triphasic_phase2fix | "
        "runtime_engine=clean_shared | "
        "dataset=3c5p21_scratchpad_v2_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return _base_to_engine_argv(cfg) + [
        "--lr-schedule-maturity-epoch-override", str(cfg.lr_schedule_maturity_epoch_override),
        "--pruning-keep-ratio", str(cfg.pruning_keep_ratio),
        "--creation-floor-per-connection", str(cfg.creation_floor_per_connection),
        "--creation-group-min-share", str(cfg.creation_group_min_share),
        "--creation-group-mix", str(cfg.creation_group_mix),
        "--creation-group-prior-enc-share", str(cfg.creation_group_prior_enc_share),
        "--creation-group-share-ema-alpha", str(cfg.creation_group_share_ema_alpha),
        "--creation-selection-weight", str(cfg.creation_selection_weight),
        "--creation-auto-budget-from-train-seq", "1" if cfg.creation_auto_budget_from_train_seq else "0",
        "--creation-train-seq-ema-alpha", str(cfg.creation_train_seq_ema_alpha),
        "--creation-train-seq-budget-gamma", str(cfg.creation_train_seq_budget_gamma),
        "--creation-train-seq-budget-min-ratio", str(cfg.creation_train_seq_budget_min_ratio),
        "--creation-train-seq-budget-trigger", str(cfg.creation_train_seq_budget_trigger),
        "--creation-train-seq-fill-ratio-threshold", str(cfg.creation_train_seq_fill_ratio_threshold),
        "--creation-train-seq-fill-ratio-streak", str(cfg.creation_train_seq_fill_ratio_streak),
        "--creation-train-seq-budget-min-epoch", str(cfg.creation_train_seq_budget_min_epoch),
        "--pruning-couple-creation-need", "1" if cfg.pruning_couple_creation_need else "0",
        "--pruning-creation-need-ema-alpha", str(cfg.pruning_creation_need_ema_alpha),
        "--pruning-creation-coupling-alpha", str(cfg.pruning_creation_coupling_alpha),
        "--enable-late-pruning-relaxation", "1" if cfg.enable_late_pruning_relaxation else "0",
        "--late-pruning-min-epoch", str(cfg.late_pruning_min_epoch),
        "--late-pruning-fill-ratio-threshold", str(cfg.late_pruning_fill_ratio_threshold),
        "--late-pruning-growth-need-threshold", str(cfg.late_pruning_growth_need_threshold),
        "--late-pruning-keep-ratio", str(cfg.late_pruning_keep_ratio),
        "--late-pruning-freeze-on-zero-growth", "1" if cfg.late_pruning_freeze_on_zero_growth else "0",
        "--collect-ffn-hidden-stats-each-epoch", "1" if cfg.collect_ffn_hidden_stats_each_epoch else "0",
        "--growth-target-primaire", str(cfg.growth_target_primaire),
        "--hebbian-threshold", str(cfg.hebbian_threshold),
        "--hebbian-threshold-min", str(cfg.hebbian_threshold_min),
        "--enable-newsyn", "1" if cfg.enable_newsyn else "0",
        "--enable-loss-masking", "1" if cfg.enable_loss_masking else "0",
        "--enable-replay-boost", "1" if cfg.enable_replay_boost else "0",
        "--enable-output-head-guard", "1" if cfg.enable_output_head_guard else "0",
    ]


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg = _base_apply_experiment_overrides(cfg, defaults)
    cfg.exp_id = str(defaults.exp_id)
    cfg.levels = tuple(str(level_name) for level_name in defaults.levels)
    cfg.start_level = str(defaults.start_level)
    cfg.lr_schedule_min_ratio = float(defaults.lr_schedule_min_ratio)
    cfg.lr_schedule_maturity_epoch_override = int(defaults.lr_schedule_maturity_epoch_override)
    cfg.pruning_keep_ratio = float(defaults.pruning_keep_ratio)
    cfg.creation_policy = str(defaults.creation_policy)
    cfg.creation_floor_per_connection = int(defaults.creation_floor_per_connection)
    cfg.creation_group_min_share = float(defaults.creation_group_min_share)
    cfg.creation_group_mix = float(defaults.creation_group_mix)
    cfg.creation_group_prior_enc_share = float(defaults.creation_group_prior_enc_share)
    cfg.creation_group_share_ema_alpha = float(defaults.creation_group_share_ema_alpha)
    cfg.creation_selection_weight = str(defaults.creation_selection_weight)
    cfg.creation_auto_budget_from_train_seq = bool(defaults.creation_auto_budget_from_train_seq)
    cfg.creation_train_seq_ema_alpha = float(defaults.creation_train_seq_ema_alpha)
    cfg.creation_train_seq_budget_gamma = float(defaults.creation_train_seq_budget_gamma)
    cfg.creation_train_seq_budget_min_ratio = float(defaults.creation_train_seq_budget_min_ratio)
    cfg.creation_train_seq_budget_trigger = int(defaults.creation_train_seq_budget_trigger)
    cfg.creation_train_seq_fill_ratio_threshold = float(defaults.creation_train_seq_fill_ratio_threshold)
    cfg.creation_train_seq_fill_ratio_streak = int(defaults.creation_train_seq_fill_ratio_streak)
    cfg.creation_train_seq_budget_min_epoch = int(defaults.creation_train_seq_budget_min_epoch)
    cfg.pruning_couple_creation_need = bool(defaults.pruning_couple_creation_need)
    cfg.pruning_creation_need_ema_alpha = float(defaults.pruning_creation_need_ema_alpha)
    cfg.pruning_creation_coupling_alpha = float(defaults.pruning_creation_coupling_alpha)
    cfg.enable_late_pruning_relaxation = bool(defaults.enable_late_pruning_relaxation)
    cfg.late_pruning_min_epoch = int(defaults.late_pruning_min_epoch)
    cfg.late_pruning_fill_ratio_threshold = float(defaults.late_pruning_fill_ratio_threshold)
    cfg.late_pruning_growth_need_threshold = float(defaults.late_pruning_growth_need_threshold)
    cfg.late_pruning_keep_ratio = float(defaults.late_pruning_keep_ratio)
    cfg.late_pruning_freeze_on_zero_growth = bool(defaults.late_pruning_freeze_on_zero_growth)
    cfg.collect_ffn_hidden_stats_each_epoch = bool(defaults.collect_ffn_hidden_stats_each_epoch)
    cfg.growth_target_primaire = float(defaults.growth_target_primaire)
    cfg.hebbian_threshold = float(defaults.hebbian_threshold)
    cfg.hebbian_threshold_min = float(defaults.hebbian_threshold_min)
    cfg.enable_newsyn = bool(defaults.enable_newsyn)
    cfg.enable_loss_masking = bool(defaults.enable_loss_masking)
    cfg.enable_replay_boost = bool(defaults.enable_replay_boost)
    cfg.enable_output_head_guard = bool(defaults.enable_output_head_guard)
    cfg.triphasic_controller_enabled = bool(defaults.triphasic_controller_enabled)
    cfg.channel_creation_enabled = bool(defaults.channel_creation_enabled)
    cfg.channel_fan_in = int(defaults.channel_fan_in)
    cfg.channel_fan_out = int(defaults.channel_fan_out)
    cfg.channel_bundle_size = int(defaults.channel_bundle_size)
    cfg.channel_candidate_pool = int(defaults.channel_candidate_pool)
    cfg.channel_bundle_init_scale = float(defaults.channel_bundle_init_scale)
    cfg.channel_underused_max_quantile = float(defaults.channel_underused_max_quantile)
    cfg.channel_require_full_bundle = bool(defaults.channel_require_full_bundle)
    cfg.block_density_cap = float(defaults.block_density_cap)
    cfg.channel_budget_fraction = float(defaults.channel_budget_fraction)
    cfg.channel_probe_threshold_ema_alpha = float(defaults.channel_probe_threshold_ema_alpha)
    cfg.max_probationary_channels_per_prefix = int(defaults.max_probationary_channels_per_prefix)
    cfg.max_probationary_channels_global = int(defaults.max_probationary_channels_global)
    cfg.max_new_probationary_channels_per_epoch = int(defaults.max_new_probationary_channels_per_epoch)
    cfg.phase1_registry_enabled = bool(defaults.phase1_registry_enabled)
    cfg.phase1_reselection_decay = float(defaults.phase1_reselection_decay)
    cfg.phase1_unexplored_bonus = float(defaults.phase1_unexplored_bonus)
    cfg.phase1_reexplored_bonus = float(defaults.phase1_reexplored_bonus)
    cfg.phase1_channel_creation_enabled = bool(defaults.phase1_channel_creation_enabled)
    cfg.phase1_pruning_enabled = bool(defaults.phase1_pruning_enabled)
    cfg.phase1_base_channel_budget = int(defaults.phase1_base_channel_budget)
    cfg.phase1_min_channel_budget = int(defaults.phase1_min_channel_budget)
    cfg.phase1_budget_coverage_onset = float(defaults.phase1_budget_coverage_onset)
    cfg.phase1_budget_min_factor = float(defaults.phase1_budget_min_factor)
    cfg.phase1_tenure_rate_ema_alpha = float(defaults.phase1_tenure_rate_ema_alpha)
    cfg.phase1_to_phase2_trigger_enabled = bool(defaults.phase1_to_phase2_trigger_enabled)
    cfg.phase1_to_phase2_min_epoch = int(defaults.phase1_to_phase2_min_epoch)
    cfg.phase1_to_phase2_coverage_global = float(defaults.phase1_to_phase2_coverage_global)
    cfg.phase1_to_phase2_min_prefix_coverage = float(defaults.phase1_to_phase2_min_prefix_coverage)
    cfg.phase1_to_phase2_force_epoch = int(defaults.phase1_to_phase2_force_epoch)
    cfg.phase1_min_evaluated_channels = int(defaults.phase1_min_evaluated_channels)
    cfg.handoff_duration = int(defaults.handoff_duration)
    cfg.handoff_channel_creation = bool(defaults.handoff_channel_creation)
    cfg.handoff_pruning = bool(defaults.handoff_pruning)
    cfg.phase2_enabled = bool(defaults.phase2_enabled)
    cfg.phase2_micro_creation_enabled = bool(defaults.phase2_micro_creation_enabled)
    cfg.phase2_micro_creation_policy = str(defaults.phase2_micro_creation_policy)
    cfg.phase2_micro_budget_target = int(defaults.phase2_micro_budget_target)
    cfg.phase2_micro_budget_scale_by_synapses = bool(defaults.phase2_micro_budget_scale_by_synapses)
    cfg.phase2_micro_budget_reference_active_synapses = int(defaults.phase2_micro_budget_reference_active_synapses)
    cfg.phase2_micro_budget_max = int(defaults.phase2_micro_budget_max)
    cfg.phase2_pruning_enabled = bool(defaults.phase2_pruning_enabled)
    cfg.phase2_pruning_start_delay = int(defaults.phase2_pruning_start_delay)
    cfg.prune_cap_from_created_window_enabled = bool(defaults.prune_cap_from_created_window_enabled)
    cfg.prune_cap_from_created_window_ratio = float(defaults.prune_cap_from_created_window_ratio)
    cfg.prune_cap_from_created_window_floor = int(defaults.prune_cap_from_created_window_floor)
    cfg.prune_cap_from_created_window_ceiling = int(defaults.prune_cap_from_created_window_ceiling)
    cfg.phase2_to_phase3_min_phase2_epochs = int(defaults.phase2_to_phase3_min_phase2_epochs)
    cfg.phase2_to_phase3_force_epoch = int(defaults.phase2_to_phase3_force_epoch)
    cfg.phase3_micro_creation_enabled = bool(defaults.phase3_micro_creation_enabled)
    cfg.phase3_pruning_enabled = bool(defaults.phase3_pruning_enabled)
    cfg.tenured_channel_grace_period = int(defaults.tenured_channel_grace_period)
    cfg.local_reinforce_enabled = bool(defaults.local_reinforce_enabled)
    cfg.local_reinforce_epochs = int(defaults.local_reinforce_epochs)
    cfg.local_reinforce_max_edges_per_epoch = int(defaults.local_reinforce_max_edges_per_epoch)
    cfg.local_reinforce_max_edges_per_channel_total = int(defaults.local_reinforce_max_edges_per_channel_total)
    cfg.local_reinforce_edges_per_channel_per_epoch = int(defaults.local_reinforce_edges_per_channel_per_epoch)
    cfg.local_reinforce_top_pool = int(defaults.local_reinforce_top_pool)
    cfg.creation_quality_mad_factor = float(defaults.creation_quality_mad_factor)
    cfg.creation_coherence_threshold = float(defaults.creation_coherence_threshold)
    cfg.creation_need_ema_alpha = float(defaults.creation_need_ema_alpha)
    return cfg
