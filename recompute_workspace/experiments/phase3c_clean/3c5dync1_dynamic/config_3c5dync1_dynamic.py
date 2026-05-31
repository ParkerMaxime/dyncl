#!/usr/bin/env python3
"""Historical config module for the 3C5dynl1 / 3C5dync1f lycee runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


CONFIG_SOURCE_OF_TRUTH = {
    "run_config": "3C5 dynamic college scratchpad defaults from 3C5dynp1b substrate.",
    "runtime_engine": "Generic engine parser/build defaults only.",
    "batch_config": "Remote transport, target selection, sync and explicit CLI overrides only.",
}


@dataclass(frozen=True)
class RunConfig:
    exp_id: str = "3C5dync1f_lycee750_from_dync1e_core4_soft_transfer"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5dync1f_lycee750_from_dync1e_core4_softtransfer_no_college_tiers_scratchpadv3_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire", "college", "lycee")
    curriculum_variant: str = "full"
    # Descriptive label for the family mix policy. The effective runtime mix is
    # still driven by the backend/profile for this runner family.
    mix_variant: str = "lycee_soft_transfer_from_college80"
    device: str = "cuda"
    seeds: tuple[int, ...] = (42,)
    epochs_per_level: int = 750
    last_level_epochs: int = 750
    samples_per_epoch: int = 16384
    batch_size: int = 64
    num_workers: int = 6
    d_model: int = 256
    d_ff: int = 1024
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    max_q_len: int = 160
    max_a_len: int = 128
    lr: float = 1e-3
    lr_schedule: str = "warmup_creation_plateau_cosine"
    lr_warmup_epochs: int = 10
    lr_schedule_min_ratio: float = 0.10
    lr_sgdr_prev_floor: float = 1e-4
    lr_sgdr_cycles: Tuple[Tuple[int, int, float, float, int], ...] = ()
    phase0_fixed_epochs: int = 50
    phase0_unfreeze_epochs: int = 20
    phase0_lr_start: float = 1e-4
    phase0_lr_end: float = 5e-5
    phase12_old_lr_target: float = 1.5e-4
    phase12_new_lr_target: float = 5e-4
    phase12_warm_restart_epochs: int = 20
    phase12_dual_lr_enabled: bool = True
    phase12_dual_lr_v_init_floor: float = 1e-8
    use_ewc: bool = False
    ewc_lambda: float = 1000.0
    ewc_fisher_samples: int = 256
    ewc_update_frequency: int = 0
    phase3_transfer_cosine_enabled: bool = True
    phase3_transfer_lr_min: float = 5e-5
    college_tier_curriculum_enabled: bool = False
    college_mixed_cap_until_epoch: int = 0
    college_mixed_cap_value: float = 1.0
    college_lcm_cap_until_epoch: int = 0
    college_lcm_cap_value: float = 1.0
    college_cap_redistribute_modules: tuple[str, ...] = (
        "numbers__div_remainder",
        "numbers__gcd",
        "numbers__is_factor",
    )
    label_smoothing: float = 0.03
    initial_connectivity_attn: float = 1.0
    initial_connectivity_ffn: float = 0.30
    checkpoint_eval_frequency: int = 250
    retention_eval_frequency: int = 50
    snapshot_eval_frequency: int = 0
    snapshot_eval_mode: str = "ar"
    checkpoint_eval_mode: str = "ar_tf"
    final_eval_mode: str = "ar_tf"
    module_eval_frequency: int = 50
    eval_module_pool_cap: int = 500
    eval_per_module_interval: int = 50
    tier_eval_frequency: int = 50
    tier_eval_samples: int = 500
    tier_ema_alpha: float = 0.0
    tier_min_epochs: int = 0
    tier_urgency_epochs: int = 0
    handoff_epochs: int = 50
    new_tier_start: float = 0.0
    new_tier_target: float = 1.0
    adaptive_module_mix_enabled: bool = True
    adaptive_module_mix_floor: float = 0.05
    freeze_module_mix_epoch: int = 0
    freeze_module_mix_window: int = 3
    subset_cache_dir: str = "data/math_subset_cache_full"
    raw_max_samples_per_module: int = 0
    raw_max_test_samples_per_module: int = 1000
    runtime_train_cap_per_module: int = 0
    runtime_eval_cap_per_module: int = 0
    retention_train_cap_per_module: int = 100000
    # Extra splits are expensive for the full 11-module college opening because
    # each module bundle is loaded separately. Keep core train/test eval only;
    # interpolate/extrapolate can be run as a targeted benchmark afterwards.
    extra_eval_splits: tuple[str, ...] = ()
    derived_cache_root: str = "data/derived_cache"
    derived_cache_name: str = "3c5dync1f_lycee_core4_scratchpad_v3"
    derived_cache_version: str = "v1"
    compile_dense_enabled: bool = False
    amp_dense_enabled: bool = True
    amp_dense_dtype: str = "bf16"
    creation_policy: str = "paired_ffn_channel_probe"
    creation_score_mode: str = "mlp_seq"
    creation_score_normalization: str = "valid_tokens"
    creation_score_alpha: float = 0.20
    creation_active_ref_quantile: float = 0.10
    creation_priority_topk: int = 128
    creation_ref_mature_only: bool = True
    max_new_synapses: int = 400
    max_new_synapses_cap: int = 4000
    pruning_frequency: int = 5
    pruning_keep_ratio: float = 0.98
    pruning_policy: str = "matrix_survival_mix"
    pruning_active_ref_quantile: float = 0.10
    pruning_priority_topk: int = 128
    pruning_ref_mature_only: bool = True
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
    enable_newsyn: bool = True
    enable_loss_masking: bool = True
    enable_replay_boost: bool = False
    enable_output_head_guard: bool = False
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
    max_probationary_channels_per_prefix: int = 72
    max_probationary_channels_global: int = 224
    max_new_probationary_channels_per_epoch: int = 16
    phase1_registry_enabled: bool = True
    phase1_reselection_decay: float = 0.50
    phase1_unexplored_bonus: float = 1.00
    phase1_reexplored_bonus: float = 0.70
    phase1_channel_creation_enabled: bool = True
    phase1_pruning_enabled: bool = False
    phase1_base_channel_budget: int = 16
    phase1_min_channel_budget: int = 2
    phase1_budget_coverage_onset: float = 0.55
    phase1_budget_min_factor: float = 0.15
    phase1_tenure_rate_ema_alpha: float = 0.10
    phase1_to_phase2_trigger_enabled: bool = True
    phase1_to_phase2_min_epoch: int = 120
    phase1_to_phase2_coverage_global: float = 0.70
    phase1_to_phase2_min_prefix_coverage: float = 0.60
    phase1_to_phase2_force_epoch: int = 240
    phase1_min_evaluated_channels: int = 128
    handoff_duration: int = 15
    handoff_channel_creation: bool = False
    handoff_pruning: bool = False
    phase2_enabled: bool = True
    phase2_micro_creation_enabled: bool = True
    phase2_micro_creation_policy: str = "need_quality_inverse_coverage"
    phase2_micro_budget_target: int = 96
    phase2_micro_budget_scale_by_synapses: bool = False
    phase2_micro_budget_reference_active_synapses: int = 0
    phase2_micro_budget_max: int = 0
    phase2_pruning_enabled: bool = True
    phase2_pruning_start_delay: int = 10
    prune_cap_from_created_window_enabled: bool = True
    prune_cap_from_created_window_ratio: float = 1.20
    prune_cap_from_created_window_floor: int = 200
    prune_cap_from_created_window_ceiling: int = 600
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
    attention_mode: str = "dense"
    dynamic_scope: str = "ffn_only"
    skip_snapshot_eval_epoch1: bool = True
    tier_train_examples_per_tier: int = 0
    tier_eval_examples_per_tier: int = 0
    init_checkpoint: str = (
        "results/phase3/3c/3c5/"
        "3c5dync1e_college750_from_dynp1b_tiers_gcd_lcm_divr_floor_mixed_flat_scratchpadv3_dm256_l2_ff1024_spe16k"
        "/dynamic/checkpoints/seed_42_college_best.pt"
    )
    resume_optimizer: bool = True
    bootstrap_completed_levels: tuple[str, ...] = ("primaire", "college")
    start_level: str = "lycee"


def default_config() -> RunConfig:
    return RunConfig()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dynamic_lycee_from_dync1e_core4_softtransfer_no_college_tiers_phase3cosine | "
        "runtime_engine=clean_shared | "
        "dataset=3c5_scratchpad_v3_primary_college_lycee_core4_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return [
        "--seeds", ",".join(str(seed) for seed in cfg.seeds),
        "--levels", ",".join(cfg.levels),
        "--device", str(cfg.device),
        "--results-dir", str(cfg.results_dir),
        "--use-subset-cache", "1",
        "--subset-cache-dir", str(cfg.subset_cache_dir),
        "--max-samples-per-module", str(cfg.raw_max_samples_per_module),
        "--max-test-samples-per-module", str(cfg.raw_max_test_samples_per_module),
        "--num-workers", str(cfg.num_workers),
        "--persistent-workers", "0",
        "--prefetch-factor", "4",
        "--d-model", str(cfg.d_model),
        "--d-ff", str(cfg.d_ff),
        "--n-enc-layers", str(cfg.n_enc_layers),
        "--n-dec-layers", str(cfg.n_dec_layers),
        "--max-q-len", str(cfg.max_q_len),
        "--max-a-len", str(cfg.max_a_len),
        "--lr", str(cfg.lr),
        "--lr-schedule", str(cfg.lr_schedule),
        "--lr-warmup-epochs", str(cfg.lr_warmup_epochs),
        "--lr-schedule-min-ratio", str(cfg.lr_schedule_min_ratio),
        "--phase0-fixed-epochs", str(cfg.phase0_fixed_epochs),
        "--phase0-unfreeze-epochs", str(cfg.phase0_unfreeze_epochs),
        "--phase0-lr-start", str(cfg.phase0_lr_start),
        "--phase0-lr-end", str(cfg.phase0_lr_end),
        "--phase12-old-lr-target", str(cfg.phase12_old_lr_target),
        "--phase12-new-lr-target", str(cfg.phase12_new_lr_target),
        "--phase12-warm-restart-epochs", str(cfg.phase12_warm_restart_epochs),
        "--phase12-dual-lr-enabled", "1" if cfg.phase12_dual_lr_enabled else "0",
        "--phase12-dual-lr-v-init-floor", str(cfg.phase12_dual_lr_v_init_floor),
        "--use-ewc", "1" if cfg.use_ewc else "0",
        "--ewc-lambda", str(cfg.ewc_lambda),
        "--ewc-fisher-samples", str(cfg.ewc_fisher_samples),
        "--ewc-update-frequency", str(cfg.ewc_update_frequency),
        "--label-smoothing", str(cfg.label_smoothing),
        "--batch-size", str(cfg.batch_size),
        "--epochs-per-level", str(cfg.epochs_per_level),
        "--last-level-epochs", str(cfg.last_level_epochs),
        "--samples-per-epoch", str(cfg.samples_per_epoch),
        "--initial-connectivity-attn", str(cfg.initial_connectivity_attn),
        "--initial-connectivity-ffn", str(cfg.initial_connectivity_ffn),
        "--creation-score-mode", str(cfg.creation_score_mode),
        "--creation-score-normalization", str(cfg.creation_score_normalization),
        "--creation-score-alpha", str(cfg.creation_score_alpha),
        "--creation-policy", str(cfg.creation_policy),
        "--creation-floor-per-connection", str(cfg.creation_floor_per_connection),
        "--creation-active-ref-quantile", str(cfg.creation_active_ref_quantile),
        "--creation-priority-topk", str(cfg.creation_priority_topk),
        "--creation-ref-mature-only", "1" if cfg.creation_ref_mature_only else "0",
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
        "--max-new-synapses", str(cfg.max_new_synapses),
        "--max-new-synapses-cap", str(cfg.max_new_synapses_cap),
        "--pruning-frequency", str(cfg.pruning_frequency),
        "--pruning-keep-ratio", str(cfg.pruning_keep_ratio),
        "--pruning-policy", str(cfg.pruning_policy),
        "--pruning-active-ref-quantile", str(cfg.pruning_active_ref_quantile),
        "--pruning-priority-topk", str(cfg.pruning_priority_topk),
        "--pruning-ref-mature-only", "1" if cfg.pruning_ref_mature_only else "0",
        "--pruning-couple-creation-need", "1" if cfg.pruning_couple_creation_need else "0",
        "--pruning-creation-need-ema-alpha", str(cfg.pruning_creation_need_ema_alpha),
        "--pruning-creation-coupling-alpha", str(cfg.pruning_creation_coupling_alpha),
        "--enable-late-pruning-relaxation", "1" if cfg.enable_late_pruning_relaxation else "0",
        "--late-pruning-min-epoch", str(cfg.late_pruning_min_epoch),
        "--late-pruning-fill-ratio-threshold", str(cfg.late_pruning_fill_ratio_threshold),
        "--late-pruning-growth-need-threshold", str(cfg.late_pruning_growth_need_threshold),
        "--late-pruning-keep-ratio", str(cfg.late_pruning_keep_ratio),
        "--late-pruning-freeze-on-zero-growth", "1" if cfg.late_pruning_freeze_on_zero_growth else "0",
        "--retention-eval-frequency", str(cfg.retention_eval_frequency),
        "--checkpoint-eval-frequency", str(cfg.checkpoint_eval_frequency),
        "--checkpoint-eval-mode", str(cfg.checkpoint_eval_mode),
        "--snapshot-eval-frequency", str(cfg.snapshot_eval_frequency),
        "--snapshot-eval-mode", str(cfg.snapshot_eval_mode),
        "--final-eval-mode", str(cfg.final_eval_mode),
        "--eval-per-module-interval", str(cfg.eval_per_module_interval),
        "--compile-dense-enabled", "1" if cfg.compile_dense_enabled else "0",
        "--amp-dense-enabled", "1" if cfg.amp_dense_enabled else "0",
        "--amp-dense-dtype", str(cfg.amp_dense_dtype),
        "--collect-ffn-hidden-stats-each-epoch", "1" if cfg.collect_ffn_hidden_stats_each_epoch else "0",
        "--growth-target-primaire", str(cfg.growth_target_primaire),
        "--hebbian-threshold", str(cfg.hebbian_threshold),
        "--hebbian-threshold-min", str(cfg.hebbian_threshold_min),
        "--enable-newsyn", "1" if cfg.enable_newsyn else "0",
        "--enable-loss-masking", "1" if cfg.enable_loss_masking else "0",
        "--enable-replay-boost", "1" if cfg.enable_replay_boost else "0",
        "--enable-output-head-guard", "1" if cfg.enable_output_head_guard else "0",
        "--init-checkpoint", str(cfg.init_checkpoint),
        "--resume-optimizer", "1" if cfg.resume_optimizer else "0",
        "--save-best-checkpoint", "1",
        "--save-last-checkpoint", "1",
    ]


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg.exp_id = str(defaults.exp_id)
    cfg.results_dir = str(cfg.results_dir or defaults.results_dir)
    cfg.levels = [str(level) for level in defaults.levels]
    cfg.curriculum_variant = str(defaults.curriculum_variant)
    cfg.mix_variant = str(defaults.mix_variant)
    cfg.max_a_len = max(int(defaults.max_a_len), int(cfg.max_a_len))
    cfg.batch_size = int(defaults.batch_size)
    cfg.num_workers = int(defaults.num_workers)
    cfg.eval_per_module_interval = int(defaults.eval_per_module_interval)
    cfg.eval_module_pool_cap = int(defaults.eval_module_pool_cap)
    cfg.tier_eval_frequency = int(defaults.tier_eval_frequency)
    cfg.tier_eval_samples = int(defaults.tier_eval_samples)
    cfg.tier_ema_alpha = float(defaults.tier_ema_alpha)
    cfg.tier_min_epochs = int(defaults.tier_min_epochs)
    cfg.tier_urgency_epochs = int(defaults.tier_urgency_epochs)
    cfg.handoff_epochs = int(defaults.handoff_epochs)
    cfg.new_tier_start = float(defaults.new_tier_start)
    cfg.new_tier_target = float(defaults.new_tier_target)
    cfg.adaptive_module_mix_enabled = bool(defaults.adaptive_module_mix_enabled)
    cfg.adaptive_module_mix_floor = float(defaults.adaptive_module_mix_floor)
    cfg.freeze_module_mix_epoch = int(defaults.freeze_module_mix_epoch)
    cfg.freeze_module_mix_window = int(defaults.freeze_module_mix_window)
    cfg.derived_cache_root = str(defaults.derived_cache_root)
    cfg.raw_max_samples_per_module = int(defaults.raw_max_samples_per_module)
    cfg.raw_max_test_samples_per_module = int(defaults.raw_max_test_samples_per_module)
    cfg.runtime_train_cap_per_module = int(defaults.runtime_train_cap_per_module)
    cfg.runtime_eval_cap_per_module = int(defaults.runtime_eval_cap_per_module)
    cfg.retention_train_cap_per_module = int(defaults.retention_train_cap_per_module)
    cfg.derived_cache_name = str(defaults.derived_cache_name)
    cfg.derived_cache_version = str(defaults.derived_cache_version)
    cfg.extra_eval_splits = tuple(defaults.extra_eval_splits)
    cfg.attention_mode = str(defaults.attention_mode)
    cfg.dynamic_scope = str(defaults.dynamic_scope)
    cfg.initial_connectivity_attn = float(defaults.initial_connectivity_attn)
    cfg.initial_connectivity_ffn = float(defaults.initial_connectivity_ffn)
    cfg.max_new_synapses = int(defaults.max_new_synapses)
    cfg.max_new_synapses_cap = int(defaults.max_new_synapses_cap)
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
    cfg.creation_score_normalization = str(defaults.creation_score_normalization)
    cfg.creation_score_alpha = float(defaults.creation_score_alpha)
    cfg.creation_active_ref_quantile = float(defaults.creation_active_ref_quantile)
    cfg.creation_priority_topk = int(defaults.creation_priority_topk)
    cfg.creation_ref_mature_only = bool(defaults.creation_ref_mature_only)
    cfg.pruning_policy = str(defaults.pruning_policy)
    cfg.pruning_active_ref_quantile = float(defaults.pruning_active_ref_quantile)
    cfg.pruning_priority_topk = int(defaults.pruning_priority_topk)
    cfg.pruning_ref_mature_only = bool(defaults.pruning_ref_mature_only)
    cfg.pruning_couple_creation_need = bool(defaults.pruning_couple_creation_need)
    cfg.pruning_creation_need_ema_alpha = float(defaults.pruning_creation_need_ema_alpha)
    cfg.pruning_creation_coupling_alpha = float(defaults.pruning_creation_coupling_alpha)
    cfg.enable_late_pruning_relaxation = bool(defaults.enable_late_pruning_relaxation)
    cfg.late_pruning_min_epoch = int(defaults.late_pruning_min_epoch)
    cfg.late_pruning_fill_ratio_threshold = float(defaults.late_pruning_fill_ratio_threshold)
    cfg.late_pruning_growth_need_threshold = float(defaults.late_pruning_growth_need_threshold)
    cfg.late_pruning_keep_ratio = float(defaults.late_pruning_keep_ratio)
    cfg.late_pruning_freeze_on_zero_growth = bool(defaults.late_pruning_freeze_on_zero_growth)
    cfg.retention_eval_frequency = int(defaults.retention_eval_frequency)
    cfg.pruning_frequency = int(defaults.pruning_frequency)
    cfg.compile_dense_enabled = bool(defaults.compile_dense_enabled)
    cfg.amp_dense_enabled = bool(defaults.amp_dense_enabled)
    cfg.amp_dense_dtype = str(defaults.amp_dense_dtype)
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
    cfg.skip_snapshot_eval_epoch1 = bool(defaults.skip_snapshot_eval_epoch1)
    cfg.tier_train_examples_per_tier = int(defaults.tier_train_examples_per_tier)
    cfg.tier_eval_examples_per_tier = int(defaults.tier_eval_examples_per_tier)
    cfg.init_checkpoint = str(defaults.init_checkpoint)
    cfg.resume_optimizer = bool(defaults.resume_optimizer)
    cfg.bootstrap_completed_levels = [str(level) for level in defaults.bootstrap_completed_levels]
    cfg.start_level = str(defaults.start_level)
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
    cfg.phase12_dual_lr_v_init_floor = float(defaults.phase12_dual_lr_v_init_floor)
    cfg.use_ewc = bool(defaults.use_ewc)
    cfg.ewc_lambda = float(defaults.ewc_lambda)
    cfg.ewc_fisher_samples = int(defaults.ewc_fisher_samples)
    cfg.ewc_update_frequency = int(defaults.ewc_update_frequency)
    cfg.phase3_transfer_cosine_enabled = bool(defaults.phase3_transfer_cosine_enabled)
    cfg.phase3_transfer_lr_min = float(defaults.phase3_transfer_lr_min)
    cfg.college_tier_curriculum_enabled = bool(defaults.college_tier_curriculum_enabled)
    cfg.college_mixed_cap_until_epoch = int(defaults.college_mixed_cap_until_epoch)
    cfg.college_mixed_cap_value = float(defaults.college_mixed_cap_value)
    cfg.college_lcm_cap_until_epoch = int(defaults.college_lcm_cap_until_epoch)
    cfg.college_lcm_cap_value = float(defaults.college_lcm_cap_value)
    cfg.college_cap_redistribute_modules = tuple(str(module) for module in defaults.college_cap_redistribute_modules)
    cfg.lr_schedule_maturity_epoch_override = 0
    cfg.save_best_checkpoint = True
    cfg.save_last_checkpoint = True
    return cfg
