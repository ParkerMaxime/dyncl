#!/usr/bin/env python3
"""Shared config and CLI parsing for clean phase 3C runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class Config:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_enc_layers: int = 1
    n_dec_layers: int = 1
    dropout: float = 0.0
    ffn_activation: str = "relu"
    tie_embeddings: bool = False
    max_q_len: int = 160
    max_a_len: int = 30
    initial_connectivity_attn: float = 0.10
    initial_connectivity_ffn: float = 0.05
    energy_budget: float = 1e9
    max_skip_distance: int = 1
    hebbian_threshold: float = 0.002
    hebbian_threshold_min: float = 1e-5
    hebbian_need_scale: float = 0.15
    protection_period: int = 15
    creation_score_mode: str = "legacy"
    creation_score_normalization: str = "batch"
    creation_score_alpha: float = 0.10
    creation_policy: str = "uniform_quota"
    creation_floor_per_connection: int = 0
    creation_active_ref_quantile: float = 0.10
    creation_gap_rel_margin: float = 0.10
    creation_gap_abs_margin: float = 1e-6
    creation_priority_topk: int = 128
    creation_ref_mature_only: bool = True
    creation_group_min_share: float = 0.40
    creation_group_mix: float = 0.50
    creation_policy_after: str = ""
    creation_policy_switch_epoch: int = 0
    creation_group_prior_enc_share: float = 0.50
    creation_group_share_ema_alpha: float = 1.0
    creation_selection_weight: str = "raw"
    creation_auto_budget_from_role_need: bool = False
    creation_role_need_ema_alpha: float = 0.20
    creation_auto_budget_min_ratio: float = 0.25
    creation_auto_budget_from_train_seq: bool = False
    creation_train_seq_ema_alpha: float = 0.20
    creation_train_seq_budget_gamma: float = 0.50
    creation_train_seq_budget_min_ratio: float = 0.25
    creation_train_seq_budget_trigger: int = 0
    creation_train_seq_fill_ratio_threshold: float = 0.0
    creation_train_seq_fill_ratio_streak: int = 0
    creation_train_seq_budget_min_epoch: int = 0
    ffn_channel_complete_k: int = 0
    lr: float = 0.001
    optimizer_name: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    lr_schedule: str = "none"
    lr_warmup_epochs: int = 0
    lr_schedule_min_ratio: float = 0.10
    lr_schedule_maturity_epoch_override: int = 0
    phase0_fixed_epochs: int = 0
    phase0_unfreeze_epochs: int = 0
    phase0_lr_start: float = 1e-4
    phase0_lr_end: float = 5e-5
    phase12_old_lr_target: float = 1.5e-4
    phase12_new_lr_target: float = 5e-4
    phase12_warm_restart_epochs: int = 20
    phase12_dual_lr_enabled: bool = False
    phase12_dual_lr_v_init_floor: float = 1e-8
    label_smoothing: float = 0.0
    batch_size: int = 128
    epochs_per_level: int = 50
    last_level_epochs: int = 200
    samples_per_epoch: int = 8000
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 4
    use_fused_adam: bool = False
    synaptogenesis_frequency: int = 1
    max_new_synapses: int = 200
    max_new_synapses_cap: int = 2000
    growth_need_boost: float = 8.0
    pruning_frequency: int = 5
    pruning_keep_ratio: float = 0.98
    pruning_policy: str = "global_usage"
    pruning_active_ref_quantile: float = 0.10
    pruning_priority_topk: int = 128
    pruning_ref_mature_only: bool = True
    pruning_couple_creation_need: bool = False
    pruning_creation_need_ema_alpha: float = 0.20
    pruning_creation_coupling_alpha: float = 1.0
    tenure_enabled: bool = False
    tenure_score_ema_alpha: float = 0.10
    tenure_min_age: int = 30
    tenure_acquire_quantile: float = 0.80
    tenure_acquire_streak: int = 20
    tenure_max_share_per_matrix: float = 0.20
    tenure_decay_on_miss: int = 1
    tenure_release_enabled: bool = False
    enable_late_pruning_relaxation: bool = False
    late_pruning_mode: str = "legacy_gate"
    late_pruning_min_epoch: int = 0
    late_pruning_fill_ratio_threshold: float = 0.75
    late_pruning_growth_need_threshold: float = 0.25
    late_pruning_keep_ratio: float = 0.995
    late_pruning_freeze_on_zero_growth: bool = False
    enable_replay_weighting: bool = True
    replay_weight_base: float = 1.0
    replay_weight_max: float = 5.0
    retention_target: float = 0.95
    retention_target_from_bootstrap: bool = False
    retention_target_bootstrap_scale: float = 1.0
    retention_gain: float = 2.0
    replay_weighting_min_epoch: int = 0
    retention_eval_frequency: int = 50
    retention_eval_mode: str = "ar"
    snapshot_eval_frequency: int = 50
    snapshot_eval_mode: str = "ar"
    final_eval_mode: str = "ar_tf"
    checkpoint_eval_frequency: int = 0
    checkpoint_eval_mode: str = "ar"
    skip_snapshot_eval_epoch1: bool = False
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    collect_ffn_hidden_stats_each_epoch: bool = True
    long_run_lite: bool = False
    eval_per_module_interval: int = 0
    compile_dense_enabled: bool = True
    compile_dense_mode: str = "reduce-overhead"
    compile_dense_backend: str = "auto"
    amp_dense_enabled: bool = True
    amp_dense_dtype: str = "bf16"
    use_ewc: bool = False
    ewc_lambda: float = 1000.0
    ewc_fisher_samples: int = 256
    ewc_update_frequency: int = 0
    enable_replay_boost: bool = True
    boost_min_level_idx: int = 3
    boost_cycle: int = 10
    boost_duration: int = 2
    boost_factor: float = 3.0
    enable_consolidation_windows: bool = False
    consolidation_min_level_idx: int = 99
    consolidation_start_epoch: int = 100
    consolidation_cycle: int = 50
    consolidation_duration: int = 3
    consolidation_mix_mode: str = "equal_seen_levels"
    consolidation_freeze_creation: bool = True
    consolidation_freeze_pruning: bool = True
    enable_gradient_tags: bool = True
    tag_decay: float = 0.995
    tag_update_rate: float = 0.03
    tag_probe_batches: int = 4
    tag_layerwise_normalize: bool = True
    tag_quantile: float = 0.70
    enable_replay_tag_protection: bool = False
    replay_tag_quantile: float = 0.70
    replay_tag_min: float = 0.02
    tagged_prune_pressure_slope: float = 0.20
    max_tagged_prune_ratio: float = 0.05
    blockb_min_epoch: int = 0
    blockb_target_bootstrap_scale: float = 1.0
    blockb_crystal_warmup_epochs: int = 0
    blockb_enable_weighted_probe: bool = False
    enable_crystallization: bool = True
    crystal_vital_floor: float = 0.05
    crystal_default_floor: float = 0.30
    crystal_beta: float = 0.70
    enable_transition_buffer: bool = True
    transition_epochs_default: int = 0
    transition_epochs_lycee: int = 2
    transition_epochs_last: int = 8
    transition_crystal_beta: float = 0.95
    transition_tag_update_rate: float = 0.06
    transition_replay_weight: float = 5.0
    enable_growth_retention: bool = True
    growth_min_factor: float = 0.10
    growth_max_factor: float = 1.00
    enable_newsyn: bool = True
    newsyn_min_level_idx: int = 3
    newsyn_epochs: int = 8
    old_synapse_grad_scale_start: float = 0.10
    disable_pruning_during_newsyn: bool = True
    enable_output_head_guard: bool = False
    enable_loss_masking: bool = True
    masking_min_level_idx: int = 3
    mask_warmup_epochs: int = 8
    mask_ramp_epochs: int = 20
    mask_penalty: float = 12.0
    mask_floor_k: float = 30.0
    mask_floor_target: float = 0.85
    mask_floor_min: float = 0.02
    mask_floor_max: float = 0.20
    mask_floor_target_from_bootstrap: bool = False
    mask_floor_target_bootstrap_scale: float = 1.0
    max_regression: float = 0.10
    final_acc_target: float = 0.80
    primaire_min_acc: float = 0.80
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    levels: List[str] = field(default_factory=lambda: ["primaire", "college", "lycee", "superieur"])
    device: str = "auto"
    results_dir: str = "results/phase3/3c/3c1/3c1_transformer"
    data_root: str = "data"
    download: bool = True
    use_subset_cache: bool = True
    subset_cache_dir: str = "data/math_subset_cache"
    curriculum_variant: str = "full"
    mix_variant: str = "default"
    math_difficulty: str = "train-easy"
    max_samples_per_module: int = 10000
    max_test_samples_per_module: int = 1000
    init_checkpoint: str = ""
    bootstrap_completed_levels: List[str] = field(default_factory=list)
    start_level: str = ""
    resume_optimizer: bool = False
    start_epoch_offset: int = 0
    enable_bootstrap_creation_reset: bool = False
    bootstrap_creation_reset_mode: str = "creation_only"
    enable_bootstrap_creation_qref_schedule: bool = False
    bootstrap_creation_qref_start: float = 0.01
    bootstrap_creation_qref_end: float = 0.10
    bootstrap_creation_qref_epochs: int = 20
    degenerate_acc_threshold: float = 0.05
    convergence_tail_window: int = 10
    drop_window_epochs: int = 10
    log_tag_layer_stats: bool = False
    growth_target_primaire: float = 0.99


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    level_order: Sequence[str],
    growth_target_primaire_default: float,
):
    p = argparse.ArgumentParser(description="3C.1 Transformer — Dynamic sparse seq2seq on DeepMind Math")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--levels", type=str, default="primaire,college,lycee,superieur")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--download", type=int, default=1)
    p.add_argument("--use-subset-cache", type=int, default=1)
    p.add_argument("--subset-cache-dir", type=str, default="data/math_subset_cache")
    p.add_argument("--math-difficulty", type=str, default="train-easy")
    p.add_argument("--max-samples-per-module", type=int, default=10000)
    p.add_argument("--max-test-samples-per-module", type=int, default=1000)
    p.add_argument("--init-checkpoint", type=str, default="")
    p.add_argument("--resume-optimizer", type=int, default=0)
    p.add_argument("--start-epoch-offset", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--persistent-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--use-fused-adam", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-enc-layers", type=int, default=1)
    p.add_argument("--n-dec-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ffn-activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--tie-embeddings", type=int, default=0)
    p.add_argument("--max-q-len", type=int, default=160)
    p.add_argument("--max-a-len", type=int, default=30)
    p.add_argument("--initial-connectivity-attn", type=float, default=0.10)
    p.add_argument("--initial-connectivity-ffn", type=float, default=0.05)
    p.add_argument("--creation-score-mode", type=str, default="legacy", choices=["legacy", "mlp_seq"])
    p.add_argument("--creation-score-normalization", type=str, default="batch", choices=["batch", "valid_tokens"])
    p.add_argument("--creation-score-alpha", type=float, default=0.10)
    p.add_argument(
        "--creation-policy",
        type=str,
        default="uniform_quota",
        choices=[
            "uniform_quota",
            "hybrid_global",
            "effect_gap_global",
            "group_balanced_effect_gap",
            "group_calibrated_weighted",
            "group_frustration_weighted",
            "need_quality_filtered",
            "need_quality_inverse_coverage",
            "paired_ffn_channel_random",
            "paired_ffn_channel_probe",
        ],
    )
    p.add_argument("--creation-floor-per-connection", type=int, default=0)
    p.add_argument("--creation-active-ref-quantile", type=float, default=0.10)
    p.add_argument("--creation-gap-rel-margin", type=float, default=0.10)
    p.add_argument("--creation-gap-abs-margin", type=float, default=1e-6)
    p.add_argument("--creation-priority-topk", type=int, default=128)
    p.add_argument("--creation-ref-mature-only", type=int, default=1)
    p.add_argument("--creation-group-min-share", type=float, default=0.40)
    p.add_argument("--creation-group-mix", type=float, default=0.50)
    p.add_argument("--creation-policy-after", type=str, default="")
    p.add_argument("--creation-policy-switch-epoch", type=int, default=0)
    p.add_argument("--creation-group-prior-enc-share", type=float, default=0.50)
    p.add_argument("--creation-group-share-ema-alpha", type=float, default=1.0)
    p.add_argument("--creation-selection-weight", type=str, default="raw", choices=["raw", "score_gap", "sqrt_score_gap"])
    p.add_argument("--creation-auto-budget-from-role-need", type=int, default=0)
    p.add_argument("--creation-role-need-ema-alpha", type=float, default=0.20)
    p.add_argument("--creation-auto-budget-min-ratio", type=float, default=0.25)
    p.add_argument("--creation-auto-budget-from-train-seq", type=int, default=0)
    p.add_argument("--creation-train-seq-ema-alpha", type=float, default=0.20)
    p.add_argument("--creation-train-seq-budget-gamma", type=float, default=0.50)
    p.add_argument("--creation-train-seq-budget-min-ratio", type=float, default=0.25)
    p.add_argument("--creation-train-seq-budget-trigger", type=int, default=0)
    p.add_argument("--creation-train-seq-fill-ratio-threshold", type=float, default=0.0)
    p.add_argument("--creation-train-seq-fill-ratio-streak", type=int, default=0)
    p.add_argument("--creation-train-seq-budget-min-epoch", type=int, default=0)
    p.add_argument("--ffn-channel-complete-k", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine", "creation_plateau_cosine", "warmup_creation_plateau_cosine"])
    p.add_argument("--lr-warmup-epochs", type=int, default=0)
    p.add_argument("--lr-schedule-min-ratio", type=float, default=0.10)
    p.add_argument("--lr-schedule-maturity-epoch-override", type=int, default=0)
    p.add_argument("--phase0-fixed-epochs", type=int, default=0)
    p.add_argument("--phase0-unfreeze-epochs", type=int, default=0)
    p.add_argument("--phase0-lr-start", type=float, default=1e-4)
    p.add_argument("--phase0-lr-end", type=float, default=5e-5)
    p.add_argument("--phase12-old-lr-target", type=float, default=1.5e-4)
    p.add_argument("--phase12-new-lr-target", type=float, default=5e-4)
    p.add_argument("--phase12-warm-restart-epochs", type=int, default=20)
    p.add_argument("--phase12-dual-lr-enabled", type=int, default=0)
    p.add_argument("--phase12-dual-lr-v-init-floor", type=float, default=1e-8)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs-per-level", type=int, default=50)
    p.add_argument("--last-level-epochs", type=int, default=200)
    p.add_argument("--samples-per-epoch", type=int, default=8000)
    p.add_argument("--max-new-synapses", type=int, default=200)
    p.add_argument("--max-new-synapses-cap", type=int, default=2000)
    p.add_argument("--pruning-frequency", type=int, default=5)
    p.add_argument("--pruning-keep-ratio", type=float, default=0.98)
    p.add_argument("--pruning-policy", type=str, default="global_usage", choices=["global_usage", "matrix_survival_mix"])
    p.add_argument("--pruning-active-ref-quantile", type=float, default=0.10)
    p.add_argument("--pruning-priority-topk", type=int, default=128)
    p.add_argument("--pruning-ref-mature-only", type=int, default=1)
    p.add_argument("--pruning-couple-creation-need", type=int, default=0)
    p.add_argument("--pruning-creation-need-ema-alpha", type=float, default=0.20)
    p.add_argument("--pruning-creation-coupling-alpha", type=float, default=1.0)
    p.add_argument("--tenure-enabled", type=int, default=0)
    p.add_argument("--tenure-score-ema-alpha", type=float, default=0.10)
    p.add_argument("--tenure-min-age", type=int, default=30)
    p.add_argument("--tenure-acquire-quantile", type=float, default=0.80)
    p.add_argument("--tenure-acquire-streak", type=int, default=20)
    p.add_argument("--tenure-max-share-per-matrix", type=float, default=0.20)
    p.add_argument("--tenure-decay-on-miss", type=int, default=1)
    p.add_argument("--tenure-release-enabled", type=int, default=0)
    p.add_argument("--enable-late-pruning-relaxation", type=int, default=0)
    p.add_argument("--late-pruning-mode", type=str, default="legacy_gate", choices=["legacy_gate", "adaptive_endogenous"])
    p.add_argument("--late-pruning-min-epoch", type=int, default=0)
    p.add_argument("--late-pruning-fill-ratio-threshold", type=float, default=0.75)
    p.add_argument("--late-pruning-growth-need-threshold", type=float, default=0.25)
    p.add_argument("--late-pruning-keep-ratio", type=float, default=0.995)
    p.add_argument("--late-pruning-freeze-on-zero-growth", type=int, default=0)
    p.add_argument("--snapshot-eval-frequency", type=int, default=50)
    p.add_argument("--retention-eval-frequency", type=int, default=50)
    p.add_argument("--retention-eval-mode", type=str, default="ar", choices=["ar", "ar_tf", "tf"])
    p.add_argument("--snapshot-eval-mode", type=str, default="ar", choices=["ar", "ar_tf", "tf"])
    p.add_argument("--final-eval-mode", type=str, default="ar_tf", choices=["ar", "ar_tf", "tf"])
    p.add_argument("--checkpoint-eval-frequency", type=int, default=0)
    p.add_argument("--checkpoint-eval-mode", type=str, default="ar", choices=["ar", "ar_tf", "tf"])
    p.add_argument("--save-best-checkpoint", type=int, default=1)
    p.add_argument("--save-last-checkpoint", type=int, default=1)
    p.add_argument("--collect-ffn-hidden-stats-each-epoch", type=int, default=1)
    p.add_argument("--long-run-lite", type=int, default=0)
    p.add_argument("--eval-per-module-interval", type=int, default=0)
    p.add_argument("--compile-dense-enabled", type=int, choices=[0, 1], default=1)
    p.add_argument("--compile-dense-mode", type=str, default="reduce-overhead")
    p.add_argument("--compile-dense-backend", type=str, default="auto")
    p.add_argument("--amp-dense-enabled", type=int, choices=[0, 1], default=1)
    p.add_argument("--amp-dense-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--use-ewc", type=int, choices=[0, 1], default=0)
    p.add_argument("--ewc-lambda", type=float, default=1000.0)
    p.add_argument("--ewc-fisher-samples", type=int, default=256)
    p.add_argument("--ewc-update-frequency", type=int, default=0)
    p.add_argument("--enable-bootstrap-creation-reset", type=int, default=0)
    p.add_argument("--bootstrap-creation-reset-mode", type=str, default="creation_only", choices=["creation_only"])
    p.add_argument("--enable-bootstrap-creation-qref-schedule", type=int, default=0)
    p.add_argument("--bootstrap-creation-qref-start", type=float, default=0.01)
    p.add_argument("--bootstrap-creation-qref-end", type=float, default=0.10)
    p.add_argument("--bootstrap-creation-qref-epochs", type=int, default=20)
    p.add_argument("--enable-newsyn", type=int, default=1)
    p.add_argument("--enable-loss-masking", type=int, default=1)
    p.add_argument("--enable-replay-boost", type=int, default=1)
    p.add_argument("--enable-output-head-guard", type=int, default=0)
    p.add_argument("--newsyn-min-level-idx", type=int, default=3)
    p.add_argument("--masking-min-level-idx", type=int, default=3)
    p.add_argument("--boost-min-level-idx", type=int, default=3)
    p.add_argument("--enable-crystallization", type=int, default=1)
    p.add_argument("--enable-gradient-tags", type=int, default=1)
    p.add_argument("--enable-transition-buffer", type=int, default=1)
    p.add_argument("--enable-growth-retention", type=int, default=1)
    p.add_argument("--enable-replay-weighting", type=int, default=1)
    p.add_argument("--newsyn-epochs", type=int, default=8)
    p.add_argument("--mask-warmup-epochs", type=int, default=8)
    p.add_argument("--mask-ramp-epochs", type=int, default=20)
    p.add_argument("--mask-floor-k", type=float, default=30.0)
    p.add_argument("--mask-floor-target", type=float, default=0.85)
    p.add_argument("--mask-floor-min", type=float, default=0.02)
    p.add_argument("--mask-floor-max", type=float, default=0.20)
    p.add_argument("--mask-floor-target-from-bootstrap", type=int, default=0)
    p.add_argument("--mask-floor-target-bootstrap-scale", type=float, default=1.0)
    p.add_argument("--boost-cycle", type=int, default=10)
    p.add_argument("--boost-duration", type=int, default=2)
    p.add_argument("--boost-factor", type=float, default=3.0)
    p.add_argument("--transition-epochs-last", type=int, default=8)
    p.add_argument("--degenerate-acc-threshold", type=float, default=0.05)
    p.add_argument("--hebbian-threshold", type=float, default=0.002)
    p.add_argument("--hebbian-threshold-min", type=float, default=1e-5)
    p.add_argument("--hebbian-need-scale", type=float, default=0.15)
    p.add_argument("--growth-target-primaire", type=float, default=growth_target_primaire_default)
    return p.parse_args(argv)


def build_config(
    args,
    *,
    level_order: Sequence[str],
    default_results_dir: str,
) -> Config:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    levels = [x.strip() for x in str(args.levels).split(",") if x.strip()]
    levels = [lv for lv in levels if lv in level_order]
    if not levels:
        raise ValueError("`--levels` doit contenir au moins un niveau valide.")
    if args.smoke:
        epochs = 3
        last_epochs = 5
    else:
        epochs = args.epochs_per_level
        last_epochs = args.last_level_epochs
    results_dir = args.results_dir or default_results_dir
    long_run_lite = bool(args.long_run_lite)
    snapshot_eval_frequency = max(0, int(args.snapshot_eval_frequency))
    checkpoint_eval_frequency = max(0, int(args.checkpoint_eval_frequency))
    collect_ffn_hidden_stats_each_epoch = bool(args.collect_ffn_hidden_stats_each_epoch)
    if long_run_lite:
        snapshot_eval_frequency = 0
        if checkpoint_eval_frequency <= 0:
            checkpoint_eval_frequency = 50
        collect_ffn_hidden_stats_each_epoch = False
    return Config(
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        n_enc_layers=int(args.n_enc_layers),
        n_dec_layers=int(args.n_dec_layers),
        dropout=float(max(0.0, args.dropout)),
        ffn_activation=str(args.ffn_activation).strip().lower(),
        tie_embeddings=bool(args.tie_embeddings),
        max_q_len=args.max_q_len,
        max_a_len=args.max_a_len,
        initial_connectivity_attn=float(np.clip(args.initial_connectivity_attn, 0.001, 1.0)),
        initial_connectivity_ffn=float(np.clip(args.initial_connectivity_ffn, 0.001, 1.0)),
        creation_score_mode=str(args.creation_score_mode).strip().lower(),
        creation_score_normalization=str(args.creation_score_normalization).strip().lower(),
        creation_score_alpha=float(max(1e-4, min(1.0, args.creation_score_alpha))),
        creation_policy=str(args.creation_policy).strip().lower(),
        creation_floor_per_connection=max(0, int(args.creation_floor_per_connection)),
        creation_active_ref_quantile=float(np.clip(args.creation_active_ref_quantile, 0.0, 1.0)),
        creation_gap_rel_margin=float(max(0.0, args.creation_gap_rel_margin)),
        creation_gap_abs_margin=float(max(0.0, args.creation_gap_abs_margin)),
        creation_priority_topk=max(1, int(args.creation_priority_topk)),
        creation_ref_mature_only=bool(args.creation_ref_mature_only),
        creation_group_min_share=float(np.clip(args.creation_group_min_share, 0.0, 0.5)),
        creation_group_mix=float(np.clip(args.creation_group_mix, 0.0, 1.0)),
        creation_policy_after=str(args.creation_policy_after).strip().lower(),
        creation_policy_switch_epoch=max(0, int(args.creation_policy_switch_epoch)),
        creation_group_prior_enc_share=float(np.clip(args.creation_group_prior_enc_share, 0.0, 1.0)),
        creation_group_share_ema_alpha=float(np.clip(args.creation_group_share_ema_alpha, 1e-6, 1.0)),
        creation_selection_weight=str(args.creation_selection_weight).strip().lower(),
        creation_auto_budget_from_role_need=bool(args.creation_auto_budget_from_role_need),
        creation_role_need_ema_alpha=float(np.clip(args.creation_role_need_ema_alpha, 1e-6, 1.0)),
        creation_auto_budget_min_ratio=float(np.clip(args.creation_auto_budget_min_ratio, 0.0, 1.0)),
        creation_auto_budget_from_train_seq=bool(args.creation_auto_budget_from_train_seq),
        creation_train_seq_ema_alpha=float(np.clip(args.creation_train_seq_ema_alpha, 1e-6, 1.0)),
        creation_train_seq_budget_gamma=float(max(1e-6, args.creation_train_seq_budget_gamma)),
        creation_train_seq_budget_min_ratio=float(np.clip(args.creation_train_seq_budget_min_ratio, 0.0, 1.0)),
        creation_train_seq_budget_trigger=max(0, int(args.creation_train_seq_budget_trigger)),
        creation_train_seq_fill_ratio_threshold=float(np.clip(args.creation_train_seq_fill_ratio_threshold, 0.0, 1.0)),
        creation_train_seq_fill_ratio_streak=max(0, int(args.creation_train_seq_fill_ratio_streak)),
        creation_train_seq_budget_min_epoch=max(0, int(args.creation_train_seq_budget_min_epoch)),
        ffn_channel_complete_k=max(0, int(args.ffn_channel_complete_k)),
        lr=args.lr,
        optimizer_name=str(args.optimizer_name).strip().lower(),
        adam_beta1=float(np.clip(args.adam_beta1, 0.0, 0.9999)),
        adam_beta2=float(np.clip(args.adam_beta2, 0.0, 0.99999)),
        adam_eps=float(max(1e-12, args.adam_eps)),
        adam_weight_decay=float(max(0.0, args.adam_weight_decay)),
        lr_schedule=str(args.lr_schedule).strip().lower(),
        lr_warmup_epochs=max(0, int(args.lr_warmup_epochs)),
        lr_schedule_min_ratio=float(np.clip(args.lr_schedule_min_ratio, 0.0, 1.0)),
        lr_schedule_maturity_epoch_override=max(0, int(args.lr_schedule_maturity_epoch_override)),
        phase0_fixed_epochs=max(0, int(args.phase0_fixed_epochs)),
        phase0_unfreeze_epochs=max(0, int(args.phase0_unfreeze_epochs)),
        phase0_lr_start=float(max(1e-8, args.phase0_lr_start)),
        phase0_lr_end=float(max(1e-8, args.phase0_lr_end)),
        phase12_old_lr_target=float(max(1e-8, args.phase12_old_lr_target)),
        phase12_new_lr_target=float(max(1e-8, args.phase12_new_lr_target)),
        phase12_warm_restart_epochs=max(0, int(args.phase12_warm_restart_epochs)),
        phase12_dual_lr_enabled=bool(args.phase12_dual_lr_enabled),
        phase12_dual_lr_v_init_floor=float(max(0.0, args.phase12_dual_lr_v_init_floor)),
        label_smoothing=float(np.clip(args.label_smoothing, 0.0, 0.3)),
        batch_size=args.batch_size,
        epochs_per_level=epochs,
        last_level_epochs=last_epochs,
        samples_per_epoch=args.samples_per_epoch,
        max_new_synapses=max(0, int(args.max_new_synapses)),
        max_new_synapses_cap=max(0, int(args.max_new_synapses_cap)),
        pruning_frequency=max(1, int(args.pruning_frequency)),
        pruning_keep_ratio=float(np.clip(args.pruning_keep_ratio, 0.90, 0.999)),
        pruning_policy=str(args.pruning_policy).strip().lower(),
        pruning_active_ref_quantile=float(np.clip(args.pruning_active_ref_quantile, 0.0, 1.0)),
        pruning_priority_topk=max(1, int(args.pruning_priority_topk)),
        pruning_ref_mature_only=bool(args.pruning_ref_mature_only),
        pruning_couple_creation_need=bool(args.pruning_couple_creation_need),
        pruning_creation_need_ema_alpha=float(np.clip(args.pruning_creation_need_ema_alpha, 1e-6, 1.0)),
        pruning_creation_coupling_alpha=float(max(0.0, args.pruning_creation_coupling_alpha)),
        tenure_enabled=bool(args.tenure_enabled),
        tenure_score_ema_alpha=float(np.clip(args.tenure_score_ema_alpha, 1e-6, 1.0)),
        tenure_min_age=max(1, int(args.tenure_min_age)),
        tenure_acquire_quantile=float(np.clip(args.tenure_acquire_quantile, 0.0, 1.0)),
        tenure_acquire_streak=max(1, int(args.tenure_acquire_streak)),
        tenure_max_share_per_matrix=float(np.clip(args.tenure_max_share_per_matrix, 0.0, 1.0)),
        tenure_decay_on_miss=max(0, int(args.tenure_decay_on_miss)),
        tenure_release_enabled=bool(args.tenure_release_enabled),
        enable_late_pruning_relaxation=bool(args.enable_late_pruning_relaxation),
        late_pruning_mode=str(args.late_pruning_mode).strip().lower(),
        late_pruning_min_epoch=max(0, int(args.late_pruning_min_epoch)),
        late_pruning_fill_ratio_threshold=float(np.clip(args.late_pruning_fill_ratio_threshold, 0.0, 1.0)),
        late_pruning_growth_need_threshold=float(np.clip(args.late_pruning_growth_need_threshold, 0.0, 1.0)),
        late_pruning_keep_ratio=float(np.clip(args.late_pruning_keep_ratio, 0.90, 1.0)),
        late_pruning_freeze_on_zero_growth=bool(args.late_pruning_freeze_on_zero_growth),
        snapshot_eval_frequency=snapshot_eval_frequency,
        retention_eval_frequency=max(0, int(args.retention_eval_frequency)),
        retention_eval_mode=str(args.retention_eval_mode),
        snapshot_eval_mode=str(args.snapshot_eval_mode),
        final_eval_mode=str(args.final_eval_mode),
        checkpoint_eval_frequency=checkpoint_eval_frequency,
        checkpoint_eval_mode=str(args.checkpoint_eval_mode),
        save_best_checkpoint=bool(args.save_best_checkpoint),
        save_last_checkpoint=bool(args.save_last_checkpoint),
        collect_ffn_hidden_stats_each_epoch=collect_ffn_hidden_stats_each_epoch,
        long_run_lite=long_run_lite,
        eval_per_module_interval=max(0, int(args.eval_per_module_interval)),
        compile_dense_enabled=bool(args.compile_dense_enabled),
        compile_dense_mode=str(args.compile_dense_mode).strip() or "reduce-overhead",
        compile_dense_backend=str(args.compile_dense_backend).strip() or "auto",
        amp_dense_enabled=bool(args.amp_dense_enabled),
        amp_dense_dtype=str(args.amp_dense_dtype).strip().lower(),
        use_ewc=bool(args.use_ewc),
        ewc_lambda=float(max(0.0, args.ewc_lambda)),
        ewc_fisher_samples=max(1, int(args.ewc_fisher_samples)),
        ewc_update_frequency=max(0, int(args.ewc_update_frequency)),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
        use_fused_adam=False,
        seeds=seeds,
        levels=levels,
        device=args.device,
        results_dir=results_dir,
        data_root=args.data_root,
        download=bool(args.download),
        use_subset_cache=bool(args.use_subset_cache),
        subset_cache_dir=str(args.subset_cache_dir),
        math_difficulty=args.math_difficulty,
        max_samples_per_module=int(max(1, args.max_samples_per_module)),
        max_test_samples_per_module=int(max(1, args.max_test_samples_per_module)),
        init_checkpoint=str(args.init_checkpoint).strip(),
        resume_optimizer=bool(args.resume_optimizer),
        start_epoch_offset=max(0, int(args.start_epoch_offset)),
        enable_newsyn=bool(args.enable_newsyn),
        enable_loss_masking=bool(args.enable_loss_masking),
        enable_replay_boost=bool(args.enable_replay_boost),
        enable_output_head_guard=bool(args.enable_output_head_guard),
        newsyn_min_level_idx=args.newsyn_min_level_idx,
        masking_min_level_idx=args.masking_min_level_idx,
        boost_min_level_idx=args.boost_min_level_idx,
        enable_crystallization=bool(args.enable_crystallization),
        enable_gradient_tags=bool(args.enable_gradient_tags),
        enable_transition_buffer=bool(args.enable_transition_buffer),
        enable_growth_retention=bool(args.enable_growth_retention),
        enable_replay_weighting=bool(args.enable_replay_weighting),
        newsyn_epochs=args.newsyn_epochs,
        mask_warmup_epochs=args.mask_warmup_epochs,
        mask_ramp_epochs=args.mask_ramp_epochs,
        mask_floor_k=float(args.mask_floor_k),
        mask_floor_target=float(args.mask_floor_target),
        mask_floor_min=float(args.mask_floor_min),
        mask_floor_max=float(args.mask_floor_max),
        mask_floor_target_from_bootstrap=bool(args.mask_floor_target_from_bootstrap),
        mask_floor_target_bootstrap_scale=float(args.mask_floor_target_bootstrap_scale),
        boost_cycle=args.boost_cycle,
        boost_duration=args.boost_duration,
        boost_factor=args.boost_factor,
        transition_epochs_last=args.transition_epochs_last,
        enable_bootstrap_creation_reset=bool(args.enable_bootstrap_creation_reset),
        bootstrap_creation_reset_mode=str(args.bootstrap_creation_reset_mode).strip().lower(),
        enable_bootstrap_creation_qref_schedule=bool(args.enable_bootstrap_creation_qref_schedule),
        bootstrap_creation_qref_start=float(np.clip(args.bootstrap_creation_qref_start, 0.0, 1.0)),
        bootstrap_creation_qref_end=float(np.clip(args.bootstrap_creation_qref_end, 0.0, 1.0)),
        bootstrap_creation_qref_epochs=max(1, int(args.bootstrap_creation_qref_epochs)),
        degenerate_acc_threshold=float(max(0.0, args.degenerate_acc_threshold)),
        hebbian_threshold=float(max(0.0, args.hebbian_threshold)),
        hebbian_threshold_min=float(max(0.0, args.hebbian_threshold_min)),
        hebbian_need_scale=float(max(0.0, args.hebbian_need_scale)),
        growth_target_primaire=float(np.clip(args.growth_target_primaire, 0.0, 1.0)),
    )
