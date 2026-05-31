#!/usr/bin/env python3
"""Configuration centralisee du runner propre 3C5p21 clean dense."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


CONFIG_SOURCE_OF_TRUTH = {
    "run_config": "Scientific experiment defaults and enforced 3C5p21 clean settings.",
    "runtime_engine": "Generic engine parser/build defaults only.",
    "batch_config": "Remote transport, target selection, sync and explicit CLI overrides only.",
}


@dataclass(frozen=True)
class RunConfig:
    exp_id: str = "3C5p21_clean_primaire500_scratchpad_v2_all4"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5p21_clean_primaire500_scratchpad_v2_all4_seed42_gpu_fullcache_dm256_l2_ff1024_spe16k"
    )
    levels: tuple[str, ...] = ("primaire",)
    curriculum_variant: str = "full"
    # Descriptive label for the family mix policy. The effective runtime mix is
    # still driven by the backend/profile for this runner family.
    mix_variant: str = "default"
    device: str = "cuda"
    seeds: tuple[int, ...] = (42,)
    epochs_per_level: int = 500
    last_level_epochs: int = 500
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
    label_smoothing: float = 0.03
    initial_connectivity_attn: float = 1.0
    initial_connectivity_ffn: float = 1.0
    checkpoint_eval_frequency: int = 250
    retention_eval_frequency: int = 0
    snapshot_eval_frequency: int = 50
    snapshot_eval_mode: str = "ar"
    checkpoint_eval_mode: str = "ar_tf"
    final_eval_mode: str = "ar_tf"
    module_eval_frequency: int = 10
    eval_module_pool_cap: int = 0
    eval_per_module_interval: int = 10
    tier_eval_frequency: int = 0
    tier_eval_samples: int = 0
    tier_ema_alpha: float = 0.0
    tier_min_epochs: int = 0
    tier_urgency_epochs: int = 0
    handoff_epochs: int = 0
    new_tier_start: float = 0.0
    new_tier_target: float = 1.0
    adaptive_module_mix_enabled: bool = True
    adaptive_module_mix_floor: float = 0.05
    freeze_module_mix_epoch: int = 0
    freeze_module_mix_window: int = 3
    subset_cache_dir: str = "data/math_subset_cache_full"
    raw_max_samples_per_module: int = 0
    raw_max_test_samples_per_module: int = 0
    runtime_train_cap_per_module: int = 0
    runtime_eval_cap_per_module: int = 0
    extra_eval_splits: tuple[str, ...] = ("interpolate", "extrapolate")
    derived_cache_root: str = "data/derived_cache"
    derived_cache_name: str = "3c5p21_clean_primaire500_scratchpad_v2_all4"
    derived_cache_version: str = "v1"
    compile_dense_enabled: bool = False
    amp_dense_enabled: bool = True
    amp_dense_dtype: str = "bf16"
    creation_policy: str = "uniform_quota"
    creation_score_mode: str = "mlp_seq"
    creation_score_normalization: str = "valid_tokens"
    creation_score_alpha: float = 0.20
    creation_active_ref_quantile: float = 0.10
    creation_priority_topk: int = 128
    creation_ref_mature_only: bool = True
    max_new_synapses: int = 0
    max_new_synapses_cap: int = 0
    pruning_frequency: int = 100000
    pruning_policy: str = "matrix_survival_mix"
    pruning_active_ref_quantile: float = 0.10
    pruning_priority_topk: int = 128
    pruning_ref_mature_only: bool = True
    attention_mode: str = "dense"
    dynamic_scope: str = "ffn_only"
    skip_snapshot_eval_epoch1: bool = True
    tier_train_examples_per_tier: int = 0
    tier_eval_examples_per_tier: int = 0


def default_config() -> RunConfig:
    return RunConfig()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_of_truth_summary() -> str:
    return (
        "run_config=science | "
        "runtime_engine=generic_defaults | "
        "batch_config=transport_or_explicit_override"
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
        "--creation-active-ref-quantile", str(cfg.creation_active_ref_quantile),
        "--creation-priority-topk", str(cfg.creation_priority_topk),
        "--creation-ref-mature-only", "1" if cfg.creation_ref_mature_only else "0",
        "--max-new-synapses", str(cfg.max_new_synapses),
        "--max-new-synapses-cap", str(cfg.max_new_synapses_cap),
        "--pruning-frequency", str(cfg.pruning_frequency),
        "--pruning-policy", str(cfg.pruning_policy),
        "--pruning-active-ref-quantile", str(cfg.pruning_active_ref_quantile),
        "--pruning-priority-topk", str(cfg.pruning_priority_topk),
        "--pruning-ref-mature-only", "1" if cfg.pruning_ref_mature_only else "0",
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
    cfg.derived_cache_name = str(defaults.derived_cache_name)
    cfg.derived_cache_version = str(defaults.derived_cache_version)
    cfg.extra_eval_splits = tuple(defaults.extra_eval_splits)
    cfg.attention_mode = str(defaults.attention_mode)
    cfg.dynamic_scope = str(defaults.dynamic_scope)
    cfg.initial_connectivity_attn = float(defaults.initial_connectivity_attn)
    cfg.initial_connectivity_ffn = float(defaults.initial_connectivity_ffn)
    cfg.max_new_synapses = int(defaults.max_new_synapses)
    cfg.max_new_synapses_cap = int(defaults.max_new_synapses_cap)
    cfg.creation_score_normalization = str(defaults.creation_score_normalization)
    cfg.creation_score_alpha = float(defaults.creation_score_alpha)
    cfg.creation_active_ref_quantile = float(defaults.creation_active_ref_quantile)
    cfg.creation_priority_topk = int(defaults.creation_priority_topk)
    cfg.creation_ref_mature_only = bool(defaults.creation_ref_mature_only)
    cfg.pruning_policy = str(defaults.pruning_policy)
    cfg.pruning_active_ref_quantile = float(defaults.pruning_active_ref_quantile)
    cfg.pruning_priority_topk = int(defaults.pruning_priority_topk)
    cfg.pruning_ref_mature_only = bool(defaults.pruning_ref_mature_only)
    cfg.retention_eval_frequency = int(defaults.retention_eval_frequency)
    cfg.pruning_frequency = int(defaults.pruning_frequency)
    cfg.compile_dense_enabled = bool(defaults.compile_dense_enabled)
    cfg.amp_dense_enabled = bool(defaults.amp_dense_enabled)
    cfg.amp_dense_dtype = str(defaults.amp_dense_dtype)
    cfg.skip_snapshot_eval_epoch1 = bool(defaults.skip_snapshot_eval_epoch1)
    cfg.tier_train_examples_per_tier = int(defaults.tier_train_examples_per_tier)
    cfg.tier_eval_examples_per_tier = int(defaults.tier_eval_examples_per_tier)
    cfg.lr_sgdr_prev_floor = float(defaults.lr_sgdr_prev_floor)
    cfg.lr_sgdr_cycles = [tuple(cycle) for cycle in defaults.lr_sgdr_cycles]
    cfg.lr_schedule_maturity_epoch_override = 0
    cfg.save_best_checkpoint = True
    cfg.save_last_checkpoint = True
    return cfg
