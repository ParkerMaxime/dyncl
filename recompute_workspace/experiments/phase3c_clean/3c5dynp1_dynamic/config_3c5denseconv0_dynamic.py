#!/usr/bin/env python3
"""Conventional dense primary baseline aligned to the 3C5dynp1b data/model budget."""

from __future__ import annotations

from dataclasses import dataclass

from config_3c5denseiso0_dynamic import (
    RunConfig as BaseRunConfig,
    apply_experiment_overrides as _base_apply_experiment_overrides,
    default_config as _base_default_config,
    to_engine_argv as _base_to_engine_argv,
)


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5denseconv0_primaire750_scratchpad_v2_adamw_cosine_fixeduniform"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5denseconv0_primaire750_scratchpad_v2_adamw_cosine_fixeduniform_dm256_l2_ff1024_spe16k"
    )
    optimizer_name: str = "adamw"
    adam_weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    lr_warmup_epochs: int = 10
    lr_schedule_min_ratio: float = 0.05
    adaptive_module_mix_enabled: bool = False
    primary_init_mix_override: dict[str, float] | None = None


def default_config() -> RunConfig:
    base = _base_default_config()
    defaults = RunConfig(
        primary_init_mix_override={
            "arithmetic__add_or_sub": 0.25,
            "arithmetic__add_sub_multiple": 0.25,
            "arithmetic__mul": 0.25,
            "arithmetic__div": 0.25,
        }
    )
    return RunConfig(
        **{
            **base.__dict__,
            **defaults.__dict__,
        }
    )


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dense_primary_conventional_adamw_cosine_fixeduniform | "
        "runtime_engine=clean_shared_with_dense_static_ffn | "
        "dataset=3c5p21_scratchpad_v2_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return _base_to_engine_argv(cfg) + [
        "--optimizer-name", str(cfg.optimizer_name),
        "--adam-weight-decay", str(cfg.adam_weight_decay),
        "--lr-schedule", str(cfg.lr_schedule),
        "--lr-warmup-epochs", str(cfg.lr_warmup_epochs),
        "--lr-schedule-min-ratio", str(cfg.lr_schedule_min_ratio),
    ]


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg = _base_apply_experiment_overrides(cfg, defaults)
    cfg.exp_id = str(defaults.exp_id)
    cfg.results_dir = str(defaults.results_dir)
    cfg.optimizer_name = str(defaults.optimizer_name)
    cfg.adam_weight_decay = float(defaults.adam_weight_decay)
    cfg.lr_schedule = str(defaults.lr_schedule)
    cfg.lr_warmup_epochs = int(defaults.lr_warmup_epochs)
    cfg.lr_schedule_min_ratio = float(defaults.lr_schedule_min_ratio)
    cfg.adaptive_module_mix_enabled = bool(defaults.adaptive_module_mix_enabled)
    cfg.primary_init_mix_override = dict(defaults.primary_init_mix_override or {})
    return cfg
