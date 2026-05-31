#!/usr/bin/env python3
"""Configuration dynp1 iso-condition avec pruning desactive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
_PHASE3C_CLEAN_ROOT = _SCRIPT_DIR.parent


def _prepend_sys_path(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


for _path in (_SCRIPT_DIR, _PHASE3C_CLEAN_ROOT):
    if _path.exists():
        _prepend_sys_path(_path)

from config_3c5dynp1_dynamic import (
    RunConfig as BaseRunConfig,
    apply_experiment_overrides as _base_apply,
    default_config as _base_default_config,
    to_engine_argv as _base_to_engine_argv,
)


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5dynp1noprun1_primaire750_ffn30_budget400"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5dynp1noprun1_primaire750_ffn30_budget400_dm256_l2_ff1024_spe16k"
    )
    phase2_pruning_enabled: bool = False


def default_config() -> RunConfig:
    base = _base_default_config()
    return RunConfig(
        **{
            **base.__dict__,
            **{
                "exp_id": RunConfig.exp_id,
                "results_dir": RunConfig.results_dir,
                "phase2_pruning_enabled": RunConfig.phase2_pruning_enabled,
            },
        }
    )


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dynamic_primary_triphasic_phase2fix_noprunev1 | "
        "runtime_engine=clean_shared_no_prune_ablation | "
        "dataset=3c5p21_scratchpad_v2_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return _base_to_engine_argv(cfg)


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg = _base_apply(cfg, defaults)
    cfg.exp_id = str(defaults.exp_id)
    cfg.results_dir = str(defaults.results_dir)
    cfg.phase2_pruning_enabled = bool(defaults.phase2_pruning_enabled)
    return cfg
