#!/usr/bin/env python3
"""Dense iso-condition primary baseline aligned to 3C5dynp1b."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


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

try:
    from .config_3c5dynp1_dynamic import RunConfig as BaseRunConfig
    from .config_3c5dynp1_dynamic import apply_experiment_overrides as _base_apply_experiment_overrides
    from .config_3c5dynp1_dynamic import default_config as _base_default_config
    from .config_3c5dynp1_dynamic import to_engine_argv as _base_to_engine_argv
except ImportError:  # pragma: no cover
    from config_3c5dynp1_dynamic import RunConfig as BaseRunConfig
    from config_3c5dynp1_dynamic import apply_experiment_overrides as _base_apply_experiment_overrides
    from config_3c5dynp1_dynamic import default_config as _base_default_config
    from config_3c5dynp1_dynamic import to_engine_argv as _base_to_engine_argv


@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    exp_id: str = "3C5denseiso0_primaire750_scratchpad_v2_iso_dynp1b"
    results_dir: str = (
        "results/phase3/3c/3c5/"
        "3c5denseiso0_primaire750_scratchpad_v2_iso_dynp1b_dm256_l2_ff1024_spe16k"
    )
    initial_connectivity_ffn: float = 1.0
    max_new_synapses: int = 0
    max_new_synapses_cap: int = 0
    pruning_frequency: int = 100000
    enable_newsyn: bool = False
    enable_loss_masking: bool = False
    enable_replay_boost: bool = False
    enable_output_head_guard: bool = False
    triphasic_controller_enabled: bool = False
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
    base = _base_default_config()
    return RunConfig(
        **{
            **base.__dict__,
            **RunConfig().__dict__,
        }
    )


def source_of_truth_summary() -> str:
    return (
        "run_config=3c5_dense_primary_iso_dynp1b | "
        "runtime_engine=clean_shared_with_dense_static_ffn | "
        "dataset=3c5p21_scratchpad_v2_module_bundles"
    )


def to_engine_argv(cfg: RunConfig) -> list[str]:
    return _base_to_engine_argv(cfg)


def apply_experiment_overrides(cfg, defaults: RunConfig):
    cfg = _base_apply_experiment_overrides(cfg, defaults)
    cfg.exp_id = str(defaults.exp_id)
    cfg.enable_newsyn = bool(defaults.enable_newsyn)
    cfg.enable_loss_masking = bool(defaults.enable_loss_masking)
    cfg.enable_replay_boost = bool(defaults.enable_replay_boost)
    cfg.enable_output_head_guard = bool(defaults.enable_output_head_guard)
    cfg.triphasic_controller_enabled = bool(defaults.triphasic_controller_enabled)
    cfg.channel_creation_enabled = bool(defaults.channel_creation_enabled)
    cfg.phase1_registry_enabled = bool(defaults.phase1_registry_enabled)
    cfg.phase1_channel_creation_enabled = bool(defaults.phase1_channel_creation_enabled)
    cfg.phase1_pruning_enabled = bool(defaults.phase1_pruning_enabled)
    cfg.handoff_channel_creation = bool(defaults.handoff_channel_creation)
    cfg.handoff_pruning = bool(defaults.handoff_pruning)
    cfg.phase2_enabled = bool(defaults.phase2_enabled)
    cfg.phase2_micro_creation_enabled = bool(defaults.phase2_micro_creation_enabled)
    cfg.phase2_pruning_enabled = bool(defaults.phase2_pruning_enabled)
    cfg.phase3_micro_creation_enabled = bool(defaults.phase3_micro_creation_enabled)
    cfg.phase3_pruning_enabled = bool(defaults.phase3_pruning_enabled)
    cfg.local_reinforce_enabled = bool(defaults.local_reinforce_enabled)
    return cfg
