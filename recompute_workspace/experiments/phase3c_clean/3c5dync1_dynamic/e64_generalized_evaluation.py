#!/usr/bin/env python3
"""Patch-1 scaffold for E64 generalized evaluation.

Scope of this first patch:
- manifest-driven checkpoint orchestration
- model / dataset loading reused from the existing E56 harness
- answer-level metrics only
- stable E64 JSON layout with step-level placeholders set to null
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
_PHASE3C_CLEAN_ROOT = _PROJECT_ROOT / "experiments" / "phase3c_clean"
_RUNTIME_ENGINE_ROOT = _PHASE3C_CLEAN_ROOT / "runtime_engine"
_P21_ROOT = _PHASE3C_CLEAN_ROOT / "3c5p21_clean_dense"
_DYNP1_ROOT = _PHASE3C_CLEAN_ROOT / "3c5dynp1_dynamic"


def _prepend(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


for _path in (
    _PROJECT_ROOT,
    _PROJECT_ROOT / "src",
    _PHASE3C_CLEAN_ROOT,
    _RUNTIME_ENGINE_ROOT,
    _SCRIPT_DIR,
    _P21_ROOT,
    _DYNP1_ROOT,
):
    if _path.exists():
        _prepend(_path)


def _load_module(alias: str, module_file: Path):
    spec = importlib.util.spec_from_file_location(alias, module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {alias}: {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


from config import build_config, parse_args
from loaders import build_loader
from results import collect_model_stats, load_training_checkpoint
from runtime import autocast_context, build_decoder_input, generate_autoregressive, get_device, resolve_amp_runtime, unwrap_compiled_model
from seed_run import _optional_model_kwargs

from e64_step_metrics import (
    answer_em,
    answer_token_acc,
    empty_module_record,
    is_fallback_target,
    is_scratchpad_module,
    level_for_module,
    parse_steps,
    score_scratchpad_pair,
)


LEVEL_ORDER = ("primaire", "college", "lycee", "superieur")
GROWTH_TARGET_PRIMAIRE_DEFAULT = 0.80
E64_EVAL_ONLY_LEVEL = "__e64_eval__"


@dataclass(frozen=True)
class RuntimeFamily:
    name: str
    config_file: Path
    dataset_pipeline_file: Path
    eval_pipeline_file: Path
    model_factory_file: Path


RUNTIME_FAMILIES: dict[str, RuntimeFamily] = {
    "dynamic_primary": RuntimeFamily(
        name="dynamic_primary",
        config_file=_DYNP1_ROOT / "config_3c5dynp1_dynamic.py",
        dataset_pipeline_file=_P21_ROOT / "dataset_pipeline.py",
        eval_pipeline_file=_P21_ROOT / "eval_pipeline.py",
        model_factory_file=_DYNP1_ROOT / "model_factory.py",
    ),
    "dynamic_primary_no_prune": RuntimeFamily(
        name="dynamic_primary_no_prune",
        config_file=_DYNP1_ROOT / "config_3c5dynp1noprun1_dynamic.py",
        dataset_pipeline_file=_P21_ROOT / "dataset_pipeline.py",
        eval_pipeline_file=_P21_ROOT / "eval_pipeline.py",
        model_factory_file=_DYNP1_ROOT / "model_factory.py",
    ),
    "dynamic_middle": RuntimeFamily(
        name="dynamic_middle",
        config_file=_SCRIPT_DIR / "config_3c5dync_baseline_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dynamic_high": RuntimeFamily(
        name="dynamic_high",
        config_file=_SCRIPT_DIR / "config_3c5dynl_baseline_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dynamic_high_historical": RuntimeFamily(
        name="dynamic_high_historical",
        config_file=_SCRIPT_DIR / "config_3c5dync1_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dynamic_joint": RuntimeFamily(
        name="dynamic_joint",
        config_file=_SCRIPT_DIR / "config_3c5dynjoint0_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dense_primary": RuntimeFamily(
        name="dense_primary",
        config_file=_DYNP1_ROOT / "config_3c5denseconv0_dynamic.py",
        dataset_pipeline_file=_P21_ROOT / "dataset_pipeline.py",
        eval_pipeline_file=_P21_ROOT / "eval_pipeline.py",
        model_factory_file=_DYNP1_ROOT / "model_factory.py",
    ),
    "dense_middle": RuntimeFamily(
        name="dense_middle",
        config_file=_SCRIPT_DIR / "config_3c5densec1_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dense_high": RuntimeFamily(
        name="dense_high",
        config_file=_SCRIPT_DIR / "config_3c5densel0_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dense_ewc_middle": RuntimeFamily(
        name="dense_ewc_middle",
        config_file=_SCRIPT_DIR / "config_3c5denseewc1_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dense_ewc_high": RuntimeFamily(
        name="dense_ewc_high",
        config_file=_SCRIPT_DIR / "config_3c5denseewcl0_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
    "dense_joint": RuntimeFamily(
        name="dense_joint",
        config_file=_SCRIPT_DIR / "config_3c5densejoint2_dynamic.py",
        dataset_pipeline_file=_SCRIPT_DIR / "dataset_pipeline.py",
        eval_pipeline_file=_SCRIPT_DIR / "eval_pipeline.py",
        model_factory_file=_SCRIPT_DIR / "model_factory.py",
    ),
}


@dataclass(frozen=True)
class ManifestEntry:
    model_id: str
    paper_label: str
    seed: int
    checkpoint_path: str
    run_family: str
    level_scope: str
    internal_run_name: str | None = None
    checkpoint_kind: str | None = None
    config_file: str | None = None
    dataset_pipeline_file: str | None = None
    eval_pipeline_file: str | None = None
    model_factory_file: str | None = None


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Patch-1 scaffold for E64 generalized evaluation")
    parser.add_argument("--manifest", required=True, help="JSON list of checkpoint entries")
    parser.add_argument(
        "--output-dir",
        default="results/phase3/_unsorted/analyses/e64_generalized_evaluation_patch1",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--eval-cap-per-module",
        type=int,
        default=0,
        help="Cap eval examples per module for smoke/debug. 0 means full pool.",
    )
    parser.add_argument(
        "--resume-skip-completed",
        type=int,
        default=1,
        help="Skip model_ids that already have a final eval_full_<model>.json in output-dir.",
    )
    parser.add_argument(
        "--tf-step-only",
        action="store_true",
        help="Compute only teacher-forcing answer metrics + TF step-level metrics; skip AR sample collection.",
    )
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> torch.device:
    raw = str(requested).strip().lower()
    if raw == "cpu":
        return get_device("cpu")
    if raw == "cuda":
        return get_device("cuda")
    return get_device("cuda" if torch.cuda.is_available() else "cpu")


def _build_cfg(config_mod, *, results_dir: str):
    defaults = config_mod.default_config()
    engine_args = parse_args(
        config_mod.to_engine_argv(defaults),
        level_order=LEVEL_ORDER,
        growth_target_primaire_default=GROWTH_TARGET_PRIMAIRE_DEFAULT,
    )
    cfg = config_mod.apply_experiment_overrides(
        build_config(
            engine_args,
            level_order=LEVEL_ORDER,
            default_results_dir=results_dir,
        ),
        defaults,
    )
    cfg.results_dir = str(results_dir)
    return cfg


def _configure_eval_runtime(cfg, device: torch.device) -> None:
    cfg.device = "cuda" if str(device).startswith("cuda") else "cpu"
    if cfg.device == "cpu":
        cfg.num_workers = 0
        cfg.pin_memory = False
        cfg.persistent_workers = False
        cfg.prefetch_factor = 0
    else:
        cfg.pin_memory = bool(getattr(cfg, "pin_memory", True))


def _build_model(model_factory_mod, cfg, train_dataset, device: torch.device):
    model_ctor = model_factory_mod.model_class()
    model = model_ctor(
        vocab_size=int(train_dataset.vocab_size),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_enc_layers=cfg.n_enc_layers,
        n_dec_layers=cfg.n_dec_layers,
        max_src_len=cfg.max_q_len,
        max_tgt_len=cfg.max_a_len,
        pad_id=int(train_dataset.pad_id),
        initial_connectivity_attn=cfg.initial_connectivity_attn,
        initial_connectivity_ffn=cfg.initial_connectivity_ffn,
        dropout=cfg.dropout,
        ffn_activation=cfg.ffn_activation,
        energy_budget=cfg.energy_budget,
        synapse_creation_cost=0.0,
        synapse_maintenance_cost=0.001,
        hebbian_threshold=cfg.hebbian_threshold,
        pruning_utility_threshold=0.01,
        protection_period=cfg.protection_period,
        tie_embeddings=cfg.tie_embeddings,
        creation_score_mode=cfg.creation_score_mode,
        creation_score_normalization=cfg.creation_score_normalization,
        creation_score_alpha=cfg.creation_score_alpha,
        creation_policy=cfg.creation_policy,
        creation_floor_per_connection=cfg.creation_floor_per_connection,
        creation_active_ref_quantile=cfg.creation_active_ref_quantile,
        creation_gap_rel_margin=cfg.creation_gap_rel_margin,
        creation_gap_abs_margin=cfg.creation_gap_abs_margin,
        creation_priority_topk=cfg.creation_priority_topk,
        creation_ref_mature_only=cfg.creation_ref_mature_only,
        creation_group_min_share=cfg.creation_group_min_share,
        creation_group_mix=cfg.creation_group_mix,
        creation_group_prior_enc_share=cfg.creation_group_prior_enc_share,
        creation_group_share_ema_alpha=cfg.creation_group_share_ema_alpha,
        creation_selection_weight=cfg.creation_selection_weight,
        creation_auto_budget_from_role_need=getattr(cfg, "creation_auto_budget_from_role_need", False),
        creation_role_need_ema_alpha=getattr(cfg, "creation_role_need_ema_alpha", 0.20),
        creation_auto_budget_min_ratio=getattr(cfg, "creation_auto_budget_min_ratio", 0.25),
        ffn_channel_complete_k=cfg.ffn_channel_complete_k,
        pruning_policy=cfg.pruning_policy,
        pruning_active_ref_quantile=cfg.pruning_active_ref_quantile,
        pruning_priority_topk=cfg.pruning_priority_topk,
        pruning_ref_mature_only=cfg.pruning_ref_mature_only,
        pruning_couple_creation_need=cfg.pruning_couple_creation_need,
        pruning_creation_need_ema_alpha=cfg.pruning_creation_need_ema_alpha,
        pruning_creation_coupling_alpha=cfg.pruning_creation_coupling_alpha,
        tenure_enabled=getattr(cfg, "tenure_enabled", False),
        tenure_score_ema_alpha=getattr(cfg, "tenure_score_ema_alpha", 0.10),
        tenure_min_age=getattr(cfg, "tenure_min_age", 30),
        tenure_acquire_quantile=getattr(cfg, "tenure_acquire_quantile", 0.80),
        tenure_acquire_streak=getattr(cfg, "tenure_acquire_streak", 20),
        tenure_max_share_per_matrix=getattr(cfg, "tenure_max_share_per_matrix", 0.20),
        tenure_decay_on_miss=getattr(cfg, "tenure_decay_on_miss", 1),
        tenure_release_enabled=getattr(cfg, "tenure_release_enabled", False),
        **_optional_model_kwargs(cfg),
    ).to(device)
    return model


def _load_checkpoint_state_strict_but_tenure_compatible(model, state_dict: Mapping[str, Any]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "tenure_mask_",
        "tenure_score_ema_",
        "tenure_streak_",
    )
    disallowed_missing = [
        str(key)
        for key in missing
        if not any(str(key).startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected:
        details: list[str] = []
        if disallowed_missing:
            details.append("missing=" + ", ".join(disallowed_missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(str(key) for key in unexpected))
        raise RuntimeError("Checkpoint incompatibility after tenure-compatible load: " + " | ".join(details))


def _subset_loader(dataset, indices: Sequence[int], cfg):
    return build_loader(
        dataset,
        np.asarray(indices, dtype=np.int64),
        int(cfg.batch_size),
        False,
        int(cfg.num_workers),
        bool(getattr(cfg, "pin_memory", False)),
        bool(getattr(cfg, "persistent_workers", False)),
        int(getattr(cfg, "prefetch_factor", 0)),
    )


def _evaluate_indices(
    eval_pipeline_mod,
    *,
    model,
    dataset,
    indices: Sequence[int],
    cfg,
    device: torch.device,
    compute_autoregressive: bool,
    compute_teacher_forcing: bool,
) -> Dict[str, float]:
    loader = _subset_loader(dataset, indices, cfg)
    return eval_pipeline_mod.evaluate_loader_with_answer_metrics(
        model=model,
        loader=loader,
        device=device,
        pad_id=int(dataset.pad_id),
        sos_id=int(dataset.sos_id),
        eos_id=int(dataset.eos_id),
        vocab_size=int(dataset.vocab_size),
        max_a_len=int(cfg.max_a_len),
        compute_autoregressive=bool(compute_autoregressive),
        compute_teacher_forcing=bool(compute_teacher_forcing),
    )


@torch.no_grad()
def _collect_ar_outputs(eval_pipeline_mod, *, model, dataset, indices: Sequence[int], cfg, device: torch.device) -> list[dict[str, Any]]:
    loader = _subset_loader(dataset, indices, cfg)
    eval_model = unwrap_compiled_model(model)
    amp_active, amp_dtype = resolve_amp_runtime(model, None, None)
    inv_vocab = {int(v): str(k) for k, v in dataset.vocab.items()}
    out: list[dict[str, Any]] = []
    for q_indices, a_indices, sample_indices in loader:
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        preds = generate_autoregressive(
            model=eval_model,
            src_tokens=q_indices,
            sos_id=int(dataset.sos_id),
            eos_id=int(dataset.eos_id),
            max_a_len=int(cfg.max_a_len),
            amp_enabled=amp_active,
            amp_dtype=amp_dtype,
        )
        batch_size = int(sample_indices.shape[0])
        for row_idx in range(batch_size):
            raw_idx = int(sample_indices[row_idx].item())
            raw = dataset.get_raw_item(raw_idx)
            target_text = eval_pipeline_mod.decode_token_row(
                a_indices[row_idx].detach().cpu().tolist(),
                inv_vocab,
                pad_id=int(dataset.pad_id),
                sos_id=int(dataset.sos_id),
                eos_id=int(dataset.eos_id),
            )
            ar_text = eval_pipeline_mod.decode_token_row(
                preds[row_idx].detach().cpu().tolist(),
                inv_vocab,
                pad_id=int(dataset.pad_id),
                sos_id=int(dataset.sos_id),
                eos_id=int(dataset.eos_id),
            )
            out.append(
                {
                    "idx": raw_idx,
                    "module_name": str(raw.get("module_name", "")),
                    "question": str(raw.get("question", "")),
                    "target_text": target_text,
                    "ar_text": ar_text,
                }
            )
    return out


def _decode_tf_text_from_gold_segment_alignment(
    *,
    target_tokens: Sequence[int],
    pred_tokens: Sequence[int],
    dataset,
) -> str:
    """Decode TF scratchpad steps from gold-aligned token positions.

    This is the paper-compatible semantics: each predicted token is scored at the
    gold target position under teacher forcing, and step boundaries are taken
    from the gold `|` delimiters rather than from a free decoding rollout.
    """
    inv_vocab = {int(v): str(k) for k, v in dataset.vocab.items()}
    pad_id = int(dataset.pad_id)
    sos_id = int(dataset.sos_id)
    eos_id = int(dataset.eos_id)
    pipe_id = int(dataset.vocab.get("|", -1))
    specials = {pad_id, sos_id, eos_id}

    aligned: list[tuple[int, int]] = []
    for idx, tgt_tok in enumerate(target_tokens):
        tgt = int(tgt_tok)
        if tgt in specials:
            continue
        pred = int(pred_tokens[idx]) if idx < len(pred_tokens) else eos_id
        aligned.append((tgt, pred))

    segments: list[str] = []
    current_chars: list[str] = []
    for tgt_tok, pred_tok in aligned:
        if tgt_tok == pipe_id:
            segment = "".join(current_chars).strip()
            if segment:
                segments.append(segment)
            current_chars = []
            continue
        if pred_tok in specials:
            continue
        current_chars.append(str(inv_vocab.get(pred_tok, "")))

    tail = "".join(current_chars).strip()
    if tail:
        segments.append(tail)
    return " | ".join(segments).strip()


@torch.no_grad()
def _collect_tf_segment_outputs(*, model, dataset, indices: Sequence[int], cfg, device: torch.device, eval_pipeline_mod) -> list[dict[str, Any]]:
    loader = _subset_loader(dataset, indices, cfg)
    amp_active, amp_dtype = resolve_amp_runtime(model, None, None)
    out: list[dict[str, Any]] = []
    for q_indices, a_indices, sample_indices in loader:
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        tgt_in = build_decoder_input(a_indices, sos_id=int(dataset.sos_id))
        eval_model = unwrap_compiled_model(model)
        with autocast_context(device, amp_active, amp_dtype):
            logits = eval_model(q_indices, tgt_in)
        tf_preds = logits.argmax(dim=-1)
        inv_vocab = {int(v): str(k) for k, v in dataset.vocab.items()}
        batch_size = int(sample_indices.shape[0])
        for row_idx in range(batch_size):
            raw_idx = int(sample_indices[row_idx].item())
            raw = dataset.get_raw_item(raw_idx)
            module_name = str(raw.get("module_name", ""))
            target_row = a_indices[row_idx].detach().cpu().tolist()
            tf_pred_row = tf_preds[row_idx].detach().cpu().tolist()
            target_text = eval_pipeline_mod.decode_token_row(
                target_row,
                inv_vocab,
                pad_id=int(dataset.pad_id),
                sos_id=int(dataset.sos_id),
                eos_id=int(dataset.eos_id),
            )
            tf_full_text = "".join(
                str(inv_vocab.get(int(tok), ""))
                for tok in tf_pred_row
                if int(tok) not in (int(dataset.pad_id), int(dataset.sos_id), int(dataset.eos_id))
            )
            tf_text = tf_full_text
            if is_scratchpad_module(module_name) and not is_fallback_target(target_text):
                tf_text = _decode_tf_text_from_gold_segment_alignment(
                    target_tokens=target_row,
                    pred_tokens=tf_pred_row,
                    dataset=dataset,
                )
            out.append(
                {
                    "idx": raw_idx,
                    "module_name": module_name,
                    "target_text": target_text,
                    "tf_text": tf_text,
                    "tf_full_text": tf_full_text,
                }
            )
    return out


def _resolve_runtime_family(entry: ManifestEntry) -> RuntimeFamily:
    if entry.config_file and entry.dataset_pipeline_file and entry.eval_pipeline_file and entry.model_factory_file:
        return RuntimeFamily(
            name=f"explicit::{entry.run_family}",
            config_file=(_PROJECT_ROOT / entry.config_file).resolve(),
            dataset_pipeline_file=(_PROJECT_ROOT / entry.dataset_pipeline_file).resolve(),
            eval_pipeline_file=(_PROJECT_ROOT / entry.eval_pipeline_file).resolve(),
            model_factory_file=(_PROJECT_ROOT / entry.model_factory_file).resolve(),
        )
    if entry.run_family not in RUNTIME_FAMILIES:
        raise KeyError(f"Unknown run_family={entry.run_family!r}. Available: {sorted(RUNTIME_FAMILIES)}")
    return RUNTIME_FAMILIES[entry.run_family]


def _parse_manifest(path: Path) -> list[ManifestEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Manifest must be a JSON list")
    out: list[ManifestEntry] = []
    for raw in payload:
        out.append(ManifestEntry(**raw))
    return out


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    rel = Path(str(checkpoint_path))
    candidates = [(_PROJECT_ROOT / rel).resolve()]

    rel_str = str(rel).replace("\\", "/")
    prefix = "results/phase3/3c/3c5/"
    if rel_str.startswith(prefix):
        suffix = rel_str[len(prefix) :]
        candidates.append(
            (
                _PROJECT_ROOT
                / "results/phase3/_unsorted/runs/cloud_pulls/phase3/3c/3c5"
                / suffix
            ).resolve()
        )
    candidates.append((_PROJECT_ROOT / "results/phase3/_unsorted/runs/cloud_pulls" / rel).resolve())

    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        raw = str(candidate)
        if raw in seen:
            continue
        seen.add(raw)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(candidate) for candidate in deduped)
    raise FileNotFoundError(f"Checkpoint introuvable. Tried:\n{tried}")


def _selected_levels_for_scope(level_scope: str) -> tuple[str, ...]:
    scope = str(level_scope).strip().lower()
    if scope == "primary_final":
        return ("primaire",)
    if scope in {"primary_ablation"}:
        return ("primaire",)
    if scope == "middle_school_final":
        return ("primaire", "college")
    if scope == "high_school_final":
        return ("primaire", "college", "lycee")
    if scope == "joint_full":
        return ("primaire", "college", "lycee")
    return ("primaire", "college", "lycee")


def _build_bundle_compat(dataset_pipeline_mod, cfg, *, selected_levels: Sequence[str]):
    build_bundle = getattr(dataset_pipeline_mod, "build_bundle")
    try:
        params = set(inspect.signature(build_bundle).parameters.keys())
    except (TypeError, ValueError):
        params = {"cfg", "progress_callback"}

    if "selected_levels_override" in params:
        return build_bundle(
            cfg,
            selected_levels_override=selected_levels,
            current_level_override=E64_EVAL_ONLY_LEVEL,
            eval_only=True,
        )
    return build_bundle(cfg)


def _prepare_entry_light(entry: ManifestEntry) -> dict[str, Any]:
    family = _resolve_runtime_family(entry)
    return {
        "entry": entry,
        "family": family,
        "checkpoint_path": str(entry.checkpoint_path),
    }


def _prepare_entry(entry: ManifestEntry, output_root: Path, device: torch.device) -> dict[str, Any]:
    family = _resolve_runtime_family(entry)
    print(f"[E64] prepare {entry.model_id} runtime={family.name}", flush=True)
    config_mod = _load_module(f"e64_config_{entry.model_id}", family.config_file)
    dataset_pipeline_mod = _load_module(f"e64_dataset_{entry.model_id}", family.dataset_pipeline_file)
    eval_pipeline_mod = _load_module(f"e64_eval_{entry.model_id}", family.eval_pipeline_file)
    model_factory_mod = _load_module(f"e64_model_{entry.model_id}", family.model_factory_file)
    cfg = _build_cfg(config_mod, results_dir=str(output_root / f"_tmp_{entry.model_id}"))
    _configure_eval_runtime(cfg, device)
    cfg.retention_train_cap_per_module = 1
    selected_levels = _selected_levels_for_scope(entry.level_scope)
    print(f"[E64] build_bundle {entry.model_id}", flush=True)
    bundle = _build_bundle_compat(
        dataset_pipeline_mod,
        cfg,
        selected_levels=selected_levels,
    )
    checkpoint_resolved = _resolve_checkpoint_path(entry.checkpoint_path)
    print(f"[E64] load_checkpoint {entry.model_id} path={checkpoint_resolved}", flush=True)
    checkpoint = load_training_checkpoint(checkpoint_resolved)
    model = _build_model(model_factory_mod, cfg, bundle.train_dataset, device)
    _load_checkpoint_state_strict_but_tenure_compatible(model, checkpoint["model_state_dict"])
    model.eval()
    return {
        "entry": entry,
        "family": family,
        "cfg": cfg,
        "bundle": bundle,
        "eval_pipeline": eval_pipeline_mod,
        "model": model,
        "checkpoint_path": str(checkpoint_resolved),
    }


def _module_record_from_metrics(module_name: str, metrics: Mapping[str, Any], raw_answers: Sequence[str]) -> dict[str, Any]:
    record = empty_module_record(module_name, n_eligible=int(metrics.get("n_samples", 0)))
    record["n_fallback_targets"] = int(sum(1 for answer in raw_answers if is_fallback_target(answer)))
    record["ar"]["answer_em"] = float(metrics.get("autoregressive_accuracy", metrics.get("eval_seq_ans", 0.0)))
    record["ar"]["answer_token_acc"] = float(
        metrics.get("autoregressive_token_accuracy", metrics.get("eval_tok_ans", metrics.get("token_accuracy", 0.0)))
    )
    record["tf"]["answer_em"] = float(metrics.get("teacher_forcing_accuracy", 0.0))
    record["tf"]["answer_token_acc"] = float(metrics.get("teacher_forcing_token_accuracy", 0.0))
    return record


def _mean_or_none(values: list[float | None]) -> float | None:
    keep = [float(v) for v in values if v is not None]
    if not keep:
        return None
    return mean(keep)


def _augment_record_with_ar_samples(record: dict[str, Any], module_name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return record
    ar_answer_vals: list[float] = []
    ar_token_vals: list[float] = []
    concept_micro_vals: list[float | None] = []
    concept_exact_vals: list[float | None] = []
    numeric_micro_vals: list[float | None] = []
    numeric_exact_vals: list[float | None] = []
    lhs_vals: list[float | None] = []
    tail_vals: list[float | None] = []
    bypass_vals: list[float | None] = []
    fallback_count = 0
    parse_fail_ar = 0
    for row in rows:
        target_text = str(row["target_text"])
        ar_text = str(row["ar_text"])
        ar_answer_vals.append(answer_em(ar_text, target_text))
        ar_token_vals.append(answer_token_acc(ar_text, target_text))
        if is_fallback_target(target_text):
            fallback_count += 1
            continue
        parsed_target = parse_steps(module_name, target_text)
        parsed_pred = parse_steps(module_name, ar_text)
        if parsed_pred.parse_status != "parsed":
            parse_fail_ar += 1
        scores = score_scratchpad_pair(parsed_target, parsed_pred)
        concept_micro_vals.append(scores["concept_step_micro"])
        concept_exact_vals.append(scores["concept_step_exact"])
        numeric_micro_vals.append(scores["numeric_step_micro"])
        numeric_exact_vals.append(scores["numeric_step_exact"])
        lhs_vals.append(scores["numeric_lhs_match"])
        tail_vals.append(scores["numeric_tail_match"])
        bypass_vals.append(scores["bypass_rate"])
    record["n_fallback_targets"] = fallback_count
    record["n_parse_fail_ar"] = parse_fail_ar
    record["ar"]["answer_em"] = mean(ar_answer_vals) if ar_answer_vals else 0.0
    record["ar"]["answer_token_acc"] = mean(ar_token_vals) if ar_token_vals else 0.0
    record["ar"]["concept_step_micro"] = _mean_or_none(concept_micro_vals)
    record["ar"]["concept_step_exact"] = _mean_or_none(concept_exact_vals)
    record["ar"]["numeric_step_micro"] = _mean_or_none(numeric_micro_vals)
    record["ar"]["numeric_step_exact"] = _mean_or_none(numeric_exact_vals)
    record["ar"]["numeric_lhs_match"] = _mean_or_none(lhs_vals)
    record["ar"]["numeric_tail_match"] = _mean_or_none(tail_vals)
    record["ar"]["bypass_rate"] = _mean_or_none(bypass_vals)
    return record


def _augment_record_with_tf_segment_samples(record: dict[str, Any], module_name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return record
    concept_micro_vals: list[float | None] = []
    concept_exact_vals: list[float | None] = []
    numeric_micro_vals: list[float | None] = []
    numeric_exact_vals: list[float | None] = []
    lhs_vals: list[float | None] = []
    tail_vals: list[float | None] = []
    bypass_vals: list[float | None] = []
    parse_fail_tf = 0
    for row in rows:
        target_text = str(row["target_text"])
        tf_text = str(row["tf_text"])
        if is_fallback_target(target_text):
            continue
        parsed_target = parse_steps(module_name, target_text)
        parsed_pred = parse_steps(module_name, tf_text)
        if parsed_pred.parse_status != "parsed":
            parse_fail_tf += 1
        scores = score_scratchpad_pair(parsed_target, parsed_pred)
        concept_micro_vals.append(scores["concept_step_micro"])
        concept_exact_vals.append(scores["concept_step_exact"])
        numeric_micro_vals.append(scores["numeric_step_micro"])
        numeric_exact_vals.append(scores["numeric_step_exact"])
        lhs_vals.append(scores["numeric_lhs_match"])
        tail_vals.append(scores["numeric_tail_match"])
        bypass_vals.append(scores["bypass_rate"])
    record["n_parse_fail_tf"] = parse_fail_tf
    record["tf"]["concept_step_micro"] = _mean_or_none(concept_micro_vals)
    record["tf"]["concept_step_exact"] = _mean_or_none(concept_exact_vals)
    record["tf"]["numeric_step_micro"] = _mean_or_none(numeric_micro_vals)
    record["tf"]["numeric_step_exact"] = _mean_or_none(numeric_exact_vals)
    record["tf"]["numeric_lhs_match"] = _mean_or_none(lhs_vals)
    record["tf"]["numeric_tail_match"] = _mean_or_none(tail_vals)
    record["tf"]["bypass_rate"] = _mean_or_none(bypass_vals)
    return record


def _balanced_level_stub() -> dict[str, Any]:
    return {
        "ar_answer_em": None,
        "tf_answer_em": None,
        "ar_concept_step_micro": None,
        "tf_concept_step_micro": None,
        "ar_numeric_step_micro": None,
        "tf_numeric_step_micro": None,
    }


def _aggregate_level_metrics(per_module: Mapping[str, Mapping[str, Any]], level_name: str) -> dict[str, Any]:
    selected = [metrics for module_name, metrics in per_module.items() if level_for_module(module_name) == level_name]
    out = _balanced_level_stub()
    if not selected:
        return out
    out["ar_answer_em"] = mean(float(item["ar"]["answer_em"]) for item in selected)
    out["tf_answer_em"] = mean(float(item["tf"]["answer_em"]) for item in selected)
    out["ar_concept_step_micro"] = _mean_or_none([item["ar"]["concept_step_micro"] for item in selected])
    out["tf_concept_step_micro"] = _mean_or_none([item["tf"]["concept_step_micro"] for item in selected])
    out["ar_numeric_step_micro"] = _mean_or_none([item["ar"]["numeric_step_micro"] for item in selected])
    out["tf_numeric_step_micro"] = _mean_or_none([item["tf"]["numeric_step_micro"] for item in selected])
    return out


def _aggregate_global_333(balanced_by_level: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out = _balanced_level_stub()
    answer_ar = [balanced_by_level[key]["ar_answer_em"] for key in ("primary", "middle_school", "high_school") if balanced_by_level[key]["ar_answer_em"] is not None]
    answer_tf = [balanced_by_level[key]["tf_answer_em"] for key in ("primary", "middle_school", "high_school") if balanced_by_level[key]["tf_answer_em"] is not None]
    out["ar_answer_em"] = mean(answer_ar) if answer_ar else None
    out["tf_answer_em"] = mean(answer_tf) if answer_tf else None
    out["ar_concept_step_micro"] = _mean_or_none([balanced_by_level[key]["ar_concept_step_micro"] for key in ("primary", "middle_school", "high_school")])
    out["tf_concept_step_micro"] = _mean_or_none([balanced_by_level[key]["tf_concept_step_micro"] for key in ("primary", "middle_school", "high_school")])
    out["ar_numeric_step_micro"] = _mean_or_none([balanced_by_level[key]["ar_numeric_step_micro"] for key in ("primary", "middle_school", "high_school")])
    out["tf_numeric_step_micro"] = _mean_or_none([balanced_by_level[key]["tf_numeric_step_micro"] for key in ("primary", "middle_school", "high_school")])
    return out


def _seed_group_key(model_id: str) -> str:
    match = re.match(r"^(.*)_(\d+)$", str(model_id))
    if not match:
        return str(model_id)
    return str(match.group(1))


def _summary_for_models(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for model_id, payload in models.items():
        grouped.setdefault(_seed_group_key(model_id), []).append(payload)
    out: dict[str, Any] = {}
    for key, items in grouped.items():
        ar_vals = [item["balanced_global_333333"]["ar_answer_em"] for item in items if item["balanced_global_333333"]["ar_answer_em"] is not None]
        tf_vals = [item["balanced_global_333333"]["tf_answer_em"] for item in items if item["balanced_global_333333"]["tf_answer_em"] is not None]
        out[key] = {
            "n_models": len(items),
            "model_ids": [item["model_id"] for item in items],
            "ar_answer_em_mean": mean(ar_vals) if ar_vals else None,
            "ar_answer_em_std": pstdev(ar_vals) if len(ar_vals) > 1 else 0.0 if ar_vals else None,
            "tf_answer_em_mean": mean(tf_vals) if tf_vals else None,
            "tf_answer_em_std": pstdev(tf_vals) if len(tf_vals) > 1 else 0.0 if tf_vals else None,
        }
    return out


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _model_output_path(output_dir: Path, model_id: str) -> Path:
    return output_dir / f"eval_full_{model_id}.json"


def _model_progress_path(output_dir: Path, model_id: str) -> Path:
    return output_dir / f"eval_full_{model_id}.partial.json"


def _summary_output_path(output_dir: Path) -> Path:
    return output_dir / "eval_full_summary.json"


def _write_summary_checkpoint(
    *,
    output_dir: Path,
    summary_payload: Mapping[str, Any],
) -> None:
    summary_with_status = dict(summary_payload)
    summary_with_status["status"] = "partial"
    summary_with_status["completed_models"] = sorted(summary_payload.get("models", {}).keys())
    _atomic_write_json(_summary_output_path(output_dir), summary_with_status)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = _resolve_device(args.device)
    manifest_path = (_PROJECT_ROOT / args.manifest).resolve()
    output_dir = (_PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _parse_manifest(manifest_path)

    if args.prepare_only:
        prepared_light = [_prepare_entry_light(entry) for entry in manifest]
        payload = {
            "status": "prepared",
            "device": str(device),
            "manifest": str(manifest_path),
            "models": [
                {
                    "model_id": item["entry"].model_id,
                    "paper_label": item["entry"].paper_label,
                    "checkpoint_path": item["checkpoint_path"],
                    "bundle_root": None,
                    "runtime_family": item["family"].name,
                }
                for item in prepared_light
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    summary_payload: dict[str, Any] = {
        "analysis_id": "E64_generalized_evaluation_patch1",
        "eval_protocol": "v1_patch3_ar_tf_segment_steps_2026_05",
        "device": str(device),
        "manifest": str(manifest_path),
        "models": {},
    }

    for entry in manifest:
        final_model_path = _model_output_path(output_dir, entry.model_id)
        progress_model_path = _model_progress_path(output_dir, entry.model_id)
        if bool(int(args.resume_skip_completed)) and final_model_path.exists():
            print(f"[E64] skip completed model={entry.model_id} path={final_model_path}", flush=True)
            try:
                summary_payload["models"][entry.model_id] = json.loads(final_model_path.read_text(encoding="utf-8"))
                _write_summary_checkpoint(output_dir=output_dir, summary_payload=summary_payload)
            except Exception:
                pass
            continue

        item = _prepare_entry(entry, output_dir, device)
        entry: ManifestEntry = item["entry"]
        cfg = item["cfg"]
        bundle = item["bundle"]
        model = item["model"]
        eval_pipeline_mod = item["eval_pipeline"]
        dataset = bundle.test_dataset
        model_stats = collect_model_stats(model, include_hidden_stats=True)
        eval_cap_per_module = max(0, int(args.eval_cap_per_module))

        per_module: dict[str, Any] = {}
        model_payload = {
            "model_id": entry.model_id,
            "paper_label": entry.paper_label,
            "checkpoint_path": item["checkpoint_path"],
            "eval_protocol": "v1_patch4_tf_only_gold_aligned_steps_2026_05" if args.tf_step_only else "v1_patch4_ar_tf_gold_aligned_steps_2026_05",
            "status": "running",
            "completed_modules": [],
            "synapses": {
                "active_total": int(model_stats.get("active_synapses", 0)),
                "by_block": model_stats.get("by_matrix_active", {}),
            },
            "per_module": per_module,
            "balanced_by_level": {
                "primary": _balanced_level_stub(),
                "middle_school": _balanced_level_stub(),
                "high_school": _balanced_level_stub(),
            },
            "balanced_global_333333": _balanced_level_stub(),
        }
        _atomic_write_json(progress_model_path, model_payload)

        for module_name, indices in dict(bundle.eval_module_pools).items():
            pool = np.asarray(indices, dtype=np.int64)
            if pool.size <= 0:
                continue
            if eval_cap_per_module > 0 and pool.size > eval_cap_per_module:
                pool = pool[:eval_cap_per_module].astype(np.int64, copy=False)
            print(
                f"[E64] module start model={entry.model_id} module={module_name} n={int(pool.size)}",
                flush=True,
            )
            metrics = _evaluate_indices(
                eval_pipeline_mod,
                model=model,
                dataset=dataset,
                indices=pool,
                cfg=cfg,
                device=device,
                compute_autoregressive=not bool(args.tf_step_only),
                compute_teacher_forcing=True,
            )
            raw_answers = [str(dataset.get_raw_item(int(idx)).get("answer", "")) for idx in pool.tolist()]
            ar_rows: list[dict[str, Any]] = []
            if not bool(args.tf_step_only):
                ar_rows = _collect_ar_outputs(
                    eval_pipeline_mod,
                    model=model,
                    dataset=dataset,
                    indices=pool,
                    cfg=cfg,
                    device=device,
                )
            tf_rows = _collect_tf_segment_outputs(
                model=model,
                dataset=dataset,
                indices=pool,
                cfg=cfg,
                device=device,
                eval_pipeline_mod=eval_pipeline_mod,
            )
            record = _module_record_from_metrics(str(module_name), metrics, raw_answers)
            if ar_rows:
                record = _augment_record_with_ar_samples(record, str(module_name), ar_rows)
            per_module[str(module_name)] = _augment_record_with_tf_segment_samples(record, str(module_name), tf_rows)
            model_payload["completed_modules"] = sorted(per_module.keys())
            _atomic_write_json(progress_model_path, model_payload)
            print(
                f"[E64] module done model={entry.model_id} module={module_name}",
                flush=True,
            )

        balanced_by_level = {
            "primary": _aggregate_level_metrics(per_module, "primary"),
            "middle_school": _aggregate_level_metrics(per_module, "middle_school"),
            "high_school": _aggregate_level_metrics(per_module, "high_school"),
        }
        balanced_global = _aggregate_global_333(balanced_by_level)

        model_payload["status"] = "completed"
        model_payload["balanced_by_level"] = balanced_by_level
        model_payload["balanced_global_333333"] = balanced_global
        summary_payload["models"][entry.model_id] = model_payload
        _atomic_write_json(final_model_path, model_payload)
        progress_model_path.unlink(missing_ok=True)
        _write_summary_checkpoint(output_dir=output_dir, summary_payload=summary_payload)
        del model
        del bundle
        del item
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_payload["aggregates"] = _summary_for_models(summary_payload["models"])
    summary_payload["status"] = "completed"
    summary_payload["completed_models"] = sorted(summary_payload["models"].keys())
    summary_path = _summary_output_path(output_dir)
    _atomic_write_json(summary_path, summary_payload)
    print(f"E64_PATCH1_OK summary={summary_path} manifest={manifest_path} device={device}")


if __name__ == "__main__":
    main()
