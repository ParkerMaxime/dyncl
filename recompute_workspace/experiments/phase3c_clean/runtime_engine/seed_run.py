#!/usr/bin/env python3
"""Shared single-seed execution for clean phase 3C runners."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .backends import get_experiment_backend
    from .loaders import build_eval_loaders
    from .results import (
        collect_model_stats,
        load_training_checkpoint,
        save_training_checkpoint,
        validate_checkpoint_compatibility,
    )
    from .runtime import get_device
    from .train import (
        compute_full_param_ewc_loss,
        compute_full_param_fisher_information,
        save_full_param_ewc_reference,
    )
    from .training_utils import (
        build_consolidation_mix,
        build_epoch_context,
        compute_convergence,
        compute_epoch_lr,
        compute_growth_need,
        evaluate_go_nogo,
        infer_lr_phase,
        resolve_mix_tables,
        set_optimizer_lr,
        update_train_seq_budget_state,
        capture_active_masks,
    )
except ImportError:  # pragma: no cover - direct module import fallback
    from backends import get_experiment_backend
    from loaders import build_eval_loaders
    from results import (
        collect_model_stats,
        load_training_checkpoint,
        save_training_checkpoint,
        validate_checkpoint_compatibility,
    )
    from runtime import get_device
    from train import (
        compute_full_param_ewc_loss,
        compute_full_param_fisher_information,
        save_full_param_ewc_reference,
    )
    from training_utils import (
        build_consolidation_mix,
        build_epoch_context,
        compute_convergence,
        compute_epoch_lr,
        compute_growth_need,
        evaluate_go_nogo,
        infer_lr_phase,
        resolve_mix_tables,
        set_optimizer_lr,
        update_train_seq_budget_state,
        capture_active_masks,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_level_token_ids(
    train_dataset,
    level_name: str,
    special_ids: Sequence[int],
) -> list[int]:
    if hasattr(train_dataset, "unique_answer_tokens_for_level"):
        toks = np.asarray(train_dataset.unique_answer_tokens_for_level(str(level_name)), dtype=np.int64)
        special = set(int(x) for x in special_ids)
        return [int(t) for t in toks.tolist() if int(t) not in special]
    level_names = np.asarray(train_dataset.level_names, dtype=object)
    idx = np.where(level_names == level_name)[0]
    if idx.size == 0:
        return []
    toks = np.unique(train_dataset.encoded_answers[idx].reshape(-1))
    special = set(int(x) for x in special_ids)
    return [int(t) for t in toks.tolist() if int(t) not in special]


def _metric_accuracy(metric: Mapping[str, Any]) -> float:
    if "primary_mastery_balanced_modules" in metric:
        return float(metric["primary_mastery_balanced_modules"])
    for key in ("accuracy", "eval_seq_ans", "eval_seq_full", "autoregressive_accuracy"):
        if key in metric:
            return float(metric[key])
    return 0.0


def _metric_token_accuracy(metric: Mapping[str, Any]) -> float:
    for key in ("token_accuracy", "eval_tok_ans", "eval_tok_full", "autoregressive_token_accuracy"):
        if key in metric:
            return float(metric[key])
    return 0.0


def _metric_autoreg_accuracy(metric: Mapping[str, Any]) -> float:
    if "primary_mastery_balanced_modules" in metric:
        return float(metric["primary_mastery_balanced_modules"])
    return float(metric.get("autoregressive_accuracy", _metric_accuracy(metric)))


def _metric_teacher_forcing_accuracy(metric: Mapping[str, Any]) -> float:
    if "primary_mastery_balanced_modules_tf" in metric:
        return float(metric["primary_mastery_balanced_modules_tf"])
    return float(metric.get("teacher_forcing_accuracy", 0.0))


def _optional_model_kwargs(cfg) -> Dict[str, Any]:
    keys = (
        "creation_quality_mad_factor",
        "creation_coherence_threshold",
        "creation_need_ema_alpha",
        "channel_creation_enabled",
        "channel_fan_in",
        "channel_fan_out",
        "channel_bundle_size",
        "channel_candidate_pool",
        "channel_bundle_init_scale",
        "channel_underused_max_quantile",
        "channel_require_full_bundle",
        "block_density_cap",
        "channel_budget_fraction",
        "channel_probe_threshold_ema_alpha",
        "max_probationary_channels_per_prefix",
        "max_probationary_channels_global",
        "max_new_probationary_channels_per_epoch",
        "phase1_registry_enabled",
        "phase1_reselection_decay",
        "phase1_unexplored_bonus",
        "phase1_reexplored_bonus",
        "tenured_channel_grace_period",
        "local_reinforce_enabled",
        "local_reinforce_epochs",
        "local_reinforce_max_edges_per_epoch",
        "local_reinforce_max_edges_per_channel_total",
        "local_reinforce_edges_per_channel_per_epoch",
        "local_reinforce_top_pool",
    )
    return {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}


def _set_model_attr_if_present(model, name: str, value: Any) -> None:
    if hasattr(model, name):
        setattr(model, name, value)


def _cosine_between(start: float, end: float, epoch_idx: int, total_epochs: int) -> float:
    total = max(1, int(total_epochs))
    denom = max(1, total - 1)
    progress = float(np.clip((int(epoch_idx) - 1) / float(denom), 0.0, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end + (start - end) * cosine)


def _initialize_new_synapse_optimizer_state(
    optimizer: optim.Optimizer,
    model: nn.Module,
    new_masks: Mapping[str, torch.Tensor],
    *,
    v_floor: float,
) -> Dict[str, Dict[str, float]]:
    init_stats: Dict[str, Dict[str, float]] = {}
    if not hasattr(model, "weights"):
        return init_stats
    floor = float(max(0.0, v_floor))
    for key in model.weights.keys():
        new_mask = new_masks.get(key)
        if new_mask is None or int(new_mask.sum().item()) <= 0:
            continue
        W = model.weights[key]
        state = optimizer.state.get(W, {})
        exp_avg_sq = state.get("exp_avg_sq")
        exp_avg = state.get("exp_avg")
        if not torch.is_tensor(exp_avg_sq):
            continue
        target_mask = new_mask.to(device=exp_avg_sq.device, dtype=torch.bool)
        if int(target_mask.sum().item()) <= 0:
            continue
        median_v = float(torch.median(exp_avg_sq.detach().reshape(-1)).item()) if int(exp_avg_sq.numel()) > 0 else 0.0
        v_init = float(max(median_v, floor))
        exp_avg_sq[target_mask] = float(v_init)
        if torch.is_tensor(exp_avg):
            exp_avg[target_mask] = 0.0
        init_stats[str(key)] = {
            "created_count": float(int(target_mask.sum().item())),
            "median_v": float(median_v),
            "v_init": float(v_init),
        }
    return init_stats


def _update_dual_lr_group2_masks(
    group2_masks: Dict[str, torch.Tensor],
    created_delta_masks: Mapping[str, torch.Tensor],
    post_pruning_masks: Mapping[str, torch.Tensor],
) -> Dict[str, int]:
    updated_counts: Dict[str, int] = {}
    for key, g2_mask in group2_masks.items():
        created_delta = created_delta_masks.get(key)
        post_mask = post_pruning_masks.get(key)
        if created_delta is None or post_mask is None:
            continue
        next_mask = (g2_mask | created_delta.to(g2_mask.device)) & post_mask.to(g2_mask.device)
        group2_masks[key] = next_mask.detach().clone()
        updated_counts[str(key)] = int(next_mask.sum().item())
    return updated_counts


def _min_prefix_coverage(coverage_by_prefix: Mapping[str, Any]) -> float:
    values = []
    for value in (coverage_by_prefix or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(min(values)) if values else 0.0


def _compute_triphasic_state(cfg, epoch: int, state: Dict[str, Any]) -> Dict[str, Any]:
    epoch_idx = int(epoch + 1)
    if not bool(getattr(cfg, "triphasic_controller_enabled", False)):
        return {
            "enabled": False,
            "phase_name": "standard",
            "creation_allowed": True,
            "pruning_allowed": True,
            "creation_policy": str(getattr(cfg, "creation_policy", "")),
            "creation_budget": None,
        }

    if not bool(state.get("phase1_to_phase2_triggered", False)):
        return {
            "enabled": True,
            "phase_name": "phase1_probe",
            "phase1_active": True,
            "handoff_active": False,
            "phase2_active": False,
            "phase3_active": False,
            "creation_allowed": bool(getattr(cfg, "phase1_channel_creation_enabled", True)),
            "pruning_allowed": bool(getattr(cfg, "phase1_pruning_enabled", False)),
            "creation_policy": "paired_ffn_channel_probe",
            "creation_budget": None,
        }

    handoff_start = int(state.get("handoff_start_epoch", 0))
    handoff_end = int(state.get("handoff_end_epoch", 0))
    if handoff_start > 0 and handoff_start <= epoch_idx < handoff_end:
        return {
            "enabled": True,
            "phase_name": "handoff",
            "phase1_active": False,
            "handoff_active": True,
            "phase2_active": False,
            "phase3_active": False,
            "creation_allowed": bool(getattr(cfg, "handoff_channel_creation", False)),
            "pruning_allowed": bool(getattr(cfg, "handoff_pruning", False)),
            "creation_policy": "paired_ffn_channel_probe",
            "creation_budget": None,
        }

    phase2_epoch = max(0, epoch_idx - max(1, handoff_end))
    min_phase2_epochs = int(max(0, getattr(cfg, "phase2_to_phase3_min_phase2_epochs", 0)))
    phase3_trigger_epoch = int(state.get("phase2_to_phase3_trigger_epoch", 0))
    if phase3_trigger_epoch > 0 and epoch_idx >= phase3_trigger_epoch and phase2_epoch >= min_phase2_epochs:
        return {
            "enabled": True,
            "phase_name": "phase3_freeze",
            "phase1_active": False,
            "handoff_active": False,
            "phase2_active": False,
            "phase3_active": True,
            "creation_allowed": bool(getattr(cfg, "phase3_micro_creation_enabled", False)),
            "pruning_allowed": bool(getattr(cfg, "phase3_pruning_enabled", False)),
            "creation_policy": str(getattr(cfg, "phase2_micro_creation_policy", "need_quality_inverse_coverage")),
            "creation_budget": int(getattr(cfg, "phase2_micro_budget_target", 0)),
        }

    pruning_delay = int(max(0, getattr(cfg, "phase2_pruning_start_delay", 0)))
    return {
        "enabled": True,
        "phase_name": "phase2_micro",
        "phase1_active": False,
        "handoff_active": False,
        "phase2_active": True,
        "phase3_active": False,
        "creation_allowed": bool(getattr(cfg, "phase2_enabled", True) and getattr(cfg, "phase2_micro_creation_enabled", True)),
        "pruning_allowed": bool(
            getattr(cfg, "phase2_enabled", True)
            and getattr(cfg, "phase2_pruning_enabled", True)
            and phase2_epoch >= pruning_delay
        ),
        "creation_policy": str(getattr(cfg, "phase2_micro_creation_policy", "need_quality_inverse_coverage")),
        "creation_budget": int(getattr(cfg, "phase2_micro_budget_target", 0)),
        "phase2_epoch": int(phase2_epoch),
    }


def _scaled_phase2_micro_budget(cfg, model, state: Mapping[str, Any]) -> int:
    base_budget = int(max(0, getattr(cfg, "phase2_micro_budget_target", 0)))
    if base_budget <= 0:
        return 0
    if not bool(getattr(cfg, "phase2_micro_budget_scale_by_synapses", False)):
        return base_budget

    try:
        stats = model.get_connectivity_stats()
        active_synapses = int(stats.total_active)
    except Exception:
        active_synapses = int(state.get("phase2_budget_reference_active_synapses", 0) or 0)
    reference_synapses = int(getattr(cfg, "phase2_micro_budget_reference_active_synapses", 0) or 0)
    if reference_synapses <= 0:
        reference_synapses = int(state.get("phase2_budget_reference_active_synapses", 0) or 0)
    if reference_synapses <= 0:
        return base_budget

    scaled = int(round(float(base_budget) * float(max(0, active_synapses)) / float(reference_synapses)))
    if base_budget > 0:
        scaled = max(1, scaled)
    max_budget = int(getattr(cfg, "phase2_micro_budget_max", 0) or 0)
    if max_budget > 0:
        scaled = min(max_budget, scaled)
    return int(scaled)


def _prune_cap_from_created_window(cfg, created_window: int) -> int | None:
    if not bool(getattr(cfg, "prune_cap_from_created_window_enabled", False)):
        return None
    floor = int(max(0, getattr(cfg, "prune_cap_from_created_window_floor", 0) or 0))
    ratio = float(max(0.0, getattr(cfg, "prune_cap_from_created_window_ratio", 1.0) or 0.0))
    ceiling = int(max(0, getattr(cfg, "prune_cap_from_created_window_ceiling", 0) or 0))
    cap = max(floor, int(round(ratio * float(max(0, int(created_window))))))
    if ceiling > 0:
        cap = min(cap, ceiling)
    return int(max(0, cap))


def run_single_seed_impl(
    seed: int,
    cfg,
    train_dataset,
    test_dataset,
    train_pools,
    eval_pools,
    output_dir: Path,
    *,
    logger,
    level_order: Sequence[str],
    growth_targets: Mapping[str, float],
    model_class,
    build_default_math_level_specs_fn,
    train_one_epoch_fn,
    evaluate_seen_levels_fn,
    run_gradient_probe_fn,
    normalize_mix_fn,
    runtime_sample_mixed_indices_fn,
    runtime_build_loader_fn,
    experiment_interpolate_mix_fn,
    final_extra_eval_fn=None,
    final_extra_eval_context=None,
    log_epoch1_data_usage_fn=None,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device(cfg.device)

    vocab_size = int(train_dataset.vocab_size)
    pad_id = int(train_dataset.pad_id)
    sos_id = int(train_dataset.sos_id)
    eos_id = int(train_dataset.eos_id)
    special_ids = (
        int(train_dataset.pad_id),
        int(train_dataset.unk_id),
        int(train_dataset.sos_id),
        int(train_dataset.eos_id),
    )
    model = model_class(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_enc_layers=cfg.n_enc_layers,
        n_dec_layers=cfg.n_dec_layers,
        max_src_len=cfg.max_q_len,
        max_tgt_len=cfg.max_a_len,
        pad_id=pad_id,
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
        creation_auto_budget_from_role_need=cfg.creation_auto_budget_from_role_need,
        creation_role_need_ema_alpha=cfg.creation_role_need_ema_alpha,
        creation_auto_budget_min_ratio=cfg.creation_auto_budget_min_ratio,
        ffn_channel_complete_k=cfg.ffn_channel_complete_k,
        pruning_policy=cfg.pruning_policy,
        pruning_active_ref_quantile=cfg.pruning_active_ref_quantile,
        pruning_priority_topk=cfg.pruning_priority_topk,
        pruning_ref_mature_only=cfg.pruning_ref_mature_only,
        pruning_couple_creation_need=cfg.pruning_couple_creation_need,
        pruning_creation_need_ema_alpha=cfg.pruning_creation_need_ema_alpha,
        pruning_creation_coupling_alpha=cfg.pruning_creation_coupling_alpha,
        tenure_enabled=cfg.tenure_enabled,
        tenure_score_ema_alpha=cfg.tenure_score_ema_alpha,
        tenure_min_age=cfg.tenure_min_age,
        tenure_acquire_quantile=cfg.tenure_acquire_quantile,
        tenure_acquire_streak=cfg.tenure_acquire_streak,
        tenure_max_share_per_matrix=cfg.tenure_max_share_per_matrix,
        tenure_decay_on_miss=cfg.tenure_decay_on_miss,
        tenure_release_enabled=cfg.tenure_release_enabled,
        **_optional_model_kwargs(cfg),
    ).to(device)

    is_dense_static_run = bool(
        float(getattr(cfg, "initial_connectivity_attn", 0.0)) >= 0.999
        and float(getattr(cfg, "initial_connectivity_ffn", 0.0)) >= 0.999
        and int(getattr(cfg, "max_new_synapses", 1)) <= 0
    )
    if (
        bool(getattr(cfg, "compile_dense_enabled", True))
        and is_dense_static_run
        and str(device).startswith("cuda")
        and hasattr(torch, "compile")
    ):
        try:
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
                torch._dynamo.config.capture_scalar_outputs = True
            inductor_cfg = getattr(getattr(torch, "_inductor", None), "config", None)
            if inductor_cfg is not None:
                if hasattr(inductor_cfg, "max_autotune_gemm"):
                    inductor_cfg.max_autotune_gemm = False
                if hasattr(inductor_cfg, "max_autotune"):
                    inductor_cfg.max_autotune = False
            triton_cfg = getattr(inductor_cfg, "triton", None)
            if triton_cfg is not None and hasattr(triton_cfg, "cudagraph_skip_dynamic_graphs"):
                triton_cfg.cudagraph_skip_dynamic_graphs = True
        except Exception:
            pass
        compile_kwargs: Dict[str, Any] = {
            "mode": str(getattr(cfg, "compile_dense_mode", "reduce-overhead") or "reduce-overhead")
        }
        backend = str(getattr(cfg, "compile_dense_backend", "auto") or "auto").strip().lower()
        if backend and backend != "auto":
            compile_kwargs["backend"] = backend
        try:
            model = torch.compile(model, **compile_kwargs)
            logger.info(
                "[seed=%s] torch.compile enabled for dense static run | mode=%s backend=%s",
                seed,
                str(compile_kwargs.get("mode")),
                str(compile_kwargs.get("backend", "auto")),
            )
        except Exception as exc:
            logger.warning("[seed=%s] torch.compile disabled after failure: %s", seed, exc)

    amp_dense_active = False
    amp_dense_dtype: Optional[torch.dtype] = None
    amp_dense_dtype_name = "off"
    amp_dense_request = str(getattr(cfg, "amp_dense_dtype", "bf16") or "bf16").strip().lower()
    if (
        bool(getattr(cfg, "amp_dense_enabled", True))
        and is_dense_static_run
        and bool(getattr(cfg, "amp_dense_cpu_enabled", False))
        and str(device) == "cpu"
        and amp_dense_request in {"bf16", "bfloat16"}
    ):
        amp_dense_active = True
        amp_dense_dtype = torch.bfloat16
        amp_dense_dtype_name = "bf16_cpu"
    if (
        bool(getattr(cfg, "amp_dense_enabled", True))
        and is_dense_static_run
        and str(device).startswith("cuda")
    ):
        if amp_dense_request in {"bf16", "bfloat16"}:
            if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                amp_dense_active = True
                amp_dense_dtype = torch.bfloat16
                amp_dense_dtype_name = "bf16"
            else:
                logger.warning("[seed=%s] AMP dense requested bf16 but CUDA bf16 is not supported; disabling AMP.", seed)
        elif amp_dense_request in {"fp16", "float16", "half"}:
            amp_dense_active = True
            amp_dense_dtype = torch.float16
            amp_dense_dtype_name = "fp16"
        else:
            logger.warning(
                "[seed=%s] Unsupported amp_dense_dtype=%r; valid values are bf16/fp16. AMP disabled.",
                seed,
                amp_dense_request,
            )
    setattr(model, "_dense_amp_enabled", bool(amp_dense_active))
    setattr(model, "_dense_amp_dtype", amp_dense_dtype)
    logger.info(
        "[seed=%s] dense AMP autocast: enabled=%s dtype=%s",
        seed,
        bool(amp_dense_active),
        amp_dense_dtype_name,
    )

    optimizer_name = str(getattr(cfg, "optimizer_name", "adam")).strip().lower()
    optimizer_kwargs = {
        "params": model.parameters(),
        "lr": cfg.lr,
        "betas": (float(cfg.adam_beta1), float(cfg.adam_beta2)),
        "eps": float(cfg.adam_eps),
    }
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            weight_decay=float(getattr(cfg, "adam_weight_decay", 0.0)),
            **optimizer_kwargs,
        )
    else:
        optimizer = optim.Adam(**optimizer_kwargs)
    rng = np.random.default_rng(seed)
    eval_loaders = build_eval_loaders(
        test_dataset, eval_pools, cfg.batch_size, cfg.num_workers, cfg.pin_memory,
        cfg.persistent_workers, cfg.prefetch_factor, getattr(cfg, "runtime_eval_cap_per_module", 0))

    selected_levels = [lv for lv in cfg.levels if lv in level_order]
    full_specs = build_default_math_level_specs_fn(curriculum_variant=cfg.curriculum_variant)
    level_specs = [s for s in full_specs if s.name in selected_levels]
    if not level_specs:
        raise RuntimeError("Aucun niveau valide sélectionné via --levels.")
    level_name_to_id = {name: idx for idx, name in enumerate(level_order)}
    bootstrap_completed_levels = [
        lv for lv in cfg.bootstrap_completed_levels
        if lv in selected_levels and lv in level_name_to_id
    ]
    if cfg.start_level:
        if cfg.start_level not in selected_levels:
            raise RuntimeError(
                f"start_level={cfg.start_level!r} absent de --levels={selected_levels}"
            )
        active_start_idx = next(i for i, spec in enumerate(level_specs) if spec.name == cfg.start_level)
    else:
        active_start_idx = 0
        for i, spec in enumerate(level_specs):
            if spec.name not in bootstrap_completed_levels:
                active_start_idx = i
                break
        else:
            raise RuntimeError("Aucun niveau actif a entrainer apres bootstrap_completed_levels.")
    expected_bootstrap_levels = [spec.name for spec in level_specs[:active_start_idx]]
    if bootstrap_completed_levels and bootstrap_completed_levels != expected_bootstrap_levels:
        raise RuntimeError(
            "bootstrap_completed_levels doit correspondre au prefixe des niveaux precedant start_level: "
            f"attendu={expected_bootstrap_levels}, recu={bootstrap_completed_levels}"
        )
    level_ids_lookup = torch.as_tensor(train_dataset.level_ids, dtype=torch.long)
    level_token_ids = {
        lv: infer_level_token_ids(train_dataset, lv, special_ids) for lv in level_order
    }
    mix_start_table, mix_end_table = resolve_mix_tables(
        cfg.mix_variant,
        {
            "primaire": {"primaire": 1.0},
            "college": {"college": 0.50, "primaire": 0.50},
            "lycee": {"lycee": 0.50, "college": 0.30, "primaire": 0.20},
            "superieur": {"superieur": 0.40, "lycee": 0.30, "college": 0.20, "primaire": 0.10},
        },
        {
            "primaire": {"primaire": 1.0},
            "college": {"college": 0.80, "primaire": 0.20},
            "lycee": {"lycee": 0.70, "college": 0.20, "primaire": 0.10},
            "superieur": {"superieur": 0.60, "lycee": 0.25, "college": 0.10, "primaire": 0.05},
        },
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"seed": seed, "levels": {}, "checkpoints": {}, "config_snapshot": {
        "enable_newsyn": cfg.enable_newsyn,
        "enable_loss_masking": cfg.enable_loss_masking,
        "enable_replay_boost": cfg.enable_replay_boost,
        "enable_crystallization": cfg.enable_crystallization,
        "enable_gradient_tags": cfg.enable_gradient_tags,
        "enable_transition_buffer": cfg.enable_transition_buffer,
        "enable_growth_retention": cfg.enable_growth_retention,
        "enable_output_head_guard": bool(cfg.enable_output_head_guard),
        "enable_consolidation_windows": bool(cfg.enable_consolidation_windows),
        "consolidation_min_level_idx": cfg.consolidation_min_level_idx,
        "consolidation_start_epoch": cfg.consolidation_start_epoch,
        "consolidation_cycle": cfg.consolidation_cycle,
        "consolidation_duration": cfg.consolidation_duration,
        "consolidation_mix_mode": str(cfg.consolidation_mix_mode),
        "consolidation_freeze_creation": bool(cfg.consolidation_freeze_creation),
        "consolidation_freeze_pruning": bool(cfg.consolidation_freeze_pruning),
        "newsyn_min_level_idx": cfg.newsyn_min_level_idx,
        "newsyn_epochs": cfg.newsyn_epochs,
        "old_synapse_grad_scale_start": cfg.old_synapse_grad_scale_start,
        "disable_pruning_during_newsyn": bool(cfg.disable_pruning_during_newsyn),
        "masking_min_level_idx": cfg.masking_min_level_idx,
        "mask_warmup_epochs": cfg.mask_warmup_epochs,
        "mask_ramp_epochs": cfg.mask_ramp_epochs,
        "mask_penalty": cfg.mask_penalty,
        "mask_floor_k": cfg.mask_floor_k,
        "mask_floor_target_base": cfg.mask_floor_target,
        "mask_floor_target": cfg.mask_floor_target,
        "mask_floor_target_effective": cfg.mask_floor_target,
        "mask_floor_min": cfg.mask_floor_min,
        "mask_floor_max": cfg.mask_floor_max,
        "mask_floor_target_from_bootstrap": bool(cfg.mask_floor_target_from_bootstrap),
        "mask_floor_target_bootstrap_scale": cfg.mask_floor_target_bootstrap_scale,
        "mask_strength_schedule": "warmup_ramp_sigmoid_floor",
        "boost_min_level_idx": cfg.boost_min_level_idx,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "n_enc_layers": cfg.n_enc_layers,
        "n_dec_layers": cfg.n_dec_layers,
        "attention_mode": str(getattr(cfg, "attention_mode", "sparse")),
        "dynamic_scope": str(getattr(cfg, "dynamic_scope", "all")),
        "compile_dense_enabled": bool(getattr(cfg, "compile_dense_enabled", True)),
        "compile_dense_mode": str(getattr(cfg, "compile_dense_mode", "reduce-overhead")),
        "compile_dense_backend": str(getattr(cfg, "compile_dense_backend", "auto")),
        "amp_dense_enabled": bool(getattr(cfg, "amp_dense_enabled", True)),
        "amp_dense_dtype": str(getattr(cfg, "amp_dense_dtype", "bf16")),
        "amp_dense_active": bool(amp_dense_active),
        "amp_dense_dtype_effective": str(amp_dense_dtype_name),
        "use_ewc": bool(getattr(cfg, "use_ewc", False)),
        "ewc_lambda": float(getattr(cfg, "ewc_lambda", 0.0)),
        "ewc_fisher_samples": int(getattr(cfg, "ewc_fisher_samples", 0)),
        "ewc_update_frequency": int(getattr(cfg, "ewc_update_frequency", 0)),
        "initial_connectivity_attn": cfg.initial_connectivity_attn,
        "initial_connectivity_ffn": cfg.initial_connectivity_ffn,
        "creation_score_mode": cfg.creation_score_mode,
        "creation_score_normalization": cfg.creation_score_normalization,
        "creation_score_alpha": cfg.creation_score_alpha,
        "creation_policy": cfg.creation_policy,
        "pruning_policy": cfg.pruning_policy,
        "lr": cfg.lr,
        "optimizer_name": optimizer_name,
        "adam_beta1": cfg.adam_beta1,
        "adam_beta2": cfg.adam_beta2,
        "adam_eps": cfg.adam_eps,
        "adam_weight_decay": float(getattr(cfg, "adam_weight_decay", 0.0)),
        "lr_schedule": cfg.lr_schedule,
        "lr_warmup_epochs": cfg.lr_warmup_epochs,
        "lr_schedule_min_ratio": cfg.lr_schedule_min_ratio,
        "label_smoothing": cfg.label_smoothing,
        "max_new_synapses": cfg.max_new_synapses,
        "max_new_synapses_cap": cfg.max_new_synapses_cap,
        "max_q_len": cfg.max_q_len,
        "max_a_len": cfg.max_a_len,
        "vocab_size": vocab_size,
        "growth_target_primaire": cfg.growth_target_primaire,
        "checkpoint_eval_frequency": cfg.checkpoint_eval_frequency,
        "checkpoint_eval_mode": cfg.checkpoint_eval_mode,
        "save_best_checkpoint": bool(cfg.save_best_checkpoint),
        "save_last_checkpoint": bool(cfg.save_last_checkpoint),
        "collect_ffn_hidden_stats_each_epoch": bool(cfg.collect_ffn_hidden_stats_each_epoch),
        "long_run_lite": bool(cfg.long_run_lite),
        "replay_weight_base": cfg.replay_weight_base,
        "replay_weight_max": cfg.replay_weight_max,
        "retention_target": cfg.retention_target,
        "retention_gain": cfg.retention_gain,
        "retention_eval_frequency": cfg.retention_eval_frequency,
        "curriculum_variant": str(cfg.curriculum_variant),
        "mix_variant": str(cfg.mix_variant),
        "init_checkpoint": str(cfg.init_checkpoint),
        "bootstrap_completed_levels": list(bootstrap_completed_levels),
        "start_level": str(cfg.start_level),
        "resume_optimizer": bool(cfg.resume_optimizer),
        "start_epoch_offset": int(getattr(cfg, "start_epoch_offset", 0)),
    }}

    experiment_backend = get_experiment_backend()
    bootstrap_payload: Optional[Dict[str, Any]] = None
    blockb_retention_targets_by_level: Dict[str, float] = {}
    if cfg.init_checkpoint:
        checkpoint_path = Path(cfg.init_checkpoint)
        bootstrap_payload = load_training_checkpoint(checkpoint_path)
        validate_checkpoint_compatibility(bootstrap_payload.get("config_snapshot", {}), cfg)
        load_result = model.load_state_dict(bootstrap_payload["model_state_dict"], strict=False)
        allowed_missing_prefixes = (
            "tenure_mask_",
            "tenure_score_ema_",
            "tenure_streak_",
        )
        missing_keys = [str(key) for key in getattr(load_result, "missing_keys", [])]
        unexpected_keys = [str(key) for key in getattr(load_result, "unexpected_keys", [])]
        bad_missing = [
            key for key in missing_keys
            if not any(str(key).startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if bad_missing or unexpected_keys:
            raise RuntimeError(
                "Bootstrap checkpoint incompatible with current model: "
                f"missing={bad_missing} unexpected={unexpected_keys}"
            )
        if cfg.resume_optimizer and "optimizer_state_dict" in bootstrap_payload:
            optimizer.load_state_dict(bootstrap_payload["optimizer_state_dict"])
        runtime_state = bootstrap_payload.get("runtime_state")
        if runtime_state and experiment_backend is not None and hasattr(experiment_backend, "restore_runtime_state"):
            experiment_backend.restore_runtime_state(runtime_state)
        result["bootstrap_checkpoint_path"] = str(checkpoint_path)
        result["bootstrap_completed_levels"] = list(bootstrap_completed_levels)
        result["bootstrap_epoch"] = int(bootstrap_payload.get("epoch", -1))
        result["bootstrap_metric_name"] = str(bootstrap_payload.get("metric_name", ""))
        result["bootstrap_metric_value"] = float(bootstrap_payload.get("metric_value", 0.0))

    if bootstrap_completed_levels:
        bootstrap_eval = evaluate_seen_levels_fn(
            model, eval_loaders, bootstrap_completed_levels, device,
            pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
            vocab_size=vocab_size, max_a_len=cfg.max_a_len,
            eval_mode=cfg.final_eval_mode,
        )
        bootstrap_acc_vals = [
            float(v["accuracy"])
            for v in bootstrap_eval.values()
            if isinstance(v, Mapping) and "accuracy" in v
        ]
        if cfg.enable_replay_weighting and cfg.retention_target_from_bootstrap and bootstrap_acc_vals:
            cfg.retention_target = float(
                max(1e-6, np.mean(bootstrap_acc_vals) * float(cfg.retention_target_bootstrap_scale))
            )
            result["config_snapshot"]["retention_target"] = cfg.retention_target
        bootstrap_stats = collect_model_stats(model, include_hidden_stats=cfg.collect_ffn_hidden_stats_each_epoch)
        for lv in bootstrap_completed_levels:
            metric = bootstrap_eval[lv]
            logger.info(
                "[seed=%s][bootstrap_only] %s | seq=%.3f tok=%.3f ar_seq=%.3f tf_seq=%.3f",
                seed,
                lv,
                float(metric["accuracy"]),
                float(metric.get("token_accuracy", 0.0)),
                float(metric.get("autoregressive_accuracy", metric["accuracy"])),
                float(metric.get("teacher_forcing_accuracy", 0.0)),
            )
            result["levels"][lv] = {
                "bootstrap_only": True,
                "new_modules": list(next(spec.new_modules for spec in level_specs if spec.name == lv)),
                "all_modules": list(next(spec.all_modules for spec in level_specs if spec.name == lv)),
                "epochs": 0,
                "transition_epochs": 0,
                "train_history": [],
                "eval_acc_current": float(metric["accuracy"]),
                "eval_token_acc_current": float(metric.get("token_accuracy", 0.0)),
                "eval_autoreg_acc_current": float(metric.get("autoregressive_accuracy", metric["accuracy"])),
                "eval_autoreg_token_acc_current": float(metric.get("autoregressive_token_accuracy", metric.get("token_accuracy", 0.0))),
                "eval_tf_acc_current": float(metric.get("teacher_forcing_accuracy", 0.0)),
                "eval_tf_token_acc_current": float(metric.get("teacher_forcing_token_accuracy", 0.0)),
                "eval_loss_current": float(metric["loss"]),
                "retention": {name: dict(values) for name, values in bootstrap_eval.items()},
                "model_stats": bootstrap_stats,
                "convergence": {},
            }

    if bool(getattr(cfg, "use_ewc", False)) and bootstrap_completed_levels:
        ewc_mix = {str(lv): 1.0 for lv in bootstrap_completed_levels}
        ewc_indices = runtime_sample_mixed_indices_fn(
            pools=train_pools,
            mix=ewc_mix,
            total_samples=int(max(1024, int(getattr(cfg, "ewc_fisher_samples", 256)))),
            rng=rng,
        )
        ewc_loader = runtime_build_loader_fn(
            train_dataset,
            ewc_indices,
            cfg.batch_size,
            False,
            cfg.num_workers,
            cfg.pin_memory,
            cfg.persistent_workers,
            cfg.prefetch_factor,
        )
        save_full_param_ewc_reference(model)
        compute_full_param_fisher_information(
            model,
            ewc_loader,
            device=device,
            pad_id=pad_id,
            sos_id=sos_id,
            vocab_size=vocab_size,
            max_a_len=cfg.max_a_len,
            n_samples=int(getattr(cfg, "ewc_fisher_samples", 256)),
            amp_enabled=amp_dense_active,
            amp_dtype=amp_dense_dtype,
        )
        logger.info(
            "[seed=%s] EWC bootstrap ready | levels=%s | lambda=%.2f | fisher_samples=%d",
            seed,
            ",".join(str(lv) for lv in bootstrap_completed_levels),
            float(getattr(cfg, "ewc_lambda", 0.0)),
            int(getattr(cfg, "ewc_fisher_samples", 0)),
        )
        post_fisher_eval = evaluate_seen_levels_fn(
            model, eval_loaders, bootstrap_completed_levels, device,
            pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
            vocab_size=vocab_size, max_a_len=cfg.max_a_len,
            eval_mode=cfg.final_eval_mode,
        )
        for lv in bootstrap_completed_levels:
            metric = post_fisher_eval[lv]
            logger.info(
                "[seed=%s][post_fisher_pre_buffer] %s | seq=%.3f tok=%.3f ar_seq=%.3f tf_seq=%.3f",
                seed,
                lv,
                float(metric["accuracy"]),
                float(metric.get("token_accuracy", 0.0)),
                float(metric.get("autoregressive_accuracy", metric["accuracy"])),
                float(metric.get("teacher_forcing_accuracy", 0.0)),
            )

    for level_idx, spec in enumerate(level_specs):
        if level_idx < active_start_idx:
            continue
        level_name = spec.name
        completed_prefix = [s.name for s in level_specs[:level_idx]]
        is_first = (len(completed_prefix) == 0)
        seen_levels = [s.name for s in level_specs[:level_idx + 1]]
        old_level_ids = sorted({int(level_name_to_id[lv]) for lv in completed_prefix})
        old_token_ids = sorted({
            int(t)
            for lv in seen_levels[:-1]
            for t in level_token_ids.get(lv, [])
        })
        default_mix = mix_start_table.get(level_name, {level_name: 1.0})
        samples = max(sum(len(train_pools.get(k, [])) for k in default_mix), 1024) if cfg.samples_per_epoch <= 0 else cfg.samples_per_epoch

        history = []
        current_replay_weight = float(cfg.replay_weight_base)
        retention_old_mean = -1.0
        last_old_ret: Dict[str, Dict[str, float]] = {}
        transition_epochs = 0
        if not is_first and cfg.enable_transition_buffer:
            is_last = (level_idx == len(level_specs) - 1)
            if is_last:
                transition_epochs = cfg.transition_epochs_last
            elif level_name == "lycee":
                transition_epochs = cfg.transition_epochs_lycee
            else:
                transition_epochs = cfg.transition_epochs_default
        if transition_epochs > 0 and completed_prefix:
            logger.info(
                "[seed=%s][%s] transition_buffer active | epochs=%d | seen_before=%s",
                seed,
                level_name,
                int(transition_epochs),
                ",".join(completed_prefix),
            )

        for t_ep in range(transition_epochs):
            t_mix = normalize_mix_fn({lv: 1.0 for lv in seen_levels[:-1]})
            t_indices = runtime_sample_mixed_indices_fn(pools=train_pools, mix=t_mix, total_samples=samples, rng=rng)
            t_loader = runtime_build_loader_fn(
                train_dataset, t_indices, cfg.batch_size, True,
                cfg.num_workers, cfg.pin_memory, cfg.persistent_workers, cfg.prefetch_factor)
            t_metrics = train_one_epoch_fn(
                model, t_loader, optimizer, device,
                pad_id=pad_id, sos_id=sos_id, vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                level_ids_lookup=level_ids_lookup,
                old_level_ids=old_level_ids if cfg.enable_replay_weighting else None,
                replay_weight=max(current_replay_weight, cfg.transition_replay_weight),
                enable_crystallization=cfg.enable_crystallization and not is_first,
                crystal_vital_floor=cfg.crystal_vital_floor, crystal_default_floor=cfg.crystal_default_floor,
                crystal_beta=cfg.transition_crystal_beta, tag_quantile=cfg.tag_quantile,
                amp_enabled=amp_dense_active, amp_dtype=amp_dense_dtype,
            )
            ms = collect_model_stats(model, include_hidden_stats=cfg.collect_ffn_hidden_stats_each_epoch)
            history.append({
                "epoch": -(transition_epochs - t_ep),
                "transition_phase": True,
                "train_loss": t_metrics["loss"],
                "train_accuracy": t_metrics["accuracy"],
                "active_synapses": ms["active_synapses"],
                "retention_snapshot": {},
            })
            if (t_ep + 1) in {1, transition_epochs, max(1, transition_epochs // 2)}:
                buffer_eval = evaluate_seen_levels_fn(
                    model, eval_loaders, completed_prefix, device,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                    vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                    eval_mode=cfg.final_eval_mode,
                )
                for lv in completed_prefix:
                    metric = buffer_eval[lv]
                    logger.info(
                        "[seed=%s][post_buffer_step=%d/%d] %s | seq=%.3f tok=%.3f ar_seq=%.3f tf_seq=%.3f",
                        seed,
                        int(t_ep + 1),
                        int(transition_epochs),
                        lv,
                        float(metric["accuracy"]),
                        float(metric.get("token_accuracy", 0.0)),
                        float(metric.get("autoregressive_accuracy", metric["accuracy"])),
                        float(metric.get("teacher_forcing_accuracy", 0.0)),
                    )

        entry_masks = capture_active_masks(model) if (
            not is_first and cfg.enable_newsyn and level_idx >= int(cfg.newsyn_min_level_idx)
        ) else None

        is_last_level = (level_idx == len(level_specs) - 1)
        epochs_this_level = cfg.last_level_epochs if is_last_level else cfg.epochs_per_level
        best_checkpoint_metric = -float("inf")
        best_checkpoint_epoch = -1
        best_checkpoint_path = ""
        train_seq_budget_ema: Optional[float] = None
        train_seq_budget_active = False
        train_seq_budget_fill_streak = 0
        train_seq_budget_last_fill_ratio = 1.0
        train_seq_budget_maturity_epoch: Optional[int] = None
        late_global_prune_created_window = 0
        triphasic_state: Dict[str, Any] = {
            "phase1_to_phase2_triggered": False,
            "phase1_to_phase2_trigger_epoch": 0,
            "phase1_to_phase2_reason": "",
            "handoff_start_epoch": 0,
            "handoff_end_epoch": 0,
            "phase2_to_phase3_trigger_epoch": int(getattr(cfg, "phase2_to_phase3_force_epoch", 0) or 0),
            "phase2_budget_reference_active_synapses": int(model.get_connectivity_stats().total_active),
            "phase1_tenure_rate_ema": None,
            "phase1_budget_modulator": 1.0,
            "phase1_effective_channel_budget": int(getattr(cfg, "phase1_base_channel_budget", 0) or 0),
            "phase1_min_budget_streak": 0,
            "phase2_effective_micro_budget": int(getattr(cfg, "phase2_micro_budget_target", 0) or 0),
            "phase2_zero_creation_streak": 0,
            "coverage_global": 0.0,
            "min_prefix_coverage": 0.0,
        }
        transfer_phase0_enabled = bool((not is_first) and int(getattr(cfg, "phase0_fixed_epochs", 0) or 0) > 0)
        phase0_last_effective_lr: Optional[float] = None
        phase12_restart_initial_lr: Optional[float] = None
        phase12_dual_lr_fused = False
        dual_lr_group2_masks: Dict[str, torch.Tensor] = {}
        if bool(getattr(cfg, "phase12_dual_lr_enabled", False)) and hasattr(model, "weights"):
            dual_lr_group2_masks = {
                key: torch.zeros_like(getattr(model, model.masks[key]), dtype=torch.bool, device=getattr(model, model.masks[key]).device)
                for key in model.weights.keys()
            }
        if transfer_phase0_enabled:
            logger.info(
                "[seed=%s][%s] Phase0/dual-LR transfer: phase0_epochs=%d lr0=%.2e->%.2e | phase12_old_lr=%.2e phase12_new_lr=%.2e warm_restart=%d dual=%s v_floor=%.1e",
                seed,
                level_name,
                int(getattr(cfg, "phase0_fixed_epochs", 0)),
                float(getattr(cfg, "phase0_lr_start", 1e-4)),
                float(getattr(cfg, "phase0_lr_end", 5e-5)),
                float(getattr(cfg, "phase12_old_lr_target", 1.5e-4)),
                float(getattr(cfg, "phase12_new_lr_target", 5e-4)),
                int(getattr(cfg, "phase12_warm_restart_epochs", 20)),
                bool(getattr(cfg, "phase12_dual_lr_enabled", False)),
                float(getattr(cfg, "phase12_dual_lr_v_init_floor", 1e-8)),
            )

        start_epoch_offset = int(max(0, min(getattr(cfg, "start_epoch_offset", 0), max(0, epochs_this_level - 1))))
        for epoch in range(start_epoch_offset, epochs_this_level):
            progress = epoch / max(1, epochs_this_level - 1)
            current_lr = compute_epoch_lr(cfg.lr, epoch, epochs_this_level, cfg, maturity_epoch=train_seq_budget_maturity_epoch)
            current_lr_phase = infer_lr_phase(epoch, epochs_this_level, cfg, maturity_epoch=train_seq_budget_maturity_epoch)
            triphasic = _compute_triphasic_state(cfg, epoch, triphasic_state)
            epoch_index = int(epoch + 1)
            phase0_active = bool(transfer_phase0_enabled and epoch_index <= int(getattr(cfg, "phase0_fixed_epochs", 0)))
            phase12_active = bool(
                transfer_phase0_enabled
                and not phase0_active
                and bool(triphasic.get("enabled", False))
                and (
                    bool(triphasic.get("phase1_active", False))
                    or bool(triphasic.get("handoff_active", False))
                    or bool(triphasic.get("phase2_active", False))
                )
            )
            phase3_transfer_active = bool(
                transfer_phase0_enabled
                and not phase0_active
                and bool(triphasic.get("phase3_active", False))
            )
            if phase0_active:
                current_lr = _cosine_between(
                    float(getattr(cfg, "phase0_lr_start", 1e-4)),
                    float(getattr(cfg, "phase0_lr_end", 5e-5)),
                    epoch_index,
                    int(getattr(cfg, "phase0_fixed_epochs", 1)),
                )
                current_lr_phase = "phase0_cosine"
                phase0_last_effective_lr = float(current_lr)
                triphasic = {
                    "enabled": True,
                    "phase_name": "phase0_stabilization",
                    "phase1_active": False,
                    "handoff_active": False,
                    "phase2_active": False,
                    "phase3_active": False,
                    "creation_allowed": False,
                    "pruning_allowed": False,
                    "creation_policy": str(getattr(cfg, "creation_policy", "")),
                    "creation_budget": None,
                }
            elif phase12_active:
                if phase12_restart_initial_lr is None:
                    phase12_restart_initial_lr = float(
                        phase0_last_effective_lr
                        if phase0_last_effective_lr is not None
                        else getattr(cfg, "phase0_lr_end", 5e-5)
                    )
                    logger.info(
                        "[seed=%s][%s] phase1 warm-restart initial_lr=%.2e (captured from phase0 end)",
                        seed,
                        level_name,
                        float(phase12_restart_initial_lr),
                    )
                local_epoch = max(1, epoch_index - int(getattr(cfg, "phase0_fixed_epochs", 0)))
                warm_epochs = int(max(0, getattr(cfg, "phase12_warm_restart_epochs", 20)))
                old_target = float(getattr(cfg, "phase12_old_lr_target", 1.5e-4))
                if warm_epochs > 0 and local_epoch <= warm_epochs:
                    warm_progress = float(local_epoch) / float(max(1, warm_epochs))
                    current_lr = float(phase12_restart_initial_lr + warm_progress * (old_target - phase12_restart_initial_lr))
                    current_lr_phase = "phase12_warm_restart"
                else:
                    current_lr = old_target
                    current_lr_phase = "phase12_plateau"
            elif phase3_transfer_active:
                if bool(getattr(cfg, "phase3_transfer_cosine_enabled", False)):
                    phase3_start = int(triphasic_state.get("phase2_to_phase3_trigger_epoch", 0) or epoch_index)
                    local_epoch = max(1, epoch_index - phase3_start + 1)
                    total_phase3_epochs = max(1, epochs_this_level - phase3_start + 1)
                    current_lr = _cosine_between(
                        float(getattr(cfg, "phase12_old_lr_target", 1.5e-4)),
                        float(getattr(cfg, "phase3_transfer_lr_min", getattr(cfg, "phase0_lr_end", 5e-5))),
                        local_epoch,
                        total_phase3_epochs,
                    )
                    current_lr_phase = "phase3_transfer_cosine"
                else:
                    current_lr = float(getattr(cfg, "phase12_old_lr_target", 1.5e-4))
                    current_lr_phase = "phase3_transfer_lr"
                if bool(getattr(cfg, "phase12_dual_lr_enabled", False)) and not phase12_dual_lr_fused:
                    for key in dual_lr_group2_masks.keys():
                        dual_lr_group2_masks[key].zero_()
                    phase12_dual_lr_fused = True
                    logger.info(
                        "[seed=%s][%s] dual-LR fusion at phase2->phase3 trigger (group2 reassigned to group1).",
                        seed,
                        level_name,
                    )
            set_optimizer_lr(optimizer, current_lr)
            if bool(triphasic.get("enabled", False)):
                _set_model_attr_if_present(model, "creation_policy", str(triphasic.get("creation_policy", cfg.creation_policy)))
                _set_model_attr_if_present(
                    model,
                    "max_new_probationary_channels_per_epoch",
                    int(triphasic_state.get("phase1_effective_channel_budget", getattr(cfg, "phase1_base_channel_budget", 0) or 0)),
                )

            need_ret = (not is_first and len(old_level_ids) > 0)
            if need_ret and (epoch == 0 or (epoch + 1) % cfg.retention_eval_frequency == 0):
                old_ret = evaluate_seen_levels_fn(
                    model, eval_loaders, seen_levels[:-1], device,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                    vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                    eval_mode=cfg.retention_eval_mode,
                )
                last_old_ret = {k: dict(v) for k, v in old_ret.items()}
                vals = [_metric_accuracy(v) for v in old_ret.values()]
                retention_old_mean = float(np.mean(vals)) if vals else -1.0
                details = " ".join(
                    f"{str(lv)}:seq={_metric_accuracy(metric):.3f}/tok={_metric_token_accuracy(metric):.3f}"
                    for lv, metric in old_ret.items()
                )
                logger.info(
                    "[retention eval][seed=%s][%s] ep %d/%d | old_mean=%.3f | %s",
                    seed,
                    level_name,
                    epoch + 1,
                    epochs_this_level,
                    float(retention_old_mean),
                    details,
                )

            gn = compute_growth_need(level_name, 0.5, growth_targets, growth_target_primaire=cfg.growth_target_primaire)
            ctx = build_epoch_context(cfg, level_idx, epoch, retention_old_mean, current_replay_weight, gn["normalized_need"])
            mix = experiment_interpolate_mix_fn(
                level_name=level_name,
                progress=progress,
                mix_start=mix_start_table,
                mix_end=mix_end_table,
            )
            if ctx.consolidation_window_active:
                mix = build_consolidation_mix(seen_levels, cfg.consolidation_mix_mode, normalize_mix_fn)
            consolidation_mix = {lv: float(mix.get(lv, 0.0)) for lv in mix}
            indices = runtime_sample_mixed_indices_fn(pools=train_pools, mix=mix, total_samples=samples, rng=rng)
            loader = runtime_build_loader_fn(
                train_dataset, indices, cfg.batch_size, True,
                cfg.num_workers, cfg.pin_memory, cfg.persistent_workers, cfg.prefetch_factor)

            plasticity_mask = None
            if ctx.newsyn_active and entry_masks and hasattr(model, "weights"):
                plasticity_mask = {}
                for key in model.weights.keys():
                    cur = (getattr(model, model.masks[key]) > 0)
                    old_m = entry_masks.get(key)
                    if old_m is not None:
                        plasticity_mask[key] = cur & (~old_m.to(cur.device))

            dual_lr_group2_mask_epoch: Optional[Mapping[str, torch.Tensor]] = None
            dual_lr_group2_scale_epoch = 1.0
            if (
                bool(getattr(cfg, "phase12_dual_lr_enabled", False))
                and bool(dual_lr_group2_masks)
                and not phase12_dual_lr_fused
                and phase12_active
            ):
                dual_lr_group2_mask_epoch = dual_lr_group2_masks
                dual_lr_group2_scale_epoch = float(
                    max(1.0, float(getattr(cfg, "phase12_new_lr_target", 5e-4)) / max(1e-8, float(current_lr)))
                )

            before_created = int(model.creation_count)
            before_pruned = int(model.pruning_count)
            metrics = train_one_epoch_fn(
                model, loader, optimizer, device,
                pad_id=pad_id, sos_id=sos_id, vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                level_ids_lookup=level_ids_lookup,
                old_level_ids=old_level_ids if cfg.enable_replay_weighting and not is_first else None,
                replay_weight=ctx.effective_replay_weight,
                enable_crystallization=ctx.crystal_active,
                crystal_vital_floor=cfg.crystal_vital_floor,
                crystal_default_floor=cfg.crystal_default_floor,
                crystal_beta=ctx.crystal_beta,
                tag_quantile=cfg.tag_quantile,
                plasticity_mask=plasticity_mask,
                old_synapse_grad_scale=ctx.old_synapse_grad_scale,
                enable_loss_masking=ctx.mask_active,
                enable_output_head_guard=bool(cfg.enable_output_head_guard and ctx.newsyn_active),
                old_token_ids=old_token_ids,
                mask_strength=ctx.mask_strength,
                mask_penalty=cfg.mask_penalty,
                label_smoothing=cfg.label_smoothing,
                amp_enabled=amp_dense_active,
                amp_dtype=amp_dense_dtype,
                dual_lr_group2_mask=dual_lr_group2_mask_epoch,
                dual_lr_group2_scale=dual_lr_group2_scale_epoch,
                ewc_loss_fn=(
                    (lambda m: compute_full_param_ewc_loss(m, float(getattr(cfg, "ewc_lambda", 0.0))))
                    if bool(getattr(cfg, "use_ewc", False)) and len(old_level_ids) > 0
                    else None
                ),
            )

            model.increment_synapse_ages()
            gn = compute_growth_need(level_name, float(metrics["accuracy"]), growth_targets, growth_target_primaire=cfg.growth_target_primaire)
            adaptive_max = int(np.clip(
                cfg.max_new_synapses * (1.0 + cfg.growth_need_boost * gn["normalized_need"]),
                cfg.max_new_synapses, cfg.max_new_synapses_cap,
            ))
            adaptive_max_nominal = int(adaptive_max)
            train_seq_budget_scale = 1.0
            train_seq_budget_candidate = int(adaptive_max)
            if cfg.creation_auto_budget_from_train_seq:
                train_seq_budget_ema, train_seq_budget_scale = update_train_seq_budget_state(
                    train_seq_budget_ema, float(metrics.get("sequence_accuracy", metrics["accuracy"])), cfg
                )
                train_seq_budget_candidate = max(1, int(round(float(adaptive_max) * float(train_seq_budget_scale))))
                if train_seq_budget_active:
                    adaptive_max = int(train_seq_budget_candidate)
            if bool(triphasic.get("enabled", False)):
                phase_budget = triphasic.get("creation_budget")
                if phase_budget is not None and int(phase_budget) > 0:
                    if bool(triphasic.get("phase2_active", False)):
                        adaptive_max = int(_scaled_phase2_micro_budget(cfg, model, triphasic_state))
                        triphasic_state["phase2_effective_micro_budget"] = int(adaptive_max)
                    else:
                        adaptive_max = int(phase_budget)

            created_epoch = pruned_epoch = 0
            pre_creation_masks = capture_active_masks(model) if dual_lr_group2_masks else {}
            hebb_thresh = max(cfg.hebbian_threshold_min, cfg.hebbian_threshold - cfg.hebbian_need_scale * gn["normalized_need"])
            if (
                not ctx.consolidation_freeze_creation
                and bool(triphasic.get("creation_allowed", True))
                and (epoch + 1) % cfg.synaptogenesis_frequency == 0
                and gn["normalized_need"] > 0
                and hasattr(model, "gradient_guided_synaptogenesis")
            ):
                n_conn = max(1, len(model.weights))
                n_dst = max(1, int(getattr(model, "n_layers", 2)) - 1)
                created_epoch = int(model.gradient_guided_synaptogenesis(
                    max_new_synapses=adaptive_max,
                    score_temperature=1.0,
                    min_score=max(1e-6, hebb_thresh * 1e-4),
                    per_connection_quota=max(1, adaptive_max // n_conn),
                    per_layer_quota=max(1, adaptive_max // n_dst),
                    init_scale=0.01,
                ))
            current_creation_fill_ratio = float(created_epoch) / float(max(1, adaptive_max_nominal))

            grad_stats = {"grad_probe_batches": 0.0, "grad_signal_mean": 0.0}
            if cfg.enable_gradient_tags and not is_first:
                grad_stats = run_gradient_probe_fn(
                    model, train_dataset, train_pools, seen_levels[:-1], None, rng,
                    cfg.batch_size, cfg.num_workers, cfg.pin_memory, cfg.persistent_workers,
                    cfg.prefetch_factor, device, cfg,
                    pad_id=pad_id, sos_id=sos_id, vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                    normalize_mix_fn=normalize_mix_fn,
                    runtime_sample_mixed_indices_fn=runtime_sample_mixed_indices_fn,
                    runtime_build_loader_fn=runtime_build_loader_fn,
                )

            late_global_prune_created_window += int(created_epoch)
            tenure_stats = {}
            if hasattr(model, "update_tenure_state"):
                tenure_stats = dict(model.update_tenure_state())
            ctx_pruning = build_epoch_context(cfg, level_idx, epoch, retention_old_mean, current_replay_weight, gn["normalized_need"])
            prune_cap_created_window = int(late_global_prune_created_window)
            prune_cap_max = _prune_cap_from_created_window(cfg, prune_cap_created_window)
            if ctx_pruning.pruning_allowed and bool(triphasic.get("pruning_allowed", True)):
                pruned_epoch = int(model.utility_based_pruning_with_protection(
                    keep_ratio=ctx_pruning.pruning_keep_ratio,
                    max_prune_count=prune_cap_max,
                    enable_replay_tag_protection=bool(cfg.enable_replay_tag_protection),
                    replay_tag_quantile=float(cfg.replay_tag_quantile),
                    replay_tag_min=float(cfg.replay_tag_min),
                    tagged_prune_pressure=float(gn["normalized_need"]),
                    tagged_prune_pressure_slope=float(cfg.tagged_prune_pressure_slope),
                    max_tagged_prune_ratio=float(cfg.max_tagged_prune_ratio),
                ))
                late_global_prune_created_window = 0

            after_created = int(model.creation_count)
            after_pruned = int(model.pruning_count)
            created_epoch = max(created_epoch, after_created - before_created)
            pruned_epoch = max(pruned_epoch, after_pruned - before_pruned)
            if dual_lr_group2_masks and int(created_epoch) > 0:
                post_masks = capture_active_masks(model)
                created_delta_masks = {
                    key: (post_masks[key] & (~pre_creation_masks.get(key, torch.zeros_like(post_masks[key])).to(post_masks[key].device)))
                    for key in post_masks.keys()
                }
                updated_counts = _update_dual_lr_group2_masks(dual_lr_group2_masks, created_delta_masks, post_masks)
                if (
                    bool(getattr(cfg, "phase12_dual_lr_enabled", False))
                    and not phase12_dual_lr_fused
                    and phase12_active
                ):
                    init_stats = _initialize_new_synapse_optimizer_state(
                        optimizer,
                        model,
                        {key: created_delta_masks[key] & post_masks[key] for key in created_delta_masks.keys()},
                        v_floor=float(getattr(cfg, "phase12_dual_lr_v_init_floor", 1e-8)),
                    )
                    if init_stats:
                        logger.info(
                            "[seed=%s][%s] dual_lr_v_init applied | created_survivors=%d | group2=%d | floor=%.1e",
                            seed,
                            level_name,
                            int(sum(int(v["created_count"]) for v in init_stats.values())),
                            int(sum(updated_counts.values())),
                            float(getattr(cfg, "phase12_dual_lr_v_init_floor", 1e-8)),
                        )
            if (
                bool(triphasic.get("enabled", False))
                and bool(triphasic.get("phase2_active", False))
                and bool(triphasic.get("creation_allowed", True))
                and int(adaptive_max) > 0
            ):
                if int(created_epoch) <= 0:
                    triphasic_state["phase2_zero_creation_streak"] = int(
                        triphasic_state.get("phase2_zero_creation_streak", 0)
                    ) + 1
                    zero_streak = int(triphasic_state["phase2_zero_creation_streak"])
                    if zero_streak == 20 or (zero_streak > 20 and zero_streak % 20 == 0):
                        creation_stats_for_warning = dict(getattr(model, "last_creation_stats", {}))
                        logger.warning(
                            "[seed=%s][%s] triphasic Phase2 has budget=%d but created 0 synapses for %d consecutive epochs | policy=%s eligible_total=%s candidate_total=%s quality_mad=%s coherence_threshold=%s",
                            seed,
                            level_name,
                            int(adaptive_max),
                            zero_streak,
                            str(triphasic.get("creation_policy", "")),
                            str(creation_stats_for_warning.get("eligible_total", creation_stats_for_warning.get("eligible_count_total", "n/a"))),
                            str(creation_stats_for_warning.get("candidate_total", "n/a")),
                            str(getattr(cfg, "creation_quality_mad_factor", "n/a")),
                            str(getattr(cfg, "creation_coherence_threshold", "n/a")),
                        )
                else:
                    triphasic_state["phase2_zero_creation_streak"] = 0
            train_seq_budget_last_fill_ratio = float(current_creation_fill_ratio)
            creation_stats = dict(getattr(model, "last_creation_stats", {}))
            pruning_stats = dict(getattr(model, "last_pruning_stats", {}))
            channel_probe_stats = {}
            if bool(triphasic.get("enabled", False)) and hasattr(model, "update_probationary_channels"):
                channel_probe_stats = dict(model.update_probationary_channels())
                coverage_global = float(channel_probe_stats.get("coverage_global", 0.0))
                min_cov = _min_prefix_coverage(dict(channel_probe_stats.get("coverage_by_prefix", {})))
                triphasic_state["coverage_global"] = float(coverage_global)
                triphasic_state["min_prefix_coverage"] = float(min_cov)
                tenure_rate_epoch = float(channel_probe_stats.get("tenure_rate_epoch", 0.0))
                evaluated_epoch = float(channel_probe_stats.get("evaluated_channels_epoch", 0.0))
                if bool(triphasic.get("phase1_active", False)) and evaluated_epoch > 0.0:
                    prev_ema = triphasic_state.get("phase1_tenure_rate_ema")
                    alpha = float(np.clip(getattr(cfg, "phase1_tenure_rate_ema_alpha", 0.10), 1e-6, 1.0))
                    triphasic_state["phase1_tenure_rate_ema"] = (
                        tenure_rate_epoch if prev_ema is None else alpha * tenure_rate_epoch + (1.0 - alpha) * float(prev_ema)
                    )
                if bool(triphasic.get("phase1_active", False)):
                    onset = float(np.clip(getattr(cfg, "phase1_budget_coverage_onset", 0.55), 0.0, 0.99))
                    min_factor = float(np.clip(getattr(cfg, "phase1_budget_min_factor", 0.15), 0.0, 1.0))
                    if coverage_global < onset:
                        coverage_factor = 1.0
                    else:
                        coverage_factor = max(min_factor, max(0.0, 1.0 - coverage_global) / max(1e-6, 1.0 - onset))
                    base_budget = int(getattr(cfg, "phase1_base_channel_budget", 16))
                    min_budget = int(getattr(cfg, "phase1_min_channel_budget", 2))
                    effective_budget = int(max(min_budget, round(float(base_budget) * float(coverage_factor))))
                    triphasic_state["phase1_budget_modulator"] = float(coverage_factor)
                    triphasic_state["phase1_effective_channel_budget"] = int(effective_budget)
                if (
                    bool(triphasic.get("phase1_active", False))
                    and not bool(triphasic_state.get("phase1_to_phase2_triggered", False))
                ):
                    epoch_idx = int(epoch + 1)
                    trigger_by_coverage = bool(
                        epoch_idx >= int(getattr(cfg, "phase1_to_phase2_min_epoch", 140))
                        and coverage_global >= float(getattr(cfg, "phase1_to_phase2_coverage_global", 0.75))
                        and min_cov >= float(getattr(cfg, "phase1_to_phase2_min_prefix_coverage", 0.65))
                    )
                    trigger_by_force = bool(epoch_idx >= int(getattr(cfg, "phase1_to_phase2_force_epoch", 220)))
                    if bool(getattr(cfg, "phase1_to_phase2_trigger_enabled", True)) and (trigger_by_coverage or trigger_by_force):
                        handoff_duration = max(0, int(getattr(cfg, "handoff_duration", 15)))
                        triphasic_state["phase1_to_phase2_triggered"] = True
                        triphasic_state["phase1_to_phase2_trigger_epoch"] = int(epoch_idx)
                        triphasic_state["phase1_to_phase2_reason"] = (
                            "force_transition_epoch" if trigger_by_force and not trigger_by_coverage else "coverage_and_min_prefix"
                        )
                        triphasic_state["handoff_start_epoch"] = int(epoch_idx + 1)
                        triphasic_state["handoff_end_epoch"] = int(epoch_idx + 1 + handoff_duration)
                        logger.info(
                            "[seed=%s][%s] triphasic phase1->phase2 trigger at ep %d | reason=%s | cov=%.3f min_cov=%.3f | handoff=[%d,%d)",
                            seed,
                            level_name,
                            epoch_idx,
                            str(triphasic_state["phase1_to_phase2_reason"]),
                            coverage_global,
                            min_cov,
                            int(triphasic_state["handoff_start_epoch"]),
                            int(triphasic_state["handoff_end_epoch"]),
                        )

            retention_snapshot = {}
            checkpoint_eval_seq = None
            checkpoint_eval_tok = None
            mix_eval_seq = None
            mix_eval_tok = None
            snapshot_every = max(0, int(cfg.snapshot_eval_frequency))
            periodic_snapshot = (snapshot_every > 0 and (epoch + 1) % snapshot_every == 0)
            skip_initial_snapshot = bool(getattr(cfg, "skip_snapshot_eval_epoch1", False))
            mix_eval_every = max(0, int(getattr(cfg, "eval_per_module_interval", 0)))
            periodic_mix_eval = (mix_eval_every > 0 and (epoch + 1) % mix_eval_every == 0)
            tier_eval_every = max(0, int(getattr(cfg, "tier_eval_frequency", 0)))
            periodic_tier_eval = (tier_eval_every > 0 and (epoch + 1) % tier_eval_every == 0)
            checkpoint_every = max(0, int(cfg.checkpoint_eval_frequency))
            checkpoint_due = (
                cfg.save_best_checkpoint
                and checkpoint_every > 0
                and ((epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs_this_level)
            )
            snap = None
            if ((epoch == 0 and not skip_initial_snapshot) or epoch + 1 == epochs_this_level or periodic_snapshot):
                snap = evaluate_seen_levels_fn(
                    model, eval_loaders, seen_levels, device,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                    vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                    eval_mode=cfg.snapshot_eval_mode,
                )
                retention_snapshot = {k: _metric_accuracy(v) for k, v in snap.items()}
                if level_name in snap:
                    checkpoint_eval_seq = _metric_accuracy(snap[level_name])
                    checkpoint_eval_tok = _metric_token_accuracy(snap[level_name])
                    if periodic_mix_eval:
                        mix_eval_seq = float(checkpoint_eval_seq)
                        mix_eval_tok = float(checkpoint_eval_tok)
            if snap is None and (periodic_mix_eval or periodic_tier_eval):
                exp_eval = evaluate_seen_levels_fn(
                    model, eval_loaders, [level_name], device,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                    vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                    eval_mode=cfg.snapshot_eval_mode,
                )
                if level_name in exp_eval and periodic_mix_eval:
                    mix_eval_seq = _metric_accuracy(exp_eval[level_name])
                    mix_eval_tok = _metric_token_accuracy(exp_eval[level_name])
            detailed_eval_snapshot = {}
            if experiment_backend is not None and hasattr(experiment_backend, "get_last_eval_details_snapshot"):
                try:
                    snapshot = experiment_backend.get_last_eval_details_snapshot()
                    if snapshot and int(snapshot.get("epoch", -1)) == int(epoch + 1):
                        detailed_eval_snapshot = snapshot
                except Exception:
                    detailed_eval_snapshot = {}
            if checkpoint_due and checkpoint_eval_seq is None:
                ck_eval = evaluate_seen_levels_fn(
                    model, eval_loaders, [level_name], device,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                    vocab_size=vocab_size, max_a_len=cfg.max_a_len,
                    eval_mode=cfg.checkpoint_eval_mode,
                )
                checkpoint_eval_seq = _metric_accuracy(ck_eval[level_name])
                checkpoint_eval_tok = _metric_token_accuracy(ck_eval[level_name])
            if cfg.save_best_checkpoint and checkpoint_eval_seq is not None and checkpoint_eval_seq > best_checkpoint_metric:
                best_checkpoint_metric = float(checkpoint_eval_seq)
                best_checkpoint_epoch = int(epoch + 1)
                best_checkpoint_path = str(checkpoint_dir / f"seed_{seed}_{level_name}_best.pt")
                save_training_checkpoint(
                    Path(best_checkpoint_path),
                    model=model,
                    optimizer=optimizer,
                    seed=seed,
                    level_name=level_name,
                    epoch=epoch + 1,
                    metric_name="eval_seq",
                    metric_value=best_checkpoint_metric,
                    config_snapshot=result["config_snapshot"],
                    runtime_state=(
                        experiment_backend.export_runtime_state()
                        if (experiment_backend is not None and hasattr(experiment_backend, "export_runtime_state"))
                        else None
                    ),
                )

            ms = collect_model_stats(model, include_hidden_stats=cfg.collect_ffn_hidden_stats_each_epoch)
            history.append({
                "epoch": epoch + 1,
                "progress": progress,
                "transition_phase": False,
                "triphasic_enabled": bool(triphasic.get("enabled", False)),
                "phase_name": str(triphasic.get("phase_name", "standard")),
                "phase1_active": bool(triphasic.get("phase1_active", False)),
                "handoff_active": bool(triphasic.get("handoff_active", False)),
                "phase2_active": bool(triphasic.get("phase2_active", False)),
                "phase3_active": bool(triphasic.get("phase3_active", False)),
                "phase1_to_phase2_triggered": bool(triphasic_state.get("phase1_to_phase2_triggered", False)),
                "phase1_to_phase2_trigger_epoch": int(triphasic_state.get("phase1_to_phase2_trigger_epoch", 0)),
                "phase1_to_phase2_reason": str(triphasic_state.get("phase1_to_phase2_reason", "")),
                "phase2_to_phase3_trigger_epoch": int(triphasic_state.get("phase2_to_phase3_trigger_epoch", 0)),
                "train_loss": metrics["loss"],
                "train_accuracy": metrics["accuracy"],
                "train_token_accuracy": metrics.get("token_accuracy", 0.0),
                "train_sequence_accuracy": metrics.get("sequence_accuracy", metrics["accuracy"]),
                "lr": current_lr,
                "lr_phase": current_lr_phase,
                "replay_weight": ctx.effective_replay_weight,
                "retention_old_mean": float(retention_old_mean),
                "consolidation_mix": dict(consolidation_mix),
                "created": created_epoch,
                "pruned": pruned_epoch,
                "active_synapses": ms["active_synapses"],
                "density": ms["density"],
                "growth_need": gn["normalized_need"],
                "creation_budget_effective_train_seq": float(adaptive_max),
                "creation_budget_nominal_train_seq": float(adaptive_max_nominal),
                "creation_fill_ratio_train_seq": float(train_seq_budget_last_fill_ratio),
                "checkpoint_eval_seq": float(checkpoint_eval_seq) if checkpoint_eval_seq is not None else None,
                "checkpoint_eval_tok": float(checkpoint_eval_tok) if checkpoint_eval_tok is not None else None,
                "mix_eval_seq": float(mix_eval_seq) if mix_eval_seq is not None else None,
                "mix_eval_tok": float(mix_eval_tok) if mix_eval_tok is not None else None,
                "detailed_eval_snapshot": detailed_eval_snapshot,
                "creation_stats": creation_stats,
                "pruning_stats": pruning_stats,
                "tenure_stats": tenure_stats,
                "channel_probe_stats": channel_probe_stats,
                "coverage_global": float(triphasic_state.get("coverage_global", 0.0)),
                "min_prefix_coverage": float(triphasic_state.get("min_prefix_coverage", 0.0)),
                "phase1_effective_channel_budget": int(triphasic_state.get("phase1_effective_channel_budget", 0)),
                "phase1_budget_modulator": float(triphasic_state.get("phase1_budget_modulator", 1.0)),
                "phase2_effective_micro_budget": int(triphasic_state.get("phase2_effective_micro_budget", 0)),
                "phase2_budget_reference_active_synapses": int(triphasic_state.get("phase2_budget_reference_active_synapses", 0)),
                "phase12_dual_lr_active": bool(dual_lr_group2_mask_epoch is not None),
                "phase12_dual_lr_scale": float(dual_lr_group2_scale_epoch),
                "phase12_group2_synapses": int(sum(int(mask.sum().item()) for mask in dual_lr_group2_masks.values())) if dual_lr_group2_masks else 0,
                "probationary_channels_epoch": float(channel_probe_stats.get("probationary_channels_epoch", 0.0)),
                "evaluated_channels_epoch": float(channel_probe_stats.get("evaluated_channels_epoch", 0.0)),
                "promoted_probe_channels_epoch": float(channel_probe_stats.get("promoted_probe_channels_epoch", channel_probe_stats.get("tenured_channels_epoch", 0.0))),
                "tenured_channels_epoch": float(channel_probe_stats.get("tenured_channels_epoch", 0.0)),
                "rejected_channels_epoch": float(channel_probe_stats.get("rejected_channels_epoch", 0.0)),
                "local_reinforce_edges_epoch": float(channel_probe_stats.get("local_reinforce_edges_epoch", 0.0)),
                "grad_probe_batches": grad_stats["grad_probe_batches"],
                "grad_signal_mean": grad_stats["grad_signal_mean"],
                "pruning_allowed": ctx_pruning.pruning_allowed,
                "pruning_keep_ratio_effective": float(ctx_pruning.pruning_keep_ratio),
                "prune_cap_from_created_window_enabled": bool(getattr(cfg, "prune_cap_from_created_window_enabled", False)),
                "prune_cap_created_window": int(prune_cap_created_window),
                "prune_cap_max_prune_count": int(prune_cap_max) if prune_cap_max is not None else None,
                "retention_snapshot": retention_snapshot,
            })

            if epoch == 0 or (epoch + 1) % 10 == 0:
                extras = []
                if bool(triphasic.get("enabled", False)):
                    extras.append(f"phase={str(triphasic.get('phase_name', 'standard'))}")
                if int(created_epoch) > 0 or int(pruned_epoch) > 0:
                    extras.append(f"cr={int(created_epoch)} pr={int(pruned_epoch)}")
                if tenure_stats:
                    tenured_total = int(tenure_stats.get("tenured_total", 0.0))
                    new_tenures_total = int(tenure_stats.get("new_tenures_total", 0.0))
                    if tenured_total > 0 or new_tenures_total > 0:
                        extras.append(f"ten={tenured_total}(+{new_tenures_total})")
                if prune_cap_max is not None:
                    extras.append(f"pcap={int(prune_cap_max)}")
                if bool(triphasic.get("phase2_active", False)):
                    extras.append(f"p2bud={int(triphasic_state.get('phase2_effective_micro_budget', adaptive_max))}")
                if dual_lr_group2_mask_epoch is not None:
                    extras.append(f"dual_lr={float(dual_lr_group2_scale_epoch):.2f}x")
                if channel_probe_stats:
                    promoted_epoch = int(
                        channel_probe_stats.get(
                            "promoted_probe_channels_epoch",
                            channel_probe_stats.get("tenured_channels_epoch", 0.0),
                        )
                    )
                    extras.append(
                        "cprobe="
                        f"p{int(channel_probe_stats.get('probationary_channels_epoch', 0.0))}"
                        f"/e{int(channel_probe_stats.get('evaluated_channels_epoch', 0.0))}"
                        f"/m{promoted_epoch}"
                        f"/r{int(channel_probe_stats.get('rejected_channels_epoch', 0.0))}"
                    )
                    extras.append(
                        f"cov={float(triphasic_state.get('coverage_global', 0.0)):.2f}"
                        f" bud={int(triphasic_state.get('phase1_effective_channel_budget', 0))}"
                    )
                logger.info(
                    "[seed=%s][%s] ep %d/%d | seq=%.3f tok=%.3f | lr=%.6g phase=%s | syn=%d%s",
                    seed,
                    level_name,
                    epoch + 1,
                    epochs_this_level,
                    metrics["accuracy"],
                    metrics.get("token_accuracy", 0.0),
                    float(current_lr),
                    str(current_lr_phase),
                    ms["active_synapses"],
                    f" | {' | '.join(extras)}" if extras else "",
                )
                if epoch == 0 and callable(log_epoch1_data_usage_fn):
                    log_epoch1_data_usage_fn(
                        seed=seed,
                        level_name=level_name,
                        epoch=int(epoch + 1),
                    )

        retention = evaluate_seen_levels_fn(
            model, eval_loaders, seen_levels, device,
            pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
            vocab_size=vocab_size, max_a_len=cfg.max_a_len,
            eval_mode=cfg.final_eval_mode,
        )
        current_eval = retention[level_name]
        level_stats = collect_model_stats(model, include_hidden_stats=cfg.collect_ffn_hidden_stats_each_epoch)
        conv = compute_convergence([h for h in history if not h.get("transition_phase")], cfg.convergence_tail_window)

        result["levels"][level_name] = {
            "new_modules": list(spec.new_modules),
            "all_modules": list(spec.all_modules),
            "epochs": epochs_this_level,
            "transition_epochs": transition_epochs,
            "train_history": history,
            "eval_acc_current": _metric_accuracy(current_eval),
            "eval_token_acc_current": _metric_token_accuracy(current_eval),
            "eval_autoreg_acc_current": _metric_autoreg_accuracy(current_eval),
            "eval_autoreg_token_acc_current": float(current_eval.get("autoregressive_token_accuracy", _metric_token_accuracy(current_eval))),
            "eval_tf_acc_current": _metric_teacher_forcing_accuracy(current_eval),
            "eval_tf_token_acc_current": float(current_eval.get("teacher_forcing_token_accuracy", 0.0)),
            "eval_loss_current": float(current_eval["loss"]),
            "retention": retention,
            "model_stats": level_stats,
            "convergence": conv,
        }
        if final_extra_eval_fn is not None and final_extra_eval_context is not None:
            result["levels"][level_name]["extra_eval"] = final_extra_eval_fn(
                model=model,
                device=device,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                vocab_size=vocab_size,
                max_a_len=cfg.max_a_len,
                eval_mode=cfg.final_eval_mode,
                context=final_extra_eval_context,
            )
        last_checkpoint_path = ""
        if cfg.save_last_checkpoint:
            last_checkpoint_path = str(checkpoint_dir / f"seed_{seed}_{level_name}_last.pt")
            save_training_checkpoint(
                Path(last_checkpoint_path),
                model=model,
                optimizer=optimizer,
                seed=seed,
                level_name=level_name,
                epoch=epochs_this_level,
                metric_name="final_eval_seq",
                metric_value=_metric_accuracy(current_eval),
                config_snapshot=result["config_snapshot"],
                runtime_state=(
                    experiment_backend.export_runtime_state()
                    if (experiment_backend is not None and hasattr(experiment_backend, "export_runtime_state"))
                    else None
                ),
            )
        result["checkpoints"][level_name] = {
            "best_path": best_checkpoint_path,
            "best_epoch": int(best_checkpoint_epoch),
            "best_metric_name": "eval_seq",
            "best_metric_value": (float(best_checkpoint_metric) if best_checkpoint_epoch > 0 else None),
            "last_path": last_checkpoint_path,
            "last_metric_name": "final_eval_seq",
            "last_metric_value": _metric_accuracy(current_eval),
        }
        logger.info(
            "[seed=%s][%s] eval_acc(ar)=%.3f | eval_acc(tf)=%.3f | syn=%d",
            seed,
            level_name,
            _metric_accuracy(current_eval),
            float(current_eval.get("teacher_forcing_accuracy", 0.0)),
            level_stats["active_synapses"],
        )

        if bool(getattr(cfg, "use_ewc", False)) and level_name != level_specs[-1].name:
            next_bootstrap_levels = [s.name for s in level_specs[: level_idx + 1]]
            ewc_mix = {str(lv): 1.0 for lv in next_bootstrap_levels}
            ewc_indices = runtime_sample_mixed_indices_fn(
                pools=train_pools,
                mix=ewc_mix,
                total_samples=int(max(1024, int(getattr(cfg, "ewc_fisher_samples", 256)))),
                rng=rng,
            )
            ewc_loader = runtime_build_loader_fn(
                train_dataset,
                ewc_indices,
                cfg.batch_size,
                False,
                cfg.num_workers,
                cfg.pin_memory,
                cfg.persistent_workers,
                cfg.prefetch_factor,
            )
            save_full_param_ewc_reference(model)
            compute_full_param_fisher_information(
                model,
                ewc_loader,
                device=device,
                pad_id=pad_id,
                sos_id=sos_id,
                vocab_size=vocab_size,
                max_a_len=cfg.max_a_len,
                n_samples=int(getattr(cfg, "ewc_fisher_samples", 256)),
                amp_enabled=amp_dense_active,
                amp_dtype=amp_dense_dtype,
            )
            logger.info(
                "[seed=%s][%s] EWC reference refreshed for next level | levels=%s",
                seed,
                level_name,
                ",".join(str(lv) for lv in next_bootstrap_levels),
            )

        if level_name in ("lycee", "college") and _metric_accuracy(current_eval) < float(cfg.degenerate_acc_threshold):
            logger.warning(
                "[seed=%s][%s] DEGENERATE trajectory detected: eval_seq_acc=%.3f < %.3f. "
                "Stopping this seed early to avoid wasting compute.",
                seed, level_name, _metric_accuracy(current_eval), cfg.degenerate_acc_threshold,
            )
            result["degenerate"] = True
            result["degenerate_level"] = level_name
            result["degenerate_acc"] = _metric_accuracy(current_eval)
            break

    last_lv = level_order[-1]
    primary_drop = -1.0
    if "lycee" in result["levels"] and last_lv in result["levels"]:
        ref = float(result["levels"]["lycee"]["eval_acc_current"])
        vals = [float(h["retention_snapshot"].get("primaire", 1.0))
                for h in result["levels"][last_lv]["train_history"]
                if not h.get("transition_phase") and h.get("epoch", 0) <= cfg.drop_window_epochs
                and "primaire" in h.get("retention_snapshot", {})]
        if vals:
            primary_drop = max(0.0, ref - min(vals))

    result["retention_diagnostics"] = {"primary_drop_first_window": primary_drop}
    result["final_model_stats"] = collect_model_stats(model, include_hidden_stats=True)
    if hasattr(model, "get_ffn_hidden_stats"):
        result["final_model_stats"]["ffn_hidden_stats_raw"] = model.get_ffn_hidden_stats(include_raw=True)
    result["go_nogo"] = evaluate_go_nogo(result, cfg, level_order)
    return result
