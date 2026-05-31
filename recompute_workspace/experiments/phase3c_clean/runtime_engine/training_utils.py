#!/usr/bin/env python3
"""Shared training loop helpers for clean phase 3C runners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class EpochContext:
    epoch: int
    is_first_level: bool
    newsyn_active: bool
    old_synapse_grad_scale: float
    mask_strength: float
    mask_active: bool
    boost_active: bool
    boost_factor: float
    effective_replay_weight: float
    crystal_active: bool
    crystal_beta: float
    creation_factor: float
    consolidation_window_active: bool
    consolidation_cycle_pos: int
    consolidation_freeze_creation: bool
    consolidation_freeze_pruning: bool
    pruning_allowed: bool
    pruning_keep_ratio: float


def compute_growth_need(
    level_name: str,
    train_accuracy: float,
    growth_targets: Mapping[str, float],
    growth_target_primaire: Optional[float] = None,
) -> Dict[str, float]:
    target = growth_targets.get(level_name, 0.90)
    if level_name == "primaire" and growth_target_primaire is not None:
        target = float(growth_target_primaire)
    shortfall = max(0.0, target - train_accuracy)
    scale = 0.10
    return {
        "normalized_need": float(np.clip(shortfall / scale, 0.0, 1.0)),
        "target_acc": target,
        "shortfall": shortfall,
    }


def update_train_seq_budget_state(
    prev_ema: Optional[float],
    train_sequence_accuracy: float,
    cfg,
) -> Tuple[float, float]:
    seq = float(np.clip(train_sequence_accuracy, 0.0, 1.0))
    alpha = float(np.clip(cfg.creation_train_seq_ema_alpha, 1e-6, 1.0))
    if prev_ema is None:
        ema = seq
    else:
        ema = alpha * seq + (1.0 - alpha) * float(prev_ema)
    gamma = float(max(1e-6, cfg.creation_train_seq_budget_gamma))
    min_ratio = float(np.clip(cfg.creation_train_seq_budget_min_ratio, 0.0, 1.0))
    scale = math.pow(max(0.0, 1.0 - ema), gamma)
    scale = float(np.clip(scale, min_ratio, 1.0))
    return float(ema), scale


def compute_epoch_lr(
    base_lr: float,
    epoch: int,
    total_epochs: int,
    cfg,
    maturity_epoch: Optional[int] = None,
) -> float:
    sgdr_cycles = getattr(cfg, "lr_sgdr_cycles", None)
    if sgdr_cycles:
        step = int(epoch) + 1
        active_cycle = None
        for cycle in sgdr_cycles:
            ep_start, ep_end, lr_max, lr_min, warmup_epochs = cycle
            start = int(ep_start)
            end = int(ep_end)
            if step < start or step > end:
                continue
            active_cycle = cycle
        if active_cycle is not None:
            ep_start, ep_end, lr_max, lr_min, warmup_epochs = active_cycle
            start = int(ep_start)
            end = int(ep_end)

            local_step = step - start
            warmup = int(max(0, warmup_epochs))
            lr_hi = float(lr_max)
            lr_lo = float(lr_min)

            if warmup > 0 and local_step <= warmup:
                if start == 0:
                    factor = float(local_step) / float(max(1, warmup))
                    return lr_hi * factor
                prev_floor = float(getattr(cfg, "lr_sgdr_prev_floor", lr_lo))
                progress = float(local_step) / float(max(1, warmup))
                return prev_floor + (lr_hi - prev_floor) * progress

            cosine_start = start + max(0, warmup)
            cosine_span = max(1, end - cosine_start)
            progress = float(step - cosine_start) / float(cosine_span)
            progress = float(np.clip(progress, 0.0, 1.0))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_lo + (lr_hi - lr_lo) * cosine

    schedule = str(getattr(cfg, "lr_schedule", "none")).strip().lower()
    maturity_override = int(max(0, getattr(cfg, "lr_schedule_maturity_epoch_override", 0)))
    if maturity_override > 0:
        maturity_epoch = maturity_override
    if schedule == "none":
        return float(base_lr)
    if schedule == "warmup_creation_plateau_cosine":
        total = max(1, int(total_epochs))
        warmup = int(max(0, min(getattr(cfg, "lr_warmup_epochs", 0), total - 1)))
        min_ratio = float(np.clip(getattr(cfg, "lr_schedule_min_ratio", 0.10), 0.0, 1.0))
        step = int(epoch) + 1
        maturity = int(max(0, maturity_epoch or 0))
        if warmup > 0 and step <= warmup:
            factor = float(step) / float(warmup)
            return float(base_lr) * float(factor)
        if maturity <= 0 or step <= maturity:
            return float(base_lr)
        decay_steps = max(1, int(total_epochs) - maturity)
        progress = float(step - maturity) / float(decay_steps)
        progress = float(np.clip(progress, 0.0, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_ratio + (1.0 - min_ratio) * cosine
        return float(base_lr) * float(factor)
    if schedule == "creation_plateau_cosine":
        min_ratio = float(np.clip(getattr(cfg, "lr_schedule_min_ratio", 0.10), 0.0, 1.0))
        step = int(epoch) + 1
        maturity = int(max(0, maturity_epoch or 0))
        if maturity <= 0 or step <= maturity:
            return float(base_lr)
        decay_steps = max(1, int(total_epochs) - maturity)
        progress = float(step - maturity) / float(decay_steps)
        progress = float(np.clip(progress, 0.0, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_ratio + (1.0 - min_ratio) * cosine
        return float(base_lr) * float(factor)
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule: {schedule!r}")

    total = max(1, int(total_epochs))
    warmup = int(max(0, min(getattr(cfg, "lr_warmup_epochs", 0), total - 1)))
    min_ratio = float(np.clip(getattr(cfg, "lr_schedule_min_ratio", 0.10), 0.0, 1.0))
    step = int(epoch) + 1

    if warmup > 0 and step <= warmup:
        factor = float(step) / float(warmup)
    else:
        decay_steps = max(1, total - warmup)
        progress = float(step - warmup) / float(decay_steps)
        progress = float(np.clip(progress, 0.0, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_ratio + (1.0 - min_ratio) * cosine
    return float(base_lr) * float(factor)


def infer_lr_phase(
    epoch: int,
    total_epochs: int,
    cfg,
    maturity_epoch: Optional[int] = None,
) -> str:
    sgdr_cycles = getattr(cfg, "lr_sgdr_cycles", None)
    if sgdr_cycles:
        step = int(epoch) + 1
        active_idx = None
        active_cycle = None
        for idx, cycle in enumerate(sgdr_cycles, start=1):
            ep_start, ep_end, _lr_max, _lr_min, warmup_epochs = cycle
            start = int(ep_start)
            end = int(ep_end)
            if step < start or step > end:
                continue
            active_idx = idx
            active_cycle = cycle
        if active_cycle is not None and active_idx is not None:
            ep_start, _ep_end, _lr_max, _lr_min, warmup_epochs = active_cycle
            start = int(ep_start)
            warmup = int(max(0, warmup_epochs))
            local_step = step - start
            if warmup > 0 and local_step <= warmup:
                return f"sgdr_warmup_c{active_idx}"
            return f"sgdr_cosine_c{active_idx}"
        return "sgdr"

    schedule = str(getattr(cfg, "lr_schedule", "none")).strip().lower()
    maturity_override = int(max(0, getattr(cfg, "lr_schedule_maturity_epoch_override", 0)))
    if maturity_override > 0:
        maturity_epoch = maturity_override
    if schedule == "none":
        return "constant"
    step = int(epoch) + 1
    total = max(1, int(total_epochs))
    warmup = int(max(0, min(getattr(cfg, "lr_warmup_epochs", 0), total - 1)))
    maturity = int(max(0, maturity_epoch or 0))
    if schedule == "cosine":
        if warmup > 0 and step <= warmup:
            return "warmup"
        return "cosine"
    if schedule == "creation_plateau_cosine":
        if maturity > 0 and step > maturity:
            return "cosine"
        return "plateau"
    if schedule == "warmup_creation_plateau_cosine":
        if warmup > 0 and step <= warmup:
            return "warmup"
        if maturity > 0 and step > maturity:
            return "cosine"
        return "plateau"
    return schedule


def set_optimizer_lr(optimizer: optim.Optimizer, lr_value: float) -> None:
    lr_value = float(max(1e-8, lr_value))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_value


def capture_active_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    if not hasattr(model, "weights") or not hasattr(model, "masks"):
        return masks
    for key in model.weights.keys():
        mask = getattr(model, model.masks[key])
        masks[key] = (mask > 0).detach().clone()
    return masks


def compute_mask_strength(epoch: int, cfg, retention_old_mean: float) -> float:
    warmup = max(0, cfg.mask_warmup_epochs)
    ramp = max(0, cfg.mask_ramp_epochs)
    if epoch <= warmup:
        return 1.0
    k = cfg.mask_floor_k
    target = cfg.mask_floor_target
    fmin, fmax = cfg.mask_floor_min, cfg.mask_floor_max
    floor = fmin + (fmax - fmin) / (1.0 + math.exp(k * (retention_old_mean - target)))
    if ramp <= 0:
        return floor
    progress = float(epoch - warmup) / float(ramp)
    return float(np.clip(max(floor, 1.0 - progress), 0.0, 1.0))


def compute_replay_weight(retention_mean: float, cfg) -> float:
    shortfall = max(0.0, cfg.retention_target - retention_mean)
    denom = max(1e-8, 0.20 * float(cfg.retention_target))
    normalized = float(np.clip(shortfall / denom, 0.0, 1.0))
    w = cfg.replay_weight_base + cfg.retention_gain * normalized
    return float(np.clip(w, cfg.replay_weight_base, cfg.replay_weight_max))


def compute_creation_factor(retention_old_mean: float, cfg) -> float:
    if retention_old_mean < 0.0:
        return 1.0
    ratio = retention_old_mean / max(1e-8, cfg.retention_target)
    return float(np.clip(ratio, cfg.growth_min_factor, cfg.growth_max_factor))


def resolve_mix_tables(
    mix_variant: str,
    mix_start_default: Mapping[str, Mapping[str, float]],
    mix_end_default: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    mix_start = {k: dict(v) for k, v in mix_start_default.items()}
    mix_end = {k: dict(v) for k, v in mix_end_default.items()}
    if str(mix_variant) == "college_soft_transfer":
        mix_start["college"] = {"primaire": 0.90, "college": 0.10}
        mix_end["college"] = {"primaire": 0.20, "college": 0.80}
    elif str(mix_variant) == "lycee_soft_transfer":
        mix_start["lycee"] = {"primaire": 0.15, "college": 0.70, "lycee": 0.15}
        mix_end["lycee"] = {"primaire": 0.10, "college": 0.20, "lycee": 0.70}
    elif str(mix_variant) == "college_oracle_equal":
        mix_start["college"] = {"primaire": 0.50, "college": 0.50}
        mix_end["college"] = {"primaire": 0.50, "college": 0.50}
    return mix_start, mix_end


def build_consolidation_mix(
    seen_levels: Sequence[str],
    mix_mode: str,
    normalize_mix_fn,
) -> Dict[str, float]:
    mode = str(mix_mode).strip().lower()
    if mode == "equal_seen_levels":
        return normalize_mix_fn({lv: 1.0 for lv in seen_levels})
    raise ValueError(f"Unsupported consolidation_mix_mode={mix_mode!r}")


def build_epoch_context(
    cfg,
    level_idx: int,
    epoch: int,
    retention_old_mean: float,
    current_replay_weight: float,
    growth_need: float,
) -> EpochContext:
    is_first = (level_idx == 0)
    newsyn_level_ok = level_idx >= int(cfg.newsyn_min_level_idx)
    masking_level_ok = level_idx >= int(cfg.masking_min_level_idx)
    boost_level_ok = level_idx >= int(cfg.boost_min_level_idx)

    newsyn_active = (
        not is_first
        and cfg.enable_newsyn
        and newsyn_level_ok
        and epoch < cfg.newsyn_epochs
    )
    if newsyn_active:
        ramp = min(1.0, epoch / max(1, cfg.newsyn_epochs - 1))
        old_grad_scale = cfg.old_synapse_grad_scale_start + ramp * (1.0 - cfg.old_synapse_grad_scale_start)
    else:
        old_grad_scale = 1.0

    mask_strength = 0.0
    if not is_first and cfg.enable_loss_masking and masking_level_ok:
        mask_strength = compute_mask_strength(epoch + 1, cfg, retention_old_mean)
    mask_active = mask_strength > 0.0

    boost_active = False
    boost_factor = 1.0
    effective_replay_weight = current_replay_weight
    if not is_first and cfg.enable_replay_boost and boost_level_ok:
        cycle_pos = (epoch + 1) % cfg.boost_cycle
        if cycle_pos == 0:
            cycle_pos = cfg.boost_cycle
        boost_active = cfg.boost_duration > 0 and cycle_pos > (cfg.boost_cycle - cfg.boost_duration)
        if boost_active:
            boost_factor = cfg.boost_factor
            effective_replay_weight *= boost_factor

    crystal_active = not is_first and cfg.enable_crystallization

    creation_factor = 1.0
    if not is_first and cfg.enable_growth_retention and retention_old_mean >= 0:
        creation_factor = compute_creation_factor(retention_old_mean, cfg)

    consolidation_window_active = False
    consolidation_cycle_pos = 0
    consolidation_level_ok = level_idx >= int(cfg.consolidation_min_level_idx)
    consolidation_start_epoch = int(max(1, cfg.consolidation_start_epoch))
    consolidation_cycle = int(max(1, cfg.consolidation_cycle))
    consolidation_duration = int(max(0, cfg.consolidation_duration))
    if (
        not is_first
        and cfg.enable_consolidation_windows
        and consolidation_level_ok
        and consolidation_duration > 0
    ):
        epoch_index = int(epoch) + 1
        if epoch_index >= consolidation_start_epoch:
            consolidation_cycle_pos = ((epoch_index - consolidation_start_epoch) % consolidation_cycle) + 1
            consolidation_window_active = consolidation_cycle_pos <= consolidation_duration

    consolidation_freeze_creation = bool(
        consolidation_window_active and bool(cfg.consolidation_freeze_creation)
    )
    consolidation_freeze_pruning = bool(
        consolidation_window_active and bool(cfg.consolidation_freeze_pruning)
    )

    pruning_allowed = ((epoch + 1) % cfg.pruning_frequency == 0)
    if newsyn_active and cfg.disable_pruning_during_newsyn:
        pruning_allowed = False
    if consolidation_freeze_pruning:
        pruning_allowed = False

    return EpochContext(
        epoch=int(epoch),
        is_first_level=bool(is_first),
        newsyn_active=bool(newsyn_active),
        old_synapse_grad_scale=float(old_grad_scale),
        mask_strength=float(mask_strength),
        mask_active=bool(mask_active),
        boost_active=bool(boost_active),
        boost_factor=float(boost_factor),
        effective_replay_weight=float(effective_replay_weight),
        crystal_active=bool(crystal_active),
        crystal_beta=float(cfg.crystal_beta),
        creation_factor=float(creation_factor),
        consolidation_window_active=bool(consolidation_window_active),
        consolidation_cycle_pos=int(consolidation_cycle_pos),
        consolidation_freeze_creation=bool(consolidation_freeze_creation),
        consolidation_freeze_pruning=bool(consolidation_freeze_pruning),
        pruning_allowed=bool(pruning_allowed),
        pruning_keep_ratio=float(np.clip(
            cfg.pruning_keep_ratio + 0.04 * growth_need,
            cfg.pruning_keep_ratio,
            0.995,
        )) if pruning_allowed else 1.0,
    )


def _linear_fit(values):
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        return {"slope": 0.0, "r2": 0.0}
    x = np.arange(arr.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, arr, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((arr - pred) ** 2))
    ss_tot = float(np.sum((arr - arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return {"slope": float(slope), "r2": float(r2)}


def compute_convergence(history, tail_window=10):
    acc = [float(h.get("train_accuracy", 0.0)) for h in history]
    loss = [float(h.get("train_loss", 0.0)) for h in history]
    af, lf = _linear_fit(acc), _linear_fit(loss)
    n, w = len(acc), max(1, tail_window)
    if n >= 2 * w:
        ad = float(np.mean(acc[-w:])) - float(np.mean(acc[-2 * w:-w]))
        ld = float(np.mean(loss[-2 * w:-w])) - float(np.mean(loss[-w:]))
    elif n >= 2:
        split = max(1, n // 2)
        ad = float(np.mean(acc[split:])) - float(np.mean(acc[:split]))
        ld = float(np.mean(loss[:split])) - float(np.mean(loss[split:]))
    else:
        ad = ld = 0.0
    effective_window = min(w, n) if n > 0 else 1
    return {
        "train_acc_r2": af["r2"],
        "train_acc_slope": af["slope"],
        "train_loss_r2": lf["r2"],
        "train_loss_slope": lf["slope"],
        "train_acc_delta_tail": ad,
        "train_loss_delta_tail": ld,
        "train_acc_std_tail": float(np.std(acc[-effective_window:])) if acc else 0.0,
        "train_loss_std_tail": float(np.std(loss[-effective_window:])) if loss else 0.0,
    }


def evaluate_go_nogo(result: Dict, cfg, level_order: Sequence[str]) -> Dict[str, Any]:
    levels = result.get("levels", {})
    last_level = level_order[-1]
    missing_core = [lv for lv in ("primaire", last_level) if lv not in levels]

    if missing_core:
        primaire_acc = float(levels.get("primaire", {}).get("eval_acc_current", 0.0))
        final_acc = float(levels.get(last_level, {}).get("eval_acc_current", 0.0))
        reasons = [f"incomplete_curriculum_missing={','.join(missing_core)}"]
        if bool(result.get("degenerate", False)):
            reasons.append(
                f"degenerate_at={result.get('degenerate_level', 'unknown')}"
                f"(acc={float(result.get('degenerate_acc', -1.0)):.3f})"
            )
        return {
            "go": False,
            "reasons": reasons,
            "primaire_acc": primaire_acc,
            "final_acc": final_acc,
            "max_regression": 1.0,
            "regressions": {},
            "targets": {
                "primaire_min_acc": cfg.primaire_min_acc,
                "final_acc_target": cfg.final_acc_target,
                "max_regression": cfg.max_regression,
            },
        }

    primaire_acc = float(levels["primaire"]["eval_acc_current"])
    final_acc = float(levels[last_level]["eval_acc_current"])
    final_ret = levels[last_level]["retention"]
    regressions = {}
    for level_name in level_order[:-1]:
        ref = float(levels[level_name]["eval_acc_current"])
        final = float(final_ret[level_name]["accuracy"])
        regressions[level_name] = max(0.0, ref - final)
    max_reg = max(regressions.values()) if regressions else 0.0

    reasons = []
    if primaire_acc < cfg.primaire_min_acc:
        reasons.append(f"primaire_acc={primaire_acc:.3f} < {cfg.primaire_min_acc:.3f}")
    if final_acc < cfg.final_acc_target:
        reasons.append(f"final_acc={final_acc:.3f} < target={cfg.final_acc_target:.3f}")
    if max_reg > cfg.max_regression:
        reasons.append(f"max_regression={max_reg:.3f} > {cfg.max_regression:.3f}")
    if bool(result.get("degenerate", False)):
        reasons.append(
            f"degenerate_at={result.get('degenerate_level', 'unknown')}"
            f"(acc={float(result.get('degenerate_acc', -1.0)):.3f})"
        )

    return {
        "go": len(reasons) == 0,
        "reasons": reasons,
        "primaire_acc": primaire_acc,
        "final_acc": final_acc,
        "max_regression": max_reg,
        "regressions": regressions,
        "targets": {
            "primaire_min_acc": cfg.primaire_min_acc,
            "final_acc_target": cfg.final_acc_target,
            "max_regression": cfg.max_regression,
        },
    }
