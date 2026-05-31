#!/usr/bin/env python3
"""Shared training helpers extracted from the legacy 3C1 engine."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from .runtime import autocast_context, build_decoder_input, resolve_amp_runtime
except ImportError:  # pragma: no cover - direct module import fallback
    from runtime import autocast_context, build_decoder_input, resolve_amp_runtime


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    pad_id: int,
    sos_id: int,
    vocab_size: int,
    max_a_len: int,
    level_ids_lookup: Optional[torch.Tensor] = None,
    old_level_ids: Optional[Sequence[int]] = None,
    replay_weight: float = 1.0,
    enable_crystallization: bool = False,
    crystal_vital_floor: float = 0.05,
    crystal_default_floor: float = 0.30,
    crystal_beta: float = 0.70,
    tag_quantile: float = 0.70,
    plasticity_mask: Optional[Mapping[str, torch.Tensor]] = None,
    old_synapse_grad_scale: float = 1.0,
    enable_loss_masking: bool = False,
    enable_output_head_guard: bool = False,
    old_token_ids: Optional[Sequence[int]] = None,
    mask_strength: float = 1.0,
    mask_penalty: float = 12.0,
    label_smoothing: float = 0.0,
    amp_enabled: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
    dual_lr_group2_mask: Optional[Mapping[str, torch.Tensor]] = None,
    dual_lr_group2_scale: float = 1.0,
    ewc_loss_fn=None,
) -> Dict[str, float]:
    model.train()
    total_loss = torch.zeros((), device=device)
    total_seq_correct = torch.zeros((), device=device)
    total_seq = torch.zeros((), device=device)
    total_tok_correct = torch.zeros((), device=device)
    total_tok = torch.zeros((), device=device)
    total_old_loss = torch.zeros((), device=device)
    total_new_loss = torch.zeros((), device=device)
    total_old_n = torch.zeros((), device=device)
    total_new_n = torch.zeros((), device=device)

    old_levels = sorted(set(int(x) for x in old_level_ids)) if old_level_ids else []
    old_levels_t = torch.tensor(old_levels, dtype=torch.long, device=device) if old_levels else None
    old_token_t = None
    if old_token_ids:
        old_token_t = torch.tensor(sorted(set(int(x) for x in old_token_ids)), dtype=torch.long, device=device)
    w_replay = float(max(1e-6, replay_weight))
    output_head_guard_rows_t = None
    output_head_guard_rows = 0
    output_head_guard_scale = float(max(0.0, min(1.0, old_synapse_grad_scale)))
    if enable_output_head_guard and old_token_t is not None and old_token_t.numel() > 0 and hasattr(model, "output_proj"):
        output_weight = getattr(model.output_proj, "weight", None)
        if output_weight is not None and int(output_weight.shape[0]) > 0:
            output_head_guard_rows_t = old_token_t[
                (old_token_t >= 0) & (old_token_t < int(output_weight.shape[0]))
            ]
            output_head_guard_rows = int(output_head_guard_rows_t.numel())
    output_head_guard_active = bool(enable_output_head_guard and output_head_guard_rows > 0)
    amp_active, amp_dtype_resolved = resolve_amp_runtime(model, amp_enabled, amp_dtype)
    loader_iter = iter(loader)
    while True:
        try:
            q_indices, a_indices, sample_indices = next(loader_iter)
        except StopIteration:
            break
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        sample_indices = sample_indices.to(dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_active, amp_dtype_resolved):
            tgt_in = build_decoder_input(a_indices, sos_id=sos_id)
            logits = model(q_indices, tgt_in)
            bsz = int(logits.size(0))

            token_ce = nn.functional.cross_entropy(
                logits.reshape(-1, int(vocab_size)),
                a_indices.reshape(-1),
                reduction="none",
                ignore_index=int(pad_id),
                label_smoothing=float(max(0.0, min(0.3, label_smoothing))),
            ).view(bsz, int(max_a_len))
            token_mask = (a_indices != int(pad_id))
            sample_den = token_mask.sum(dim=1).clamp(min=1).to(token_ce.dtype)
            per_sample = (token_ce * token_mask.to(token_ce.dtype)).sum(dim=1) / sample_den

            if (level_ids_lookup is None) or (old_levels_t is None) or (old_levels_t.numel() == 0):
                loss = per_sample.mean()
                old_mask = torch.zeros((bsz,), dtype=torch.bool, device=device)
                new_mask = ~old_mask
            else:
                batch_level_ids = level_ids_lookup[sample_indices].to(device, non_blocking=True)
                old_mask = torch.isin(batch_level_ids, old_levels_t)
                new_mask = ~old_mask

                if enable_loss_masking and old_token_t is not None and old_token_t.numel() > 0:
                    sidx = torch.nonzero(new_mask, as_tuple=True)[0]
                    if sidx.numel() > 0:
                        penalty = float(max(0.0, mask_strength)) * float(max(0.0, mask_penalty))
                        ml = logits[sidx].clone()
                        ml[:, :, old_token_t] = ml[:, :, old_token_t] - penalty
                        tl = nn.functional.cross_entropy(
                            ml.reshape(-1, int(vocab_size)),
                            a_indices[sidx].reshape(-1),
                            reduction="none",
                            ignore_index=int(pad_id),
                            label_smoothing=float(max(0.0, min(0.3, label_smoothing))),
                        ).view(sidx.numel(), int(max_a_len))
                        tm = token_mask[sidx].to(tl.dtype)
                        sd = tm.sum(dim=1).clamp(min=1.0)
                        per_sample[sidx] = (tl * tm).sum(dim=1) / sd

                ow = old_mask.to(per_sample.dtype)
                nw = new_mask.to(per_sample.dtype)
                oc = ow.sum()
                nc_ = nw.sum()
                ol = (per_sample * ow).sum() / torch.clamp(oc, min=1.0)
                nl = (per_sample * nw).sum() / torch.clamp(nc_, min=1.0)
                has_old = oc > 0
                has_new = nc_ > 0
                loss = torch.where(
                    has_old & has_new,
                    nl + w_replay * ol,
                    torch.where(has_old, w_replay * ol, nl),
                )

                total_old_loss += ol.detach() * oc
                total_new_loss += nl.detach() * nc_
                total_old_n += oc
                total_new_n += nc_
            if ewc_loss_fn is not None:
                loss = loss + ewc_loss_fn(model)
        loss.backward()

        if enable_crystallization and hasattr(model, "weights") and hasattr(model, "_get_replay_tag"):
            for key in model.weights.keys():
                W = model.weights[key]
                if W.grad is None:
                    continue
                tag = model._get_replay_tag(key).to(W.grad.device).detach()
                tag = torch.clamp(tag, 0.0, 1.0)
                mask_t = getattr(model, model.masks[key]).to(W.grad.device)
                active = mask_t > 0
                q = float(max(0.0, min(1.0, tag_quantile)))
                threshold = (
                    torch.quantile(tag[active], q)
                    if int(active.sum().item()) > 0
                    else torch.tensor(1.0, device=tag.device)
                )
                is_vital = (tag >= threshold) & active
                beta = float(max(0.0, min(1.0, crystal_beta)))
                nv_floor = float(max(0.50, crystal_default_floor))
                nv_scaling = torch.clamp(1.0 - beta * tag, min=nv_floor, max=1.0)
                v_scaling = torch.full_like(tag, float(crystal_vital_floor))
                scaling = torch.where(is_vital, v_scaling, nv_scaling)
                W.grad.data.mul_(scaling)

        if plasticity_mask is not None and hasattr(model, "weights"):
            osc = float(max(0.0, min(1.0, old_synapse_grad_scale)))
            for key in model.weights.keys():
                W = model.weights[key]
                if W.grad is None or key not in plasticity_mask:
                    continue
                pm = plasticity_mask[key].to(W.grad.device)
                gs = torch.where(pm, torch.ones_like(W.grad), torch.full_like(W.grad, osc))
                W.grad.data.mul_(gs)

        if dual_lr_group2_mask is not None and hasattr(model, "weights"):
            group2_scale = float(max(1.0, dual_lr_group2_scale))
            for key in model.weights.keys():
                W = model.weights[key]
                if W.grad is None or key not in dual_lr_group2_mask:
                    continue
                g2_mask = dual_lr_group2_mask[key].to(W.grad.device)
                if int(g2_mask.sum().item()) <= 0:
                    continue
                W.grad.data[g2_mask] *= group2_scale

        if output_head_guard_active:
            out_proj = getattr(model, "output_proj", None)
            if out_proj is not None and output_head_guard_rows_t is not None:
                if getattr(out_proj, "weight", None) is not None and out_proj.weight.grad is not None:
                    guard_rows = output_head_guard_rows_t.to(device=out_proj.weight.grad.device)
                    guarded_weight_grad = out_proj.weight.grad.index_select(0, guard_rows) * output_head_guard_scale
                    out_proj.weight.grad.index_copy_(0, guard_rows, guarded_weight_grad)
                if getattr(out_proj, "bias", None) is not None and out_proj.bias.grad is not None:
                    guard_rows = output_head_guard_rows_t.to(device=out_proj.bias.grad.device)
                    guarded_bias_grad = out_proj.bias.grad.index_select(0, guard_rows) * output_head_guard_scale
                    out_proj.bias.grad.index_copy_(0, guard_rows, guarded_bias_grad)
        optimizer.step()

        bs = float(bsz)
        preds = logits.argmax(dim=-1)
        seq_ok = ((preds == a_indices) | (~token_mask)).all(dim=1)
        tok_ok = ((preds == a_indices) & token_mask).sum().to(torch.float32)
        tok_n = token_mask.sum().to(torch.float32)

        total_loss += loss.detach() * bs
        total_seq_correct += seq_ok.sum().to(torch.float32)
        total_seq += bs
        total_tok_correct += tok_ok
        total_tok += tok_n

    n_seq = int(total_seq.item())
    if n_seq == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sequence_accuracy": 0.0,
            "token_accuracy": 0.0,
            "loss_old": 0.0,
            "loss_new": 0.0,
            "n_old": 0.0,
            "n_new": 0.0,
            "output_head_guard_active": False,
            "output_head_guard_rows": 0,
            "output_head_guard_scale": 1.0,
        }
    n_old = int(total_old_n.item())
    n_new = int(total_new_n.item())
    return {
        "loss": float(total_loss.item()) / n_seq,
        "accuracy": float(total_seq_correct.item()) / n_seq,
        "sequence_accuracy": float(total_seq_correct.item()) / n_seq,
        "token_accuracy": float(total_tok_correct.item()) / max(1.0, float(total_tok.item())),
        "loss_old": float(total_old_loss.item()) / max(1, n_old),
        "loss_new": float(total_new_loss.item()) / max(1, n_new),
        "n_old": float(n_old),
        "n_new": float(n_new),
        "output_head_guard_active": bool(output_head_guard_active),
        "output_head_guard_rows": int(output_head_guard_rows),
        "output_head_guard_scale": float(output_head_guard_scale if output_head_guard_active else 1.0),
    }


def save_full_param_ewc_reference(model: nn.Module) -> None:
    refs = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        refs[str(name)] = param.detach().clone()
    setattr(model, "_ewc_reference_params", refs)


def compute_full_param_ewc_loss(model: nn.Module, lambda_ewc: float = 1000.0) -> torch.Tensor:
    refs = getattr(model, "_ewc_reference_params", None) or {}
    fisher = getattr(model, "_ewc_fisher_params", None) or {}
    if not refs or not fisher:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ref = refs.get(str(name))
        info = fisher.get(str(name))
        if ref is None or info is None:
            continue
        penalty = penalty + (info.to(param.device) * (param - ref.to(param.device)).pow(2)).sum()
    return penalty * float(lambda_ewc / 2.0)


def compute_full_param_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
    sos_id: int,
    vocab_size: int,
    max_a_len: int,
    n_samples: int = 256,
    amp_enabled: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> None:
    was_training = model.training
    model.train()
    fisher = {
        str(name): torch.zeros_like(param, device=param.device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    seen = 0
    loader_iter = iter(dataloader)
    while seen < int(n_samples):
        try:
            q_indices, a_indices, _sample_indices = next(loader_iter)
        except StopIteration:
            break
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with autocast_context(device, *(resolve_amp_runtime(model, amp_enabled, amp_dtype))):
            tgt_in = build_decoder_input(a_indices, sos_id=sos_id)
            logits = model(q_indices, tgt_in)
            bsz = int(logits.size(0))
            token_ce = nn.functional.cross_entropy(
                logits.reshape(-1, int(vocab_size)),
                a_indices.reshape(-1),
                reduction="none",
                ignore_index=int(pad_id),
            ).view(bsz, int(max_a_len))
            token_mask = (a_indices != int(pad_id))
            sample_den = token_mask.sum(dim=1).clamp(min=1).to(token_ce.dtype)
            loss = ((token_ce * token_mask.to(token_ce.dtype)).sum(dim=1) / sample_den).mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[str(name)].add_(param.grad.detach().pow(2))
        seen += bsz
    denom = float(max(1, seen))
    for name in fisher:
        fisher[name].div_(denom)
    setattr(model, "_ewc_fisher_params", fisher)
    if not was_training:
        model.eval()


def run_gradient_probe(
    model,
    train_dataset,
    train_pools,
    old_levels,
    level_mix,
    rng,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    device,
    cfg,
    pad_id: int,
    sos_id: int,
    vocab_size: int,
    max_a_len: int,
    *,
    normalize_mix_fn,
    runtime_sample_mixed_indices_fn,
    runtime_build_loader_fn,
) -> Dict[str, float]:
    if (not cfg.enable_gradient_tags or cfg.tag_probe_batches <= 0 or not old_levels or not hasattr(model, "weights")):
        return {"grad_probe_batches": 0.0, "grad_signal_mean": 0.0}

    if level_mix:
        filtered_mix = {
            str(lv): float(w)
            for lv, w in level_mix.items()
            if str(lv) in old_levels and float(w) > 0.0
        }
        mix = normalize_mix_fn(filtered_mix) if filtered_mix else normalize_mix_fn({lv: 1.0 for lv in old_levels})
    else:
        mix = normalize_mix_fn({lv: 1.0 for lv in old_levels})
    total = max(batch_size, batch_size * cfg.tag_probe_batches)
    indices = runtime_sample_mixed_indices_fn(pools=train_pools, mix=mix, total_samples=total, rng=rng)
    loader = runtime_build_loader_fn(
        train_dataset,
        indices,
        batch_size,
        False,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )

    was_training = model.training
    model.train()
    grad_accum = {k: torch.zeros_like(model.weights[k].data) for k in model.weights.keys()}
    model.zero_grad(set_to_none=True)
    batches = 0
    for q_indices, a_indices, _sample_indices in loader:
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        tgt_in = build_decoder_input(a_indices, sos_id=sos_id)
        logits = model(q_indices, tgt_in)
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, int(vocab_size)),
            a_indices.reshape(-1),
            reduction="none",
            ignore_index=int(pad_id),
            label_smoothing=float(max(0.0, min(0.3, getattr(cfg, "label_smoothing", 0.0)))),
        ).view(a_indices.size(0), int(max_a_len))
        token_mask = (a_indices != int(pad_id)).to(ce.dtype)
        per_sample = (ce * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
        loss = per_sample.mean()
        loss.backward()
        for k in model.weights.keys():
            if model.weights[k].grad is not None:
                grad_accum[k].add_(model.weights[k].grad.detach().abs())
        model.zero_grad(set_to_none=True)
        batches += 1
        if batches >= cfg.tag_probe_batches:
            break

    d, a = float(cfg.tag_decay), float(cfg.tag_update_rate)
    signal_means = []
    with torch.no_grad():
        for k in model.weights.keys():
            mask = getattr(model, model.masks[k])
            tag = model._get_replay_tag(k).to(mask.device)
            signal = grad_accum[k].to(mask.device)
            if batches > 0:
                signal = signal / float(batches)
            signal = signal * mask
            if cfg.tag_layerwise_normalize:
                mx = float(signal.max().item())
                signal = signal / mx if mx > 1e-8 else torch.zeros_like(signal)
            active = mask > 0
            if int(active.sum().item()) > 0:
                signal_means.append(float(signal[active].mean().item()))
            tag.copy_(d * tag + a * signal)
            tag.clamp_(0.0, 1.0)

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return {
        "grad_probe_batches": float(batches),
        "grad_signal_mean": float(np.mean(signal_means)) if signal_means else 0.0,
    }
