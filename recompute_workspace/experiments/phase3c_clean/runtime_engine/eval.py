#!/usr/bin/env python3
"""Shared evaluation helpers for clean phase 3C runners."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .runtime import (
        autocast_context,
        build_decoder_input,
        generate_autoregressive,
        resolve_amp_runtime,
        unwrap_compiled_model,
    )
except ImportError:  # pragma: no cover - direct module import fallback
    from runtime import (
        autocast_context,
        build_decoder_input,
        generate_autoregressive,
        resolve_amp_runtime,
        unwrap_compiled_model,
    )


def resolve_eval_mode(eval_mode: str) -> Tuple[bool, bool]:
    mode = str(eval_mode).strip().lower()
    if mode in {"ar", "autoregressive"}:
        return True, False
    if mode in {"ar_tf", "both", "full"}:
        return True, True
    if mode in {"tf", "teacher_forcing", "teacher"}:
        return False, True
    raise ValueError(f"Unsupported eval mode: {eval_mode!r}. Use one of: ar, ar_tf, tf.")


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    pad_id: int,
    sos_id: int,
    eos_id: int,
    vocab_size: int,
    max_a_len: int,
    compute_autoregressive: bool = True,
    compute_teacher_forcing: bool = True,
    amp_enabled: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    model.eval()
    eval_model = unwrap_compiled_model(model)
    total_loss = 0.0
    total_seq_correct = 0.0
    total_seq = 0.0
    total_tok_correct = 0.0
    total_tok = 0.0
    total_tf_seq_correct = 0.0
    total_tf_tok_correct = 0.0
    total_tf_tok = 0.0
    total_samples = 0.0
    amp_active, amp_dtype_resolved = resolve_amp_runtime(model, amp_enabled, amp_dtype)
    for q_indices, a_indices, _sample_indices in loader:
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        token_mask = (a_indices != int(pad_id))
        bs = float(a_indices.size(0))
        total_samples += bs
        tok_n = token_mask.sum().item()

        if compute_teacher_forcing:
            tgt_in = build_decoder_input(a_indices, sos_id=sos_id)
            with autocast_context(device, amp_active, amp_dtype_resolved):
                logits = eval_model(q_indices, tgt_in)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, int(vocab_size)),
                    a_indices.reshape(-1),
                    reduction="none",
                    ignore_index=int(pad_id),
                ).view(a_indices.size(0), int(max_a_len))
            sample_den = token_mask.sum(dim=1).clamp(min=1).to(ce.dtype)
            sample_loss = (ce * token_mask.to(ce.dtype)).sum(dim=1) / sample_den
            loss = sample_loss.mean()
            total_loss += loss.item() * bs

            tf_preds = logits.argmax(dim=-1)
            tf_seq_ok = ((tf_preds == a_indices) | (~token_mask)).all(dim=1)
            tf_tok_ok = ((tf_preds == a_indices) & token_mask).sum().item()
            total_tf_seq_correct += float(tf_seq_ok.sum().item())
            total_tf_tok_correct += float(tf_tok_ok)
            total_tf_tok += float(tok_n)

        if compute_autoregressive:
            preds = generate_autoregressive(
                model=eval_model,
                src_tokens=q_indices,
                sos_id=sos_id,
                eos_id=eos_id,
                max_a_len=max_a_len,
                amp_enabled=amp_active,
                amp_dtype=amp_dtype_resolved,
            )
            seq_ok = ((preds == a_indices) | (~token_mask)).all(dim=1)
            tok_ok = ((preds == a_indices) & token_mask).sum().item()
            total_seq_correct += float(seq_ok.sum().item())
            total_seq += bs
            total_tok_correct += float(tok_ok)
            total_tok += float(tok_n)

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sequence_accuracy": 0.0,
            "token_accuracy": 0.0,
            "autoregressive_accuracy": 0.0,
            "autoregressive_sequence_accuracy": 0.0,
            "autoregressive_token_accuracy": 0.0,
            "teacher_forcing_accuracy": 0.0,
            "teacher_forcing_sequence_accuracy": 0.0,
            "teacher_forcing_token_accuracy": 0.0,
            "n_samples": 0,
        }
    ar_seq = (total_seq_correct / total_seq) if (compute_autoregressive and total_seq > 0) else 0.0
    ar_tok = (total_tok_correct / max(1.0, total_tok)) if (compute_autoregressive and total_tok > 0) else 0.0
    tf_seq = (total_tf_seq_correct / total_samples) if compute_teacher_forcing else 0.0
    tf_tok = (total_tf_tok_correct / max(1.0, total_tf_tok)) if (compute_teacher_forcing and total_tf_tok > 0) else 0.0
    primary_seq = ar_seq if compute_autoregressive else tf_seq
    primary_tok = ar_tok if compute_autoregressive else tf_tok
    mean_loss = (total_loss / total_samples) if compute_teacher_forcing else 0.0
    return {
        "loss": mean_loss,
        "accuracy": primary_seq,
        "sequence_accuracy": primary_seq,
        "token_accuracy": primary_tok,
        "autoregressive_accuracy": ar_seq,
        "autoregressive_sequence_accuracy": ar_seq,
        "autoregressive_token_accuracy": ar_tok,
        "teacher_forcing_accuracy": tf_seq,
        "teacher_forcing_sequence_accuracy": tf_seq,
        "teacher_forcing_token_accuracy": tf_tok,
        "n_samples": int(total_samples),
    }
