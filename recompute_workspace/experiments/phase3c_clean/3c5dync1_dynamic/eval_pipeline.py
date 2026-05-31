#!/usr/bin/env python3
"""Evaluation helpers for the clean 3C5p4 runner."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.scratchpad_utils import extract_answer
from runtime import (
    autocast_context,
    build_decoder_input,
    generate_autoregressive,
    resolve_amp_runtime,
    unwrap_compiled_model,
)
from eval import resolve_eval_mode
from backends import runtime_build_loader


def module_eval_due(epoch_idx: int, frequency: int, final_epoch: int) -> bool:
    if frequency > 0 and epoch_idx % frequency == 0:
        return True
    return epoch_idx >= final_epoch


def tier_eval_due(epoch_idx: int, frequency: int) -> bool:
    if epoch_idx <= 1:
        return False
    if frequency <= 0:
        return False
    return epoch_idx % frequency == 0


def unwrap_indexed_subset_dataset(loader: DataLoader):
    dataset = getattr(loader, "dataset", None)
    while dataset is not None and hasattr(dataset, "dataset"):
        dataset = getattr(dataset, "dataset")
    return dataset


def decode_token_row(
    token_row,
    inv_vocab: Mapping[int, str],
    *,
    pad_id: int,
    sos_id: int,
    eos_id: int,
) -> str:
    chars = []
    for token in token_row:
        t = int(token)
        if t in (pad_id, sos_id):
            continue
        if t == eos_id:
            break
        chars.append(str(inv_vocab.get(t, "")))
    return "".join(chars)


def accumulate_answer_metrics(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    inv_vocab: Mapping[int, str],
    *,
    pad_id: int,
    sos_id: int,
    eos_id: int,
) -> Tuple[float, float, float]:
    seq_correct = 0.0
    tok_correct = 0.0
    tok_total = 0.0

    pred_rows = pred_tokens.detach().cpu().tolist()
    target_rows = target_tokens.detach().cpu().tolist()
    for pred_row, tgt_row in zip(pred_rows, target_rows):
        pred_text = decode_token_row(pred_row, inv_vocab, pad_id=pad_id, sos_id=sos_id, eos_id=eos_id)
        tgt_text = decode_token_row(tgt_row, inv_vocab, pad_id=pad_id, sos_id=sos_id, eos_id=eos_id)
        pred_ans = extract_answer(pred_text)
        tgt_ans = extract_answer(tgt_text)
        if pred_ans == tgt_ans:
            seq_correct += 1.0
        width = max(1, len(pred_ans), len(tgt_ans))
        tok_total += float(width)
        for idx in range(width):
            pred_char = pred_ans[idx] if idx < len(pred_ans) else ""
            tgt_char = tgt_ans[idx] if idx < len(tgt_ans) else ""
            if pred_char == tgt_char:
                tok_correct += 1.0
    return seq_correct, tok_correct, tok_total


@torch.no_grad()
def evaluate_loader_with_answer_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    sos_id: int,
    eos_id: int,
    vocab_size: int,
    max_a_len: int,
    compute_autoregressive: bool = True,
    compute_teacher_forcing: bool = True,
) -> Dict[str, float]:
    model.eval()
    eval_model = unwrap_compiled_model(model)
    amp_active, amp_dtype = resolve_amp_runtime(model, None, None)

    total_loss = 0.0
    total_samples = 0.0
    total_seq_correct_full = 0.0
    total_seq_full = 0.0
    total_tok_correct_full = 0.0
    total_tok_full = 0.0
    total_tf_seq_correct_full = 0.0
    total_tf_tok_correct_full = 0.0
    total_tf_tok_full = 0.0
    total_seq_correct_ans = 0.0
    total_tok_correct_ans = 0.0
    total_tok_ans = 0.0
    total_tf_seq_correct_ans = 0.0
    total_tf_tok_correct_ans = 0.0
    total_tf_tok_ans = 0.0

    base_dataset = unwrap_indexed_subset_dataset(loader)
    vocab_obj = getattr(base_dataset, "vocab", None)
    inv_vocab: Optional[Dict[int, str]] = None
    if isinstance(vocab_obj, Mapping):
        inv_vocab = {int(v): str(k) for k, v in vocab_obj.items()}

    for q_indices, a_indices, _sample_indices in loader:
        q_indices = q_indices.to(device, non_blocking=True)
        a_indices = a_indices.to(device, non_blocking=True)
        token_mask = (a_indices != int(pad_id))
        batch_size = float(a_indices.size(0))
        tok_n = float(token_mask.sum().item())
        total_samples += batch_size

        if compute_teacher_forcing:
            tgt_in = build_decoder_input(a_indices, sos_id=sos_id)
            with autocast_context(device, amp_active, amp_dtype):
                logits = eval_model(q_indices, tgt_in)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, int(vocab_size)),
                    a_indices.reshape(-1),
                    reduction="none",
                    ignore_index=int(pad_id),
                ).view(a_indices.size(0), int(max_a_len))
            sample_den = token_mask.sum(dim=1).clamp(min=1).to(ce.dtype)
            sample_loss = (ce * token_mask.to(ce.dtype)).sum(dim=1) / sample_den
            total_loss += float(sample_loss.mean().item()) * batch_size

            tf_preds = logits.argmax(dim=-1)
            tf_seq_ok = ((tf_preds == a_indices) | (~token_mask)).all(dim=1)
            tf_tok_ok = ((tf_preds == a_indices) & token_mask).sum().item()
            total_tf_seq_correct_full += float(tf_seq_ok.sum().item())
            total_tf_tok_correct_full += float(tf_tok_ok)
            total_tf_tok_full += tok_n

            if inv_vocab is not None:
                seq_ans, tok_ans, tok_ans_n = accumulate_answer_metrics(
                    tf_preds,
                    a_indices,
                    inv_vocab,
                    pad_id=pad_id,
                    sos_id=sos_id,
                    eos_id=eos_id,
                )
                total_tf_seq_correct_ans += float(seq_ans)
                total_tf_tok_correct_ans += float(tok_ans)
                total_tf_tok_ans += float(tok_ans_n)

        if compute_autoregressive:
            preds = generate_autoregressive(
                model=eval_model,
                src_tokens=q_indices,
                sos_id=sos_id,
                eos_id=eos_id,
                max_a_len=max_a_len,
                amp_enabled=amp_active,
                amp_dtype=amp_dtype,
            )
            seq_ok = ((preds == a_indices) | (~token_mask)).all(dim=1)
            tok_ok = ((preds == a_indices) & token_mask).sum().item()
            total_seq_correct_full += float(seq_ok.sum().item())
            total_seq_full += batch_size
            total_tok_correct_full += float(tok_ok)
            total_tok_full += tok_n

            if inv_vocab is not None:
                seq_ans, tok_ans, tok_ans_n = accumulate_answer_metrics(
                    preds,
                    a_indices,
                    inv_vocab,
                    pad_id=pad_id,
                    sos_id=sos_id,
                    eos_id=eos_id,
                )
                total_seq_correct_ans += float(seq_ans)
                total_tok_correct_ans += float(tok_ans)
                total_tok_ans += float(tok_ans_n)

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "sequence_accuracy": 0.0,
            "token_accuracy": 0.0,
            "eval_seq_ans": 0.0,
            "eval_tok_ans": 0.0,
            "eval_seq_full": 0.0,
            "eval_tok_full": 0.0,
            "autoregressive_accuracy": 0.0,
            "autoregressive_sequence_accuracy": 0.0,
            "autoregressive_token_accuracy": 0.0,
            "teacher_forcing_accuracy": 0.0,
            "teacher_forcing_sequence_accuracy": 0.0,
            "teacher_forcing_token_accuracy": 0.0,
            "n_samples": 0,
        }

    ar_seq_full = (total_seq_correct_full / total_seq_full) if (compute_autoregressive and total_seq_full > 0.0) else 0.0
    ar_tok_full = (total_tok_correct_full / max(1.0, total_tok_full)) if (compute_autoregressive and total_tok_full > 0.0) else 0.0
    tf_seq_full = (total_tf_seq_correct_full / total_samples) if compute_teacher_forcing else 0.0
    tf_tok_full = (total_tf_tok_correct_full / max(1.0, total_tf_tok_full)) if (compute_teacher_forcing and total_tf_tok_full > 0.0) else 0.0

    if inv_vocab is None:
        ar_seq_ans = ar_seq_full
        ar_tok_ans = ar_tok_full
        tf_seq_ans = tf_seq_full
        tf_tok_ans = tf_tok_full
    else:
        ar_seq_ans = (total_seq_correct_ans / total_seq_full) if (compute_autoregressive and total_seq_full > 0.0) else 0.0
        ar_tok_ans = (total_tok_correct_ans / max(1.0, total_tok_ans)) if (compute_autoregressive and total_tok_ans > 0.0) else 0.0
        tf_seq_ans = (total_tf_seq_correct_ans / total_samples) if compute_teacher_forcing else 0.0
        tf_tok_ans = (total_tf_tok_correct_ans / max(1.0, total_tf_tok_ans)) if (compute_teacher_forcing and total_tf_tok_ans > 0.0) else 0.0

    primary_seq_ans = ar_seq_ans if compute_autoregressive else tf_seq_ans
    primary_tok_ans = ar_tok_ans if compute_autoregressive else tf_tok_ans
    primary_seq_full = ar_seq_full if compute_autoregressive else tf_seq_full
    primary_tok_full = ar_tok_full if compute_autoregressive else tf_tok_full
    mean_loss = (total_loss / total_samples) if compute_teacher_forcing else 0.0

    return {
        "loss": float(mean_loss),
        "accuracy": float(primary_seq_ans),
        "sequence_accuracy": float(primary_seq_ans),
        "token_accuracy": float(primary_tok_ans),
        "eval_seq_ans": float(primary_seq_ans),
        "eval_tok_ans": float(primary_tok_ans),
        "eval_seq_full": float(primary_seq_full),
        "eval_tok_full": float(primary_tok_full),
        "autoregressive_accuracy": float(ar_seq_ans),
        "autoregressive_sequence_accuracy": float(ar_seq_ans),
        "autoregressive_token_accuracy": float(ar_tok_ans),
        "teacher_forcing_accuracy": float(tf_seq_ans),
        "teacher_forcing_sequence_accuracy": float(tf_seq_ans),
        "teacher_forcing_token_accuracy": float(tf_tok_ans),
        "n_samples": int(total_samples),
    }


def build_extra_eval_loaders(
    *,
    bundle,
    cfg,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split_name, dataset in dict(getattr(bundle, "extra_eval_datasets", {})).items():
        level_pools = dict(getattr(bundle, "extra_eval_level_pools", {}).get(str(split_name), {}))
        if "primaire" in level_pools and len(level_pools["primaire"]) > 0:
            pool = level_pools["primaire"]
        elif level_pools:
            _level_name, pool = next(iter(level_pools.items()))
        else:
            continue
        loaders[str(split_name)] = runtime_build_loader(
            dataset,
            pool,
            int(cfg.batch_size),
            False,
            int(cfg.num_workers),
            bool(getattr(cfg, "pin_memory", True)),
            bool(getattr(cfg, "persistent_workers", False)),
            int(getattr(cfg, "prefetch_factor", 0)),
        )
    return loaders


def evaluate_extra_splits(
    *,
    model: nn.Module,
    device: torch.device,
    pad_id: int,
    sos_id: int,
    eos_id: int,
    vocab_size: int,
    max_a_len: int,
    eval_mode: str,
    context,
) -> Dict[str, Dict[str, float]]:
    compute_autoregressive, compute_teacher_forcing = resolve_eval_mode(eval_mode)
    loaders = dict(context or {})
    results: Dict[str, Dict[str, float]] = {}
    for split_name, loader in loaders.items():
        results[str(split_name)] = evaluate_loader_with_answer_metrics(
            model=model,
            loader=loader,
            device=device,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            vocab_size=vocab_size,
            max_a_len=max_a_len,
            compute_autoregressive=compute_autoregressive,
            compute_teacher_forcing=compute_teacher_forcing,
        )
    return results
