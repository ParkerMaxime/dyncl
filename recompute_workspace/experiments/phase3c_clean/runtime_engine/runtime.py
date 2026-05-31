#!/usr/bin/env python3
"""Runtime helpers extracted from the legacy 3C1 engine."""

from __future__ import annotations
from contextlib import nullcontext
from typing import Optional, Set, Tuple

import torch
import torch.nn as nn


def get_device(device_config: str = "auto") -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def build_decoder_input(answer_tokens: torch.Tensor, sos_id: int) -> torch.Tensor:
    """Teacher forcing input: [SOS] + answer[:-1]."""
    bsz = int(answer_tokens.size(0))
    sos = torch.full((bsz, 1), int(sos_id), dtype=answer_tokens.dtype, device=answer_tokens.device)
    return torch.cat([sos, answer_tokens[:, :-1]], dim=1)


def resolve_amp_runtime(
    model: nn.Module,
    amp_enabled: Optional[bool],
    amp_dtype: Optional[torch.dtype],
) -> Tuple[bool, Optional[torch.dtype]]:
    enabled = bool(getattr(model, "_dense_amp_enabled", False)) if amp_enabled is None else bool(amp_enabled)
    dtype = getattr(model, "_dense_amp_dtype", None) if amp_dtype is None else amp_dtype
    if dtype not in (torch.bfloat16, torch.float16):
        return False, None
    return bool(enabled), dtype


def autocast_context(
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
):
    if bool(amp_enabled) and str(device).startswith("cuda") and amp_dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    if bool(amp_enabled) and str(device) == "cpu" and amp_dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the eager module behind torch.compile wrappers when available."""
    base = model
    seen: Set[int] = set()
    while hasattr(base, "_orig_mod"):
        nxt = getattr(base, "_orig_mod", None)
        if nxt is None or not isinstance(nxt, nn.Module):
            break
        current_id = id(base)
        if current_id in seen:
            break
        seen.add(current_id)
        base = nxt
    return base


@torch.no_grad()
def generate_autoregressive(
    model: nn.Module,
    src_tokens: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_a_len: int,
    amp_enabled: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Autoregressive decoder for evaluation."""
    bsz = int(src_tokens.size(0))
    generated = torch.full((bsz, 1), int(sos_id), dtype=torch.long, device=src_tokens.device)
    done = torch.zeros((bsz,), dtype=torch.bool, device=src_tokens.device)
    amp_active, amp_dtype_resolved = resolve_amp_runtime(model, amp_enabled, amp_dtype)
    ar_model = unwrap_compiled_model(model)

    for _ in range(int(max_a_len)):
        with autocast_context(src_tokens.device, amp_active, amp_dtype_resolved):
            logits = ar_model(src_tokens, generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        done |= (next_token.squeeze(1) == int(eos_id))
        if bool(done.all().item()):
            break

    out = generated[:, 1:]
    if out.size(1) < int(max_a_len):
        pad = torch.full(
            (bsz, int(max_a_len) - out.size(1)),
            int(eos_id),
            dtype=torch.long,
            device=src_tokens.device,
        )
        out = torch.cat([out, pad], dim=1)
    elif out.size(1) > int(max_a_len):
        out = out[:, : int(max_a_len)]
    return out
