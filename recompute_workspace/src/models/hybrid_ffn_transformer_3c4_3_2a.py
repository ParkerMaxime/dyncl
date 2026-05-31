"""
Hybrid Transformer seq2seq with dense attention and dynamic sparse FFN only (3C4.3.2a variant).

Goal:
- 3C4.3.2a variant, same hybrid FFN mechanics as 3C4.3.1e
- keep the 3B/3C dynamic API on FFN matrices
- make all attention projections dense and static
- preserve compatibility with the existing 3C.1 training loop
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import torch
import torch.nn as nn

from .sparse_transformer_3c4_2e_lite import SparseTransformerSeq2Seq

logger = logging.getLogger(__name__)


class HybridDynamicFFNTransformerSeq2Seq(SparseTransformerSeq2Seq):
    """
    Transformer with:
    - dense attention projections (Q/K/V/O)
    - dynamic sparse FFN projections only

    The dynamic interface is intentionally preserved on FFN matrices through
    `self.weights`, `self.masks`, replay tags, creation, pruning, and age/usage
    tracking. This lets the existing 3C.1 mechanism stack operate on the FFN
    only, without touching the attention routing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_mode = "dense"
        self.dynamic_scope = "ffn_only"
        stats = self.get_connectivity_stats()
        logger.info(
            "HybridDynamicFFNTransformerSeq2Seq ready: dense attention + dynamic FFN only"
        )
        logger.info(
            "  Dynamic FFN synapses: %d/%d (%.2f%%)",
            stats.total_active,
            stats.total_possible,
            100.0 * stats.overall_density,
        )
        logger.info(
            "  Dense attention parameters: %d",
            self.count_dense_attention_parameters(),
        )

    def _init_sparse_transformer_connectivity(
        self,
        initial_connectivity_attn: float,
        initial_connectivity_ffn: float,
    ) -> None:
        """
        Register:
        - dense attention projections in `self.dense_weights`
        - dynamic sparse FFN projections in `self.weights`

        `initial_connectivity_attn` is ignored by design because attention is
        fully dense in 3C2.0.
        """
        _ = float(initial_connectivity_attn)

        self.dense_weights = nn.ParameterDict()
        self.dense_attention_keys: List[str] = []

        group = 1

        for l in range(self.n_enc_layers):
            p = f"enc{l}"
            self._add_dense_weight(f"{p}_Q", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_K", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_V", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_O", self.d_model, self.d_model)
            self._add_sparse_weight(f"{p}_FFN_up", self.d_model, self.d_ff, initial_connectivity_ffn, group)
            self._add_sparse_weight(f"{p}_FFN_down", self.d_ff, self.d_model, initial_connectivity_ffn, group)
            self._apply_ffn_channel_complete(p)
            self._dst_groups.append(group)
            group += 1

        for l in range(self.n_dec_layers):
            p = f"dec{l}"
            self._add_dense_weight(f"{p}_self_Q", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_self_K", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_self_V", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_self_O", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_cross_Q", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_cross_K", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_cross_V", self.d_model, self.d_model)
            self._add_dense_weight(f"{p}_cross_O", self.d_model, self.d_model)
            self._add_sparse_weight(f"{p}_FFN_up", self.d_model, self.d_ff, initial_connectivity_ffn, group)
            self._add_sparse_weight(f"{p}_FFN_down", self.d_ff, self.d_model, initial_connectivity_ffn, group)
            self._apply_ffn_channel_complete(p)
            self._dst_groups.append(group)
            group += 1

    def _add_dense_weight(self, key: str, in_dim: int, out_dim: int) -> None:
        std = math.sqrt(2.0 / float(max(1, in_dim + out_dim)))
        w = torch.randn(in_dim, out_dim) * std
        self.dense_weights[key] = nn.Parameter(w)
        self.dense_attention_keys.append(key)

    def _dense_linear(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return torch.matmul(x, self.dense_weights[key])

    def _encoder_self_attention(
        self,
        x: torch.Tensor,
        layer_idx: int,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        p = f"enc{layer_idx}"
        q = self._split_heads(self._dense_linear(x, f"{p}_Q"))
        k = self._split_heads(self._dense_linear(x, f"{p}_K"))
        v = self._split_heads(self._dense_linear(x, f"{p}_V"))
        out = self._scaled_dot_product_attention(q, k, v, causal_mask=None, key_padding_mask=key_padding_mask)
        out = self._merge_heads(out)
        return self._dense_linear(out, f"{p}_O")

    def _decoder_self_attention(
        self,
        x: torch.Tensor,
        layer_idx: int,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        p = f"dec{layer_idx}_self"
        q = self._split_heads(self._dense_linear(x, f"{p}_Q"))
        k = self._split_heads(self._dense_linear(x, f"{p}_K"))
        v = self._split_heads(self._dense_linear(x, f"{p}_V"))
        out = self._scaled_dot_product_attention(
            q, k, v, causal_mask=causal_mask, key_padding_mask=key_padding_mask
        )
        out = self._merge_heads(out)
        return self._dense_linear(out, f"{p}_O")

    def _decoder_cross_attention(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        layer_idx: int,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        p = f"dec{layer_idx}_cross"
        q = self._split_heads(self._dense_linear(x, f"{p}_Q"))
        k = self._split_heads(self._dense_linear(memory, f"{p}_K"))
        v = self._split_heads(self._dense_linear(memory, f"{p}_V"))
        out = self._scaled_dot_product_attention(q, k, v, causal_mask=None, key_padding_mask=memory_padding_mask)
        out = self._merge_heads(out)
        return self._dense_linear(out, f"{p}_O")

    def count_dense_attention_parameters(self) -> int:
        return int(sum(int(p.numel()) for p in self.dense_weights.values()))

    def count_dynamic_parameters(self) -> int:
        return int(sum(int(getattr(self, self.masks[key]).sum().item()) for key in self.weights.keys()))

    def count_dense_parameters(self) -> int:
        total = 0
        for name, p in self.named_parameters():
            if name.startswith("weights."):
                continue
            total += int(p.numel())
        return int(total)
