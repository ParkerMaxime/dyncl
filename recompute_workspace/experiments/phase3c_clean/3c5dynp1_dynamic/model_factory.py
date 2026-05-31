#!/usr/bin/env python3
"""Model construction for 3C5 dynamic primary triphasic run."""

from __future__ import annotations

from models.hybrid_ffn_transformer_3c4_3_2a import HybridDynamicFFNTransformerSeq2Seq


def model_class():
    """Return the 3C4 triphasic-capable hybrid model variant."""
    return HybridDynamicFFNTransformerSeq2Seq
