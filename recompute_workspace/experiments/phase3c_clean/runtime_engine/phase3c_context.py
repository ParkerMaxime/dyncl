#!/usr/bin/env python3
"""Shared phase 3C context for clean runners."""

from __future__ import annotations

import logging
import sys
from typing import Dict

from curriculum.math_curriculum import MATH_LEVEL_ORDER, normalize_mix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)

LEVEL_ORDER = MATH_LEVEL_ORDER
GROWTH_TARGETS: Dict[str, float] = {
    "primaire": 0.99,
    "college": 0.95,
    "lycee": 0.90,
    "superieur": 0.86,
}

__all__ = ["logger", "LEVEL_ORDER", "GROWTH_TARGETS", "normalize_mix"]
