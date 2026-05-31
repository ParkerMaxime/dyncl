#!/usr/bin/env python3
"""Scratchpad conversion and answer-audit helpers for clean 3C5p4."""

from __future__ import annotations

from typing import Iterable

from data.scratchpad_utils import (
    apply_all_scratchpad_to_dataset,
    add_or_sub_scratchpad,
    add_sub_multiple_scratchpad,
    div_remainder_scratchpad,
    div_scratchpad,
    extract_answer,
    gcd_scratchpad,
    is_factor_scratchpad,
    lcm_scratchpad,
    mixed_scratchpad,
    mul_scratchpad,
    run_scratchpad_selftest,
)


def run_answer_audit() -> None:
    """Checks the extract_answer invariant on the scratchpad formats we rely on."""
    run_scratchpad_selftest()
    cases: Iterable[tuple[str, str]] = (
        (add_or_sub_scratchpad("What is 15 - 7?", "8"), "8"),
        (add_sub_multiple_scratchpad("Evaluate 12 + 7 - 4.", "15"), "15"),
        (mul_scratchpad(12, 34), "408"),
        (div_scratchpad("Divide 144 by 12.", "12"), "12"),
        (gcd_scratchpad("Calculate the greatest common factor of 21 and 14.", "7"), "7"),
        (lcm_scratchpad("What is the lowest common multiple of 9 and 6?", "18"), "18"),
        (div_remainder_scratchpad("Calculate the remainder when 29 is divided by 6.", "5"), "5"),
        (is_factor_scratchpad("Is 6 a factor of 42?", "True"), "True"),
        (mixed_scratchpad("What is 8 + 3 * 4?", "20"), "20"),
    )
    for scratchpad_text, expected in cases:
        got = extract_answer(scratchpad_text)
        if got != expected:
            raise AssertionError(
                f"extract_answer invariant failed: expected={expected!r} got={got!r}"
            )


def apply_scratchpad_postprocess(
    train_dataset,
    test_dataset,
    extra_eval_datasets,
    *,
    progress_callback=None,
):
    train_stats = apply_all_scratchpad_to_dataset(
        train_dataset,
        progress_callback=progress_callback,
        dataset_label="train",
    )
    test_stats = apply_all_scratchpad_to_dataset(
        test_dataset,
        progress_callback=progress_callback,
        dataset_label="test",
    )
    extra_stats = {}
    for split_name, dataset in dict(extra_eval_datasets).items():
        extra_stats[str(split_name)] = apply_all_scratchpad_to_dataset(
            dataset,
            progress_callback=progress_callback,
            dataset_label=f"extra_eval:{str(split_name)}",
        )
    return {
        "train": train_stats,
        "test": test_stats,
        "extra_eval": extra_stats,
    }
