#!/usr/bin/env python3
"""Shared backend registries and wrappers for clean phase 3C runners."""

from __future__ import annotations

from typing import Any, Callable

_RUNTIME_DATASET_BACKEND: Any | None = None
_EXPERIMENT_BACKEND: Any | None = None


def set_runtime_dataset_backend(backend: Any | None) -> None:
    global _RUNTIME_DATASET_BACKEND
    _RUNTIME_DATASET_BACKEND = backend


def clear_runtime_dataset_backend() -> None:
    set_runtime_dataset_backend(None)


def get_runtime_dataset_backend() -> Any | None:
    return _RUNTIME_DATASET_BACKEND


def set_experiment_backend(backend: Any | None) -> None:
    global _EXPERIMENT_BACKEND
    _EXPERIMENT_BACKEND = backend


def clear_experiment_backend() -> None:
    set_experiment_backend(None)


def get_experiment_backend() -> Any | None:
    return _EXPERIMENT_BACKEND


def _default_sample_mixed_indices(*, pools, mix, total_samples, rng):
    from curriculum.math_curriculum import sample_mixed_indices

    return sample_mixed_indices(pools=pools, mix=mix, total_samples=total_samples, rng=rng)


def _default_build_loader(
    dataset,
    indices,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
):
    try:
        from .loaders import build_loader
    except ImportError:  # pragma: no cover - direct module import fallback
        from loaders import build_loader

    return build_loader(
        dataset,
        indices,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )


def _default_interpolate_mix(level_name, progress, mix_start, mix_end):
    from curriculum.math_curriculum import interpolate_mix

    return interpolate_mix(level_name, progress, mix_start, mix_end)


def _default_evaluate_seen_levels(
    model,
    eval_loaders,
    seen_levels,
    device,
    pad_id,
    sos_id,
    eos_id,
    vocab_size,
    max_a_len,
    eval_mode="ar_tf",
):
    try:
        from .eval import evaluate_loader, resolve_eval_mode
    except ImportError:  # pragma: no cover - direct module import fallback
        from eval import evaluate_loader, resolve_eval_mode

    compute_autoregressive, compute_teacher_forcing = resolve_eval_mode(eval_mode)
    return {
        lv: evaluate_loader(
            model,
            eval_loaders[lv],
            device,
            pad_id,
            sos_id,
            eos_id,
            vocab_size,
            max_a_len,
            compute_autoregressive=compute_autoregressive,
            compute_teacher_forcing=compute_teacher_forcing,
        )
        for lv in seen_levels
    }


def runtime_sample_mixed_indices(*, pools, mix, total_samples, rng, fallback: Callable[..., Any] | None = None):
    backend = get_runtime_dataset_backend()
    if backend is not None:
        sampled = backend.sample_indices(
            pools=pools,
            mix=mix,
            total_samples=total_samples,
            rng=rng,
        )
        if sampled is not None:
            return sampled
    resolved_fallback = fallback or _default_sample_mixed_indices
    return resolved_fallback(pools=pools, mix=mix, total_samples=total_samples, rng=rng)


def runtime_build_loader(
    dataset,
    indices,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    *,
    fallback: Callable[..., Any] | None = None,
):
    backend = get_runtime_dataset_backend()
    if backend is not None:
        loader = backend.build_loader(
            dataset,
            indices,
            batch_size,
            shuffle,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )
        if loader is not None:
            return loader
    resolved_fallback = fallback or _default_build_loader
    return resolved_fallback(
        dataset,
        indices,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )


def experiment_interpolate_mix(
    level_name,
    progress,
    mix_start,
    mix_end,
    *,
    fallback: Callable[..., Any] | None = None,
):
    backend = get_experiment_backend()
    if backend is not None:
        mix = backend.interpolate_mix(
            level_name=level_name,
            progress=progress,
            mix_start=mix_start,
            mix_end=mix_end,
            fallback=fallback or _default_interpolate_mix,
        )
        if mix is not None:
            return mix
    resolved_fallback = fallback or _default_interpolate_mix
    return resolved_fallback(level_name, progress, mix_start, mix_end)


def experiment_evaluate_seen_levels(
    model,
    eval_loaders,
    seen_levels,
    device,
    *,
    pad_id,
    sos_id,
    eos_id,
    vocab_size,
    max_a_len,
    eval_mode,
    fallback: Callable[..., Any] | None = None,
):
    backend = get_experiment_backend()
    if backend is not None:
        evaluated = backend.evaluate_seen_levels(
            model=model,
            eval_loaders=eval_loaders,
            seen_levels=seen_levels,
            device=device,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            vocab_size=vocab_size,
            max_a_len=max_a_len,
            eval_mode=eval_mode,
            fallback=fallback or _default_evaluate_seen_levels,
        )
        if evaluated is not None:
            return evaluated
    resolved_fallback = fallback or _default_evaluate_seen_levels
    return resolved_fallback(
        model,
        eval_loaders,
        seen_levels,
        device,
        pad_id,
        sos_id,
        eos_id,
        vocab_size,
        max_a_len,
        eval_mode=eval_mode,
    )
