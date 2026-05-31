#!/usr/bin/env python3
"""Derived-cache metadata helpers for clean dataset bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from curriculum.math_curriculum import MathSeq2SeqDataset

from .types import CachedMathSeq2SeqDataset, DerivedCacheSpec


def cache_dir(spec: DerivedCacheSpec) -> Path:
    return Path(spec.cache_root) / f"{spec.cache_name}_{spec.version}"


def metadata_path(spec: DerivedCacheSpec) -> Path:
    return cache_dir(spec) / "meta.json"


def build_signature_payload(spec: DerivedCacheSpec) -> Dict[str, Any]:
    return {
        "cache_name": spec.cache_name,
        "version": spec.version,
        "max_q_len": int(spec.max_q_len),
        "max_a_len": int(spec.max_a_len),
        "raw_max_samples_per_module": int(spec.raw_max_samples_per_module),
        "raw_max_test_samples_per_module": int(spec.raw_max_test_samples_per_module),
        "runtime_train_cap_per_module": int(spec.runtime_train_cap_per_module),
        "runtime_eval_cap_per_module": int(spec.runtime_eval_cap_per_module),
        "scratchpad_policy": str(spec.scratchpad_policy),
        "tiering_policy": str(spec.tiering_policy),
        "extra_eval_splits": list(spec.extra_eval_splits),
        "modules_by_level": {
            str(level_name): list(modules)
            for level_name, modules in sorted(dict(spec.modules_by_level).items())
        },
    }


def build_dataset_identity_payload(spec: DerivedCacheSpec) -> Dict[str, Any]:
    """Dataset identity independent from experiment/run cache naming."""
    payload = build_signature_payload(spec)
    payload.pop("cache_name", None)
    return payload


def build_signature(spec: DerivedCacheSpec) -> str:
    payload = json.dumps(build_dataset_identity_payload(spec), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_metadata(spec: DerivedCacheSpec) -> Dict[str, Any]:
    payload = build_signature_payload(spec)
    payload["signature"] = build_signature(spec)
    return payload


def metadata_matches(spec: DerivedCacheSpec, on_disk: Dict[str, Any]) -> bool:
    expected = build_dataset_identity_payload(spec)
    payload = dict(on_disk)
    if "modules_by_level" not in payload and spec.modules_by_level:
        payload["modules_by_level"] = {
            str(level_name): list(modules)
            for level_name, modules in sorted(dict(spec.modules_by_level).items())
        }
    return all(payload.get(key) == value for key, value in expected.items())


def compatible_cache_specs(spec: DerivedCacheSpec) -> list[DerivedCacheSpec]:
    """Return exact cache first, then other cache dirs with the same dataset identity."""
    candidates: list[DerivedCacheSpec] = []
    seen: set[str] = set()

    def _add(candidate: DerivedCacheSpec) -> None:
        root_key = str(cache_dir(candidate).resolve())
        if root_key in seen:
            return
        seen.add(root_key)
        candidates.append(candidate)

    if bundle_is_compatible(spec) and bundle_files_present(spec):
        _add(spec)

    root = Path(spec.cache_root)
    if not root.exists():
        return candidates
    suffix = f"_{spec.version}"
    for meta_path in sorted(root.glob(f"*{suffix}/meta.json")):
        cache_root = meta_path.parent
        cache_dir_name = cache_root.name
        if not cache_dir_name.endswith(suffix):
            continue
        cache_name = cache_dir_name[: -len(suffix)]
        if not cache_name:
            continue
        candidate = DerivedCacheSpec(
            cache_root=spec.cache_root,
            cache_name=cache_name,
            version=spec.version,
            max_q_len=spec.max_q_len,
            max_a_len=spec.max_a_len,
            raw_max_samples_per_module=spec.raw_max_samples_per_module,
            raw_max_test_samples_per_module=spec.raw_max_test_samples_per_module,
            runtime_train_cap_per_module=spec.runtime_train_cap_per_module,
            runtime_eval_cap_per_module=spec.runtime_eval_cap_per_module,
            scratchpad_policy=spec.scratchpad_policy,
            tiering_policy=spec.tiering_policy,
            extra_eval_splits=spec.extra_eval_splits,
            modules_by_level=spec.modules_by_level,
        )
        if bundle_is_compatible(candidate) and bundle_files_present(candidate):
            _add(candidate)
    return candidates


def _dataset_file_path(spec: DerivedCacheSpec, split: str) -> Path:
    return cache_dir(spec) / f"{split}_dataset.npz"


def _dataset_dir_path(spec: DerivedCacheSpec, split: str) -> Path:
    return cache_dir(spec) / f"{split}_dataset"


def _pool_file_path(spec: DerivedCacheSpec, name: str) -> Path:
    return cache_dir(spec) / f"{name}.npz"


def save_metadata(spec: DerivedCacheSpec) -> Dict[str, Any]:
    root = cache_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    payload = expected_metadata(spec)
    metadata_path(spec).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_metadata(spec: DerivedCacheSpec) -> Dict[str, Any] | None:
    path = metadata_path(spec)
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def bundle_is_compatible(spec: DerivedCacheSpec) -> bool:
    payload = load_metadata(spec)
    if payload is None:
        return False
    return metadata_matches(spec, payload)


def save_dataset(spec: DerivedCacheSpec, split: str, dataset: MathSeq2SeqDataset) -> None:
    """Persist large datasets as memory-mappable arrays.

    Older caches used a single compressed npz. That minimized disk usage but
    forced full decompression into RAM at run startup. The directory format
    below keeps large token arrays in standalone .npy files so np.load can use
    mmap_mode="r" during training.
    """
    root = _dataset_dir_path(spec, split)
    root.mkdir(parents=True, exist_ok=True)
    (root / "vocab.json").write_text(json.dumps(dataset.vocab, sort_keys=True), encoding="utf-8")
    np.save(root / "level_ids.npy", np.asarray(dataset.level_ids, dtype=np.int64), allow_pickle=False)
    np.save(
        root / "encoded_questions.npy",
        np.asarray(dataset.encoded_questions, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        root / "encoded_answers.npy",
        np.asarray(dataset.encoded_answers, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(root / "level_names.npy", np.asarray(dataset.level_names, dtype=str), allow_pickle=False)
    np.save(root / "module_names.npy", np.asarray(dataset.module_names, dtype=str), allow_pickle=False)

    raw_meta = {
        "format": "npy_memmap_v1",
        "length": int(len(dataset)),
        "max_q_len": int(spec.max_q_len),
        "max_a_len": int(spec.max_a_len),
        "has_raw_text": False,
    }
    (root / "meta.json").write_text(json.dumps(raw_meta, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_dir_files_present(spec: DerivedCacheSpec, split: str) -> bool:
    root = _dataset_dir_path(spec, split)
    required = (
        "meta.json",
        "vocab.json",
        "level_ids.npy",
        "encoded_questions.npy",
        "encoded_answers.npy",
        "level_names.npy",
        "module_names.npy",
    )
    return root.is_dir() and all((root / name).exists() for name in required)


def _load_dataset_dir(
    spec: DerivedCacheSpec,
    split: str,
    vocab: Dict[str, int] | None = None,
) -> CachedMathSeq2SeqDataset:
    root = _dataset_dir_path(spec, split)
    effective_vocab = dict(vocab) if vocab else json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    return CachedMathSeq2SeqDataset(
        vocab=effective_vocab,
        max_q_len=int(spec.max_q_len),
        max_a_len=int(spec.max_a_len),
        level_ids=np.load(root / "level_ids.npy", mmap_mode="r", allow_pickle=False),
        encoded_questions=np.load(root / "encoded_questions.npy", mmap_mode="r", allow_pickle=False),
        encoded_answers=np.load(root / "encoded_answers.npy", mmap_mode="r", allow_pickle=False),
        level_names=np.load(root / "level_names.npy", mmap_mode="r", allow_pickle=False),
        module_names=np.load(root / "module_names.npy", mmap_mode="r", allow_pickle=False),
    )


def load_dataset(spec: DerivedCacheSpec, split: str, vocab: Dict[str, int] | None = None) -> MathSeq2SeqDataset:
    if _dataset_dir_files_present(spec, split):
        return _load_dataset_dir(spec, split, vocab)

    payload = np.load(_dataset_file_path(spec, split), allow_pickle=True)
    effective_vocab = dict(vocab) if vocab else json.loads(str(payload["vocab_json"].tolist()))
    dataset = MathSeq2SeqDataset(
        questions=payload["questions"].tolist(),
        answers=payload["answers"].tolist(),
        level_names=payload["level_names"].tolist(),
        module_names=payload["module_names"].tolist(),
        level_ids=payload["level_ids"].astype(np.int64),
        vocab=effective_vocab,
        max_q_len=int(spec.max_q_len),
        max_a_len=int(spec.max_a_len),
    )
    dataset.encoded_questions = payload["encoded_questions"].astype(np.int64, copy=False)
    dataset.encoded_answers = payload["encoded_answers"].astype(np.int64, copy=False)
    return dataset


def bundle_files_present(spec: DerivedCacheSpec) -> bool:
    required = [
        metadata_path(spec),
        _pool_file_path(spec, "train_level_pools"),
        _pool_file_path(spec, "eval_level_pools"),
        _pool_file_path(spec, "train_module_pools"),
        _pool_file_path(spec, "eval_module_pools"),
        _pool_file_path(spec, "train_tier_pools"),
        _pool_file_path(spec, "eval_tier_pools"),
    ]
    required_datasets = ["train", "test"]
    for split_name in spec.extra_eval_splits:
        required_datasets.append(f"eval_{split_name}")
        required.append(_pool_file_path(spec, f"eval_level_pools__{split_name}"))
        required.append(_pool_file_path(spec, f"eval_module_pools__{split_name}"))
    return all(path.exists() for path in required) and all(
        _dataset_dir_files_present(spec, split_name) or _dataset_file_path(spec, split_name).exists()
        for split_name in required_datasets
    )


def save_pools(spec: DerivedCacheSpec, name: str, pools: Dict[str, np.ndarray]) -> None:
    serializable = {
        str(key): np.asarray(value, dtype=np.int64)
        for key, value in pools.items()
        if len(np.asarray(value, dtype=np.int64)) > 0
    }
    np.savez_compressed(_pool_file_path(spec, name), **serializable)


def load_pools(spec: DerivedCacheSpec, name: str) -> Dict[str, np.ndarray]:
    path = _pool_file_path(spec, name)
    if not path.exists():
        return {}
    payload = np.load(path, allow_pickle=False)
    return {str(key): np.asarray(payload[key], dtype=np.int64) for key in payload.files}


def save_nested_pools(spec: DerivedCacheSpec, name: str, pools: Dict[str, Dict[str, np.ndarray]]) -> None:
    flattened: Dict[str, np.ndarray] = {}
    for outer_key, inner in pools.items():
        for inner_key, values in inner.items():
            flattened[f"{outer_key}::{inner_key}"] = np.asarray(values, dtype=np.int64)
    np.savez_compressed(_pool_file_path(spec, name), **flattened)


def load_nested_pools(spec: DerivedCacheSpec, name: str) -> Dict[str, Dict[str, np.ndarray]]:
    path = _pool_file_path(spec, name)
    if not path.exists():
        return {}
    payload = np.load(path, allow_pickle=False)
    nested: Dict[str, Dict[str, np.ndarray]] = {}
    for key in payload.files:
        outer_key, inner_key = str(key).split("::", 1)
        nested.setdefault(outer_key, {})[inner_key] = np.asarray(payload[key], dtype=np.int64)
    return nested
