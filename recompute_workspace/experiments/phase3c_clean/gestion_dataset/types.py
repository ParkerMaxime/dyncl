#!/usr/bin/env python3
"""Dataclasses shared by the clean dataset-management layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


IndexArray = np.ndarray


@dataclass(frozen=True)
class DerivedCacheSpec:
    cache_root: Path
    cache_name: str
    version: str
    max_q_len: int
    max_a_len: int
    raw_max_samples_per_module: int
    raw_max_test_samples_per_module: int
    runtime_train_cap_per_module: int
    runtime_eval_cap_per_module: int
    scratchpad_policy: str
    tiering_policy: str
    extra_eval_splits: tuple[str, ...] = ()
    modules_by_level: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    train_dataset: object
    test_dataset: object
    extra_eval_datasets: Dict[str, object] = field(default_factory=dict)
    train_level_pools: Dict[str, IndexArray] = field(default_factory=dict)
    eval_level_pools: Dict[str, IndexArray] = field(default_factory=dict)
    extra_eval_level_pools: Dict[str, Dict[str, IndexArray]] = field(default_factory=dict)
    train_module_pools: Dict[str, IndexArray] = field(default_factory=dict)
    eval_module_pools: Dict[str, IndexArray] = field(default_factory=dict)
    extra_eval_module_pools: Dict[str, Dict[str, IndexArray]] = field(default_factory=dict)
    train_tier_pools: Dict[str, Dict[str, IndexArray]] = field(default_factory=dict)
    eval_tier_pools: Dict[str, Dict[str, IndexArray]] = field(default_factory=dict)
    extra_eval_tier_pools: Dict[str, Dict[str, Dict[str, IndexArray]]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


class CachedMathSeq2SeqDataset(Dataset):
    """Dataset backed by on-disk NumPy arrays, optionally memory-mapped."""

    def __init__(
        self,
        *,
        vocab: Mapping[str, int],
        max_q_len: int,
        max_a_len: int,
        level_ids: np.ndarray,
        encoded_questions: np.ndarray,
        encoded_answers: np.ndarray,
        level_names: Sequence[str] | np.ndarray,
        module_names: Sequence[str] | np.ndarray,
        questions: Sequence[str] | None = None,
        answers: Sequence[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.max_q_len = int(max_q_len)
        self.max_a_len = int(max_a_len)
        self.level_ids = np.asarray(level_ids, dtype=np.int64)
        self.targets = self.level_ids
        self.encoded_questions = encoded_questions
        self.encoded_answers = encoded_answers
        self.level_names = level_names
        self.module_names = module_names
        self.questions = questions
        self.answers = answers

        self.pad_id = int(self.vocab.get("<PAD>", 0))
        self.unk_id = int(self.vocab.get("<UNK>", 1))
        self.sos_id = int(self.vocab.get("<SOS>", 2))
        self.eos_id = int(self.vocab.get("<EOS>", 3))

    def __len__(self) -> int:
        return int(self.encoded_questions.shape[0])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __getitem__(self, idx: int):
        q = torch.as_tensor(np.array(self.encoded_questions[int(idx)], copy=True), dtype=torch.long)
        a = torch.as_tensor(np.array(self.encoded_answers[int(idx)], copy=True), dtype=torch.long)
        return q, a

    def get_raw_item(self, idx: int) -> Dict[str, object]:
        idx_int = int(idx)
        question = "" if self.questions is None else str(self.questions[idx_int])
        answer = "" if self.answers is None else str(self.answers[idx_int])
        return {
            "question": question,
            "answer": answer,
            "level_name": str(self.level_names[idx_int]),
            "module_name": str(self.module_names[idx_int]),
            "level_id": int(self.level_ids[idx_int]),
        }


class CompositeMathSeq2SeqDataset(Dataset):
    """Lazy view over module datasets without concatenating encoded arrays."""

    def __init__(self, datasets: Sequence[object]) -> None:
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        if not self.datasets:
            raise ValueError("CompositeMathSeq2SeqDataset requires at least one non-empty dataset.")

        first = self.datasets[0]
        self.vocab = dict(first.vocab)
        self.max_q_len = int(first.max_q_len)
        self.max_a_len = int(first.max_a_len)
        self.pad_id = int(first.pad_id)
        self.unk_id = int(first.unk_id)
        self.sos_id = int(first.sos_id)
        self.eos_id = int(first.eos_id)
        self.targets = None

        offsets = []
        total = 0
        level_ids = []
        level_names = []
        module_names = []
        for dataset in self.datasets:
            if dict(dataset.vocab) != self.vocab:
                raise RuntimeError("Composite dataset vocab mismatch.")
            if int(dataset.max_q_len) != self.max_q_len or int(dataset.max_a_len) != self.max_a_len:
                raise RuntimeError("Composite dataset sequence length mismatch.")
            offsets.append(total)
            total += len(dataset)
            level_ids.append(np.asarray(dataset.level_ids, dtype=np.int64))
            level_names.extend(list(dataset.level_names))
            module_names.extend(list(dataset.module_names))

        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.lengths = np.asarray([len(dataset) for dataset in self.datasets], dtype=np.int64)
        self.total_len = int(total)
        self.level_ids = np.concatenate(level_ids).astype(np.int64, copy=False)
        self.targets = self.level_ids
        self.level_names = level_names
        self.module_names = module_names
        self.questions = None
        self.answers = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return int(self.total_len)

    def _locate(self, idx: int) -> tuple[int, int]:
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= self.total_len:
            raise IndexError(idx_int)
        dataset_idx = int(np.searchsorted(self.offsets, idx_int, side="right") - 1)
        local_idx = int(idx_int - int(self.offsets[dataset_idx]))
        return dataset_idx, local_idx

    def __getitem__(self, idx: int):
        dataset_idx, local_idx = self._locate(int(idx))
        return self.datasets[dataset_idx][local_idx]

    def unique_answer_tokens_for_level(self, level_name: str) -> np.ndarray:
        chunks = []
        for dataset in self.datasets:
            level_names = np.asarray(dataset.level_names)
            idx = np.where(level_names == str(level_name))[0]
            if idx.size == 0:
                continue
            chunks.append(np.unique(dataset.encoded_answers[idx].reshape(-1)))
        if not chunks:
            return np.asarray([], dtype=np.int64)
        return np.unique(np.concatenate(chunks).astype(np.int64, copy=False))

    def get_raw_item(self, idx: int) -> Dict[str, object]:
        dataset_idx, local_idx = self._locate(int(idx))
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_raw_item"):
            return dataset.get_raw_item(local_idx)
        q, a = dataset[local_idx]
        return {
            "question": q if isinstance(q, str) else "",
            "answer": a if isinstance(a, str) else "",
            "level_name": str(dataset.level_names[local_idx]),
            "module_name": str(dataset.module_names[local_idx]),
            "level_id": int(dataset.level_ids[local_idx]),
        }


@dataclass
class SamplerPlan:
    top_level_mix: Dict[str, float]
    nested_mix: Dict[str, Mapping[str, float]] = field(default_factory=dict)
    total_samples: int = 0


@dataclass(frozen=True)
class RuntimeIndexSelection:
    indices: IndexArray
    shuffle_seed: int | None = None
    source: str = ""


class RuntimeDatasetBackend(Protocol):
    def sample_indices(
        self,
        *,
        pools: Mapping[str, IndexArray],
        mix: Mapping[str, float],
        total_samples: int,
        rng: Any,
    ) -> RuntimeIndexSelection | None:
        ...

    def build_loader(
        self,
        dataset: object,
        indices: object,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
    ) -> object | None:
        ...
