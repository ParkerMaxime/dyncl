"""
Shared curriculum and character-level tokenization utilities used by the
published paper pipelines.

This module provides:
    - the ordered curriculum level definitions;
    - the deterministic character vocabulary used for question/answer encoding;
    - dataset wrappers for seq2seq training and evaluation;
    - helpers for local dataset availability, compact subset caches, and
      index-pool construction across curriculum levels.

The code is copied from the paper run family and kept here so the release
deposit remains self-contained.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The `datasets` package is required. Install it with "
        "`mamba run -n ML pip install datasets==2.19.2 --break-system-packages`."
    ) from exc


MATH_LEVEL_ORDER: Tuple[str, ...] = ("primaire", "college", "lycee", "superieur")

MATH_LEVEL_MODULES: Dict[str, Tuple[str, ...]] = {
    "primaire": (
        "arithmetic__add_or_sub",
        "arithmetic__add_sub_multiple",
        "arithmetic__mul",
        "arithmetic__div",
    ),
    "college": (
        "arithmetic__mixed",
        "numbers__gcd",
        "numbers__lcm",
        "numbers__div_remainder",
        "numbers__is_factor",
        "numbers__place_value",
        "numbers__round_number",
    ),
    "lycee": (
        "algebra__linear_1d",
        "algebra__linear_2d",
        "algebra__sequence_next_term",
        "algebra__sequence_nth_term",
        "polynomials__add",
        "polynomials__evaluate",
    ),
    "superieur": (
        "calculus__differentiate",
        "calculus__differentiate_composed",
        "polynomials__compose",
        "probability__swr_p_sequence",
    ),
}

MATH_LEVEL_MODULE_VARIANTS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "full": MATH_LEVEL_MODULES,
    "primaire_mul_only": {
        "primaire": ("arithmetic__mul",),
        "college": (),
        "lycee": (),
        "superieur": (),
    },
    "gcd_only": {
        "primaire": MATH_LEVEL_MODULES["primaire"],
        "college": ("numbers__gcd",),
        "lycee": (),
        "superieur": (),
    },
    "college_core": {
        **MATH_LEVEL_MODULES,
        "college": (
            "arithmetic__mixed",
            "numbers__div_remainder",
            "numbers__is_factor",
            "numbers__place_value",
        ),
    },
    "lycee_core_1": {
        **MATH_LEVEL_MODULES,
        "college": (
            "arithmetic__mixed",
            "numbers__div_remainder",
            "numbers__is_factor",
            "numbers__place_value",
        ),
        "lycee": (
            "algebra__linear_1d",
            "algebra__sequence_next_term",
            "polynomials__evaluate",
        ),
    },
}

MATH_LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "primaire": "Basic arithmetic operations",
    "college": "Composite arithmetic and number theory",
    "lycee": "Intermediate algebra and polynomial tasks",
    "superieur": "Symbolic calculus, composition, and probability",
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
MATH_SPECIAL_TOKENS: Tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN)

# Base character vocabulary used by the released math question/answer format.
_BASE_MATH_CHARS = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "+-*/=().,?:;'\"<>[]{}_^%|&!#@$~\\ "
)


def build_default_math_vocab(extra_chars: Sequence[str] | None = None) -> Dict[str, int]:
    """Build the deterministic character-level vocabulary used by the release."""
    vocab: Dict[str, int] = {}
    for tok in MATH_SPECIAL_TOKENS:
        vocab[tok] = len(vocab)

    ordered_chars = list(dict.fromkeys(_BASE_MATH_CHARS + "".join(extra_chars or ())))
    for ch in ordered_chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def _encode_text_to_indices(
    text: str,
    vocab: Mapping[str, int],
    max_len: int,
    *,
    add_sos: bool,
    add_eos: bool,
) -> np.ndarray:
    """Encode text into token ids with padding and truncation."""
    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")

    pad_id = int(vocab[PAD_TOKEN])
    unk_id = int(vocab[UNK_TOKEN])
    sos_id = int(vocab[SOS_TOKEN])
    eos_id = int(vocab[EOS_TOKEN])

    ids: List[int] = []
    if add_sos:
        ids.append(sos_id)

    reserve_eos = 1 if add_eos else 0
    max_content = max(0, max_len - len(ids) - reserve_eos)
    for ch in text[:max_content]:
        ids.append(int(vocab.get(ch, unk_id)))

    if add_eos and len(ids) < max_len:
        ids.append(eos_id)

    arr = np.full((max_len,), fill_value=pad_id, dtype=np.int64)
    if ids:
        n = min(len(ids), max_len)
        arr[:n] = np.asarray(ids[:n], dtype=np.int64)
    return arr


def decode_indices(indices: Sequence[int], vocab: Mapping[str, int]) -> str:
    """Decode token ids back to text, ignoring PAD/SOS and stopping at EOS."""
    inv_vocab = {int(v): str(k) for k, v in vocab.items()}
    pad_id = int(vocab[PAD_TOKEN])
    sos_id = int(vocab[SOS_TOKEN])
    eos_id = int(vocab[EOS_TOKEN])
    unk_id = int(vocab[UNK_TOKEN])

    chars: List[str] = []
    for idx in indices:
        idx_int = int(idx)
        if idx_int in (pad_id, sos_id):
            continue
        if idx_int == eos_id:
            break
        token = inv_vocab.get(idx_int, UNK_TOKEN)
        if idx_int == unk_id:
            chars.append("?")
        elif token in MATH_SPECIAL_TOKENS:
            continue
        else:
            chars.append(token)
    return "".join(chars)


@dataclass(frozen=True)
class MathLevelSpec:
    """Definition of one curriculum level and its active module set."""

    name: str
    modules: Tuple[str, ...]
    new_modules: Tuple[str, ...]
    all_modules: Tuple[str, ...]
    description: str


class MathSeq2SeqDataset(Dataset):
    """
    Character-level seq2seq dataset for DeepMind Mathematics style examples.

    `__getitem__` returns:
        (question_indices, answer_indices)
    where:
        question_indices: LongTensor [max_q_len]
        answer_indices:   LongTensor [max_a_len]

    The raw text stays available through `get_raw_item(idx)` for inspection.
    """

    def __init__(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        level_names: Sequence[str],
        module_names: Sequence[str],
        level_ids: Sequence[int],
        vocab: Mapping[str, int] | None = None,
        max_q_len: int = 160,
        max_a_len: int = 30,
        add_sos_to_question: bool = False,
        add_eos_to_question: bool = True,
        add_sos_to_answer: bool = False,
        add_eos_to_answer: bool = True,
    ) -> None:
        if not (len(questions) == len(answers) == len(level_names) == len(module_names) == len(level_ids)):
            raise ValueError("All input sequences must have the same length.")

        self.vocab: Dict[str, int] = dict(vocab) if vocab is not None else build_default_math_vocab()
        for required in MATH_SPECIAL_TOKENS:
            if required not in self.vocab:
                raise ValueError(f"Invalid vocabulary: missing special token {required!r}")

        self.max_q_len = int(max_q_len)
        self.max_a_len = int(max_a_len)
        self.add_sos_to_question = bool(add_sos_to_question)
        self.add_eos_to_question = bool(add_eos_to_question)
        self.add_sos_to_answer = bool(add_sos_to_answer)
        self.add_eos_to_answer = bool(add_eos_to_answer)

        self.questions: List[str] = list(questions)
        self.answers: List[str] = list(answers)
        self.level_names: List[str] = list(level_names)
        self.module_names: List[str] = list(module_names)
        self.level_ids = np.asarray(level_ids, dtype=np.int64)

        # Preserve the common `dataset.targets` convention expected downstream.
        self.targets = self.level_ids

        self.pad_id = int(self.vocab[PAD_TOKEN])
        self.unk_id = int(self.vocab[UNK_TOKEN])
        self.sos_id = int(self.vocab[SOS_TOKEN])
        self.eos_id = int(self.vocab[EOS_TOKEN])

        # Pre-encode once to reduce CPU work during training and evaluation.
        n = len(self.questions)
        if n == 0:
            self.encoded_questions = np.zeros((0, self.max_q_len), dtype=np.int64)
            self.encoded_answers = np.zeros((0, self.max_a_len), dtype=np.int64)
        else:
            self.encoded_questions = np.stack(
                [
                    _encode_text_to_indices(
                        q,
                        self.vocab,
                        self.max_q_len,
                        add_sos=self.add_sos_to_question,
                        add_eos=self.add_eos_to_question,
                    )
                    for q in self.questions
                ],
                axis=0,
            ).astype(np.int64, copy=False)
            self.encoded_answers = np.stack(
                [
                    _encode_text_to_indices(
                        a,
                        self.vocab,
                        self.max_a_len,
                        add_sos=self.add_sos_to_answer,
                        add_eos=self.add_eos_to_answer,
                    )
                    for a in self.answers
                ],
                axis=0,
            ).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q = torch.from_numpy(self.encoded_questions[idx]).to(dtype=torch.long)
        a = torch.from_numpy(self.encoded_answers[idx]).to(dtype=torch.long)
        return q, a

    def get_raw_item(self, idx: int) -> Dict[str, object]:
        """Return the raw text and metadata view for one example."""
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx],
            "level_name": self.level_names[idx],
            "module_name": self.module_names[idx],
            "level_id": int(self.level_ids[idx]),
        }


def build_default_math_level_specs(
    modules_by_level: Mapping[str, Sequence[str]] | None = None,
    curriculum_variant: str = "full",
) -> List[MathLevelSpec]:
    """Build curriculum level specs with cumulative active-module scopes."""
    if modules_by_level is not None:
        mapping = modules_by_level
    else:
        if curriculum_variant not in MATH_LEVEL_MODULE_VARIANTS:
            raise ValueError(
                f"Unknown curriculum_variant: {curriculum_variant!r}. "
                f"Valid choices: {sorted(MATH_LEVEL_MODULE_VARIANTS)}"
            )
        mapping = MATH_LEVEL_MODULE_VARIANTS[curriculum_variant]
    specs: List[MathLevelSpec] = []
    active: List[str] = []

    for level_name in MATH_LEVEL_ORDER:
        new_modules = tuple(mapping[level_name])
        active.extend(new_modules)
        all_modules = tuple(active)
        specs.append(
            MathLevelSpec(
                name=level_name,
                modules=new_modules,
                new_modules=new_modules,
                all_modules=all_modules,
                description=MATH_LEVEL_DESCRIPTIONS.get(level_name, ""),
            )
        )

    return specs


def describe_math_level_specs(specs: Sequence[MathLevelSpec]) -> Dict[str, Dict[str, object]]:
    """Return a JSON-serializable description of the curriculum levels."""
    desc: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        desc[spec.name] = {
            "modules": list(spec.modules),
            "new_modules": list(spec.new_modules),
            "all_modules": list(spec.all_modules),
            "description": spec.description,
        }
    return desc


def list_required_math_modules(
    modules_by_level: Mapping[str, Sequence[str]] | None = None,
    curriculum_variant: str = "full",
) -> List[str]:
    """
    Return the deduplicated list of modules required by the curriculum.
    The order follows the curriculum from `primaire` to `superieur`.
    """
    specs = build_default_math_level_specs(modules_by_level, curriculum_variant=curriculum_variant)
    modules: List[str] = []
    for spec in specs:
        modules.extend(spec.modules)
    return list(dict.fromkeys(modules))


def _set_hf_offline_mode(enabled: bool) -> Dict[str, str | None]:
    """Temporarily enable or disable HF offline mode and return the previous state."""
    keys = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
    prev = {k: os.environ.get(k) for k in keys}
    if enabled:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        for k in keys:
            if k in os.environ:
                del os.environ[k]
    return prev


def _restore_hf_offline_mode(prev: Mapping[str, str | None]) -> None:
    for key, value in prev.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def check_math_dataset_local_availability(
    modules: Sequence[str],
    *,
    cache_dir: str | None = None,
    probe_train_rows: int = 1,
    probe_test_rows: int = 1,
) -> Dict[str, List[str]]:
    """
    Check which modules are already available locally with HF forced offline.

    A module is considered present if `train[:probe]` and `test[:probe]`
    can be loaded without network access.
    """
    present: List[str] = []
    missing: List[str] = []
    uniq_modules = list(dict.fromkeys(str(m) for m in modules))

    prev_env = _set_hf_offline_mode(True)
    try:
        for module_name in uniq_modules:
            try:
                train_expr = f"train[:{max(1, int(probe_train_rows))}]"
                test_expr = f"test[:{max(1, int(probe_test_rows))}]"
                load_dataset(
                    "deepmind/math_dataset",
                    module_name,
                    split=train_expr,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                load_dataset(
                    "deepmind/math_dataset",
                    module_name,
                    split=test_expr,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                present.append(module_name)
            except Exception:
                missing.append(module_name)
    finally:
        _restore_hf_offline_mode(prev_env)

    return {"present": present, "missing": missing}


def ensure_math_dataset_available(
    modules: Sequence[str],
    *,
    cache_dir: str | None = None,
    auto_download_missing: bool = True,
    trust_remote_code: bool = True,
) -> Dict[str, object]:
    """
    Ensure module availability through a simple local-cache-first workflow:
    - check the local cache;
    - download only missing modules when allowed;
    - verify availability again before returning.
    """
    first_check = check_math_dataset_local_availability(modules, cache_dir=cache_dir)
    missing = list(first_check["missing"])

    downloaded: List[str] = []
    if missing and auto_download_missing:
        for module_name in missing:
            # Materialize the full module locally once so later loads can stay offline.
            load_dataset(
                "deepmind/math_dataset",
                module_name,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
            downloaded.append(module_name)

    second_check = check_math_dataset_local_availability(modules, cache_dir=cache_dir)
    still_missing = list(second_check["missing"])
    ok = len(still_missing) == 0

    return {
        "ok": ok,
        "requested_count": len(list(dict.fromkeys(str(m) for m in modules))),
        "present_count": len(second_check["present"]),
        "missing_count": len(still_missing),
        "downloaded_count": len(downloaded),
        "downloaded_modules": downloaded,
        "missing_modules": still_missing,
    }


def _module_subset_paths(subset_cache_dir: str, module_name: str) -> Tuple[Path, Path, Path]:
    module_dir = Path(subset_cache_dir) / str(module_name)
    return module_dir / "train.jsonl", module_dir / "test.jsonl", module_dir / "meta.json"


def _read_jsonl_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if limit <= 0:
        limit = 10**12
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({
                "question": _to_text(item.get("question", "")),
                "answer": _to_text(item.get("answer", "")),
            })
    return rows


def _write_jsonl_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {
                "question": _to_text(row.get("question", "")),
                "answer": _to_text(row.get("answer", "")),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def check_math_subset_cache_availability(
    modules: Sequence[str],
    *,
    subset_cache_dir: str,
    min_train_rows: int = 1,
    min_test_rows: int = 1,
    min_train_rows_dict: Dict[str, int] | None = None,
) -> Dict[str, List[str]]:
    """
    Check the availability of a compact per-module local subset cache.
    """
    present: List[str] = []
    missing: List[str] = []
    uniq_modules = list(dict.fromkeys(str(m) for m in modules))

    for module_name in uniq_modules:
        train_path, test_path, meta_path = _module_subset_paths(subset_cache_dir, module_name)
        if not (train_path.exists() and test_path.exists() and meta_path.exists()):
            missing.append(module_name)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            train_n = int(meta.get("train_rows", 0))
            test_n = int(meta.get("test_rows", 0))
        except Exception:
            missing.append(module_name)
            continue

        eff_train_min = (
            int(min_train_rows_dict[module_name])
            if min_train_rows_dict is not None and module_name in min_train_rows_dict
            else int(min_train_rows)
        )
        if train_n >= eff_train_min and test_n >= int(min_test_rows):
            present.append(module_name)
        else:
            missing.append(module_name)

    return {"present": present, "missing": missing}


def ensure_math_subset_cache(
    modules: Sequence[str],
    *,
    subset_cache_dir: str,
    max_train_rows: int,
    max_test_rows: int,
    max_train_rows_dict: Dict[str, int] | None = None,
    difficulty: str = "train-easy",
    cache_dir: str | None = None,
    auto_download_missing: bool = True,
    trust_remote_code: bool = True,
    progress_callback: Callable[[Dict[str, object]], None] | None = None,
) -> Dict[str, object]:
    """
    Prepare a compact per-module local subset cache:
    - verify local files;
    - stream only the rows needed when a cache entry is missing.
    """
    requested = list(dict.fromkeys(str(m) for m in modules))
    train_target_raw = int(max_train_rows)
    train_target = max(1, train_target_raw) if train_target_raw > 0 else 0
    test_target = max(1, int(max_test_rows))
    train_split = _resolve_train_split(difficulty)

    first = check_math_subset_cache_availability(
        requested,
        subset_cache_dir=subset_cache_dir,
        min_train_rows=train_target,
        min_test_rows=test_target,
        min_train_rows_dict=max_train_rows_dict,
    )
    missing = list(first["missing"])

    downloaded: List[str] = []
    if missing and auto_download_missing:
        for module_name in missing:
            eff_train_target = (
                int(max_train_rows_dict[module_name])
                if max_train_rows_dict is not None and module_name in max_train_rows_dict
                else train_target
            )
            eff_train_target = max(1, eff_train_target) if eff_train_target > 0 else 0
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "module_start",
                        "module": str(module_name),
                        "train_target": int(eff_train_target),
                        "test_target": int(test_target),
                    }
                )
            ds_stream = load_dataset(
                "deepmind/math_dataset",
                module_name,
                streaming=True,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )

            train_rows: List[Dict[str, str]] = []
            progress_stride = 50_000
            for idx, row in enumerate(
                _iter_limited_rows_from_split(ds_stream[train_split], eff_train_target if eff_train_target > 0 else 10**12),
                start=1,
            ):
                train_rows.append(
                    {
                        "question": _to_text(row.get("question", "")),
                        "answer": _to_text(row.get("answer", "")),
                    }
                )
                if progress_callback is not None and idx % progress_stride == 0:
                    progress_callback(
                        {
                            "stage": "train_progress",
                            "module": str(module_name),
                            "rows": int(idx),
                            "target": int(eff_train_target),
                        }
                    )

            test_rows: List[Dict[str, str]] = []
            for idx, row in enumerate(_iter_limited_rows_from_split(ds_stream["test"], test_target), start=1):
                test_rows.append(
                    {
                        "question": _to_text(row.get("question", "")),
                        "answer": _to_text(row.get("answer", "")),
                    }
                )
                if progress_callback is not None and idx % 200 == 0:
                    progress_callback(
                        {
                            "stage": "test_progress",
                            "module": str(module_name),
                            "rows": int(idx),
                            "target": int(test_target),
                        }
                    )

            train_path, test_path, meta_path = _module_subset_paths(subset_cache_dir, module_name)
            _write_jsonl_rows(train_path, train_rows)
            _write_jsonl_rows(test_path, test_rows)
            meta = {
                "module": module_name,
                "train_split": train_split,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
            }
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            downloaded.append(module_name)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "module_done",
                        "module": str(module_name),
                        "train_rows": int(len(train_rows)),
                        "test_rows": int(len(test_rows)),
                    }
                )

    second = check_math_subset_cache_availability(
        requested,
        subset_cache_dir=subset_cache_dir,
        min_train_rows=train_target,
        min_test_rows=test_target,
        min_train_rows_dict=max_train_rows_dict,
    )
    still_missing = list(second["missing"])
    ok = len(still_missing) == 0

    return {
        "ok": ok,
        "requested_count": len(requested),
        "present_count": len(second["present"]),
        "missing_count": len(still_missing),
        "downloaded_count": len(downloaded),
        "downloaded_modules": downloaded,
        "missing_modules": still_missing,
        "subset_cache_dir": str(subset_cache_dir),
        "train_rows_target": train_target_raw,
        "test_rows_target": test_target,
    }


def _resolve_train_split(difficulty: str) -> str:
    """
    Resolve the requested training split name.

    DeepMind Mathematics typically exposes `train` and `test`. The helper
    also accepts `train-easy|medium|hard` for compatibility with older configs.
    """
    d = (difficulty or "train").strip().lower()
    if d.startswith("train"):
        return "train"
    if d.startswith("test"):
        return "test"
    if d in {"easy", "medium", "hard"}:
        return "train"
    if d in {"train", "test"}:
        return d
    raise ValueError(
        f"Unsupported difficulty/split: {difficulty!r}. Use `train`, `test`, or `train-*`."
    )


def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        text = value.strip()
        # Some cached splits expose stringified byte literals instead of bytes.
        if (text.startswith("b'") and text.endswith("'")) or (text.startswith('b"') and text.endswith('"')):
            try:
                raw = ast.literal_eval(text)
                if isinstance(raw, bytes):
                    return raw.decode("utf-8", errors="replace").strip()
            except Exception:
                pass
        return text
    return str(value).strip()


def _iter_limited_rows_from_split(split_iterable, limit: int):
    """Iterate over a split with an optional row limit."""
    if limit <= 0:
        yield from split_iterable
        return
    count = 0
    for row in split_iterable:
        yield row
        count += 1
        if count >= limit:
            break


def build_math_datasets(
    modules_by_level: Mapping[str, Sequence[str]] | None = None,
    max_samples_per_module: int = 10_000,
    max_samples_per_module_dict: Dict[str, int] | None = None,
    difficulty: str = "train-easy",
    max_test_samples_per_module: int | None = None,
    vocab: Mapping[str, int] | None = None,
    max_q_len: int = 160,
    max_a_len: int = 30,
    cache_dir: str | None = None,
    streaming: bool = False,
    use_subset_cache: bool = False,
    subset_cache_dir: str | None = None,
) -> Tuple[MathSeq2SeqDataset, MathSeq2SeqDataset]:
    """
    Load DeepMind Mathematics modules and return `(train_dataset, test_dataset)`.

    Args:
        modules_by_level: mapping from curriculum level to module list
        max_samples_per_module: training cap per module
        difficulty: requested training split name
        max_test_samples_per_module: evaluation cap per module
        use_subset_cache: load examples from `subset_cache_dir/<module>/*.jsonl`
        subset_cache_dir: local subset-cache directory
    """
    specs = build_default_math_level_specs(modules_by_level)
    train_split = _resolve_train_split(difficulty)
    effective_vocab = dict(vocab) if vocab is not None else build_default_math_vocab()
    test_cap = (
        int(max_test_samples_per_module)
        if max_test_samples_per_module is not None
        else max(200, int(max_samples_per_module) // 10)
    )

    level_to_id = {name: idx for idx, name in enumerate(MATH_LEVEL_ORDER)}

    train_questions: List[str] = []
    train_answers: List[str] = []
    train_levels: List[str] = []
    train_modules: List[str] = []
    train_level_ids: List[int] = []

    test_questions: List[str] = []
    test_answers: List[str] = []
    test_levels: List[str] = []
    test_modules: List[str] = []
    test_level_ids: List[int] = []

    # Performance note:
    # - streaming=False loads from the local Arrow cache when available.
    # - streaming=True falls back to remote streaming when needed.
    module_cache: Dict[str, object] = {}

    for spec in specs:
        level_name = spec.name
        level_id = level_to_id[level_name]
        for module_name in spec.modules:
            eff_cap = (
                int(max_samples_per_module_dict[module_name])
                if max_samples_per_module_dict is not None and module_name in max_samples_per_module_dict
                else int(max_samples_per_module)
            )
            eff_test_cap = (
                int(max_test_samples_per_module)
                if max_test_samples_per_module is not None
                else max(200, int(eff_cap) // 10)
            )
            if use_subset_cache:
                if not subset_cache_dir:
                    raise ValueError("subset_cache_dir requis quand use_subset_cache=True")
                train_path, test_path, _ = _module_subset_paths(subset_cache_dir, module_name)
                if not train_path.exists() or not test_path.exists():
                    raise FileNotFoundError(
                        f"Subset cache manquant pour module={module_name}: "
                        f"{train_path} / {test_path}. "
                        "Execute ensure_math_subset_cache(...) avant build_math_datasets."
                    )
                train_iter = _read_jsonl_rows(train_path, int(eff_cap))
                test_iter = _read_jsonl_rows(test_path, int(eff_test_cap))
            elif streaming:
                if module_name not in module_cache:
                    module_cache[module_name] = load_dataset(
                        "deepmind/math_dataset",
                        module_name,
                        streaming=True,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                    )
                module_ds = module_cache[module_name]
                train_iter = _iter_limited_rows_from_split(module_ds[train_split], int(eff_cap))
                test_iter = _iter_limited_rows_from_split(module_ds["test"], int(eff_test_cap))
            else:
                train_limit = int(eff_cap)
                train_split_expr = train_split if train_limit <= 0 else f"{train_split}[:{train_limit}]"
                test_split_expr = "test" if int(eff_test_cap) <= 0 else f"test[:{int(eff_test_cap)}]"
                train_iter = load_dataset(
                    "deepmind/math_dataset",
                    module_name,
                    split=train_split_expr,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                test_iter = load_dataset(
                    "deepmind/math_dataset",
                    module_name,
                    split=test_split_expr,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )

            for row in train_iter:
                train_questions.append(_to_text(row["question"]))
                train_answers.append(_to_text(row["answer"]))
                train_levels.append(level_name)
                train_modules.append(module_name)
                train_level_ids.append(level_id)

            for row in test_iter:
                test_questions.append(_to_text(row["question"]))
                test_answers.append(_to_text(row["answer"]))
                test_levels.append(level_name)
                test_modules.append(module_name)
                test_level_ids.append(level_id)

    train_dataset = MathSeq2SeqDataset(
        questions=train_questions,
        answers=train_answers,
        level_names=train_levels,
        module_names=train_modules,
        level_ids=train_level_ids,
        vocab=effective_vocab,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )
    test_dataset = MathSeq2SeqDataset(
        questions=test_questions,
        answers=test_answers,
        level_names=test_levels,
        module_names=test_modules,
        level_ids=test_level_ids,
        vocab=effective_vocab,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
    )
    return train_dataset, test_dataset


_EVAL_SPLIT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "test": ("test",),
    "interpolate": ("interpolate", "interpolation"),
    "extrapolate": ("extrapolate", "extrapolation"),
}


def _row_partition_bucket(row: Mapping[str, object], buckets: int = 2) -> int:
    payload = (
        f"{_to_text(row.get('question', ''))}\n{_to_text(row.get('answer', ''))}"
    ).encode("utf-8")
    return int(hashlib.sha1(payload).digest()[0]) % max(1, int(buckets))


def _synthesize_benchmark_split_from_test_rows(
    test_rows: Sequence[Mapping[str, object]],
    split_name: str,
    limit: int,
) -> List[Mapping[str, object]]:
    normalized = _normalize_eval_split_name(split_name)
    canonical = "interpolate" if normalized in {"interpolate", "interpolation"} else "extrapolate"
    target_bucket = 0 if canonical == "interpolate" else 1
    hard_limit = max(1, int(limit))

    selected: List[Mapping[str, object]] = []
    for row in test_rows:
        if _row_partition_bucket(row, buckets=2) == target_bucket:
            selected.append(row)
            if len(selected) >= hard_limit:
                break

    if selected:
        return selected
    return list(test_rows[:hard_limit])


def _normalize_eval_split_name(split_name: str) -> str:
    return str(split_name).strip().lower()


def _resolve_extra_eval_splits(extra_eval_splits: Sequence[str] | None) -> List[str]:
    if not extra_eval_splits:
        return []
    resolved: List[str] = []
    seen = set()
    for raw in extra_eval_splits:
        split_name = _normalize_eval_split_name(str(raw))
        if not split_name or split_name == "test" or split_name in seen:
            continue
        seen.add(split_name)
        resolved.append(split_name)
    return resolved


def _iter_hf_module_split(
    module_name: str,
    split_name: str,
    limit: int,
    *,
    cache_dir: str | None,
    streaming: bool,
):
    if streaming:
        module_ds = load_dataset(
            "deepmind/math_dataset",
            module_name,
            streaming=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        if split_name not in module_ds:
            raise KeyError(split_name)
        return _iter_limited_rows_from_split(module_ds[split_name], int(limit))

    split_expr = split_name if int(limit) <= 0 else f"{split_name}[:{int(limit)}]"
    return load_dataset(
        "deepmind/math_dataset",
        module_name,
        split=split_expr,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


def _load_extra_eval_split_rows(
    module_name: str,
    split_name: str,
    limit: int,
    *,
    cache_dir: str | None,
    streaming: bool,
):
    candidates = _EVAL_SPLIT_ALIASES.get(split_name, (split_name,))
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            return _iter_hf_module_split(
                module_name,
                candidate,
                limit,
                cache_dir=cache_dir,
                streaming=streaming,
            )
        except Exception as exc:  # pragma: no cover - depends on remote split availability
            last_exc = exc
            continue

    normalized = _normalize_eval_split_name(split_name)
    if normalized in {"interpolate", "interpolation", "extrapolate", "extrapolation"}:
        try:
            test_probe_limit = max(256, int(limit) * 4)
            test_rows = list(
                _iter_hf_module_split(
                    module_name,
                    "test",
                    test_probe_limit,
                    cache_dir=cache_dir,
                    streaming=streaming,
                )
            )
            return _synthesize_benchmark_split_from_test_rows(
                test_rows=test_rows,
                split_name=normalized,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - depends on remote availability
            last_exc = exc

    if last_exc is None:
        raise RuntimeError(f"Aucun split resolvable pour {split_name!r}")
    raise RuntimeError(
        f"Impossible de charger split={split_name!r} pour module={module_name!r}"
    ) from last_exc


def build_math_datasets_with_eval_splits(
    modules_by_level: Mapping[str, Sequence[str]] | None = None,
    max_samples_per_module: int = 10_000,
    max_samples_per_module_dict: Dict[str, int] | None = None,
    difficulty: str = "train-easy",
    max_test_samples_per_module: int | None = None,
    max_eval_samples_per_module: int | None = None,
    vocab: Mapping[str, int] | None = None,
    max_q_len: int = 160,
    max_a_len: int = 30,
    cache_dir: str | None = None,
    streaming: bool = False,
    use_subset_cache: bool = False,
    subset_cache_dir: str | None = None,
    extra_eval_splits: Sequence[str] | None = None,
) -> Tuple[MathSeq2SeqDataset, Dict[str, MathSeq2SeqDataset]]:
    """
    Variant of `build_math_datasets` that returns multiple evaluation splits.

    Returns:
        `(train_dataset, eval_datasets)` with `eval_datasets["test"]` always present.
    """
    train_dataset, test_dataset = build_math_datasets(
        modules_by_level=modules_by_level,
        max_samples_per_module=max_samples_per_module,
        max_samples_per_module_dict=max_samples_per_module_dict,
        difficulty=difficulty,
        max_test_samples_per_module=max_test_samples_per_module,
        vocab=vocab,
        max_q_len=max_q_len,
        max_a_len=max_a_len,
        cache_dir=cache_dir,
        streaming=streaming,
        use_subset_cache=use_subset_cache,
        subset_cache_dir=subset_cache_dir,
    )
    eval_datasets: Dict[str, MathSeq2SeqDataset] = {"test": test_dataset}

    requested_extra_splits = _resolve_extra_eval_splits(extra_eval_splits)
    if not requested_extra_splits:
        return train_dataset, eval_datasets

    specs = build_default_math_level_specs(modules_by_level)
    level_to_id = {name: idx for idx, name in enumerate(MATH_LEVEL_ORDER)}
    eval_cap = (
        int(max_eval_samples_per_module)
        if max_eval_samples_per_module is not None
        else (
            int(max_test_samples_per_module)
            if max_test_samples_per_module is not None
            else max(200, int(max_samples_per_module) // 10)
        )
    )

    split_buffers: Dict[str, Dict[str, List[object]]] = {}
    for split_name in requested_extra_splits:
        split_buffers[split_name] = {
            "questions": [],
            "answers": [],
            "levels": [],
            "modules": [],
            "level_ids": [],
        }

    for spec in specs:
        level_name = spec.name
        level_id = level_to_id[level_name]
        for module_name in spec.modules:
            for split_name in requested_extra_splits:
                normalized_split = _normalize_eval_split_name(split_name)
                if use_subset_cache and subset_cache_dir and normalized_split in {
                    "interpolate",
                    "interpolation",
                    "extrapolate",
                    "extrapolation",
                }:
                    _, test_path, _ = _module_subset_paths(subset_cache_dir, module_name)
                    if test_path.exists():
                        test_probe_limit = max(256, int(eval_cap) * 4)
                        test_rows = _read_jsonl_rows(test_path, int(test_probe_limit))
                        split_iter = _synthesize_benchmark_split_from_test_rows(
                            test_rows=test_rows,
                            split_name=normalized_split,
                            limit=eval_cap,
                        )
                        buf = split_buffers[split_name]
                        for row in split_iter:
                            buf["questions"].append(_to_text(row["question"]))
                            buf["answers"].append(_to_text(row["answer"]))
                            buf["levels"].append(level_name)
                            buf["modules"].append(module_name)
                            buf["level_ids"].append(level_id)
                        continue
                try:
                    split_iter = _load_extra_eval_split_rows(
                        module_name,
                        split_name,
                        eval_cap,
                        cache_dir=cache_dir,
                        streaming=streaming,
                    )
                except Exception:
                    continue
                buf = split_buffers[split_name]
                for row in split_iter:
                    buf["questions"].append(_to_text(row["question"]))
                    buf["answers"].append(_to_text(row["answer"]))
                    buf["levels"].append(level_name)
                    buf["modules"].append(module_name)
                    buf["level_ids"].append(level_id)

    for split_name in requested_extra_splits:
        buf = split_buffers[split_name]
        if not buf["questions"]:
            continue
        eval_datasets[split_name] = MathSeq2SeqDataset(
            questions=buf["questions"],
            answers=buf["answers"],
            level_names=buf["levels"],
            module_names=buf["modules"],
            level_ids=buf["level_ids"],
            vocab=dict(train_dataset.vocab),
            max_q_len=max_q_len,
            max_a_len=max_a_len,
        )
    return train_dataset, eval_datasets


def _extract_module_names(dataset_or_modules) -> np.ndarray:
    if isinstance(dataset_or_modules, np.ndarray):
        return dataset_or_modules.astype(str)

    if isinstance(dataset_or_modules, torch.Tensor):
        return dataset_or_modules.detach().cpu().numpy().astype(str)

    modules = getattr(dataset_or_modules, "module_names", None)
    if modules is not None:
        return np.asarray(modules, dtype=str)

    return np.asarray(dataset_or_modules, dtype=str)


def _indices_for_modules(module_names: np.ndarray, modules: Sequence[str]) -> np.ndarray:
    if len(modules) == 0:
        return np.empty(0, dtype=np.int64)
    mask = np.isin(module_names, np.asarray(modules, dtype=str))
    return np.where(mask)[0].astype(np.int64)


def build_train_index_pools(
    dataset_or_module_names,
    specs: Sequence[MathLevelSpec],
) -> Dict[str, np.ndarray]:
    """Build level-level training pools from the newly introduced modules."""
    module_names = _extract_module_names(dataset_or_module_names)
    pools: Dict[str, np.ndarray] = {}
    for spec in specs:
        pools[spec.name] = _indices_for_modules(module_names, spec.new_modules)
    return pools


def build_train_index_pools_per_module(
    dataset_or_module_names,
    spec: MathLevelSpec,
) -> Dict[str, np.ndarray]:
    """Build per-module training pools for the current level."""
    module_names = _extract_module_names(dataset_or_module_names)
    pools: Dict[str, np.ndarray] = {}
    for module_name in spec.new_modules:
        pools[str(module_name)] = _indices_for_modules(module_names, [module_name])
    return pools


def build_eval_index_pools(
    dataset_or_module_names,
    specs: Sequence[MathLevelSpec],
) -> Dict[str, np.ndarray]:
    """Build level-level evaluation pools from cumulative module scopes."""
    module_names = _extract_module_names(dataset_or_module_names)
    pools: Dict[str, np.ndarray] = {}
    for spec in specs:
        pools[spec.name] = _indices_for_modules(module_names, spec.all_modules)
    return pools


def build_eval_index_pools_per_module(
    dataset_or_module_names,
    spec: MathLevelSpec,
) -> Dict[str, np.ndarray]:
    """Build per-module evaluation pools for the current level."""
    module_names = _extract_module_names(dataset_or_module_names)
    pools: Dict[str, np.ndarray] = {}
    for module_name in spec.new_modules:
        pools[str(module_name)] = _indices_for_modules(module_names, [module_name])
    return pools


def normalize_mix(mix: Mapping[str, float]) -> Dict[str, float]:
    """Normalize a weighted mix into probabilities summing to one."""
    clean = {k: max(float(v), 0.0) for k, v in mix.items() if float(v) > 0.0}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Invalid mix: total weight must be > 0.")
    return {k: v / total for k, v in clean.items()}


def interpolate_mix(
    level_name: str,
    progress: float,
    mix_start: Mapping[str, Mapping[str, float]],
    mix_end: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Linearly interpolate the replay mix for one curriculum level."""
    p = float(np.clip(progress, 0.0, 1.0))

    base_start = normalize_mix(mix_start.get(level_name, {level_name: 1.0}))
    base_end = normalize_mix(mix_end.get(level_name, base_start))

    keys = sorted(set(base_start.keys()) | set(base_end.keys()))
    blended: Dict[str, float] = {}
    for key in keys:
        start_val = base_start.get(key, 0.0)
        end_val = base_end.get(key, 0.0)
        value = (1.0 - p) * start_val + p * end_val
        if value > 1e-12:
            blended[key] = value

    if not blended:
        blended = {level_name: 1.0}

    return normalize_mix(blended)


def sample_mixed_indices(
    pools: Mapping[str, np.ndarray],
    mix: Mapping[str, float],
    total_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a mixed batch of indices according to `mix`."""
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")

    mix_norm = normalize_mix(mix)
    keys = list(mix_norm.keys())

    expected = np.array([mix_norm[k] * total_samples for k in keys], dtype=np.float64)
    counts = np.floor(expected).astype(np.int64)

    remainder = int(total_samples - counts.sum())
    if remainder > 0:
        fractions = expected - counts
        order = np.argsort(-fractions)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

    chunks: List[np.ndarray] = []
    for key, count in zip(keys, counts.tolist()):
        if count <= 0:
            continue

        pool = np.asarray(pools.get(key, np.empty(0, dtype=np.int64)), dtype=np.int64)
        if pool.size == 0:
            continue

        replace = count > pool.size
        sampled = rng.choice(pool, size=count, replace=replace)
        chunks.append(sampled.astype(np.int64))

    if not chunks:
        raise ValueError("Impossible d'echantillonner: aucun pool utilisable pour ce mix")

    indices = np.concatenate(chunks)
    rng.shuffle(indices)
    return indices


__all__ = [
    "PAD_TOKEN",
    "UNK_TOKEN",
    "SOS_TOKEN",
    "EOS_TOKEN",
    "MATH_SPECIAL_TOKENS",
    "MATH_LEVEL_ORDER",
    "MATH_LEVEL_MODULES",
    "MATH_LEVEL_MODULE_VARIANTS",
    "MathLevelSpec",
    "MathSeq2SeqDataset",
    "build_default_math_vocab",
    "decode_indices",
    "build_default_math_level_specs",
    "describe_math_level_specs",
    "list_required_math_modules",
    "check_math_dataset_local_availability",
    "ensure_math_dataset_available",
    "check_math_subset_cache_availability",
    "ensure_math_subset_cache",
    "build_math_datasets",
    "build_math_datasets_with_eval_splits",
    "build_train_index_pools",
    "build_train_index_pools_per_module",
    "build_eval_index_pools",
    "build_eval_index_pools_per_module",
    "normalize_mix",
    "interpolate_mix",
    "sample_mixed_indices",
]
