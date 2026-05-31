#!/usr/bin/env python3
"""
Build a release-ready dataset for the paper directly from DeepMind Mathematics.

The script is designed to be simple to run from the published deposit:
    1. select a paper profile (`primary` or `curriculum`);
    2. download only the required module subsets;
    3. build train/eval datasets with the release char-level tokenizer;
    4. apply the release scratchpad formatting in place;
    5. export JSONL files plus a compact manifest.

Example:
    mamba run -n ML python build_paper_dataset.py \
        --profile curriculum \
        --output-dir ./paper_dataset_release
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import paper_math_curriculum as curriculum
import paper_scratchpad_utils as scratchpad


PRIMARY_PROFILE_MODULES: Dict[str, tuple[str, ...]] = {
    "primaire": (
        "arithmetic__add_or_sub",
        "arithmetic__add_sub_multiple",
        "arithmetic__mul",
        "arithmetic__div",
    ),
    "college": (),
    "lycee": (),
    "superieur": (),
}

CURRICULUM_PROFILE_MODULES: Dict[str, tuple[str, ...]] = {
    "primaire": PRIMARY_PROFILE_MODULES["primaire"],
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
        "algebra__sequence_next_term",
        "algebra__sequence_nth_term",
        "polynomials__evaluate",
    ),
    "superieur": (),
}

PROFILE_MODULES: Dict[str, Dict[str, tuple[str, ...]]] = {
    "primary": PRIMARY_PROFILE_MODULES,
    "curriculum": CURRICULUM_PROFILE_MODULES,
}

DEFAULT_EXTRA_EVAL_SPLITS: tuple[str, ...] = ("interpolate", "extrapolate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_MODULES),
        default="curriculum",
        help="Dataset profile to build.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the exported dataset and metadata will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--subset-cache-dir",
        default=None,
        help="Optional explicit subset-cache directory. Defaults to <output-dir>/cache/subset.",
    )
    parser.add_argument(
        "--train-cap-per-module",
        type=int,
        default=50000,
        help="Maximum number of train examples to export per module.",
    )
    parser.add_argument(
        "--eval-cap-per-module",
        type=int,
        default=1000,
        help="Maximum number of evaluation examples to export per module.",
    )
    parser.add_argument(
        "--difficulty",
        default="train-easy",
        help="Training split selector passed to the builder.",
    )
    parser.add_argument(
        "--max-q-len",
        type=int,
        default=160,
        help="Maximum encoded question length.",
    )
    parser.add_argument(
        "--max-a-len",
        type=int,
        default=128,
        help="Maximum encoded answer length.",
    )
    parser.add_argument(
        "--extra-eval-splits",
        nargs="*",
        default=list(DEFAULT_EXTRA_EVAL_SPLITS),
        help="Additional evaluation splits to export alongside `test`.",
    )
    parser.add_argument(
        "--skip-scratchpad",
        action="store_true",
        help="Export the raw answer targets without scratchpad conversion.",
    )
    return parser.parse_args()


def dataset_to_rows(dataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(len(dataset)):
        rows.append(dataset.get_raw_item(idx))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_vocab(path: Path, vocab: Mapping[str, int]) -> None:
    ordered = sorted(((token, int(idx)) for token, idx in vocab.items()), key=lambda item: item[1])
    payload = {
        "vocab_size": len(vocab),
        "ordered_tokens": [{"token": token, "index": idx} for token, idx in ordered],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def relative_to_output(path: Path, output_dir: Path) -> str:
    return os.path.relpath(str(path), start=str(output_dir))


def report(message: str) -> None:
    print(f"[build_paper_dataset] {message}", flush=True)


def subset_progress(event: Mapping[str, object]) -> None:
    stage = str(event.get("stage", ""))
    module = str(event.get("module", ""))
    if stage == "module_start":
        report(
            f"download start module={module} "
            f"train_target={event.get('train_target')} test_target={event.get('test_target')}"
        )
    elif stage == "train_progress":
        report(
            f"download train module={module} "
            f"rows={event.get('rows')}/{event.get('target')}"
        )
    elif stage == "test_progress":
        report(
            f"download test module={module} "
            f"rows={event.get('rows')}/{event.get('target')}"
        )
    elif stage == "module_done":
        report(
            f"download done module={module} "
            f"train_rows={event.get('train_rows')} test_rows={event.get('test_rows')}"
        )


def scratchpad_progress(event: Mapping[str, object]) -> None:
    stage = str(event.get("stage", ""))
    dataset = str(event.get("dataset", ""))
    if stage == "scratchpad_dataset_start":
        report(
            f"scratchpad start split={dataset} scanned={event.get('scanned')}"
        )
    elif stage == "scratchpad_module_progress":
        report(
            f"scratchpad progress split={dataset} "
            f"module={event.get('module')} rows={event.get('rows')}"
        )
    elif stage == "scratchpad_dataset_done":
        report(
            f"scratchpad done split={dataset} converted={event.get('converted')}"
        )


def apply_release_scratchpad(datasets: Mapping[str, object]) -> dict[str, dict[str, int]]:
    scratchpad.run_scratchpad_selftest()
    stats: dict[str, dict[str, int]] = {}
    for split_name, dataset in datasets.items():
        report(f"running scratchpad conversion for split={split_name}")
        stats[split_name] = scratchpad.apply_all_scratchpad_to_dataset(
            dataset,
            dataset_label=split_name,
            progress_callback=scratchpad_progress,
        )
    return stats


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    subset_cache_dir = (
        Path(args.subset_cache_dir).expanduser().resolve()
        if args.subset_cache_dir
        else output_dir / "cache" / "subset"
    )
    dataset_dir = output_dir / "dataset"
    metadata_dir = output_dir / "metadata"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    modules_by_level = PROFILE_MODULES[args.profile]
    required_modules = curriculum.list_required_math_modules(modules_by_level=modules_by_level)
    report(
        f"profile={args.profile} modules={len(required_modules)} "
        f"train_cap={args.train_cap_per_module} eval_cap={args.eval_cap_per_module}"
    )
    report(f"output_dir={output_dir}")

    report("preparing local subset cache")
    subset_status = curriculum.ensure_math_subset_cache(
        required_modules,
        subset_cache_dir=str(subset_cache_dir),
        max_train_rows=int(args.train_cap_per_module),
        max_test_rows=int(args.eval_cap_per_module),
        difficulty=str(args.difficulty),
        cache_dir=args.cache_dir,
        auto_download_missing=True,
        trust_remote_code=True,
        progress_callback=subset_progress,
    )
    if not bool(subset_status.get("ok", False)):
        raise RuntimeError(
            "Subset cache preparation failed for one or more modules: "
            f"{subset_status.get('missing_modules', [])}"
        )
    report(
        f"subset cache ready present={subset_status.get('present_count')} "
        f"downloaded={subset_status.get('downloaded_count')}"
    )

    report("building in-memory datasets")
    train_dataset, eval_datasets = curriculum.build_math_datasets_with_eval_splits(
        modules_by_level=modules_by_level,
        max_samples_per_module=int(args.train_cap_per_module),
        difficulty=str(args.difficulty),
        max_test_samples_per_module=int(args.eval_cap_per_module),
        max_eval_samples_per_module=int(args.eval_cap_per_module),
        vocab=None,
        max_q_len=int(args.max_q_len),
        max_a_len=int(args.max_a_len),
        cache_dir=args.cache_dir,
        streaming=False,
        use_subset_cache=True,
        subset_cache_dir=str(subset_cache_dir),
        extra_eval_splits=tuple(args.extra_eval_splits),
    )
    report(
        f"dataset build done train_rows={len(train_dataset)} "
        f"eval_splits={list(eval_datasets.keys())}"
    )

    all_datasets: dict[str, object] = {"train": train_dataset, **eval_datasets}
    scratchpad_stats: dict[str, dict[str, int]] = {}
    if not args.skip_scratchpad:
        report("applying scratchpad conversion")
        scratchpad_stats = apply_release_scratchpad(all_datasets)
        report("scratchpad conversion complete")
    else:
        report("scratchpad conversion skipped")

    row_counts: dict[str, int] = {}
    for split_name, dataset in all_datasets.items():
        report(f"exporting split={split_name}")
        row_counts[split_name] = write_jsonl(dataset_dir / f"{split_name}.jsonl", dataset_to_rows(dataset))
        report(f"exported split={split_name} rows={row_counts[split_name]}")

    report("writing vocabulary and level metadata")
    write_vocab(metadata_dir / "vocab.json", train_dataset.vocab)
    specs = curriculum.build_default_math_level_specs(modules_by_level=modules_by_level)
    (metadata_dir / "levels.json").write_text(
        json.dumps(curriculum.describe_math_level_specs(specs), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    portable_subset_status = dict(subset_status)
    if "subset_cache_dir" in portable_subset_status:
        portable_subset_status["subset_cache_dir"] = relative_to_output(
            subset_cache_dir,
            output_dir,
        )

    manifest = {
        "profile": args.profile,
        "modules_by_level": {level: list(modules) for level, modules in modules_by_level.items()},
        "required_modules": required_modules,
        "train_cap_per_module": int(args.train_cap_per_module),
        "eval_cap_per_module": int(args.eval_cap_per_module),
        "difficulty": str(args.difficulty),
        "max_q_len": int(args.max_q_len),
        "max_a_len": int(args.max_a_len),
        "scratchpad_applied": not bool(args.skip_scratchpad),
        "extra_eval_splits": list(args.extra_eval_splits),
        "subset_cache_dir": relative_to_output(subset_cache_dir, output_dir),
        "row_counts": row_counts,
        "scratchpad_stats": scratchpad_stats,
        "subset_cache_status": portable_subset_status,
        "outputs": {
            "dataset_dir": "dataset/",
            "metadata_dir": "metadata/",
            "train_jsonl": "dataset/train.jsonl",
            "eval_jsonl": {
                split_name: f"dataset/{split_name}.jsonl"
                for split_name in all_datasets
                if split_name != "train"
            },
            "vocab": "metadata/vocab.json",
            "levels": "metadata/levels.json",
        },
    }
    (metadata_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report("manifest written")

    print(json.dumps({"ok": True, "output_dir": str(output_dir), "row_counts": row_counts}, indent=2))


if __name__ == "__main__":
    main()
