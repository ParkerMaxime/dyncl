#!/usr/bin/env python3
"""E65 rollout-fidelity audit on algebra__linear_1d for selected paper models.

Goal:
- isolate AR rollout failure modes on a fixed shared subset
- keep TF local competence side-by-side
- produce a compact paper-friendly summary plus constrained annotation CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
_PHASE3C_CLEAN_ROOT = _PROJECT_ROOT / "experiments" / "phase3c_clean"
_RUNTIME_ENGINE_ROOT = _PHASE3C_CLEAN_ROOT / "runtime_engine"

for _path in (
    _PROJECT_ROOT,
    _PROJECT_ROOT / "src",
    _PHASE3C_CLEAN_ROOT,
    _RUNTIME_ENGINE_ROOT,
    _SCRIPT_DIR,
):
    raw = str(_path)
    if _path.exists() and raw not in sys.path:
        sys.path.insert(0, raw)

from e64_generalized_evaluation import (  # noqa: E402
    ManifestEntry,
    _prepare_entry,
    _resolve_device,
)
from e64_step_metrics import (  # noqa: E402
    _NUMBER_TOKEN_RE,
    answer_em,
    answer_token_acc,
    canon_answer,
    extract_answer,
    parse_steps,
    score_scratchpad_pair,
)
from runtime import (  # noqa: E402
    build_decoder_input,
    generate_autoregressive,
    resolve_amp_runtime,
    unwrap_compiled_model,
)


MODEL_IDS = ("dynH_42", "dynH_43", "dynH_44", "denseH_42", "dyn_joint_42")
TARGET_MODULE = "algebra__linear_1d"
CANONICAL_MODEL_ID = "dynH_42"
LABELS = ("format", "early_copy", "late_numeric", "conceptual", "other")
OPERAND_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:/\d+)?(?:\.\d+)?")


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="E65 rollout-fidelity audit for algebra__linear_1d")
    parser.add_argument(
        "--manifest",
        default="experiments/phase3c_clean/3c5dync1_dynamic/e64_paper1_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase3/_unsorted/analyses/e65_linear_1d_rollout_audit",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, ManifestEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, ManifestEntry] = {}
    for item in payload:
        entry = ManifestEntry(
            model_id=str(item["model_id"]),
            paper_label=str(item.get("paper_label", item["model_id"])),
            seed=int(item["seed"]),
            checkpoint_path=str(item["checkpoint_path"]),
            run_family=str(item["run_family"]),
            level_scope=str(item["level_scope"]),
            internal_run_name=item.get("internal_run_name"),
            checkpoint_kind=item.get("checkpoint_kind"),
            config_file=item.get("config_file"),
            dataset_pipeline_file=item.get("dataset_pipeline_file"),
            eval_pipeline_file=item.get("eval_pipeline_file"),
            model_factory_file=item.get("model_factory_file"),
        )
        entries[entry.model_id] = entry
    return entries


def _canonical_indices(prepared: Mapping[str, Mapping[str, Any]], module_name: str, limit: int) -> list[int]:
    canonical = prepared[CANONICAL_MODEL_ID]
    pool = canonical["bundle"].eval_module_pools.get(module_name, ())
    indices = np.asarray(pool, dtype=np.int64).tolist()
    return indices[:limit]


def _decode_token_row(token_row, inv_vocab: Mapping[int, str], *, pad_id: int, sos_id: int, eos_id: int) -> str:
    chars: list[str] = []
    for token in token_row:
        t = int(token)
        if t in (pad_id, sos_id):
            continue
        if t == eos_id:
            break
        chars.append(str(inv_vocab.get(t, "")))
    return "".join(chars)


@torch.no_grad()
def _run_model_on_tokens(*, model, q_indices, a_indices, dataset_meta, cfg, device: torch.device) -> dict[str, Any]:
    q_indices = q_indices.unsqueeze(0).to(device)
    a_indices = a_indices.unsqueeze(0).to(device)
    eval_model = unwrap_compiled_model(model)
    amp_active, amp_dtype = resolve_amp_runtime(model, None, None)

    tgt_in = build_decoder_input(a_indices, sos_id=int(dataset_meta["sos_id"]))
    logits = eval_model(q_indices, tgt_in)
    tf_preds = logits.argmax(dim=-1)
    ar_preds = generate_autoregressive(
        model=eval_model,
        src_tokens=q_indices,
        sos_id=int(dataset_meta["sos_id"]),
        eos_id=int(dataset_meta["eos_id"]),
        max_a_len=int(cfg.max_a_len),
        amp_enabled=amp_active,
        amp_dtype=amp_dtype,
    )

    raw_item = dataset_meta["raw_item"]
    vocab = dataset_meta["vocab"]
    inv_vocab = {int(v): str(k) for k, v in vocab.items()}
    target_text = _decode_token_row(
        a_indices[0].detach().cpu().tolist(),
        inv_vocab,
        pad_id=int(dataset_meta["pad_id"]),
        sos_id=int(dataset_meta["sos_id"]),
        eos_id=int(dataset_meta["eos_id"]),
    )
    tf_text = _decode_token_row(
        tf_preds[0].detach().cpu().tolist(),
        inv_vocab,
        pad_id=int(dataset_meta["pad_id"]),
        sos_id=int(dataset_meta["sos_id"]),
        eos_id=int(dataset_meta["eos_id"]),
    )
    ar_text = _decode_token_row(
        ar_preds[0].detach().cpu().tolist(),
        inv_vocab,
        pad_id=int(dataset_meta["pad_id"]),
        sos_id=int(dataset_meta["sos_id"]),
        eos_id=int(dataset_meta["eos_id"]),
    )
    return {
        "question": str(raw_item.get("question", "")),
        "target_text": target_text,
        "tf_text": tf_text,
        "ar_text": ar_text,
    }


def _edit_distance_chars(lhs: str, rhs: str) -> int:
    left = str(lhs)
    right = str(rhs)
    if not left:
        return len(right)
    if not right:
        return len(left)
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        dp[i][0] = i
    for j in range(len(right) + 1):
        dp[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[-1][-1])


def _first_divergence_char_pos(target: str, prediction: str) -> int | None:
    lhs = str(target)
    rhs = str(prediction)
    width = min(len(lhs), len(rhs))
    for idx in range(width):
        if lhs[idx] != rhs[idx]:
            return idx
    if len(lhs) == len(rhs):
        return None
    return width


def _extract_operands(question: str) -> list[str]:
    return [match.group(0) for match in OPERAND_RE.finditer(str(question))]


def _extract_linear_1d_operands_from_target(target_text: str) -> list[str]:
    first_segment = str(target_text).split("|", 1)[0].strip()
    return [match.group(0) for match in OPERAND_RE.finditer(first_segment)]


def _extract_numbers(text: str) -> list[str]:
    out: list[str] = []
    for tok in _NUMBER_TOKEN_RE.findall(str(text)):
        if any(ch.isdigit() for ch in tok):
            out.append(tok)
    return out


def _number_copy_accuracy(operands: Sequence[str], prediction_text: str) -> float | None:
    operands_list = list(operands)
    if not operands_list:
        return None
    pred_counter = Counter(_extract_numbers(prediction_text))
    gold_counter = Counter(operands_list)
    matched = 0
    total = sum(gold_counter.values())
    for token, count in gold_counter.items():
        matched += min(count, pred_counter.get(token, 0))
    return float(matched) / float(max(1, total))


def _operand_preservation_accuracy(operands: Sequence[str], prediction_text: str) -> float | None:
    operand_set = list(dict.fromkeys(str(x) for x in operands))
    if not operand_set:
        return None
    pred_set = set(_extract_numbers(prediction_text))
    matched = sum(1 for token in operand_set if token in pred_set)
    return float(matched) / float(len(operand_set))


def _format_error(target_text: str, ar_text: str, parsed_ar) -> bool:
    if parsed_ar.parse_status != "parsed":
        return True
    if "=" not in str(ar_text):
        return True
    if "|" not in str(ar_text):
        return True
    target_answer = canon_answer(target_text)
    ar_answer = canon_answer(ar_text)
    return not bool(ar_answer) and bool(target_answer)


def _conceptual_skeleton_correct(parsed_target, parsed_pred) -> float | None:
    scores = score_scratchpad_pair(parsed_target, parsed_pred)
    exact = scores.get("concept_step_exact")
    if exact is None:
        return None
    return float(exact)


def _numeric_propagation_error(parsed_target, parsed_pred, ar_text: str, target_text: str) -> float | None:
    concept_ok = _conceptual_skeleton_correct(parsed_target, parsed_pred)
    if concept_ok is None:
        return None
    if concept_ok < 1.0:
        return 0.0
    return float(answer_em(ar_text, target_text) < 1.0)


def _suggest_label(*, format_error: bool, first_error_pos: int | None, concept_ok: float | None, num_copy: float | None, num_prop: float | None) -> str:
    if format_error:
        return "format"
    if concept_ok is not None and concept_ok < 1.0:
        if first_error_pos is not None and first_error_pos <= 12:
            return "early_copy"
        return "conceptual"
    if num_prop is not None and num_prop >= 1.0:
        return "late_numeric"
    if num_copy is not None and num_copy < 1.0 and (first_error_pos is not None and first_error_pos <= 12):
        return "early_copy"
    return "other"


def _mean(values: Sequence[float | None]) -> float | None:
    keep = [float(v) for v in values if v is not None]
    if not keep:
        return None
    return float(mean(keep))


def _median(values: Sequence[float | int | None]) -> float | None:
    keep = [float(v) for v in values if v is not None]
    if not keep:
        return None
    return float(median(keep))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = (_PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = _load_manifest((_PROJECT_ROOT / args.manifest).resolve())
    device = _resolve_device(args.device)

    selected_entries = {model_id: manifest_entries[model_id] for model_id in MODEL_IDS}
    prepared = {
        model_id: _prepare_entry(entry, output_dir, device)
        for model_id, entry in selected_entries.items()
    }

    canonical_dataset = prepared[CANONICAL_MODEL_ID]["bundle"].test_dataset
    canonical_indices = _canonical_indices(prepared, TARGET_MODULE, int(args.samples))
    if len(canonical_indices) < int(args.samples):
        raise RuntimeError(
            f"E65 canonical subset too small for {TARGET_MODULE}: requested={int(args.samples)} available={len(canonical_indices)}."
        )
    rows_by_model: dict[str, list[dict[str, Any]]] = {model_id: [] for model_id in MODEL_IDS}

    for sample_idx, dataset_idx in enumerate(canonical_indices):
        q_indices, a_indices = canonical_dataset[int(dataset_idx)]
        raw_item = canonical_dataset.get_raw_item(int(dataset_idx))
        dataset_meta = {
            "raw_item": raw_item,
            "vocab": canonical_dataset.vocab,
            "pad_id": canonical_dataset.pad_id,
            "sos_id": canonical_dataset.sos_id,
            "eos_id": canonical_dataset.eos_id,
        }
        for model_id, item in prepared.items():
            sample = _run_model_on_tokens(
                model=item["model"],
                q_indices=q_indices,
                a_indices=a_indices,
                dataset_meta=dataset_meta,
                cfg=item["cfg"],
                device=device,
            )
            target_text = str(sample["target_text"])
            tf_text = str(sample["tf_text"])
            ar_text = str(sample["ar_text"])
            question = str(sample["question"])
            parsed_target = parse_steps(TARGET_MODULE, target_text)
            parsed_tf = parse_steps(TARGET_MODULE, tf_text)
            parsed_ar = parse_steps(TARGET_MODULE, ar_text)
            tf_scores = score_scratchpad_pair(parsed_target, parsed_tf)
            ar_scores = score_scratchpad_pair(parsed_target, parsed_ar)
            operands = _extract_linear_1d_operands_from_target(target_text)
            format_error = _format_error(target_text, ar_text, parsed_ar)
            first_error_pos = _first_divergence_char_pos(target_text, ar_text)
            digit_edit_distance = _edit_distance_chars(canon_answer(target_text), canon_answer(ar_text))
            number_copy_accuracy = _number_copy_accuracy(operands, ar_text)
            operand_preservation_accuracy = _operand_preservation_accuracy(operands, ar_text)
            conceptual_ok = _conceptual_skeleton_correct(parsed_target, parsed_ar)
            numeric_prop = _numeric_propagation_error(parsed_target, parsed_ar, ar_text, target_text)
            label_suggested = _suggest_label(
                format_error=format_error,
                first_error_pos=first_error_pos,
                concept_ok=conceptual_ok,
                num_copy=number_copy_accuracy,
                num_prop=numeric_prop,
            )
            rows_by_model[model_id].append(
                {
                    "sample_idx": sample_idx,
                    "sample_key": f"{TARGET_MODULE}||{question}",
                    "question": question,
                    "target_scratchpad": target_text,
                    "ar_generation": ar_text,
                    "tf_generation": tf_text,
                    "target_answer": extract_answer(target_text),
                    "ar_answer": extract_answer(ar_text),
                    "tf_answer": extract_answer(tf_text),
                    "ar_answer_em": answer_em(ar_text, target_text),
                    "tf_answer_em": answer_em(tf_text, target_text),
                    "tf_answer_token_acc": answer_token_acc(tf_text, target_text),
                    "tf_concept_step_micro": tf_scores["concept_step_micro"],
                    "tf_numeric_step_micro": tf_scores["numeric_step_micro"],
                    "ar_concept_step_micro": ar_scores["concept_step_micro"],
                    "ar_numeric_step_micro": ar_scores["numeric_step_micro"],
                    "first_error_position": first_error_pos,
                    "gold_length": len(target_text),
                    "generated_length": len(ar_text),
                    "digit_edit_distance": digit_edit_distance,
                    "number_copy_accuracy": number_copy_accuracy,
                    "operand_preservation_accuracy": operand_preservation_accuracy,
                    "conceptual_skeleton_correct": conceptual_ok,
                    "numeric_propagation_error": numeric_prop,
                    "format_error": float(format_error),
                    "failure_label_suggested": label_suggested,
                    "failure_label_manual": "",
                }
            )

    summary_models: dict[str, Any] = {}
    for model_id, rows in rows_by_model.items():
        summary_models[model_id] = {
            "n_samples": len(rows),
            "ar_answer_em": _mean([row["ar_answer_em"] for row in rows]),
            "tf_answer_em": _mean([row["tf_answer_em"] for row in rows]),
            "tf_concept_step_micro": _mean([row["tf_concept_step_micro"] for row in rows]),
            "mean_first_error_position": _mean([row["first_error_position"] for row in rows]),
            "median_first_error_position": _median([row["first_error_position"] for row in rows]),
            "format_error_rate": _mean([row["format_error"] for row in rows]),
            "number_copy_accuracy": _mean([row["number_copy_accuracy"] for row in rows]),
            "operand_preservation_accuracy": _mean([row["operand_preservation_accuracy"] for row in rows]),
            "conceptual_skeleton_correct": _mean([row["conceptual_skeleton_correct"] for row in rows]),
            "numeric_propagation_error_rate": _mean([row["numeric_propagation_error"] for row in rows]),
            "median_digit_edit_distance": _median([row["digit_edit_distance"] for row in rows]),
            "label_histogram_suggested": {
                label: sum(1 for row in rows if row["failure_label_suggested"] == label)
                for label in LABELS
            },
        }

    summary_payload = {
        "analysis_id": "E65_linear_1d_rollout_audit",
        "module": TARGET_MODULE,
        "sample_count_shared": len(canonical_indices),
        "canonical_source_model": CANONICAL_MODEL_ID,
        "models": summary_models,
        "notes": {
            "first_error_position_unit": "character_index_on_decoded_char_tokens",
            "number_copy_accuracy": "multiset recall of numeric literals from question into AR scratchpad",
            "operand_preservation_accuracy": "set recall of numeric literals from question into AR scratchpad",
            "conceptual_skeleton_correct": "concept_step_exact on parsed scratchpad skeleton",
            "numeric_propagation_error_rate": "conceptually exact skeleton but wrong final extracted answer",
            "annotation_labels": list(LABELS),
        },
    }

    summary_path = output_dir / "linear_1d_rollout_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    for model_id, rows in rows_by_model.items():
        json_path = output_dir / f"linear_1d_rollout_{model_id}.json"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        csv_path = output_dir / f"linear_1d_rollout_{model_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    payload = {
        "status": "ok",
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "models": list(summary_models.keys()),
        "n_shared_samples": len(canonical_indices),
    }
    print(json.dumps(payload, indent=2))
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
