#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation"
TABLE_DIR = REPO_ROOT / "tables" / "generated"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_metric(row: dict, base: str) -> str:
    mean_key = f"{base}_mean"
    std_key = f"{base}_std"
    if mean_key in row:
        std = row.get(std_key)
        if std is None:
            return f"{row[mean_key]:.4f}"
        return f"{row[mean_key]:.4f} +/- {std:.4f}"
    value = row.get(base)
    if value is None:
        return ""
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")


def build_main_results() -> tuple[list[dict], list[str]]:
    raw_rows = load_json(EVAL_DIR / "aggregated" / "main_results_aggregate.json")["rows"]
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "label": row["label"],
                "seeds": ",".join(str(seed) for seed in row.get("seeds", [])),
                "active_ffn_synapses": row.get("active_ffn_synapses", ""),
                "effective_parameters_millions": row.get("effective_parameters_millions", ""),
                "primary_ar": normalize_metric(row, "primary_ar"),
                "middle_ar": normalize_metric(row, "middle_ar"),
                "high_ar": normalize_metric(row, "high_ar"),
                "global_ar": normalize_metric(row, "global_ar"),
                "global_tf": normalize_metric(row, "global_tf"),
            }
        )
    fieldnames = [
        "label",
        "seeds",
        "active_ffn_synapses",
        "effective_parameters_millions",
        "primary_ar",
        "middle_ar",
        "high_ar",
        "global_ar",
        "global_tf",
    ]
    return rows, fieldnames


def build_balanced_global() -> tuple[list[dict], list[str]]:
    raw_rows = load_json(EVAL_DIR / "aggregated" / "balanced_global_aggregate.json")["rows"]
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "label": row["label"],
                "seeds": ",".join(str(seed) for seed in row.get("seeds", [])),
                "global_ar": normalize_metric(row, "global_ar"),
                "global_tf": normalize_metric(row, "global_tf"),
            }
        )
    fieldnames = ["label", "seeds", "global_ar", "global_tf"]
    return rows, fieldnames


def build_linear_audit() -> tuple[list[dict], list[str]]:
    summary = load_json(EVAL_DIR / "audit" / "linear_1d_audit_summary.json")
    rows = []
    for label, metrics in summary["models"].items():
        hist = metrics.get("label_histogram_suggested", {})
        rows.append(
            {
                "label": label,
                "n_samples": metrics.get("n_samples", ""),
                "ar_answer_em": metrics.get("ar_answer_em", ""),
                "tf_answer_em": metrics.get("tf_answer_em", ""),
                "tf_concept_step_micro": metrics.get("tf_concept_step_micro", ""),
                "mean_first_error_position": metrics.get("mean_first_error_position", ""),
                "median_first_error_position": metrics.get("median_first_error_position", ""),
                "number_copy_accuracy": metrics.get("number_copy_accuracy", ""),
                "operand_preservation_accuracy": metrics.get("operand_preservation_accuracy", ""),
                "conceptual_skeleton_correct": metrics.get("conceptual_skeleton_correct", ""),
                "numeric_propagation_error_rate": metrics.get("numeric_propagation_error_rate", ""),
                "median_digit_edit_distance": metrics.get("median_digit_edit_distance", ""),
                "label_histogram": json.dumps(hist, sort_keys=True),
            }
        )
    fieldnames = [
        "label",
        "n_samples",
        "ar_answer_em",
        "tf_answer_em",
        "tf_concept_step_micro",
        "mean_first_error_position",
        "median_first_error_position",
        "number_copy_accuracy",
        "operand_preservation_accuracy",
        "conceptual_skeleton_correct",
        "numeric_propagation_error_rate",
        "median_digit_edit_distance",
        "label_histogram",
    ]
    return rows, fieldnames


def export_table(name: str, builder) -> None:
    rows, fieldnames = builder()
    write_csv(TABLE_DIR / f"{name}.csv", rows, fieldnames)
    write_markdown(TABLE_DIR / f"{name}.md", rows, fieldnames)


def main() -> None:
    ensure_dir(TABLE_DIR)
    export_table("main_results", build_main_results)
    export_table("balanced_global", build_balanced_global)
    export_table("linear_1d_audit", build_linear_audit)
    print(f"Saved tables to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
