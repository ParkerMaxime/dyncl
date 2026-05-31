#!/usr/bin/env python3
"""Shared model stats, checkpoint, and aggregation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def collect_model_stats(model, include_hidden_stats: bool = True) -> Dict[str, Any]:
    stats = model.get_connectivity_stats()
    dynamic_parameters = int(
        getattr(model, "count_dynamic_parameters", lambda: int(stats.total_active))()
    )
    dense_parameters = int(
        getattr(
            model,
            "count_dense_parameters",
            lambda: max(0, int(model.count_parameters()) - dynamic_parameters),
        )()
    )
    return {
        "active_synapses": int(stats.total_active),
        "possible_synapses": int(stats.total_possible),
        "density": float(stats.overall_density),
        "energy": float(model.compute_energy()),
        "effective_parameters": int(model.count_parameters()),
        "dynamic_parameters": dynamic_parameters,
        "dense_parameters": dense_parameters,
        "dense_attention_parameters": int(
            getattr(model, "count_dense_attention_parameters", lambda: 0)()
        ),
        "attention_mode": str(getattr(model, "attention_mode", "sparse")),
        "dynamic_scope": str(getattr(model, "dynamic_scope", "all")),
        "created_total": int(model.creation_count),
        "pruned_total": int(model.pruning_count),
        "tenured_total": int((getattr(model, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
        "last_creation_stats": dict(getattr(model, "last_creation_stats", {})),
        "last_pruning_stats": dict(getattr(model, "last_pruning_stats", {})),
        "last_tenure_stats": dict(getattr(model, "last_tenure_stats", {})),
        "ffn_hidden_stats": (
            dict(getattr(model, "get_ffn_hidden_stats", lambda include_raw=False: {})())
            if include_hidden_stats
            else {}
        ),
    }


def save_training_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    seed: int,
    level_name: str,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config_snapshot: Mapping[str, Any],
    runtime_state: Mapping[str, Any] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "level": str(level_name),
        "epoch": int(epoch),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "config_snapshot": dict(config_snapshot),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if runtime_state:
        payload["runtime_state"] = dict(runtime_state)
    torch.save(payload, checkpoint_path)


def load_training_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise RuntimeError(f"Checkpoint invalide (model_state_dict manquant): {checkpoint_path}")
    return payload


def _normalize_checkpoint_cfg_value(key: str, value: Any) -> Any:
    if key == "ffn_activation":
        return "relu" if value in (None, "", "relu") else str(value)
    if key in {"attention_mode", "dynamic_scope"}:
        return None if value is None else str(value)
    return value


def validate_checkpoint_compatibility(
    checkpoint_cfg: Mapping[str, Any],
    cfg,
) -> None:
    critical = {
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "n_enc_layers": cfg.n_enc_layers,
        "n_dec_layers": cfg.n_dec_layers,
        "max_q_len": cfg.max_q_len,
        "max_a_len": cfg.max_a_len,
        "attention_mode": getattr(cfg, "attention_mode", None),
        "dynamic_scope": getattr(cfg, "dynamic_scope", None),
        "ffn_activation": cfg.ffn_activation,
    }
    mismatches: List[str] = []
    for key, expected in critical.items():
        got = _normalize_checkpoint_cfg_value(key, checkpoint_cfg.get(key))
        exp = _normalize_checkpoint_cfg_value(key, expected)
        if got != exp:
            mismatches.append(f"{key}: checkpoint={got!r} current={exp!r}")
    if mismatches:
        raise RuntimeError(
            "Checkpoint incompatible avec la config courante:\n- " + "\n- ".join(mismatches)
        )


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())} if arr.size else {"mean": 0.0, "std": 0.0}


def aggregate_results(all_results, level_order: Sequence[str]) -> Dict[str, Any]:
    summary = {
        "n_seeds": len(all_results),
        "seeds": [int(r["seed"]) for r in all_results],
        "go_nogo_summary": {
            "go_votes": sum(1 for r in all_results if r["go_nogo"]["go"]),
            "total": len(all_results),
        },
    }
    summary["per_seed"] = []
    for result in all_results:
        go = result["go_nogo"]
        summary["per_seed"].append({
            "seed": result["seed"],
            "go": go["go"],
            "max_regression": go["max_regression"],
            "superieur_accuracy": go["final_acc"],
            "primaire_accuracy": go["primaire_acc"],
            "final_active_synapses": float(result["final_model_stats"]["active_synapses"]),
        })

    for key in ["max_regression", "final_acc", "primaire_acc"]:
        stats = mean_std(float(r["go_nogo"][key]) for r in all_results)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_std"] = stats["std"]

    for level_name in level_order:
        accuracies = [
            float(r["levels"][level_name]["eval_acc_current"])
            for r in all_results
            if level_name in r.get("levels", {})
        ]
        if accuracies:
            stats = mean_std(accuracies)
            summary[f"{level_name}_accuracy_mean"] = stats["mean"]
            summary[f"{level_name}_accuracy_std"] = stats["std"]
        else:
            summary[f"{level_name}_accuracy_mean"] = None
            summary[f"{level_name}_accuracy_std"] = None

    for level_name in level_order[:-1]:
        stats = mean_std(float(r["go_nogo"]["regressions"].get(level_name, 0.0)) for r in all_results)
        summary[f"{level_name}_regression_mean"] = stats["mean"]
        summary[f"{level_name}_regression_std"] = stats["std"]

    syn_stats = mean_std(float(r["final_model_stats"]["active_synapses"]) for r in all_results)
    summary["final_active_synapses_mean"] = syn_stats["mean"]
    summary["final_active_synapses_std"] = syn_stats["std"]

    drops = [
        float(r.get("retention_diagnostics", {}).get("primary_drop_first_window", -1))
        for r in all_results
    ]
    drops = [value for value in drops if value >= 0]
    if drops:
        stats = mean_std(drops)
        summary["primary_drop_first_window_mean"] = stats["mean"]
        summary["primary_drop_first_window_std"] = stats["std"]

    extra_split_names = sorted({
        str(split_name)
        for result in all_results
        for level_data in dict(result.get("levels", {})).values()
        for split_name in dict(level_data.get("extra_eval", {})).keys()
    })
    if extra_split_names:
        summary["extra_eval"] = {}
        for split_name in extra_split_names:
            seq_vals = []
            tok_vals = []
            full_seq_vals = []
            full_tok_vals = []
            for result in all_results:
                last_level_name = None
                for candidate in reversed(level_order):
                    if candidate in result.get("levels", {}):
                        last_level_name = candidate
                        break
                if last_level_name is None:
                    continue
                split_metrics = (
                    result.get("levels", {})
                    .get(last_level_name, {})
                    .get("extra_eval", {})
                    .get(split_name, {})
                )
                if not split_metrics:
                    continue
                seq_vals.append(float(split_metrics.get("eval_seq_ans", split_metrics.get("accuracy", 0.0))))
                tok_vals.append(float(split_metrics.get("eval_tok_ans", split_metrics.get("token_accuracy", 0.0))))
                full_seq_vals.append(float(split_metrics.get("eval_seq_full", 0.0)))
                full_tok_vals.append(float(split_metrics.get("eval_tok_full", 0.0)))
            if seq_vals:
                summary["extra_eval"][split_name] = {
                    "eval_seq_ans_mean": mean_std(seq_vals)["mean"],
                    "eval_seq_ans_std": mean_std(seq_vals)["std"],
                    "eval_tok_ans_mean": mean_std(tok_vals)["mean"],
                    "eval_tok_ans_std": mean_std(tok_vals)["std"],
                    "eval_seq_full_mean": mean_std(full_seq_vals)["mean"],
                    "eval_seq_full_std": mean_std(full_seq_vals)["std"],
                    "eval_tok_full_mean": mean_std(full_tok_vals)["mean"],
                    "eval_tok_full_std": mean_std(full_tok_vals)["std"],
                }

    return summary
