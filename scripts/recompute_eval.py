#!/usr/bin/env python3
"""Public wrapper for the vendored E64 checkpoint reevaluation code."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "evaluation" / "e64_supported_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "recomputed" / "e64"
E64_SCRIPT = (
    REPO_ROOT
    / "recompute_workspace"
    / "experiments"
    / "phase3c_clean"
    / "3c5dync1_dynamic"
    / "e64_generalized_evaluation.py"
)


def _collect_missing_checkpoints(entries: list[dict], checkpoint_root: Path | None) -> list[str]:
    if checkpoint_root is None:
        return []
    missing: list[str] = []
    for item in entries:
        rel = Path(str(item["checkpoint_path"]))
        if rel.parts and rel.parts[0] == "checkpoints":
            rel = Path(*rel.parts[1:])
        candidate = (checkpoint_root / rel).resolve()
        if not candidate.exists():
            missing.append(str(candidate))
    return missing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute E64 evaluation from released checkpoints.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--checkpoint-root",
        default="",
        help="Optional root directory containing the checkpoint tree referenced by the manifest.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tf-step-only", action="store_true")
    parser.add_argument("--eval-cap-per-module", type=int, default=0)
    parser.add_argument("--resume-skip-completed", type=int, default=1)
    return parser.parse_args()


def _normalize_manifest(manifest_path: Path, checkpoint_root: Path | None) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Manifest must be a JSON list: {manifest_path}")

    normalized: list[dict] = []
    for raw in payload:
        item = dict(raw)
        item.pop("evaluation_path", None)
        model_id = str(item["model_id"])
        item.setdefault("paper_label", model_id)
        item.setdefault("internal_run_name", None)
        item.setdefault("config_file", None)
        item.setdefault("dataset_pipeline_file", None)
        item.setdefault("eval_pipeline_file", None)
        item.setdefault("model_factory_file", None)

        if checkpoint_root is not None:
            rel = Path(str(item["checkpoint_path"]))
            if rel.parts and rel.parts[0] == "checkpoints":
                rel = Path(*rel.parts[1:])
            item["checkpoint_path"] = str((checkpoint_root / rel).resolve())

        normalized.append(item)
    return normalized


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve() if args.checkpoint_root else None

    normalized = _normalize_manifest(manifest_path, checkpoint_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.prepare_only:
        effective_checkpoint_root = checkpoint_root or (REPO_ROOT / "checkpoints")
        missing = _collect_missing_checkpoints(normalized, effective_checkpoint_root)
        if missing:
            sample = "\n".join(missing[:10])
            raise FileNotFoundError(
                "Checkpoint files are missing for recompute_eval.\n"
                f"Expected under: {effective_checkpoint_root}\n"
                f"Examples:\n{sample}\n"
                "Place checkpoints under dyncl/checkpoints or pass --checkpoint-root."
            )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
        handle.flush()
        temp_manifest = Path(handle.name)

    cmd = [
        sys.executable,
        str(E64_SCRIPT),
        "--manifest",
        str(temp_manifest),
        "--output-dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--eval-cap-per-module",
        str(args.eval_cap_per_module),
        "--resume-skip-completed",
        str(args.resume_skip_completed),
    ]
    if args.prepare_only:
        cmd.append("--prepare-only")
    if args.tf_step_only:
        cmd.append("--tf-step-only")

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    finally:
        temp_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
