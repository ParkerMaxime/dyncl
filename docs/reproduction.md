# Reproduction Guide

This repository is designed to support artifact reproduction at two levels.

## Level A: paper figures and tables without retraining

This is the primary supported path.

From the repository root:

```bash
mamba env create -f environment.yml
mamba run -n dyncl python scripts/make_tables.py
mamba run -n dyncl python scripts/make_figures.py
```

This rebuilds the released paper-facing tables and figure PDFs from the JSON artifacts shipped in:

- `evaluation/raw_json/`
- `evaluation/aggregated/`
- `evaluation/audit/`

## Level B: checkpoint reevaluation without retraining

This release supports checkpoint reevaluation for the cleaned `E64` protocol.

From the repository root:

```bash
mamba run -n dyncl python scripts/recompute_eval.py --prepare-only
```

If the checkpoint tree is stored externally, pass:

```bash
mamba run -n dyncl python scripts/recompute_eval.py \
  --checkpoint-root /path/to/checkpoints
```

The wrappers normalize the release manifests and call the vendored evaluation
workspace in `recompute_workspace/`.

Supported reevaluation families in this release:

- dynamic primary / middle / high
- dynamic primary no-prune ablation
- dense primary / middle / high
- dense EWC middle / high
- dynamic joint
- dense joint

Static sparse reevaluation is intentionally out of scope for this bounded
workspace.

## Level C: independent training reruns

Full training is compute-intensive and depends on the main project codebase,
the underlying dataset pipeline, and the run-family configs used during the
paper campaign.

The manifests copied into:

- `configs/run_configs/`
- `configs/evaluation/`

provide the stable naming bridge between paper labels and internal run families.

This release does not claim that full retraining is lightweight. It does claim that all reported paper figures and summary tables can be reproduced from the released evaluation artifacts.

## Notes on provenance

Some aggregate JSON files are paper-facing reconstructions, not direct raw pipeline dumps. The direct/reconstructed split is documented in:

- `docs/provenance.md`

This is intentional: the release aligns artifact naming and summary layers with the paper nomenclature.
