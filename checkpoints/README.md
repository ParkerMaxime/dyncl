# Checkpoints

This directory is intentionally lightweight in the publication-facing repository.

During local staging, the full paper checkpoint set is available in the sibling
artifact deposit under:

- `../docs/analyse_papier/article/depot/checkpoints/`

For publication, the intended pattern is:

1. keep this repository focused on code, configs, figures, tables, and JSON evaluation artifacts;
2. host heavy checkpoints separately on a release-oriented artifact backend such as Zenodo or Hugging Face Hub;
3. document those permanent URLs here before public release.

The paper figures and tables in this repository do not require the checkpoints to be present locally.

If you want to rerun the released `E64` evaluation code, either:

1. place checkpoints under this directory with the manifest-compatible layout, for example:
   - `checkpoints/dynamic/dynH_seed42.pt`
   - `checkpoints/dense/denseH_42.pt`
   - `checkpoints/ewc/dense_EWC_H_42.pt`
2. or keep them elsewhere and pass `--checkpoint-root /path/to/checkpoints` to:
   - `scripts/recompute_eval.py`
