# Tokenization Release Notes

This directory contains the **exact tokenization-side code family used by the paper runs**, plus a compact description of the effective vocabulary configuration.

## Main point

The paper runs packaged in this release use a **deterministic char-level tokenizer**, not a released SentencePiece model.

This matters because some historical file names still contain strings such as `spe16k`, but the effective 3C5 paper pipeline tokenizes at the character level through `paper_math_curriculum.py`.

## Included code

## `tokenizer_code/`

- `paper_math_curriculum.py`
  - exact shared curriculum/tokenization source used by the paper family
- `paper_scratchpad_utils.py`
  - local scratchpad conversion helpers used by the release dataset builder
- `primary_dataset_pipeline.py`
  - exact primary dense/static-sparse paper-family dataset pipeline
- `curriculum_dataset_pipeline.py`
  - exact college/high-school dynamic paper-family dataset pipeline
- `build_paper_dataset.py`
  - release-facing CLI that downloads the required DeepMind Mathematics examples, applies scratchpad formatting, and exports JSONL datasets plus metadata

These files are copied from the project sources used by the paper runs.

## `vocab_or_config/`

This directory contains a small paper-facing description of:

- the special tokens;
- the base character vocabulary;
- the effective tokenization rules used by the release.

## Which version is represented here

This package is intentionally restricted to the versions used by the paper artifacts already copied into this repository:

- `dynP`, `dynM`, `dynH`, `dyn_joint`
- `denseP`, `denseM`, `denseH`, `dense_joint`
- `sparseP`, `sparseM`, `sparseH`, `sparse_joint`
- `dense_EWC_M`, `dense_EWC_H`
- `dynP_noprune`

It does **not** attempt to document older exploratory tokenization branches.
