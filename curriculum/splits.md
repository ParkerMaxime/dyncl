# Curriculum Splits and Regimes

This file explains how the paper runs use the same module inventory under different training regimes.

## Source of truth

- [`primary_dataset_pipeline.py`](../tokenization/tokenizer_code/primary_dataset_pipeline.py)
- [`curriculum_dataset_pipeline.py`](../tokenization/tokenizer_code/curriculum_dataset_pipeline.py)
- paper tables and run labels in [`../article/paper.pdf`](../article/paper.pdf)

## Sequential curriculum runs

### `dynP`, `denseP`, `sparseP`

Only the primary level is active:

- `primaire`: 4 modules
- `college`: empty
- `lycee`: empty

### `dynM`, `denseM`, `sparseM`, `dense_EWC_M`

Primary and middle-school are active:

- `primaire`: 4 modules
- `college`: 7 modules
- `lycee`: empty

### `dynH`, `denseH`, `sparseH`, `dense_EWC_H`

The full paper curriculum is active:

- `primaire`: 4 modules
- `college`: 7 modules
- `lycee`: 4 modules

## Joint runs

### `dyn_joint_42`, `dense_joint_42`, `sparse_joint_42`

The full 15-module set is trained jointly from the start.

## Ablation

### `dynP_noprune_42`

This is a primary-only dynamic ablation attached to the same primary module set as `dynP`.

## Not included in the paper release scope

- `superieur` modules are not part of the paper experiments.
- broader historical curriculum variants in `paper_math_curriculum.py` are not used as headline paper runs.
