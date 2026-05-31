# Sampling Protocol

This file documents the sampling conventions used by the paper runs at the level needed to interpret the released artifacts.

## Source of truth

- paper appendix in [`../article/paper.pdf`](../article/paper.pdf)
- primary dynamic run manifest: [`../manifests/run_configs/dynP_config.json`](../manifests/run_configs/dynP_config.json)
- curriculum dynamic run manifests: [`../manifests/run_configs/dynM_config.json`](../manifests/run_configs/dynM_config.json) and [`../manifests/run_configs/dynH_config.json`](../manifests/run_configs/dynH_config.json)
- dense controls: `config_3c5densec1_dynamic.py`, `config_3c5densel0_dynamic.py`

## Evaluation sampling

The paper states that headline module-level AR/TF evaluations are drawn from the held-out DeepMind Mathematics test split, typically at:

- `1,000` examples per module for headline evaluations

and that some selected validation or diagnostic runs use larger held-out evaluations, up to:

- `4,000` examples per module

The linear_1d targeted rollout audit is separate and uses:

- one fixed held-out subset of `100` examples for every audited model

## Config-level caps used by the paper family

For the college/high-school 3C5 dynamic family and the dense/joint/EWC controls built on the same plumbing, the default config-level setting is:

- `raw_max_test_samples_per_module = 1000`

For the primary dynamic paper family:

- the primary config sets `runtime_eval_cap_per_module = 500`

This is a run-family implementation detail of the primary pipeline and should not be confused with the paper-facing module-level reporting layer.

## Training-side sampling

The copied pipelines expose:

- raw sample caps per module
- runtime train caps per module
- runtime eval caps per module
- pool-based sampler mixes by module and by level

These controls are part of the exact pipeline code shipped in:

- [`primary_dataset_pipeline.py`](../tokenization/tokenizer_code/primary_dataset_pipeline.py)
- [`curriculum_dataset_pipeline.py`](../tokenization/tokenizer_code/curriculum_dataset_pipeline.py)

## Mix profiles used by the effective paper pipelines

Representative module-level initial mixes are defined in the pipeline code:

- `PRIMAIRE_INIT_MIX`
- `COLLEGE_INIT_MIX`
- `COLLEGE_RETENTION_MIX`
- `LYCEE_INIT_MIX`

These mixes are part of the released curriculum implementation and should be read from the copied pipeline code rather than reconstructed from the article prose alone.
