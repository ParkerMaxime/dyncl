# Dataset Note

## Base dataset

The paper uses modules sampled from the DeepMind Mathematics dataset:

- Saxton et al., 2019

The release package here does **not** redistribute the full DeepMind Mathematics corpus. It documents:

- the module inventory used by the paper runs;
- the curriculum structure;
- the tokenization and scratchpad conventions;
- the evaluation artifacts derived from those runs.

## What is release-local vs external

### Included in this release

- model checkpoints
- per-run and aggregated evaluation JSON files
- linear_1d audit artifacts
- paper-facing manifests
- scratchpad format documentation
- exact tokenization-side code used by the paper runs
- curriculum documentation

### External dependency

- DeepMind Mathematics raw data

## Practical implication

To reproduce the training data exactly, a user still needs access to the upstream DeepMind Mathematics dataset and must rebuild the derived caches using the copied pipeline code and the matching run configurations.

