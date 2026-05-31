# Provenance Notes

This file distinguishes between:

1. artifacts copied directly from project outputs or sources;
2. artifacts reconstructed in paper-facing form for the release deposit.

## Direct copies

The following are copied directly from project files:

- `article/paper.pdf`
- all copied checkpoint `.pt` files
- all present per-run evaluation JSON files
- the `linear_1d` audit source artifacts
- copied tokenization-side code files
- copied figure source `.tex` files
- copied figure PDF exports

## Paper-facing reconstructed artifacts

The following were created during deposit packaging so that the release matches the article nomenclature and tables:

- `eval_json/aggregated/main_results_aggregate.json`
- `eval_json/aggregated/balanced_global_aggregate.json`
- `eval_json/aggregated/dynP_multiseed_summary.json`
- `eval_json/aggregated/dynM_multiseed_summary.json`
- `eval_json/aggregated/dynH_multiseed_summary.json`
- `eval_json/aggregated/figure_data_primary.json`
- `eval_json/aggregated/figure_data_middle.json`
- `eval_json/aggregated/figure_data_trajectory.json`
- paper-facing manifests under `manifests/`
- scratchpad documentation under `scratchpad/`
- curriculum documentation under `curriculum/`
- tokenization release notes under `tokenization/vocab_or_config/`

## Why some reconstruction was necessary

The release contains:

- article-facing run labels;
- summary tables and figure inputs that are not stored as one canonical raw JSON file;
- a small number of documentation layers whose purpose is to make the deposit readable without access to the original experiment workspace.

The reconstructed files therefore act as a stable bridge between:

- the raw copied artifacts shipped in this deposit;
- the claims, tables, and figures reported in the paper.

## Current state

The deposit now contains:

- the direct per-run exported evaluation JSON for `dynH_seed44`;
- the copied checkpoint and figure assets;
- the paper-facing aggregate and manifest layer needed to match the article.

The remaining distinctions are therefore about provenance class
(direct copy vs paper-facing reconstruction), not about missing core
paper artifacts.
