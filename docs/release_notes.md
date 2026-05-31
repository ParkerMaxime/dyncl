# Release Notes

## Scope

This release package is built around the paper-facing 3C5 curriculum-learning artifact set used by `dyn_CL_1`.

It includes:

- renamed checkpoints matching the paper labels;
- copied per-run evaluation artifacts when available;
- paper-facing aggregate JSON files for tables and figures;
- the targeted `linear_1d` audit artifacts;
- exact tokenization-side code used by the paper family;
- scratchpad and curriculum documentation derived from the exact release code.

## Naming policy

Internal project names and paper names differ in several places. This deposit standardizes on the **paper nomenclature** for release stability.

## Release caveat

The main remaining caveat is not a missing file, but the hybrid nature of the release:

- some artifacts are copied directly from project outputs;
- some aggregate JSON files and manifests are reconstructed in paper-facing form so that the release matches the nomenclature and tables used in the article.

This is documented explicitly in:

- [`provenance.md`](provenance.md)

## Budget and checkpoint note

The release uses different effective training budgets across families for methodological reasons tied to the actual project behavior:

- dynamic sparse sequential runs needed longer budgets because convergence includes structural adaptation, not only weight optimisation;
- dense controls converged faster in this regime, so shorter budgets were sufficient for the headline controls;
- dense controls were also the family most exposed to training-collapse behavior in this project setup, which is why best-checkpoint selection matters mainly for dense and dense+EWC controls rather than for the dynamic sparse runs.

## Priority follow-up

A priority next step is to rerun the dense controls with a more robust dense-specific recipe in order to establish their true limits more confidently. At the current stage, the dense family is the main one for which optimisation instability can still confound interpretation of the control ceiling.
