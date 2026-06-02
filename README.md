# DynCL: Bio-Inspired Structural Plasticity as a Continual Learning Substrate for Curriculum Transformers

This repository is the publication-facing reproduction package for the DynCL paper.

It contains:

- the paper PDF;
- raw per-run evaluation JSON files;
- paper-facing aggregate JSON files;
- the `linear_1d` rollout audit artifacts;
- curriculum and tokenization documentation;
- lightweight scripts to regenerate the released figures and summary tables without retraining.

The package is organized for artifact reproduction first. Full retraining
remains possible from the main project codebase, but is not required to
rebuild the reported paper figures and tables.

## Paper

- [`article/paper.pdf`](article/paper.pdf)

## Repository layout

- [`article/`](article/)
  - paper PDF
- [`evaluation/raw_json/`](evaluation/raw_json/)
  - per-run evaluation JSON files
- [`evaluation/aggregated/`](evaluation/aggregated/)
  - paper-facing aggregates used to rebuild tables and figures
- [`evaluation/audit/`](evaluation/audit/)
  - targeted `linear_1d` rollout audit
- [`figures/paper/`](figures/paper/)
  - released paper figure PDFs
- [`figures/generated/`](figures/generated/)
  - regenerated figures created by `scripts/make_figures.py`
- [`configs/`](configs/)
  - evaluation and run manifests
- [`curriculum/`](curriculum/)
  - curriculum documentation
- [`tokenization/`](tokenization/)
  - tokenization notes and exact helper code copied for the release
- [`docs/`](docs/)
  - provenance, reproducibility checklist, and release notes

## Quick reproduction

Create the environment:

```bash
cd dyncl
mamba env create -f environment.yml
mamba run -n dyncl python scripts/make_tables.py
mamba run -n dyncl python scripts/make_figures.py
```

Outputs:

- tables: `tables/generated/`
- figures: `figures/generated/`

## What the scripts regenerate

`scripts/make_tables.py` regenerates:

- `main_results.csv`
- `main_results.md`
- `balanced_global.csv`
- `balanced_global.md`
- `linear_1d_audit.csv`
- `linear_1d_audit.md`

`scripts/make_figures.py` regenerates:

- `pareto_primary_rebuilt.pdf`
- `pareto_primary_rebuilt.png`
- `pareto_middle_rebuilt.pdf`
- `pareto_middle_rebuilt.png`
- `cl_trajectory_rebuilt.pdf`
- `cl_trajectory_rebuilt.png`

## Full retraining

This repository is not optimized as a standalone training repo. The released
artifacts are sufficient to reproduce the reported paper figures and tables
without rerunning expensive training.

For full retraining, use the main project codebase and the manifests in:

- [`configs/run_configs/`](configs/run_configs/)
- [`configs/evaluation/`](configs/evaluation/)

## Data

The DeepMind Mathematics dataset is not redistributed here. See:

- [`curriculum/dataset_note.md`](curriculum/dataset_note.md)

## Checkpoints

Checkpoint handling is documented in:

- [`checkpoints/README.md`](checkpoints/README.md)

Archived paper checkpoints:

- [Zenodo DOI 10.5281/zenodo.20479133](https://doi.org/10.5281/zenodo.20479133)

## Provenance and release notes

- [`docs/provenance.md`](docs/provenance.md)
- [`docs/release_notes.md`](docs/release_notes.md)
- [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md)
- [`docs/reproduction.md`](docs/reproduction.md)

## Citation

See:

- [`CITATION.cff`](CITATION.cff)

## License

This repository is released under the MIT license:

- [`LICENSE`](LICENSE)

The upstream dataset remains subject to its own terms.
