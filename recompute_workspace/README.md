# Vendored Evaluation Workspace

This directory contains the bounded code subset required to rerun the released
evaluation checkpoints with the publication-facing E64 protocol.

Scope:

- `E64`: generalized checkpoint reevaluation

Design constraints:

- this is not a full training workspace;
- only the imports needed by the evaluation stack were copied;
- the public entry points for release users are:
  - `scripts/recompute_eval.py`

The vendored code is intentionally kept behind those wrappers so the top-level
reproduction interface stays stable even if the internal workspace layout
changes later.

Note:

- additional internal evaluation code may remain archived here for provenance,
  but it is not part of the supported public interface of this release.
