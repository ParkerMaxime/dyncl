# Hyperparameter Audit Against Retrieved Artifacts

This note verifies the hyperparameters explicitly mentioned in the article against the files already packaged in this deposit.

## Scope

The goal is not to restate every internal config field. The goal is to verify that the **hyperparameters named in the article** are backed by retrieved source files, and to document the few places where the article intentionally compresses family-specific variation.

## Main article table: dynamic shared hyperparameters

The article appendix lists the following as the main hyperparameters shared across dynamic runs.

## Architecture

- `d_model = 256`
  - confirmed in:
    - [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
    - [`dense_config.json`](../manifests/run_configs/dense_config.json)
- `d_ff = 1024`
  - confirmed in the same config sources
- `n_heads = 4`
  - confirmed in:
    - the copied run-config manifests for the paper families, especially [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
- `n_enc = n_dec = 2`
  - confirmed in the primary and dynamic 3C5 config families

## Sequence and tokenization

- `max_q_len = 160`
  - confirmed in the copied paper-family config sources
- `max_a_len = 128`
  - confirmed in the copied paper-family config sources
- `vocab size = 98`
  - consistent with the copied char-level tokenizer family:
    - [`paper_math_curriculum.py`](../tokenization/tokenizer_code/paper_math_curriculum.py)
    - [`special_tokens.json`](../tokenization/vocab_or_config/special_tokens.json)
    - [`base_math_chars.txt`](../tokenization/vocab_or_config/base_math_chars.txt)
  - release interpretation:
    - deterministic char-level vocabulary
    - 4 special tokens plus the fixed base character inventory

## Optimization

- `batch size = 64`
  - confirmed in the effective paper-family configs:
    - primary dynamic
    - college/high-school dynamic
    - dense controls inherited from the same family
- `Adam beta1, beta2 = 0.9, 0.999`
  - confirmed in:
    - the copied run-config manifests, especially [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
- `label smoothing = 0.03`
  - confirmed in:
    - [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
    - [`dense_config.json`](../manifests/run_configs/dense_config.json)

## Dynamic sparse substrate

- `FFN init connectivity = 30%`
  - confirmed in:
    - [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
    - inherited college/high-school dynamic families
- `Creation budget = 400 (cap 4000)`
  - confirmed for the primary dynamic paper family in:
    - [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
  - nuance:
    - later dynamic families keep the same top-level scale but also define phase-specific controller budgets in their own config chain
- `LR schedule = warmup-plateau-cosine`
  - article wording matches the effective dynamic family recipe
  - confirmed through:
    - [`dynP_config.json`](../manifests/run_configs/dynP_config.json)
    - [`dynM_config.json`](../manifests/run_configs/dynM_config.json)
    - [`dynH_config.json`](../manifests/run_configs/dynH_config.json)
- `LR floor ratio = 0.05`
  - confirmed for the primary dynamic paper family
  - nuance:
    - joint and some dense control families use different floor ratios, so the article is correctly framing this as a **dynamic shared** table, not a universal release-wide constant

## Static sparse controls

The article states:

- same architecture
- same optimizer family
- same curriculum ordering
- same batch size
- same training budget
- same dense-attention configuration
- same evaluation protocol
- same per-matrix initial FFN density
- no dynamic structure updates

This is consistent with the copied static sparse config files:

- [`static_sparse_config.json`](../manifests/run_configs/static_sparse_config.json)

Important nuance already reflected in the article:

- the fixed-mask controls do **not** reuse the dynamic structural-plasticity schedule blindly;
- they disable triphasic structure updates and use the stable fixed-mask control recipe captured by their own config family.

## EWC controls

The article states:

- diagonal Fisher estimate
- `8192` Fisher samples
- `lambda = 30` at primary-to-middle
- `lambda = 100` at middle-to-high

This is confirmed in:

- [`ewc_config.json`](../manifests/run_configs/ewc_config.json)

## Training budgets mentioned in the article

The article intentionally compresses family-specific variation into a short appendix paragraph. The retrieved configs support the following reading.

## Why the budgets differ across families

Two practical constraints shaped the effective paper budgets.

### 1. Dense controls converged faster

The dense controls do not carry the same structural dynamics as DynCL:

- no creation updates
- no pruning updates
- no triphasic structural adaptation

In practice, this means the dense runs typically reached their useful convergence regime faster than the dynamic sparse curriculum runs. This is the main reason the dense sequential controls were run with shorter effective budgets than the `750`-epoch dynamic and static-sparse sequential curricula.

### 2. Dense training was also less stable in this project regime

In this project, the dense controls were the main family that showed training-collapse issues under some recipes. This is why the dense controls are also the only family for which the selected paper checkpoint is not assumed to coincide systematically with the last checkpoint.

Operationally:

- dynamic sparse runs were stable enough that the manifest checkpoint typically coincided with the final useful model state;
- dense controls were more collapse-prone in this setup, so the best retained checkpoint could be more reliable than the last checkpoint.

This is not presented as a general claim about dense models. It is a release-specific observation about the exact training regime used for this paper family.

Because of this, one high-priority follow-up is to rerun the dense controls under a more robust dense-specific training recipe, so that their real ceiling and failure limits can be established with higher confidence. This is the main remaining methodological priority on the control side.

### Sequential dynamic

- `dynP`, `dynM`, `dynH`
  - batch size `64`
  - `750` epochs per level

### Sequential static sparse

- `sparseP`, `sparseM`, `sparseH`
  - batch size `64`
  - `750` epochs per level

### Dense sequential controls

- `denseP`
  - primary dense control uses its own conventional dense recipe
  - primary paper-family config still keeps batch size `64`
- `denseM`, `denseH`
  - `200` epochs per level
  - `adamw`
  - `cosine`
  - best-checkpoint selection is especially relevant for this family because these were the runs most exposed to late-training collapse in the project regime

### Dense + EWC sequential controls

- `dense_EWC_M`, `dense_EWC_H`
  - `200` epochs per level
  - `adamw`
  - `cosine`
  - EWC enabled with the verified `lambda`/Fisher settings above
  - these controls inherit the same dense-training stability caveat as the non-EWC dense family

### Joint controls

- `dyn_joint`
  - `900` epochs
- `dense_joint`
  - `900` epochs in the copied paper config family
- `sparse_joint`
  - fixed sparse joint control with its own copied config family

## One useful clarification not obvious from the article alone

The article’s hyperparameter appendix is accurate, but it is intentionally **family-compressed**:

- the shared dynamic table describes the dynamic sparse substrate;
- dense, EWC, and joint controls use their own config families;
- the deposit therefore needs both:
  - the article-facing summary;
  - the copied config files that preserve run-family specifics.

That condition is now satisfied by the current deposit structure.

## Conclusion

The hyperparameter audit is fully backed by copied config sources, and
the current deposit now also contains the previously missing
`dynH_seed44` per-run evaluation JSON. The remaining release nuance is
therefore documentary rather than experimental: some artifacts are
direct copies, while others are paper-facing reconstructions recorded in
the provenance notes.
