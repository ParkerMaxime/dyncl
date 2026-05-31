# Curriculum Modules Used by the Paper Runs

This file documents the **effective module set used by the paper artifacts**.

## Source of truth

- Paper appendix: [`../article/paper.pdf`](../article/paper.pdf)
- Primary-only pipeline:
  [`primary_dataset_pipeline.py`](../tokenization/tokenizer_code/primary_dataset_pipeline.py)
- College/high-school pipeline:
  [`curriculum_dataset_pipeline.py`](../tokenization/tokenizer_code/curriculum_dataset_pipeline.py)

## Effective paper curriculum

The paper uses a **15-module curriculum** split across three levels.

## Primary

- `arithmetic__add_or_sub`
- `arithmetic__add_sub_multiple`
- `arithmetic__mul`
- `arithmetic__div`

## Middle-school

- `arithmetic__mixed`
- `numbers__gcd`
- `numbers__lcm`
- `numbers__div_remainder`
- `numbers__is_factor`
- `numbers__place_value`
- `numbers__round_number`

## High-school

- `algebra__linear_1d`
- `algebra__sequence_next_term`
- `algebra__sequence_nth_term`
- `polynomials__evaluate`

## Important scope note

The shared low-level curriculum source `paper_math_curriculum.py` contains a broader historical level map, including modules such as:

- `algebra__linear_2d`
- `polynomials__add`
- several `superieur` modules

These are **not** part of the effective paper curriculum represented by the copied 3C5 paper pipelines.

## Target format by module family

The paper distinguishes:

- direct-answer targets
- scratchpad targets

However, the effective release code also contains answer-only fallback behavior for some scratchpad-managed modules. See:

- [`../scratchpad/format_spec.md`](../scratchpad/format_spec.md)
- [`../scratchpad/operator_vocab.md`](../scratchpad/operator_vocab.md)
