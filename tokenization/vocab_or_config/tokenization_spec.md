# Effective Tokenization Specification

## Source of truth

- [`paper_math_curriculum.py`](../tokenizer_code/paper_math_curriculum.py)

## Effective tokenizer type

The released paper runs use a **deterministic character-level tokenizer**.

Encoding is performed by:

- `build_default_math_vocab(...)`
- `_encode_text_to_indices(...)`
- `decode_indices(...)`

from `paper_math_curriculum.py`.

## Special tokens

The vocabulary starts with:

1. `<PAD>`
2. `<UNK>`
3. `<SOS>`
4. `<EOS>`

## Base character inventory

The base vocabulary is the ordered character string stored in:

- `vocab_or_config/base_math_chars.txt`

This includes:

- digits
- lowercase letters
- uppercase letters
- arithmetic operators
- punctuation used in DeepMind Mathematics questions and scratchpad targets
- the segment separator `|`
- space

## Encoding rules

- text is encoded character by character;
- unknown characters map to `<UNK>`;
- `<SOS>` is optionally prepended;
- `<EOS>` is optionally appended if room remains;
- the sequence is padded or truncated to `max_len`.

## Decoding rules

- `<PAD>` and `<SOS>` are ignored;
- decoding stops at `<EOS>`;
- `<UNK>` is rendered as `?` by the helper decoder.

## Pipeline usage in the paper family

- `primary_dataset_pipeline.py`
  - passes `vocab=None` to the bundle builder, so the default char-level vocabulary is used
- `curriculum_dataset_pipeline.py`
  - likewise relies on the same shared default vocabulary through the dataset builder

## Negative statement

No separate released SentencePiece model, `.model`, or `.vocab` artifact was used by the packaged paper runs represented in this deposit.
