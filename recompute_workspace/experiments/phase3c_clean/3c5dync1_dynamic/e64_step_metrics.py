#!/usr/bin/env python3
"""Shared metric helpers for E64 generalized evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Any, Iterable


PRIMARY_DIRECT_MODULES = {
    "arithmetic__add_or_sub",
    "arithmetic__mul",
    "arithmetic__div",
}

MIDDLE_DIRECT_MODULES = {
    "numbers__place_value",
    "numbers__round_number",
    "numbers__is_factor",
}

PRIMARY_SCRATCHPAD_MODULES = {
    "arithmetic__add_sub_multiple",
}

MIDDLE_SCRATCHPAD_MODULES = {
    "arithmetic__mixed",
    "numbers__gcd",
    "numbers__lcm",
    "numbers__div_remainder",
}

HIGH_SCRATCHPAD_MODULES = {
    "algebra__linear_1d",
    "algebra__sequence_next_term",
    "algebra__sequence_nth_term",
    "polynomials__evaluate",
}

MODULE_TO_LEVEL = {
    **{name: "primary" for name in PRIMARY_DIRECT_MODULES | PRIMARY_SCRATCHPAD_MODULES},
    **{name: "middle_school" for name in MIDDLE_DIRECT_MODULES | MIDDLE_SCRATCHPAD_MODULES},
    **{name: "high_school" for name in HIGH_SCRATCHPAD_MODULES},
}

DIRECT_MODULES = PRIMARY_DIRECT_MODULES | MIDDLE_DIRECT_MODULES
SCRATCHPAD_MODULES = PRIMARY_SCRATCHPAD_MODULES | MIDDLE_SCRATCHPAD_MODULES | HIGH_SCRATCHPAD_MODULES


_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_TOKEN_RE = re.compile(r"[A-Za-z]+|[+\-]?\d+(?:/\d+)?(?:\.\d+)?|[=+\-*/()%]")


@dataclass(frozen=True)
class ParsedStep:
    raw: str
    concept_signature: tuple[str, ...]
    numeric_signature: tuple[str, ...]
    lhs_signature: tuple[str, ...]
    tail_signature: tuple[str, ...]


@dataclass(frozen=True)
class ParsedScratchpad:
    module_name: str
    is_scratchpad: bool
    raw_text: str
    parse_status: str
    steps: tuple[ParsedStep, ...]


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub("", str(text).strip())


def _normalize_signs(text: str) -> str:
    out = str(text)
    while True:
        prev = out
        out = out.replace("+-", "-").replace("-+", "-").replace("--", "+").replace("++", "+")
        if out == prev:
            return out


def _normalize_fraction_token(token: str) -> str:
    raw = normalize_whitespace(token)
    if "/" not in raw:
        return raw
    parts = raw.split("/")
    if len(parts) != 2:
        return raw
    num_raw, den_raw = parts
    try:
        num = int(num_raw)
        den = int(den_raw)
    except ValueError:
        return raw
    if den == 0:
        return raw
    frac = Fraction(num, den)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def canon_number(token: str) -> str:
    raw = _normalize_signs(normalize_whitespace(token))
    if raw in {"", "+", "-"}:
        return raw
    if raw in {"-0", "+0", "-0.0", "+0.0", "0.0"}:
        return "0"
    frac = _normalize_fraction_token(raw)
    if frac != raw:
        raw = frac
    try:
        value = float(raw)
    except ValueError:
        return raw
    if math.isfinite(value) and value == 0.0:
        return "0"
    return raw


def canon_text(text: str) -> str:
    return _normalize_signs(normalize_whitespace(text))


def extract_answer(text: str) -> str:
    raw = str(text).strip()
    pos = raw.rfind("=")
    if pos < 0:
        return raw
    return raw[pos + 1 :].strip()


def canon_answer(text: str) -> str:
    return canon_number(extract_answer(text))


def answer_em(prediction: str, target: str) -> float:
    return float(canon_answer(prediction) == canon_answer(target))


def answer_token_acc(prediction: str, target: str) -> float:
    lhs = canon_answer(prediction)
    rhs = canon_answer(target)
    width = max(1, len(lhs), len(rhs))
    correct = 0
    for idx in range(width):
        lch = lhs[idx] if idx < len(lhs) else ""
        rch = rhs[idx] if idx < len(rhs) else ""
        if lch == rch:
            correct += 1
    return float(correct) / float(width)


def level_for_module(module_name: str) -> str | None:
    return MODULE_TO_LEVEL.get(str(module_name))


def is_scratchpad_module(module_name: str) -> bool:
    return str(module_name) in SCRATCHPAD_MODULES


def is_direct_module(module_name: str) -> bool:
    return str(module_name) in DIRECT_MODULES


def is_fallback_target(target_text: str) -> bool:
    text = str(target_text).strip()
    if not text:
        return True
    if "|" not in text:
        return True
    return text.startswith("result=")


def _split_segments(text: str) -> list[str]:
    return [segment.strip() for segment in str(text).split("|") if str(segment).strip()]


def _top_level_operator(expr: str) -> str:
    depth = 0
    compact = canon_text(expr)
    for idx, ch in enumerate(compact):
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if depth != 0:
            continue
        if ch in "+*/":
            return ch
        if ch == "-" and idx > 0:
            prev = compact[idx - 1]
            if prev not in "+-*/=(":
                return ch
    return ""


def _tokenize_numeric(text: str) -> tuple[str, ...]:
    return tuple(canon_number(tok) if any(ch.isdigit() for ch in tok) else canon_text(tok) for tok in _NUMBER_TOKEN_RE.findall(str(text)))


def _signature_result(segment: str) -> ParsedStep:
    value = canon_number(segment.split("=", 1)[1])
    return ParsedStep(
        raw=segment,
        concept_signature=("assign_result",),
        numeric_signature=("assign_result", value),
        lhs_signature=("assign_result",),
        tail_signature=(value,),
    )


def _signature_assignment(segment: str) -> ParsedStep | None:
    match = re.fullmatch(r"\s*([A-Za-z]+)\s*=\s*(.+)\s*", segment)
    if not match:
        return None
    variable = canon_text(match.group(1))
    value = canon_number(match.group(2))
    return ParsedStep(
        raw=segment,
        concept_signature=("assign",),
        numeric_signature=("assign", variable, value),
        lhs_signature=(variable,),
        tail_signature=(value,),
    )


def _signature_decomposition(segment: str) -> ParsedStep | None:
    compact = canon_text(segment)
    match = re.fullmatch(r"(.+?)=(.+?)\*(.+?)\+(.+)", compact)
    if not match:
        return None
    a, q, b, r = (canon_number(part) for part in match.groups())
    return ParsedStep(
        raw=segment,
        concept_signature=("euclid_decompose",),
        numeric_signature=("euclid_decompose", a, b, q, r),
        lhs_signature=(a,),
        tail_signature=(b, q, r),
    )


def _signature_mod(segment: str) -> ParsedStep | None:
    compact = canon_text(segment)
    match = re.fullmatch(r"(.+?)%(.+?)=(.+)", compact)
    if not match:
        return None
    a, b, r = (canon_number(part) for part in match.groups())
    return ParsedStep(
        raw=segment,
        concept_signature=("euclid_mod",),
        numeric_signature=("euclid_mod", a, b, r),
        lhs_signature=(a,),
        tail_signature=(b, r),
    )


def _signature_terminal(prefix: str, segment: str) -> ParsedStep | None:
    if not segment.startswith(prefix):
        return None
    value = canon_number(segment[len(prefix) :])
    tag = prefix.rstrip("=")
    return ParsedStep(
        raw=segment,
        concept_signature=(tag,),
        numeric_signature=(tag, value),
        lhs_signature=(tag,),
        tail_signature=(value,),
    )


def _signature_binary_step(segment: str, concept_kind: str = "eval_expr") -> ParsedStep | None:
    compact = canon_text(segment)
    if "=" not in compact:
        return None
    lhs, rhs = compact.rsplit("=", 1)
    op = _top_level_operator(lhs)
    op_tag = op if op else "?"
    return ParsedStep(
        raw=segment,
        concept_signature=(concept_kind, op_tag),
        numeric_signature=("binop", *_tokenize_numeric(lhs), "=", canon_number(rhs)),
        lhs_signature=tuple(_tokenize_numeric(lhs)),
        tail_signature=(op_tag, canon_number(rhs)),
    )


def _copy_prefix_step(segment: str, concept_kind: str) -> ParsedStep:
    tokens = _tokenize_numeric(segment)
    return ParsedStep(
        raw=segment,
        concept_signature=(concept_kind,),
        numeric_signature=(concept_kind, *tokens),
        lhs_signature=(concept_kind,),
        tail_signature=tokens,
    )


def _arith_chain_step(segment: str) -> ParsedStep:
    if segment.startswith("result="):
        return _signature_result(segment)
    if "=" in segment:
        step = _signature_binary_step(segment, "eval_expr")
        if step is not None:
            return step
    return _copy_prefix_step(segment, "copy_expr")


def _number_theory_step(segment: str) -> ParsedStep:
    for prefix, concept in (
        ("gcd=", "gcd_terminal"),
        ("lcm=", "lcm_formula"),
        ("remainder=", "div_remainder_terminal"),
        ("result=", "assign_result"),
    ):
        step = _signature_terminal(prefix, segment)
        if step is not None:
            if concept != step.concept_signature[0]:
                return ParsedStep(
                    raw=step.raw,
                    concept_signature=(concept,),
                    numeric_signature=(concept, *step.numeric_signature[1:]),
                    lhs_signature=(concept,),
                    tail_signature=step.tail_signature,
                )
            return step
    step = _signature_mod(segment)
    if step is not None:
        return step
    step = _signature_decomposition(segment)
    if step is not None:
        return step
    if "=" in segment:
        parsed = _signature_binary_step(segment, "eval_expr")
        if parsed is not None:
            return parsed
    return _copy_prefix_step(segment, "copy_expr")


def _linear_step(segment: str, *, step_idx: int) -> ParsedStep:
    assign = _signature_assignment(segment)
    if assign is not None:
        return assign
    if "=" in segment:
        concept = "copy_eq" if step_idx == 0 else "transform"
        parsed = _signature_binary_step(segment, concept)
        if parsed is not None:
            return parsed
    return _copy_prefix_step(segment, "copy_eq" if step_idx == 0 else "transform")


def _sequence_step(segment: str, *, nth: bool) -> ParsedStep:
    if segment.startswith("result="):
        return _signature_result(segment)
    if "=" in segment:
        concept = "transform" if nth else "extrapolate"
        parsed = _signature_binary_step(segment, concept)
        if parsed is not None:
            return parsed
    return _copy_prefix_step(segment, "copy_seq")


def _poly_step(segment: str, *, step_idx: int) -> ParsedStep:
    if segment.startswith("result="):
        return _signature_result(segment)
    assign = _signature_assignment(segment)
    if assign is not None:
        return ParsedStep(
            raw=assign.raw,
            concept_signature=("substitute",) if step_idx == 0 else assign.concept_signature,
            numeric_signature=assign.numeric_signature,
            lhs_signature=assign.lhs_signature,
            tail_signature=assign.tail_signature,
        )
    if "=" in segment:
        concept = "substitute" if step_idx == 0 else "eval_expr"
        parsed = _signature_binary_step(segment, concept)
        if parsed is not None:
            return parsed
    return _copy_prefix_step(segment, "copy_expr")


def parse_steps(module_name: str, text: str) -> ParsedScratchpad:
    module = str(module_name)
    raw = str(text).strip()
    if not is_scratchpad_module(module):
        return ParsedScratchpad(module, False, raw, "not_applicable", ())
    if not raw:
        return ParsedScratchpad(module, True, raw, "empty", ())
    segments = _split_segments(raw)
    if not segments:
        return ParsedScratchpad(module, True, raw, "empty_after_split", ())
    steps: list[ParsedStep] = []
    try:
        for idx, segment in enumerate(segments):
            if module in PRIMARY_SCRATCHPAD_MODULES or module == "arithmetic__mixed":
                steps.append(_arith_chain_step(segment))
            elif module in {"numbers__gcd", "numbers__lcm", "numbers__div_remainder"}:
                steps.append(_number_theory_step(segment))
            elif module == "algebra__linear_1d":
                steps.append(_linear_step(segment, step_idx=idx))
            elif module == "algebra__sequence_next_term":
                steps.append(_sequence_step(segment, nth=False))
            elif module == "algebra__sequence_nth_term":
                steps.append(_sequence_step(segment, nth=True))
            elif module == "polynomials__evaluate":
                steps.append(_poly_step(segment, step_idx=idx))
            else:
                steps.append(_copy_prefix_step(segment, "parse_fail"))
    except Exception:
        return ParsedScratchpad(module, True, raw, "parse_fail", ())
    return ParsedScratchpad(module, True, raw, "parsed", tuple(steps))


def _edit_distance(lhs: Iterable[tuple[str, ...]], rhs: Iterable[tuple[str, ...]]) -> int:
    left = list(lhs)
    right = list(rhs)
    if not left:
        return len(right)
    if not right:
        return len(left)
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        dp[i][0] = i
    for j in range(len(right) + 1):
        dp[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[-1][-1])


def _micro_score(lhs: Iterable[tuple[str, ...]], rhs: Iterable[tuple[str, ...]]) -> float | None:
    left = list(lhs)
    right = list(rhs)
    denom = max(len(left), len(right))
    if denom <= 0:
        return None
    distance = _edit_distance(left, right)
    return float(max(0.0, 1.0 - (float(distance) / float(denom))))


def _exact_score(lhs: Iterable[tuple[str, ...]], rhs: Iterable[tuple[str, ...]]) -> float | None:
    left = list(lhs)
    right = list(rhs)
    denom = max(len(left), len(right))
    if denom <= 0:
        return None
    return float(_edit_distance(left, right) == 0)


def score_scratchpad_pair(target: ParsedScratchpad, prediction: ParsedScratchpad) -> dict[str, float | None]:
    if not target.is_scratchpad:
        return {
            "concept_step_micro": None,
            "concept_step_exact": None,
            "numeric_step_micro": None,
            "numeric_step_exact": None,
            "numeric_lhs_match": None,
            "numeric_tail_match": None,
            "bypass_rate": None,
        }
    if target.parse_status != "parsed":
        return {
            "concept_step_micro": None,
            "concept_step_exact": None,
            "numeric_step_micro": None,
            "numeric_step_exact": None,
            "numeric_lhs_match": None,
            "numeric_tail_match": None,
            "bypass_rate": None,
        }
    target_concept = [step.concept_signature for step in target.steps]
    pred_concept = [step.concept_signature for step in prediction.steps]
    target_numeric = [step.numeric_signature for step in target.steps]
    pred_numeric = [step.numeric_signature for step in prediction.steps]
    target_lhs = [step.lhs_signature for step in target.steps]
    pred_lhs = [step.lhs_signature for step in prediction.steps]
    target_tail = [step.tail_signature for step in target.steps]
    pred_tail = [step.tail_signature for step in prediction.steps]
    return {
        "concept_step_micro": _micro_score(target_concept, pred_concept),
        "concept_step_exact": _exact_score(target_concept, pred_concept),
        "numeric_step_micro": _micro_score(target_numeric, pred_numeric),
        "numeric_step_exact": _exact_score(target_numeric, pred_numeric),
        "numeric_lhs_match": _micro_score(target_lhs, pred_lhs),
        "numeric_tail_match": _micro_score(target_tail, pred_tail),
        "bypass_rate": float(len(target.steps) >= 2 and len(prediction.steps) <= 1),
    }


def empty_mode_metric_block() -> dict[str, Any]:
    return {
        "answer_em": 0.0,
        "concept_step_micro": None,
        "concept_step_exact": None,
        "numeric_step_micro": None,
        "numeric_step_exact": None,
        "numeric_lhs_match": None,
        "numeric_tail_match": None,
        "bypass_rate": None,
        "answer_token_acc": 0.0,
    }


def empty_module_record(module_name: str, *, n_eligible: int = 0) -> dict[str, Any]:
    return {
        "is_scratchpad": is_scratchpad_module(module_name),
        "n_eligible": int(n_eligible),
        "n_fallback_targets": 0,
        "n_parse_fail_ar": 0,
        "n_parse_fail_tf": 0,
        "ar": empty_mode_metric_block(),
        "tf": empty_mode_metric_block(),
    }
