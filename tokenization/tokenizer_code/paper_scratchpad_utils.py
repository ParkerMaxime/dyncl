"""
Scratchpad generation helpers used by the published paper dataset builder.

This file is a local copy of the release scratchpad logic with imports rewritten
so it can run directly from the artifact deposit. It exposes:
    - per-module scratchpad converters;
    - `extract_answer(...)` for answer-level evaluation;
    - `apply_all_scratchpad_to_dataset(...)` for in-place dataset conversion;
    - `run_scratchpad_selftest()` for deterministic smoke checks.
"""

from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Dict, List, Tuple

import paper_math_curriculum as math_curriculum_mod

# The published runs use max answer length 128. Keep converters conservative so
# exported JSONL datasets do not rely on downstream truncation behavior.
_MAX_SCRATCHPAD_LEN = 128

_NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?")
_MUL_INLINE_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\*\s*([+-]?\d+(?:\.\d+)?)")
_ADD_SUB_INLINE_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*([+-])\s*([+-]?\d+(?:\.\d+)?)")
_DIV_INLINE_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)")
_EXPR_PREFIX_PATTERN = re.compile(
    r"^(what is the value of|what is|evaluate|calculate|work out)\s+",
    flags=re.IGNORECASE,
)
_ALLOWED_EXPR_PATTERN = re.compile(r"^[0-9+\-().\s]+$")
_ALLOWED_MIXED_EXPR_PATTERN = re.compile(r"^[0-9+\-*/().\s]+$")
_INT_TOKEN_PATTERN = re.compile(r"[+-]?\d+")


def _clean_text(text: str) -> str:
    return str(text).strip().rstrip(".?!")


def _extract_first_two_numbers(text: str) -> Tuple[str, str]:
    nums = _NUMBER_PATTERN.findall(str(text))
    if len(nums) < 2:
        raise ValueError(f"Unable to extract two numeric operands from {text!r}")
    return nums[0], nums[1]


def _extract_first_two_ints(text: str) -> Tuple[int, int]:
    nums = _INT_TOKEN_PATTERN.findall(str(text))
    if len(nums) < 2:
        raise ValueError(f"Unable to extract two integer operands from {text!r}")
    return int(nums[0]), int(nums[1])


def _is_int_token(token: str) -> bool:
    return bool(_INT_TOKEN_PATTERN.fullmatch(str(token).strip()))


def _fallback_answer(answer: str) -> str:
    return f"result={str(answer).strip()}"


def _finalize_scratchpad(candidate: str, answer: str) -> str:
    answer_str = str(answer).strip()
    scratch = str(candidate).strip()
    if not scratch:
        return _fallback_answer(answer_str)
    if extract_answer(scratch) != answer_str:
        if "=" in scratch:
            scratch = f"{scratch}|result={answer_str}"
        else:
            scratch = f"{scratch}={answer_str}"
    if len(scratch) > _MAX_SCRATCHPAD_LEN:
        return _fallback_answer(answer_str)
    return scratch


def _encode_dataset_answer(dataset, idx: int, scratch: str) -> None:
    safe_scratch = str(scratch).strip()
    if len(safe_scratch) > int(dataset.max_a_len):
        safe_scratch = _fallback_answer(extract_answer(safe_scratch))

    dataset.answers[idx] = safe_scratch
    dataset.encoded_answers[idx] = math_curriculum_mod._encode_text_to_indices(  # noqa: SLF001
        safe_scratch,
        dataset.vocab,
        dataset.max_a_len,
        add_sos=dataset.add_sos_to_answer,
        add_eos=dataset.add_eos_to_answer,
    )


def mul_scratchpad(a: int, b: int) -> str:
    """
    Build scratchpad decomposition for integer multiplication.

    Example:
      123 * 456 ->
      "123*6=738|123*50=6150|123*400=49200|738+6150+49200=56088"
    """
    b_int = int(b)
    b_sign = -1 if b_int < 0 else 1
    b_digits = str(abs(b_int))
    steps = []
    partials = []

    for i, digit_char in enumerate(reversed(b_digits)):
        d = int(digit_char)
        shift = 10**i
        scaled = b_sign * d * shift
        partial = int(a) * int(scaled)
        partials.append(partial)
        steps.append(f"{int(a)}*{scaled}={partial}")

    if len(partials) > 1:
        final_sum = "+".join(str(p) for p in partials)
        steps.append(f"{final_sum}={sum(partials)}")

    return "|".join(steps)


def extract_answer(sequence: str) -> str:
    """Return text after the last '=' (or the whole string if no '=')"""
    s = str(sequence).strip()
    pos = s.rfind("=")
    if pos < 0:
        return s
    return s[pos + 1 :].strip()


def _parse_mul_operands_text(question: str) -> Tuple[str, str]:
    text = _clean_text(question)
    match = _MUL_INLINE_PATTERN.search(text)
    if match:
        return match.group(1), match.group(2)
    return _extract_first_two_numbers(text)


def parse_mul_operands(question: str) -> Tuple[int, int]:
    """Parse integer `a` and `b` from a multiplication question."""
    a_str, b_str = _parse_mul_operands_text(question)
    if _is_int_token(a_str) and _is_int_token(b_str):
        return int(a_str), int(b_str)
    raise ValueError(f"Unable to parse integer multiplication operands from question={question!r}")


def _mul_scratchpad_from_question(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        a_str, b_str = _parse_mul_operands_text(question)
    except Exception:
        return _fallback_answer(answer_str)

    if _is_int_token(a_str) and _is_int_token(b_str):
        candidate = mul_scratchpad(int(a_str), int(b_str))
    else:
        candidate = f"{a_str}*{b_str}={answer_str}"
    return _finalize_scratchpad(candidate, answer_str)


def parse_add_or_sub_operands(question: str) -> Tuple[str, str, str]:
    """
    Return (a_str, op, b_str) from add/sub question.

    op is one of: '+' or '-'.
    """
    text = _clean_text(question)
    lowered = text.lower()

    inline_match = _ADD_SUB_INLINE_PATTERN.search(text)
    if inline_match:
        return inline_match.group(1), inline_match.group(2), inline_match.group(3)

    if "subtract" in lowered and " from " in lowered:
        a_str, b_str = _extract_first_two_numbers(text)
        return b_str, "-", a_str

    if " less than " in lowered:
        a_str, b_str = _extract_first_two_numbers(text)
        return b_str, "-", a_str

    if "difference between" in lowered or "distance between" in lowered:
        a_str, b_str = _extract_first_two_numbers(text)
        return a_str, "-", b_str

    if "take away" in lowered or "minus" in lowered:
        a_str, b_str = _extract_first_two_numbers(text)
        return a_str, "-", b_str

    if (
        "plus" in lowered
        or "total of" in lowered
        or "add together" in lowered
        or "sum of" in lowered
    ):
        a_str, b_str = _extract_first_two_numbers(text)
        return a_str, "+", b_str

    a_str, b_str = _extract_first_two_numbers(text)
    return a_str, "+", b_str


def _try_decimal(token: str) -> Decimal | None:
    try:
        return Decimal(str(token))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _resolve_difference_order(a_str: str, b_str: str, answer: str) -> Tuple[str, str]:
    da = _try_decimal(a_str)
    db = _try_decimal(b_str)
    dr = _try_decimal(answer)
    if da is None or db is None or dr is None:
        return a_str, b_str
    if da - db == dr:
        return a_str, b_str
    if db - da == dr:
        return b_str, a_str
    if abs(da - db) == abs(dr):
        return (a_str, b_str) if (da - db) >= Decimal("0") else (b_str, a_str)
    return a_str, b_str


def add_or_sub_scratchpad(question: str, answer: str) -> str:
    """
    Build "a OP b = result" scratchpad for arithmetic__add_or_sub.
    """
    answer_str = str(answer).strip()
    try:
        a_str, op, b_str = parse_add_or_sub_operands(question)
        lowered = _clean_text(question).lower()
        if "difference between" in lowered or "distance between" in lowered:
            a_str, b_str = _resolve_difference_order(a_str, b_str, answer_str)
            op = "-"
        candidate = f"{a_str}{op}{b_str}={answer_str}"
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def _extract_add_sub_expression(question: str) -> str:
    text = _clean_text(question)
    text = _EXPR_PREFIX_PATTERN.sub("", text).strip()
    if not text:
        raise ValueError("Empty add/sub expression")
    if not _ALLOWED_EXPR_PATTERN.fullmatch(text):
        raise ValueError(f"Unsupported characters in expression={text!r}")
    return text


def _decimal_to_text(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.to_integral_value())
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "+0", ""}:
        return "0"
    return text


def _constant_token(expr: str, node: ast.Constant) -> str:
    token = expr[node.col_offset : node.end_col_offset].strip()
    if token:
        return token
    return str(node.value)


def _eval_add_sub_steps(expr: str) -> List[str]:
    tree = ast.parse(expr, mode="eval")
    steps: List[str] = []

    def _eval_node(node: ast.AST) -> Tuple[Decimal, str]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left_val, left_text = _eval_node(node.left)
            right_val, right_text = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                value = left_val + right_val
                op = "+"
            else:
                value = left_val - right_val
                op = "-"
            value_text = _decimal_to_text(value)
            expr_text = f"{left_text}{op}{right_text}"
            steps.append(f"{expr_text}={value_text}")
            return value, value_text

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand_val, _operand_text = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                value = -operand_val
            else:
                value = operand_val
            return value, _decimal_to_text(value)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            token = _constant_token(expr, node)
            value = Decimal(token)
            return value, _decimal_to_text(value)

        raise ValueError(f"Unsupported node type in add/sub expression: {type(node).__name__}")

    _eval_node(tree.body)
    return steps


def add_sub_multiple_scratchpad(question: str, answer: str) -> str:
    """
    Build step-by-step scratchpad for arithmetic__add_sub_multiple.
    """
    answer_str = str(answer).strip()
    try:
        expr = _extract_add_sub_expression(question)
        steps = _eval_add_sub_steps(expr)
        if not steps:
            return _fallback_answer(answer_str)
        candidate = "|".join([*steps, f"result={answer_str}"])
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def parse_div_operands(question: str) -> Tuple[str, str]:
    """Extract (a_str, b_str) from division question."""
    text = _clean_text(question)
    match = _DIV_INLINE_PATTERN.search(text)
    if match:
        return match.group(1), match.group(2)
    return _extract_first_two_numbers(text)


def div_scratchpad(question: str, answer: str) -> str:
    """
    Build scratchpad for arithmetic__div.

    Exact division:
      a/b=q*b=a|result=q
    Fractional answer:
      a/b=result=fraction
    """
    answer_str = str(answer).strip()
    try:
        a_str, b_str = parse_div_operands(question)
    except Exception:
        return _fallback_answer(answer_str)

    if "/" in answer_str:
        candidate = f"{a_str}/{b_str}=result={answer_str}"
    else:
        candidate = f"{a_str}/{b_str}|{answer_str}*{b_str}={a_str}|result={answer_str}"
    return _finalize_scratchpad(candidate, answer_str)


def _euclid_steps(a: int, b: int, *, include_zero_step: bool = True) -> Tuple[List[str], int]:
    aa = abs(int(a))
    bb = abs(int(b))
    if bb == 0:
        raise ValueError("Euclidean steps require non-zero divisor")

    steps: List[str] = []
    while bb != 0:
        q, r = divmod(aa, bb)
        if include_zero_step or r != 0:
            steps.append(f"{aa}={q}*{bb}+{r}")
        aa, bb = bb, r
    return steps, aa


def gcd_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        a, b = _extract_first_two_ints(_clean_text(question))
        steps, g = _euclid_steps(a, b, include_zero_step=False)
        candidate = "|".join([*steps, f"gcd={g}"])
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def lcm_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        a, b = _extract_first_two_ints(_clean_text(question))
        aa = abs(int(a))
        bb = abs(int(b))
        if aa == 0 or bb == 0:
            candidate = f"{aa}*{bb}=0|lcm=0"
            return _finalize_scratchpad(candidate, answer_str)

        steps, g = _euclid_steps(aa, bb, include_zero_step=False)
        product = aa * bb
        lcm_value = product // g
        candidate = "|".join(
            [
                *steps,
                f"gcd={g}",
                f"{aa}*{bb}={product}",
                f"{product}/{g}={lcm_value}",
            ]
        )
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def div_remainder_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        a, b = _extract_first_two_ints(_clean_text(question))
        if int(b) == 0:
            return _fallback_answer(answer_str)
        q, r = divmod(int(a), int(b))
        mul_result = q * int(b)
        candidate = f"{a}/{b}|{b}*{q}={mul_result}|{a}-{mul_result}={r}|remainder={r}"
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def parse_is_factor_operands(question: str) -> Tuple[int, int]:
    """
    Return (target, factor) so that factor divides target.
    """
    text = _clean_text(question)
    lowered = text.lower()
    n1, n2 = _extract_first_two_ints(text)

    if "multiple of" in lowered or "divisible by" in lowered:
        return int(n1), int(n2)
    if "factor of" in lowered or "divisor of" in lowered:
        return int(n2), int(n1)

    if abs(int(n1)) >= abs(int(n2)):
        return int(n1), int(n2)
    return int(n2), int(n1)


def is_factor_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        target, factor = parse_is_factor_operands(question)
        if int(factor) == 0:
            return _fallback_answer(answer_str)
        q, r = divmod(int(target), int(factor))
        mul_result = q * int(factor)
        if r == 0:
            candidate = (
                f"{target}/{factor}|{factor}*{q}={mul_result}|"
                f"remainder={r}|is_factor={answer_str}"
            )
        else:
            candidate = (
                f"{target}/{factor}|{factor}*{q}={mul_result}|"
                f"{target}-{mul_result}={r}|remainder={r}|is_factor={answer_str}"
            )
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def _fraction_to_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _extract_mixed_expression(question: str) -> str:
    text = _clean_text(question)
    text = _EXPR_PREFIX_PATTERN.sub("", text).strip()
    if not text:
        raise ValueError("Empty mixed expression")
    if not _ALLOWED_MIXED_EXPR_PATTERN.fullmatch(text):
        raise ValueError(f"Unsupported characters in mixed expression={text!r}")
    return text


def _constant_to_fraction(expr: str, node: ast.Constant) -> Fraction:
    token = _constant_token(expr, node)
    return Fraction(token)


def _format_operand_for_op(token: str, op: str) -> str:
    text = str(token).strip()
    needs_wrap = False

    if op in {"*", "/"} and "/" in text:
        needs_wrap = True
    if len(text) > 1 and text.startswith("-"):
        needs_wrap = True
    if ("+" in text[1:]) or ("-" in text[1:]):
        needs_wrap = True

    if needs_wrap and not (text.startswith("(") and text.endswith(")")):
        return f"({text})"
    return text


def _eval_mixed_steps(expr: str) -> List[str]:
    tree = ast.parse(expr, mode="eval")
    steps: List[str] = []

    def _eval_node(node: ast.AST) -> Tuple[Fraction, str]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left_val, left_text = _eval_node(node.left)
            right_val, right_text = _eval_node(node.right)

            if isinstance(node.op, ast.Add):
                value = left_val + right_val
                op = "+"
            elif isinstance(node.op, ast.Sub):
                value = left_val - right_val
                op = "-"
            elif isinstance(node.op, ast.Mult):
                value = left_val * right_val
                op = "*"
            else:
                if right_val == 0:
                    raise ValueError("Division by zero in mixed expression")
                value = left_val / right_val
                op = "/"

            left_out = _format_operand_for_op(left_text, op)
            right_out = _format_operand_for_op(right_text, op)
            value_text = _fraction_to_text(value)
            steps.append(f"{left_out}{op}{right_out}={value_text}")
            return value, value_text

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand_val, _operand_text = _eval_node(node.operand)
            value = -operand_val if isinstance(node.op, ast.USub) else operand_val
            return value, _fraction_to_text(value)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = _constant_to_fraction(expr, node)
            return value, _fraction_to_text(value)

        raise ValueError(f"Unsupported node type in mixed expression: {type(node).__name__}")

    _eval_node(tree.body)
    return steps


def mixed_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        expr = _extract_mixed_expression(question)
        steps = _eval_mixed_steps(expr)
        if not steps:
            return _fallback_answer(answer_str)
        candidate = "|".join([*steps, f"result={answer_str}"])
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def _direct_result_scratchpad(_question: str, answer: str) -> str:
    return _finalize_scratchpad(f"result={str(answer).strip()}", str(answer).strip())


def _normalize_symbolic_text(text: str) -> str:
    return str(text).strip().replace("**", "^").replace(" ", "")


def _safe_eval_numeric_expr(expr: str) -> Fraction:
    tree = ast.parse(str(expr).replace("^", "**"), mode="eval")

    def _eval_node(node: ast.AST) -> Fraction:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            if right.denominator != 1:
                raise ValueError("fractional exponent")
            return left ** int(right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = _eval_node(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            token = ast.get_source_segment(str(expr).replace("^", "**"), node)
            return Fraction(str(token if token is not None else node.value))
        raise ValueError(f"unsupported numeric node: {type(node).__name__}")

    return _eval_node(tree.body)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _parse_int_sequence(question: str) -> List[int]:
    return [int(v) for v in re.findall(r"-?\d+", str(question))]


def _finite_difference_rows(values: List[int]) -> List[List[int]]:
    rows: List[List[int]] = [list(values)]
    while len(rows[-1]) > 1:
        prev = rows[-1]
        cur = [int(prev[i + 1] - prev[i]) for i in range(len(prev) - 1)]
        rows.append(cur)
        if len(set(cur)) == 1:
            break
    return rows


def linear_1d_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        q = _clean_text(question)
        match = re.search(r"^Solve\s+(.+?)\s+for\s+([A-Za-z])$", q)
        if not match:
            return _fallback_answer(answer_str)
        equation = match.group(1)
        var_name = match.group(2)
        if "=" not in equation:
            return _fallback_answer(answer_str)
        left_raw, right_raw = equation.split("=", 1)
        # Evaluate the equation at x=0 and x=1 to recover coeff*x + const = 0.
        expr = f"({left_raw})-({right_raw})"
        const = _safe_eval_numeric_expr(expr.replace(var_name, "0"))
        at_one = _safe_eval_numeric_expr(expr.replace(var_name, "1"))
        coeff = at_one - const
        if coeff == 0:
            return _fallback_answer(answer_str)
        numerator = -const
        solution = numerator / coeff
        candidate = (
            f"{_format_fraction(coeff)}*{var_name}+{_format_fraction(const)}=0|"
            f"0-({_format_fraction(const)})={_format_fraction(numerator)}|"
            f"{_format_fraction(numerator)}/{_format_fraction(coeff)}={_format_fraction(solution)}|"
            f"{var_name}={answer_str}"
        )
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def sequence_next_term_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        values = _parse_int_sequence(question)
        if len(values) < 3:
            return _fallback_answer(answer_str)
        rows = _finite_difference_rows(values)
        highest = rows[-1][-1]
        highest_name = f"d{len(rows) - 1}"
        steps = [f"{highest_name}={highest}"]
        next_delta = highest
        for order in range(len(rows) - 2, 0, -1):
            current = rows[order][-1]
            propagated = current + next_delta
            name = "d" if order == 1 else f"d{order}"
            steps.append(f"{name}={current}+{next_delta}={propagated}")
            next_delta = propagated
        steps.append(f"{values[-1]}+{next_delta}={answer_str}")
        return _finalize_scratchpad("|".join([*steps, f"result={answer_str}"]), answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def sequence_nth_term_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        q = _clean_text(question)
        var_match = re.search(r"([A-Za-z])'th term", q)
        var_name = var_match.group(1) if var_match else "n"
        values = _parse_int_sequence(question)
        if len(values) < 3:
            return _fallback_answer(answer_str)
        rows = _finite_difference_rows(values)
        steps: List[str] = []
        for row in rows[:-1]:
            if len(row) >= 2:
                steps.append(f"{row[1]}-{row[0]}={row[1] - row[0]}")
        formula = _normalize_symbolic_text(answer_str)
        candidate = "|".join([*steps, f"a_{var_name}={formula}", f"result={answer_str}"])
        return _finalize_scratchpad(candidate, answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def _extract_simple_function_defs(question: str) -> Dict[str, Tuple[str, str]]:
    q = _clean_text(question)
    definitions: Dict[str, Tuple[str, str]] = {}
    for match in re.finditer(r"Let\s+([A-Za-z])\(([A-Za-z])\)\s*=\s*(.*?)(?:\.|$)", q):
        definitions[match.group(1)] = (match.group(2), match.group(3).strip())
    return definitions


def _poly_coeffs(expr: str, var_name: str) -> Dict[int, Fraction]:
    # Minimal polynomial parser for expanded DeepMind expressions. It handles
    # sums of terms like -3*x**2, x, and constants. More complex expressions
    # intentionally fall back to result=answer.
    clean = str(expr).replace(" ", "")
    if not clean:
        raise ValueError("empty polynomial")
    normalized = clean.replace("-", "+-")
    if normalized.startswith("+"):
        normalized = normalized[1:]
    coeffs: Dict[int, Fraction] = {}
    for raw_term in normalized.split("+"):
        term = raw_term.strip()
        if not term:
            continue
        if var_name in term:
            if f"{var_name}**" in term:
                prefix, degree_raw = term.split(f"{var_name}**", 1)
                degree = int(degree_raw)
            else:
                prefix = term.split(var_name, 1)[0]
                degree = 1
            prefix = prefix.rstrip("*")
            if prefix in {"", "+"}:
                coeff = Fraction(1)
            elif prefix == "-":
                coeff = Fraction(-1)
            else:
                coeff = _safe_eval_numeric_expr(prefix)
        else:
            degree = 0
            coeff = _safe_eval_numeric_expr(term)
        coeffs[degree] = coeffs.get(degree, Fraction(0)) + coeff
    return {d: c for d, c in coeffs.items() if c != 0}


def polynomials_evaluate_scratchpad(question: str, answer: str) -> str:
    answer_str = str(answer).strip()
    try:
        definitions = _extract_simple_function_defs(question)
        tail = re.split(r"(?:What is|Determine|Calculate|Give)\s+", _clean_text(question))[-1]
        call = re.search(r"([A-Za-z])\((-?\d+)\)", tail)
        if not call:
            return _fallback_answer(answer_str)
        func_name, x0_raw = call.group(1), call.group(2)
        if func_name not in definitions:
            return _fallback_answer(answer_str)
        var_name, expr = definitions[func_name]
        coeffs = _poly_coeffs(expr, var_name)
        x0 = int(x0_raw)
        max_degree = max(coeffs) if coeffs else 0
        powers = {1: x0}
        steps: List[str] = []
        for degree in range(2, max_degree + 1):
            powers[degree] = int(x0**degree)
            steps.append(f"{x0}^{degree}={powers[degree]}")
        term_values: List[Fraction] = []
        for degree in range(max_degree, -1, -1):
            coeff = coeffs.get(degree, Fraction(0))
            if coeff == 0:
                continue
            base = Fraction(1 if degree == 0 else powers.get(degree, x0))
            value = coeff * base
            term_values.append(value)
            if degree == 0:
                steps.append(f"{_format_fraction(coeff)}={_format_fraction(value)}")
            elif degree == 1:
                steps.append(f"{_format_fraction(coeff)}*{x0}={_format_fraction(value)}")
            else:
                steps.append(f"{_format_fraction(coeff)}*{powers[degree]}={_format_fraction(value)}")
        if term_values:
            sum_expr = "+".join(
                f"({_format_fraction(v)})" if v < 0 else _format_fraction(v)
                for v in term_values
            )
            steps.append(f"{sum_expr}={answer_str}")
        return _finalize_scratchpad("|".join([*steps, f"result={answer_str}"]), answer_str)
    except Exception:
        return _fallback_answer(answer_str)


def polynomials_add_scratchpad(question: str, answer: str) -> str:
    # Raw DeepMind polynomials__add frequently contains auxiliary constants,
    # derivatives, systems, and nested function definitions. Until a complete
    # symbolic resolver is implemented, keep this module explicit answer-only.
    return _fallback_answer(str(answer).strip())


def apply_mul_scratchpad_to_dataset(
    dataset,
    *,
    module_name: str = "arithmetic__mul",
) -> Dict[str, int]:
    """
    In-place conversion of dataset answers for one module to scratchpad format.

    Returns:
      dict with counters: scanned / converted.
    """
    converted = 0
    scanned = len(getattr(dataset, "module_names", ()))

    for idx, mod in enumerate(getattr(dataset, "module_names", ())):
        if str(mod) != str(module_name):
            continue
        question = dataset.questions[idx]
        answer = dataset.answers[idx]
        scratch = _mul_scratchpad_from_question(question, answer)
        _encode_dataset_answer(dataset, idx, scratch)
        converted += 1

    return {"scanned": int(scanned), "converted": int(converted)}


def apply_all_scratchpad_to_dataset(
    dataset,
    *,
    progress_callback=None,
    dataset_label: str = "",
) -> Dict[str, int]:
    """
    Convert in-place all supported primaire+college modules to scratchpad targets.

    Returns:
      dict counters, including total scanned/converted and per-module counts.
    """
    dispatch = {
        "arithmetic__mul": _mul_scratchpad_from_question,
        "arithmetic__div": div_scratchpad,
        "arithmetic__add_or_sub": add_or_sub_scratchpad,
        "arithmetic__add_sub_multiple": add_sub_multiple_scratchpad,
        "numbers__gcd": gcd_scratchpad,
        "numbers__lcm": lcm_scratchpad,
        "numbers__div_remainder": div_remainder_scratchpad,
        "numbers__is_factor": is_factor_scratchpad,
        "arithmetic__mixed": mixed_scratchpad,
        "numbers__place_value": _direct_result_scratchpad,
        "numbers__round_number": _direct_result_scratchpad,
        "algebra__linear_1d": linear_1d_scratchpad,
        "algebra__sequence_next_term": sequence_next_term_scratchpad,
        "algebra__sequence_nth_term": sequence_nth_term_scratchpad,
        "polynomials__evaluate": polynomials_evaluate_scratchpad,
        "polynomials__add": polynomials_add_scratchpad,
    }
    per_module = {name: 0 for name in dispatch}

    converted = 0
    scanned = len(getattr(dataset, "module_names", ()))
    module_counts = {}
    for module_name in getattr(dataset, "module_names", ()):
        module_key = str(module_name)
        module_counts[module_key] = int(module_counts.get(module_key, 0)) + 1

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "scratchpad_dataset_start",
                "dataset": str(dataset_label),
                "scanned": int(scanned),
                "module_counts": dict(module_counts),
            }
        )

    module_progress = {name: 0 for name in dispatch}
    for idx, module_name in enumerate(getattr(dataset, "module_names", ())):
        module_key = str(module_name)
        converter = dispatch.get(module_key)
        if converter is None:
            continue
        question = dataset.questions[idx]
        answer = dataset.answers[idx]
        scratch = converter(question, answer)
        _encode_dataset_answer(dataset, idx, scratch)
        per_module[module_key] += 1
        module_progress[module_key] += 1
        converted += 1
        if (
            progress_callback is not None
            and module_progress[module_key] > 0
            and module_progress[module_key] % 50000 == 0
        ):
            progress_callback(
                {
                    "stage": "scratchpad_module_progress",
                    "dataset": str(dataset_label),
                    "module": str(module_key),
                    "rows": int(module_progress[module_key]),
                }
            )

    out = {"scanned": int(scanned), "converted": int(converted)}
    out.update({k: int(v) for k, v in per_module.items()})
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "scratchpad_dataset_done",
                "dataset": str(dataset_label),
                "converted": int(converted),
                "per_module": {k: int(v) for k, v in per_module.items()},
            }
        )
    return out


def run_scratchpad_selftest() -> None:
    """Deterministic unit-like checks used by smoke commands."""
    assert mul_scratchpad(7, 8) == "7*8=56"
    assert mul_scratchpad(12, 34) == "12*4=48|12*30=360|48+360=408"
    assert mul_scratchpad(23, 45) == "23*5=115|23*40=920|115+920=1035"
    assert (
        mul_scratchpad(123, 456)
        == "123*6=738|123*50=6150|123*400=49200|738+6150+49200=56088"
    )
    assert mul_scratchpad(99, 99) == "99*9=891|99*90=8910|891+8910=9801"
    assert mul_scratchpad(5, 0) == "5*0=0"
    assert extract_answer("23*5=115|23*40=920|115+920=1035") == "1035"
    assert extract_answer("7*8=56") == "56"
    assert parse_mul_operands("123 * 456") == (123, 456)
    assert parse_mul_operands("What is 12*34?") == (12, 34)

    add_sc = add_or_sub_scratchpad("Total of 0.06 and -1977321735.", "-1977321734.94")
    assert extract_answer(add_sc) == "-1977321734.94"
    asm_sc = add_sub_multiple_scratchpad("Evaluate 60 - ((36 - 34) + 33).", "25")
    assert extract_answer(asm_sc) == "25"
    div_sc = div_scratchpad("Divide -57912 by -57.", "1016")
    assert extract_answer(div_sc) == "1016"
    div_frac_sc = div_scratchpad("Divide 3383266 by -5.", "-3383266/5")
    assert extract_answer(div_frac_sc) == "-3383266/5"
    div_exact_sc = div_scratchpad("Divide 144 by 12.", "12")
    assert div_exact_sc == "144/12|12*12=144|result=12"
    assert extract_answer(div_exact_sc) == "12"

    gcd_sc = gcd_scratchpad("Calculate the greatest common factor of 48 and 18.", "6")
    assert gcd_sc == "48=2*18+12|18=1*12+6|gcd=6"
    assert extract_answer(gcd_sc) == "6"
    gcd_equal_sc = gcd_scratchpad("Calculate the greatest common factor of 12 and 12.", "12")
    assert gcd_equal_sc == "gcd=12"
    assert extract_answer(gcd_equal_sc) == "12"

    lcm_sc = lcm_scratchpad("What is the lowest common multiple of 12 and 8?", "24")
    assert lcm_sc == "12=1*8+4|gcd=4|12*8=96|96/4=24"
    assert extract_answer(lcm_sc) == "24"
    lcm_equal_sc = lcm_scratchpad("What is the lowest common multiple of 12 and 12?", "12")
    assert lcm_equal_sc == "gcd=12|12*12=144|144/12=12"
    assert extract_answer(lcm_equal_sc) == "12"

    drm_sc = div_remainder_scratchpad("Calculate the remainder when 17 is divided by 5.", "2")
    assert drm_sc == "17/5|5*3=15|17-15=2|remainder=2"
    assert extract_answer(drm_sc) == "2"

    isf_true = is_factor_scratchpad("Is 4 a factor of 36?", "True")
    assert isf_true == "36/4|4*9=36|remainder=0|is_factor=True"
    assert extract_answer(isf_true) == "True"
    isf_false = is_factor_scratchpad("Is 3 a factor of 17?", "False")
    assert isf_false == "17/3|3*5=15|17-15=2|remainder=2|is_factor=False"
    assert extract_answer(isf_false) == "False"

    mixed_sc = mixed_scratchpad("What is (2 + 3) * 4?", "20")
    assert extract_answer(mixed_sc) == "20"

    lin_sc = linear_1d_scratchpad("Solve 24 = 1601*c - 1605*c for c.", "-6")
    assert lin_sc == "4*c+24=0|0-(24)=-24|-24/4=-6|c=-6"
    assert extract_answer(lin_sc) == "-6"
    seq_next = sequence_next_term_scratchpad("What is next in -6525, -6520, -6515, -6510?", "-6505")
    assert seq_next == "d1=5|-6510+5=-6505|result=-6505"
    assert extract_answer(seq_next) == "-6505"
    seq_nth = sequence_nth_term_scratchpad(
        "What is the f'th term of 2298, 2334, 2372, 2412, 2454?",
        "f**2 + 33*f + 2264",
    )
    assert extract_answer(seq_nth) == "f**2 + 33*f + 2264"
    poly_eval = polynomials_evaluate_scratchpad("Let z(p) = -18*p - 920. Give z(-35).", "-290")
    assert extract_answer(poly_eval) == "-290"


__all__ = [
    "add_or_sub_scratchpad",
    "add_sub_multiple_scratchpad",
    "apply_all_scratchpad_to_dataset",
    "apply_mul_scratchpad_to_dataset",
    "div_remainder_scratchpad",
    "div_scratchpad",
    "extract_answer",
    "gcd_scratchpad",
    "is_factor_scratchpad",
    "linear_1d_scratchpad",
    "lcm_scratchpad",
    "mixed_scratchpad",
    "mul_scratchpad",
    "parse_add_or_sub_operands",
    "parse_div_operands",
    "parse_is_factor_operands",
    "parse_mul_operands",
    "polynomials_add_scratchpad",
    "polynomials_evaluate_scratchpad",
    "run_scratchpad_selftest",
    "sequence_next_term_scratchpad",
    "sequence_nth_term_scratchpad",
]
