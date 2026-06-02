"""Reward function for deepscaler data with LaTeX ground truths.

Handles answer extraction from multiple formats (####, \\boxed{}, Answer:)
and compares with LaTeX ground truths using numerical evaluation.
"""
import re
from fractions import Fraction


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute reward for deepscaler math problems.

    Args:
        data_source: Dataset source identifier.
        solution_str: Model output string.
        ground_truth: Expected answer (may be LaTeX).
        extra_info: Optional extra information.

    Returns:
        float: 1.0 for correct, 0.0 for incorrect.
    """
    solution_tail = solution_str[-300:] if len(solution_str) > 300 else solution_str

    pred = _extract_answer(solution_tail)
    if pred is None:
        return 0.0

    # Numerical comparison
    pred_val = _parse_number(pred)
    gt_val = _parse_number(ground_truth)
    if pred_val is not None and gt_val is not None:
        return 1.0 if abs(pred_val - gt_val) < 1e-6 else 0.0

    # String comparison after normalization
    if _normalize(pred) == _normalize(ground_truth):
        return 1.0

    return 0.0


def _extract_answer(text):
    """Extract answer from model output."""
    patterns = [
        r"####\s*(.+?)(?:\n|$)",
        r"\\boxed\{([^}]+)\}",
        r"(?i)Answer\s*:\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.findall(pattern, text)
        if match:
            return match[-1].strip()
    return None


def _parse_number(s):
    """Parse a math expression (possibly LaTeX) to a float."""
    expr = _latex_to_arithmetic(s.strip())
    if expr is None:
        return None
    try:
        return float(Fraction(expr))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(expr)
    except (ValueError, TypeError):
        pass
    return None


def _latex_to_arithmetic(s):
    """Convert LaTeX math expression to arithmetic string."""
    # \frac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    # \frac ab -> a/b (shorthand: two single chars after \frac)
    s = re.sub(r"\\frac(\d)(\d)", r"\1/\2", s)
    s = re.sub(r"\\frac(-?\d+\.?\d*)(-?\d+\.?\d*)", r"\1/\2", s)
    # Remove remaining LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Remove decoration
    s = s.replace("{", "").replace("}", "").replace("$", "").replace("(", "").replace(")", "").replace(" ", "")
    # Handle ratio format a:b -> a/b
    if re.fullmatch(r"-?[\d.]+:-?[\d.]+", s):
        s = s.replace(":", "/")
    if not s or not re.search(r"\d", s):
        return None
    return s


def _normalize(s):
    """Basic normalization for fallback string comparison."""
    s = s.strip().lower().replace(" ", "")
    s = s.replace("{", "").replace("}", "").replace("\\", "").replace("$", "")
    return s
