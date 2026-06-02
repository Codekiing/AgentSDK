"""Custom reward function for GSM8K dataset with data_source="gsm8k".

VERL's built-in default_compute_score only handles "openai/gsm8k".
This wrapper delegates to the VERL gsm8k reward for our "gsm8k" data_source.
"""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute reward for GSM8K math problems.

    Args:
        data_source: Dataset source identifier (expected: "gsm8k").
        solution_str: Model output string.
        ground_truth: Expected answer (numeric string).
        extra_info: Optional extra information.

    Returns:
        float: 1.0 for correct, 0.0 for incorrect.
    """
    if data_source not in ("gsm8k", "openai/gsm8k"):
        return 0.0

    # Extract the final answer from the solution
    # GSM8K answers are numbers, potentially with commas
    answer = _extract_gsm8k_answer(solution_str)
    if answer is None:
        return 0.0

    # Normalize and compare
    try:
        pred_val = _normalize_number(answer)
        gt_val = _normalize_number(str(ground_truth))
        return 1.0 if abs(pred_val - gt_val) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return 0.0


def _extract_gsm8k_answer(text: str):
    """Extract the final numeric answer from GSM8K-style model output.

    GSM8K answers typically appear after '####' or as the last number.
    Also handles various answer formats.
    """
    if not text:
        return None

    # Pattern 1: #### marker (standard GSM8K format)
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        return match.group(1)

    # Pattern 2: \boxed{} format
    match = re.search(r'\\boxed\{(-?[\d,]+(?:\.\d+)?)\}', text)
    if match:
        return match.group(1)

    # Pattern 3: "The answer is X" or "= X" at the end
    match = re.search(r'(?:answer\s*(?:is|=|:)?\s*)(-?[\d,]+(?:\.\d+)?)\s*$', text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 4: Last number in the text
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    return None


def _normalize_number(s: str):
    """Normalize a number string: remove commas, convert to float."""
    s = str(s).strip().replace(',', '').replace(' ', '')
    # Handle fractions like "1/2"
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1])
    return float(s)
