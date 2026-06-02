"""Wrapper around math_dapo with strict_box_verify=True.

VERL default calls math_dapo.compute_score without strict_box_verify,
which uses the Minerva pattern ("Answer: NUMBER") and fails for
\\boxed{} format. This wrapper forces strict_box_verify=True so
the reward function correctly extracts \\boxed{answer} from outputs.
"""

from verl.utils.reward_score.math_dapo import compute_score as _math_dapo_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute reward using math_dapo with strict box verification.

    Extracts \\boxed{answer} from the last 100 chars of the solution
    and compares to ground_truth using math_dapo's normalize_final_answer.
    """
    result = _math_dapo_compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        strict_box_verify=True,
    )
    # Fix: replace None pred with string to avoid VERL validation crash
    if result.get("pred") is None:
        result["pred"] = "[NO_BOXED]"
    return result
