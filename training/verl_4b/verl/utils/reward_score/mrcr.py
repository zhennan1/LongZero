"""
MRCR (Multi-Round Co-reference Resolution) Reward Score

This module computes the reward score for MRCR tasks.
The evaluation metric is the SequenceMatcher ratio as implemented in Python's difflib.

The model must:
1. Prepend a specific random alphanumeric string to its answer
2. The rest of the response should match the ground truth needle content
"""

import os
from difflib import SequenceMatcher


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute score for MRCR task.

    Args:
        solution_str (str): Model's response string
        ground_truth (str or dict): Expected answer (with random prefix) or dict containing answer
        extra_info (dict, optional): Additional info including random_string_to_prepend

    Returns:
        float: Score between 0 and 1 (SequenceMatcher ratio)
    """
    if solution_str is None:
        return 0.0

    solution_str = solution_str.strip()

    # Handle thinking model output: remove <think>...</think> section
    end_tag = "</think>"
    pos = solution_str.rfind(end_tag) if isinstance(solution_str, str) else -1
    if pos != -1:
        solution_str = solution_str[pos + len(end_tag):]
    solution_str = solution_str.lstrip()

    # Get random string to prepend
    if extra_info is not None and "random_string_to_prepend" in extra_info:
        random_string_to_prepend = extra_info["random_string_to_prepend"]
    elif isinstance(ground_truth, dict) and "random_string_to_prepend" in ground_truth:
        random_string_to_prepend = ground_truth["random_string_to_prepend"]
        ground_truth = ground_truth.get("answer", ground_truth.get("ground_truth", ""))
    else:
        # Try to extract from ground_truth if it's a string (first 10 chars are typically the prefix)
        random_string_to_prepend = ground_truth[:10] if isinstance(ground_truth, str) and len(ground_truth) >= 10 else ""

    # Check if response starts with the required random string
    if not solution_str.startswith(random_string_to_prepend):
        return 0.0

    # Remove the prefix for comparison
    response_content = solution_str.removeprefix(random_string_to_prepend)

    if isinstance(ground_truth, str):
        answer_content = ground_truth.removeprefix(random_string_to_prepend)
    else:
        answer_content = str(ground_truth)

    # Compute SequenceMatcher ratio
    score = float(SequenceMatcher(None, response_content, answer_content).ratio())

    return score


def compute_score_strict(solution_str, ground_truth, extra_info=None, threshold=0.8):
    """
    Compute binary score for MRCR task with strict matching.

    Args:
        solution_str (str): Model's response string
        ground_truth (str or dict): Expected answer
        extra_info (dict, optional): Additional info
        threshold (float): Minimum SequenceMatcher ratio for a positive score

    Returns:
        float: 1.0 if ratio >= threshold, 0.0 otherwise
    """
    ratio = compute_score(solution_str, ground_truth, extra_info)
    return 1.0 if ratio >= threshold else 0.0


def compute_score_scaled(solution_str, ground_truth, extra_info=None, min_threshold=0.5):
    """
    Compute scaled score for MRCR task.
    Returns 0 if ratio < min_threshold, otherwise returns the ratio.

    Args:
        solution_str (str): Model's response string
        ground_truth (str or dict): Expected answer
        extra_info (dict, optional): Additional info
        min_threshold (float): Minimum ratio to get a non-zero score

    Returns:
        float: Score between 0 and 1
    """
    ratio = compute_score(solution_str, ground_truth, extra_info)
    return ratio if ratio >= min_threshold else 0.0
