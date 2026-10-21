"""LongBench v2 multiple-choice reward: the extraction logic is verbatim identical to extract_answer in the official pred.py.

Two properties of the official implementation are deliberately kept so that the reward and the final evaluation use the same metric:
  1. `*` is removed first (the official code does this to handle markdown bold)
  2. re.search takes the **first** match, not the last one
Point 2 has a side effect for thinking models: if the reasoning first writes "The correct answer is (B)" and only
concludes (C) at the end, B is scored. The evaluation side behaves the same way, and taking the last match would
misalign training and evaluation, so it is left unchanged; this is a deliberate consistency choice.
"""
import re

CHOICES = ("A", "B", "C", "D")


def extract_answer(response):
    response = (response or "").replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


def compute_score(solution_str, ground_truth):
    """1.0 iff the extracted option matches the gold label; 0 if nothing can be extracted (treated as not following the format)."""
    if isinstance(ground_truth, (list, tuple)):
        golds = [str(g).strip().upper() for g in ground_truth]
    else:
        golds = [str(ground_truth).strip().upper()]
    pred = extract_answer(solution_str)
    if pred is None:
        return 0.0
    return 1.0 if pred.strip().upper() in golds else 0.0
