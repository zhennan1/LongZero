import re
import os
from collections import Counter
import string



def compute_score(solution_str, ground_truth):
    """
    Compute score for longcontext_qa
    Args:
        solution_str (str): Solution string
        ground_truth (str): Ground truth string
    Returns:
        float: Score
    """
    solution_str = solution_str.strip()
    reward_calc_type = os.getenv("REWARD_CALC_TYPE", "pure_exact_match")
    if reward_calc_type == "pure_exact_match":
        retval = _pure_exact_match_in_string(solution_str, ground_truth)
    elif reward_calc_type == "f1_score":
        raise NotImplementedError("F1 score is not implemented yet")
    elif reward_calc_type == "format_exact_match":
        retval = _format_exact_match_in_string(solution_str, ground_truth)
    elif reward_calc_type == "mcq_letter_match":
        retval = _mcq_letter_match(solution_str, ground_truth)
    elif reward_calc_type == "answer_line_exact_match":
        retval = _answer_line_exact_match(solution_str, ground_truth)
    elif reward_calc_type == "answer_line_exact_match_v2":
        retval = _answer_line_exact_match_v2(solution_str, ground_truth)
    elif reward_calc_type == "format_f1_score":
        retval = _format_f1_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Unknown reward_calc_type: {reward_calc_type}")

    return retval

def _format_f1_score(solution_str, ground_truth):
    """
    export REWARD_CALC_TYPE=format_f1_score
    export ANSWER_OVER_FLOW_LIMIT=128
    export EOT_OVER_FLOW_LIMIT=32
    export MAX_BOXED_LIMIT=1
    export PUNISH_MULTIPLE_BRACES=1
    export F1_SCORE_THRESHOLD=0.5
    """
    max_boxed_limit = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_multiple_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if max_boxed_limit > 0:
                if solution_str.count("\\boxed") > max_boxed_limit:
                    boxed_occurs = solution_str.count("\\boxed")
                    raise ValueError(f"Too many boxed parts in solution_str: {boxed_occurs} > {max_boxed_limit}")
                    # return 0

            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if punish_multiple_braces > 0:
                    if pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1:
                        print(f"Multiple braces found in pred: {pred}")
                        raise ValueError(f"Multiple braces found in pred: {pred}")
                        # return 0
                assert isinstance(pred, str), f"pred should be a string, got {type(pred)} instead"
                assert isinstance(truth, str), f"truth should be a string, got {type(truth)} instead"
                f1_score = qa_f1_score(pred, truth)
                f1_score_threshold = float(os.getenv("F1_SCORE_THRESHOLD", 0.5))
                if f1_score > f1_score_threshold:
                    retval = f1_score
                    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
                    if len(pred) - len(truth) > answer_over_flow_limit:
                        retval = 0
                    eot_over_flow_limit = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))
                    if len(solution_str) - (solution_str.rfind(pred)+len(pred)) > eot_over_flow_limit:
                        retval = 0
                    max_retval = max(max_retval, retval)
                    
        except Exception as e:
            print(f"Error encountered: {e}")
            return max_retval
                
    return max_retval

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)
    
        
def _pure_exact_match_in_string(solution_str, ground_truth):
    if isinstance(ground_truth, str):  # More Pythonic way to check type
        ground_truth = [ground_truth]
    
    retval = 0  # Default return value
    
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if is_gt_in_pred(pred, truth):
                    retval = 1.0
                    return retval  # Early return if a match is found
        except Exception as e:
            print(f"Error encountered: {e}")
            return retval  # Return 0 if an exception occurs

    return retval  # Return 0 if no match is found

def _format_exact_match_in_string(solution_str, ground_truth):
    """
    export REWARD_CALC_TYPE=format_exact_match
    export ANSWER_OVER_FLOW_LIMIT=128
    export EOT_OVER_FLOW_LIMIT=32
    export MAX_BOXED_LIMIT=1
    export PUNISH_MULTIPLE_BRACES=1
    """
    max_boxed_limit = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_multiple_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if max_boxed_limit > 0:
                if solution_str.count("\\boxed") > max_boxed_limit:
                    boxed_occurs = solution_str.count("\\boxed")
                    raise ValueError(f"Too many boxed parts in solution_str: {boxed_occurs} > {max_boxed_limit}")
                    # return 0

            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if punish_multiple_braces > 0:
                    if pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1:
                        print(f"Multiple braces found in pred: {pred}")
                        raise ValueError(f"Multiple braces found in pred: {pred}")
                        # return 0
                assert isinstance(pred, str), f"pred should be a string, got {type(pred)} instead"
                assert isinstance(truth, str), f"truth should be a string, got {type(truth)} instead"
                if is_gt_in_pred(pred, truth):
                    retval = 1.0
                    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
                    if len(pred) - len(truth) > answer_over_flow_limit:
                        retval -= 1.0
                    eot_over_flow_limit = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))
                    if len(solution_str) - (solution_str.rfind(pred)+len(pred)) > eot_over_flow_limit:
                        retval -= 1.0
                    max_retval = max(max_retval, retval)
                    
        except Exception as e:
            print(f"Error encountered: {e}")
            return max_retval
                
    return max_retval





def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def normalize_text(text):
    """
    Normalize text by lowercasing and removing special characters
    """
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def is_gt_in_pred(pred, ground_truth, verbose=False):
    if pred is None and ground_truth is None:
        print("WARNING: Both None")
        return True
    if pred is None or ground_truth is None:
        return False

    try:
        # normalize string
        ss1 = normalize_text(pred)
        ss2 = normalize_text(ground_truth)
        if verbose:
            print(ss1, ss2)
        return ss2 in ss1
    except Exception:
        return ground_truth in pred


ANSWER_LINE_RE = re.compile(r"(?i)the answer is\s*:?\s*(.+?)(?:\.\s*$|\n|$)")


def extract_answer_line(solution_str):
    """Pull the answer out of the last `the answer is X` in the whole response.

    Case-insensitive, anywhere in the generation: the final mention is the
    model's last word on the answer, whether or not it made it past </think>.
    """
    matches = ANSWER_LINE_RE.findall(solution_str)
    return matches[-1].strip() if matches else None


def _answer_line_exact_match(solution_str, ground_truth):
    """
    export REWARD_CALC_TYPE=answer_line_exact_match
    export ANSWER_OVER_FLOW_LIMIT=128

    Measured on the base model (40 val questions x 2 rollouts): it produces the
    answer-line marker 98.8% of the time versus 47.5% for \boxed{}, so this mode
    scores whether the question was answered instead of whether the marker was
    remembered.
    """
    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    pred = extract_answer_line(solution_str)
    if pred is None:
        return 0
    for truth in ground_truth:
        try:
            if is_gt_in_pred(pred, truth):
                if len(pred) - len(truth) > answer_over_flow_limit:
                    continue
                return 1.0
        except Exception as e:
            print(f"Error encountered: {e}")
            return 0
    return 0

_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


def is_gt_in_pred_v2(pred, ground_truth):
    """Whole-word version of is_gt_in_pred (2026-10-07): curly quotes unified, and the
    normalized truth must appear as a whole-token span, so "15" no longer matches "5"
    and "mid-1950s" no longer matches "1950"."""
    if pred is None or ground_truth is None:
        return False
    ss1 = normalize_text(str(pred).translate(_QUOTES))
    ss2 = normalize_text(str(ground_truth).translate(_QUOTES))
    return bool(ss2) and f" {ss2} " in f" {ss1} "


def _answer_line_exact_match_v2(solution_str, ground_truth):
    """
    export REWARD_CALC_TYPE=answer_line_exact_match_v2
    Same as answer_line_exact_match but with is_gt_in_pred_v2 (whole-word, curly quotes).
    Paired with the cleaned data whose ground_truth keeps only answer-equivalent aliases.
    """
    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    pred = extract_answer_line(solution_str)
    if pred is None:
        return 0
    for truth in ground_truth:
        if is_gt_in_pred_v2(pred, truth) and len(pred) - len(truth) <= answer_over_flow_limit:
            return 1.0
    return 0


LETTER_RE = re.compile(r"^[\(\[\s]*([A-Da-d])[\)\].,:;\s]*$")


def _mcq_letter_match(solution_str, ground_truth):
    """Four-choice: ground_truth = [correct letter, correct option text].

    Judge by letter first; if the model writes the option text instead of the letter, it is also correct (a lenient rule
    that avoids repeating v1, where the `\\boxed` anchor scored nearly half of the correct answers as 0).
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    gt = list(ground_truth)
    letter = str(gt[0]).strip().upper() if gt else ""
    text = str(gt[1]) if len(gt) > 1 else ""

    pred = extract_answer_line(solution_str)
    if not pred:
        return 0.0
    pred = pred.strip()

    m = LETTER_RE.match(pred)
    if m:
        return 1.0 if m.group(1).upper() == letter else 0.0

    # not a bare letter: allow forms such as "C) 1995" / "1995"
    m2 = re.match(r"^[\(\[\s]*([A-Da-d])[\)\].,:;]\s+\S", pred)
    if m2:
        return 1.0 if m2.group(1).upper() == letter else 0.0
    if text and is_gt_in_pred(pred, text):
        return 1.0
    return 0.0
