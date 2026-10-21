#!/usr/bin/env python3
"""LongBench v1 evaluation under the official protocol (three Multi-Doc QA subsets: hotpotqa / 2wikimqa / musique).

Aligned with the official pred.py/eval.py: prompts use the official dataset2prompt templates verbatim, scoring takes the
max of the official metrics.qa_f1_score over answers, and the context is not truncated (the 262k window far exceeds the
v1 lengths; the official truncation was a capacity compromise for 32k models).

Deviations (required for thinking models, documented): the official max_gen=32 is unusable for thinking-only models
(32 tokens cannot even hold the thinking), so a 16384-token budget is used and the text after the last </think> is scored
as the prediction by the official metric. Official greedy decoding is unusable for Qwen3 thinking models (the model card
explicitly warns that greedy decoding causes thinking loops, which caused 5.5-17.5% invalid outputs in a first greedy run),
so temperature 0.7 / top_p 0.95 is used; without </think> resample at most 2 more times, otherwise score 0
(the resample count is recorded in the output).
"""
import argparse, json, os, re, string, sys, pathlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os as _os
# NO_THINK=1: the evaluated model does not think (instruct); the whole output is the prediction, no resampling
NO_THINK = _os.getenv("NO_THINK") == "1"

# ---- English QA F1 from the official metrics.py, inlined verbatim (importing the full metrics would pull in jieba and other dependencies) ----
def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)

def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(normalize_answer(prediction).split(),
                    normalize_answer(ground_truth).split())
# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base_url", required=True)
ap.add_argument("--data_dir", default="assets/LongBench/v1data")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()

D2P = json.load(open(str(pathlib.Path(__file__).with_name("lb1_dataset2prompt.json"))))
DATASETS = ["hotpotqa", "2wikimqa", "musique"]

client = OpenAI(base_url=args.base_url, api_key="EMPTY")

def one(job):
    ds, idx, row = job
    prompt = D2P[ds].format(context=row["context"], input=row["input"])
    txt, tries, ctoks = "", 0, 0
    for attempt in range(3):  # no </think> (thinking did not converge): resample at most 2 more times
        tries = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.7, top_p=0.95, max_tokens=16384)
            txt = resp.choices[0].message.content or ""
            ctoks = getattr(resp.usage, "completion_tokens", 0)
        except Exception as e:
            return {"dataset": ds, "idx": idx, "error": repr(e), "f1": 0.0,
                    "tries": tries}
        if NO_THINK or "</think>" in txt:
            break
    if NO_THINK:
        pred, valid = txt.strip(), True
    elif "</think>" in txt:
        pred = txt.rsplit("</think>", 1)[1].strip()
        valid = True
    else:
        pred, valid = "", False
    f1 = max((qa_f1_score(pred, gt) for gt in row["answers"]), default=0.0) if valid else 0.0
    return {"dataset": ds, "idx": idx, "f1": f1, "pred": pred[:300],
            "answers": row["answers"], "valid": valid,
            "tries": tries, "completion_tokens": ctoks}

jobs = []
for ds in DATASETS:
    rows = [json.loads(l) for l in open(f"{args.data_dir}/{ds}.jsonl", encoding="utf-8")]
    jobs += [(ds, i, r) for i, r in enumerate(rows)]
print(f"{len(jobs)} questions", flush=True)

results = []
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for i, r in enumerate(ex.map(one, jobs)):
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(jobs)}", flush=True)

with open(args.out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import defaultdict
agg = defaultdict(list)
inval = defaultdict(int)
err = 0
for r in results:
    agg[r["dataset"]].append(r["f1"])
    if "error" in r: err += 1
    elif not r.get("valid", True): inval[r["dataset"]] += 1
allf = []
for ds in DATASETS:
    s = agg[ds]; allf += s
    print(f"{ds}: F1={100*sum(s)/len(s):.2f} (n={len(s)}, invalid={inval[ds]})", flush=True)
retried = sum(1 for r in results if r.get("tries", 1) > 1)
print(f"OVERALL: F1={100*sum(allf)/len(allf):.2f}  errors={err}  "
      f"retried={retried}", flush=True)
