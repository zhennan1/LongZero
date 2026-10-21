#!/usr/bin/env python3
"""LongReason (lz1bytedance/LongReason, 794 five-choice questions, 32k split by default) evaluation.

Protocol: the dataset's own prompt field is used verbatim as the user message (official template; its last line asks for
"The answer is" + option). The official setup uses greedy decoding; Qwen3 thinking models are not decoded greedily (the
model card explicitly warns against it), so, as for the other benchmarks, temperature 0.7 / top_p 0.95 / max_tokens 16384
are used, and without </think> we resample at most 2 more times.
Extraction: in the text after </think>, take the first A-E letter after the last "the answer is"
(the official code takes the character immediately after it; here spaces, brackets, asterisks and similar decoration are
skipped, which is more lenient but identical for all models).
Each question is sampled --k times (default 2).
"""
import argparse, glob, json, os, re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from openai import OpenAI
import os as _os
# NO_THINK=1: the evaluated model does not think (instruct); the whole output is the prediction, no resampling
NO_THINK = _os.getenv("NO_THINK") == "1"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base_url", required=True)
ap.add_argument("--data_dir", required=True, help="LongReason snapshot directory (contains data/*.parquet)")
ap.add_argument("--split", default="32k")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

parts = sorted(glob.glob(os.path.join(args.data_dir, "data", f"{args.split}-*.parquet")))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
rows = df.to_dict("records")
if args.limit:
    rows = rows[:args.limit]

PAT = re.compile(r"the answer is[\s:：\*\(\[\"'“]*([A-E])(?![A-Za-z])", re.I)


def extract(pred):
    m = PAT.findall(pred)
    return m[-1].upper() if m else ""


def one(job):
    i, r, s = job
    rec = {"idx": i, "example_idx": r["example_idx"], "sample": s, "answer": r["answer"]}
    txt, tries, ctoks, fin = "", 0, 0, ""
    for attempt in range(3):
        tries = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": r["prompt"]}],
                temperature=0.7, top_p=0.95, max_tokens=16384)
            txt = resp.choices[0].message.content or ""
            ctoks = getattr(resp.usage, "completion_tokens", 0)
            fin = resp.choices[0].finish_reason
        except Exception as e:
            rec.update(error=repr(e), valid=False, choice="", acc=0.0, tries=tries)
            return rec
        if NO_THINK or "</think>" in txt:
            break
    valid = NO_THINK or "</think>" in txt
    pred = (txt.strip() if NO_THINK else txt.rsplit("</think>", 1)[1].strip()) if valid else ""
    choice = extract(pred)
    rec.update(tries=tries, completion_tokens=ctoks, finish_reason=fin, valid=valid,
               pred_tail=pred[-300:], choice=choice, acc=float(valid and choice == r["answer"]))
    return rec


client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
jobs = [(i, r, s) for s in range(args.k) for i, r in enumerate(rows)]
print(f"LongReason-{args.split}: {len(rows)} questions × k={args.k} = {len(jobs)} generations", flush=True)
results = []
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for n, rec in enumerate(ex.map(one, jobs)):
        results.append(rec)
        if (n + 1) % 100 == 0:
            print(f"{n+1}/{len(jobs)}", flush=True)

with open(args.out, "w") as f:
    for rec in results:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
per_k = [100 * sum(r["acc"] for r in results if r["sample"] == s) / len(rows) for s in range(args.k)]
print(f"LongReason-{args.split} acc={sum(per_k)/len(per_k):.2f} per_sample={[round(x, 2) for x in per_k]} "
      f"n={len(results)} invalid={sum(1 for r in results if not r.get('valid'))} "
      f"no_choice={sum(1 for r in results if r.get('valid') and not r.get('choice'))} "
      f"errors={sum(1 for r in results if 'error' in r)}", flush=True)
