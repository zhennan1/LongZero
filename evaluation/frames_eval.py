#!/usr/bin/env python3
"""FRAMES (google/frames-benchmark, 824 questions) evaluation in the oracle setting.

Protocol: following the official oracle setting, all gold Wikipedia articles of a question are placed in the prompt
together with the question; the model reasons freely and gives a final answer. The original paper scores with a Gemini
autorater; instead of an LLM judge we use automatic scoring with two levels, normalized exact match and bidirectional
containment (the same criterion for every evaluated model, so paired comparisons are valid; absolute values are underestimated).

Thinking-model settings: temperature 0.7 / top_p 0.95 (greedy decoding is explicitly discouraged for Qwen3 thinking models:
thinking loops without converging, which caused 9-21% invalid outputs in a first greedy run), max_tokens 16384; the text
after the last </think> is the prediction; without </think> resample at most 2 more times, otherwise count as invalid
(resample count and completion tokens are recorded in the output). When the context exceeds --ctx_budget_chars
(default 980,000 characters, about 245k tokens at 4 characters/token), the longest articles are truncated first; the
truncation is recorded in the output.
"""
import argparse, json, sys, urllib.parse, re, string
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os as _os
# NO_THINK=1: the evaluated model does not think (instruct); the whole output is the prediction, no resampling
NO_THINK = _os.getenv("NO_THINK") == "1"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base_url", required=True)
ap.add_argument("--data", default="assets/lczero/frames/frames_clean.jsonl")
ap.add_argument("--wiki", default="assets/lczero/frames/frames_wiki_text.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--ctx_budget_chars", type=int, default=980_000)  # ~245k tokens
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

wiki = {}
for l in open(args.wiki, encoding="utf-8"):
    d = json.loads(l)
    wiki[d["url"]] = d

rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
if args.limit:
    rows = rows[:args.limit]

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def build_prompt(r):
    links = r["links"]
    arts, missing = [], []
    for u in links:
        d = wiki.get(u)
        if d and not d.get("missing") and d["text"]:
            arts.append((d["title"], d["text"]))
        else:
            missing.append(u)
    total = sum(len(t) for _, t in arts)
    truncated = False
    while total > args.ctx_budget_chars and arts:
        # truncate the longest article first, each time down to the share the remaining budget allows
        arts.sort(key=lambda x: -len(x[1]))
        title, txt = arts[0]
        cut = max(len(txt) // 2, len(txt) - (total - args.ctx_budget_chars))
        arts[0] = (title, txt[:cut])
        total = sum(len(t) for _, t in arts)
        truncated = True
    body = "\n\n".join(f"# Article: {t}\n{x}" for t, x in arts)
    prompt = (f"You are given several Wikipedia articles. Answer the question "
              f"based on the information in these articles.\n\n{body}\n\n"
              f"Question: {r['prompt']}\n"
              f"Give your final answer concisely.")
    return prompt, missing, truncated

def one(job):
    i, r = job
    prompt, missing, truncated = build_prompt(r)
    rec = {"idx": i, "answer": r["answer"], "reasoning_types": r["reasoning_types"],
           "missing_articles": len(missing), "truncated": truncated}
    txt, tries, ctoks, fin = "", 0, 0, ""
    for attempt in range(3):  # no </think> (thinking did not converge): resample at most 2 more times
        tries = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.7, top_p=0.95, max_tokens=16384)
            txt = resp.choices[0].message.content or ""
            ctoks = getattr(resp.usage, "completion_tokens", 0)
            fin = resp.choices[0].finish_reason
        except Exception as e:
            rec.update(error=repr(e), acc_em=0.0, acc_contain=0.0, valid=False,
                       tries=tries)
            return rec
        if NO_THINK or "</think>" in txt:
            break
    rec.update(tries=tries, completion_tokens=ctoks, finish_reason=fin)
    if NO_THINK:
        pred, valid = txt.strip(), True
    elif "</think>" in txt:
        pred = txt.rsplit("</think>", 1)[1].strip()
        valid = True
    else:
        pred, valid = "", False
    np_, na = norm(pred), norm(r["answer"])
    rec.update(pred=pred[:400], valid=valid,
               acc_em=float(valid and np_ == na),
               acc_contain=float(valid and (na in np_ or (np_ != "" and np_ in na))))
    return rec

client = OpenAI(base_url=args.base_url, api_key="EMPTY")
jobs = list(enumerate(rows))
print(f"{len(jobs)} questions", flush=True)
results = []
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for k, rec in enumerate(ex.map(one, jobs)):
        results.append(rec)
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{len(jobs)}", flush=True)

with open(args.out, "w") as f:
    for rec in results:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

n = len(results)
em = sum(r["acc_em"] for r in results) / n
ct = sum(r["acc_contain"] for r in results) / n
inval = sum(1 for r in results if not r.get("valid", False))
err = sum(1 for r in results if "error" in r)
trunc = sum(1 for r in results if r["truncated"])
miss = sum(1 for r in results if r["missing_articles"])
retried = sum(1 for r in results if r.get("tries", 1) > 1)
print(f"EM={100*em:.2f} contain={100*ct:.2f} n={n} invalid={inval} "
      f"errors={err} retried={retried} truncated={trunc} "
      f"rows_with_missing_articles={miss}", flush=True)
