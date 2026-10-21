# -*- coding: utf-8 -*-
"""
Per-question consistency probe: run each question K times (thinking mode,
temperature sampling) and record pass counts.

Answers the key RLVR question: is the ~50% aggregate pass rate
  (a) bimodal  — half the questions always solved, half never  -> no GRPO signal
  (b) stochastic — most questions sometimes right, sometimes wrong -> real signal

Also enables a thinking-mode sieve: keep questions by per-question pass@K band.

Usage:
  python repeat_eval.py --samples worlds_scale_mixed.jsonl --n-questions 100 \
      --k 8 --workers 8 --out repeat_results.json
"""

import argparse, json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OPEN_BOOK_TMPL = """You are given several documents, each wrapped in <CHUNK_i> tags. Answer the question using ONLY information from the documents.

{context}

Question: {question}

First identify which chunks contain the evidence needed, then give a short final answer. Respond in exactly this format:
<useful_chunks> CHUNK_i, CHUNK_j, ... </useful_chunks>
<answer> your short answer </answer>"""


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def norm(s):
    return re.sub(r"[\s,，、．.。'\"“”]+", "", s.lower())


MATCH_V2 = False  # enabled by --match v2: same criterion as the verl reward answer_line_exact_match_v2


def _wnorm(t):
    t = str(t).translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})).lower()
    return re.sub(r"\s+", " ", re.sub(r"[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/]", " ", t)).strip()


def answer_correct(pred, gold, aliases):
    if MATCH_V2:  # 2026-10-08: whole-word matching + curly-quote normalization ("15"≠"5", "mid-1950s"≠"1950", O’Dwyer=O'Dwyer)
        p = f" {_wnorm(pred)} "
        return any(_wnorm(g) and f" {_wnorm(g)} " in p for g in [gold] + list(aliases))
    p = norm(pred)
    for g in [gold] + list(aliases):
        parts = [norm(x) for x in re.split(r"[,，]", g) if x.strip()]
        if parts and all(part in p for part in parts):
            return True
    return False


def extract(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--match", choices=["v1", "v2"], default="v1",
                    help="v2 = same criterion as the reward answer_line_exact_match_v2 (whole word + curly quotes); used by the fixed v2 recipe")
    ap.add_argument("--k1", type=int, default=0,
                    help="adaptive cap: run in waves of 2; a question stops at its "
                         "FIRST mixed result (kept as unstable); still-uniform "
                         "questions stop after k1 runs (dropped as stable)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="repeat_results.json")
    args = ap.parse_args()
    global MATCH_V2
    MATCH_V2 = args.match == "v2"

    bases = [b.strip() for b in args.base.split(",") if b.strip()]
    with urllib.request.urlopen(f"{bases[0]}/models", timeout=30) as r:
        model = args.model or json.loads(r.read())["data"][0]["id"]
    think_only = "thinking" in model.lower()
    print(f"model: {model}, k={args.k}, thinking_only={think_only}")

    pool = []
    for line in open(args.samples, encoding="utf-8"):
        s = json.loads(line)
        for q in s["questions"]:
            pool.append((s["sample_id"], q, s["context"]))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pool = pool[:args.n_questions]
    print(f"selected {len(pool)} questions from {args.samples}")

    import itertools, threading
    _rr = itertools.cycle(bases)
    _lock = threading.Lock()

    def run_once(item):
        sid, q, ctx = item
        with _lock:
            base = next(_rr)
        payload = {"model": model,
                   "messages": [{"role": "user", "content":
                                 OPEN_BOOK_TMPL.format(context=ctx,
                                                       question=q["question"])}],
                   "max_tokens": args.max_tokens,
                   "temperature": 0.7, "top_p": 0.95}
        resp = post(f"{base}/chat/completions", payload)
        text = resp["choices"][0]["message"].get("content") or ""
        if think_only:
            if "</think>" not in text:  # truncated mid-thought: no valid answer
                return False, "[TRUNCATED-THINKING]"
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        pred = extract("answer", text)
        return answer_correct(pred, q["answer"], q.get("aliases", [])), pred

    rows = {}

    def run_many(items, k_runs):
        tasks = [(sid, q, ctx, rep) for sid, q, ctx in items for rep in range(k_runs)]
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_once, (sid, q, ctx)): (sid, q, rep)
                    for sid, q, ctx, rep in tasks}
            for fut in as_completed(futs):
                sid, q, rep = futs[fut]
                key = f"{sid}/{q['qid']}"
                rows.setdefault(key, {"question": q["question"], "answer": q["answer"],
                                      "passes": [], "preds": []})
                try:
                    ok, pred = fut.result()
                except Exception as e:
                    ok, pred = None, f"REQUEST FAILED: {e}"
                rows[key]["passes"].append(ok)
                rows[key]["preds"].append(pred[:120])
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{len(tasks)} calls done")

    # first-disagreement stopping: selection only needs "mixed vs uniform",
    # so a question stops the moment it shows both a pass and a fail
    cap = args.k1 if 0 < args.k1 <= args.k else args.k
    active, done_runs = list(pool), 0
    while active and done_runs < cap:
        step = min(2, cap - done_runs)
        run_many(active, step)
        done_runs += step
        still = []
        for sid, q, ctx in active:
            sc = [p for p in rows.get(f"{sid}/{q['qid']}", {}).get("passes", [])
                  if p is not None]
            if sc and 0 < sum(sc) < len(sc):
                continue  # mixed -> unstable, no more runs needed
            still.append((sid, q, ctx))
        print(f"wave done ({done_runs}/{cap} runs): "
              f"{len(active) - len(still)} newly mixed, {len(still)} still uniform")
        active = still
    k1 = cap  # for the summary banner

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    bands = {"always-right": 0, "always-wrong": 0, "unstable": 0}
    tot_pass = tot_runs = 0
    for v in rows.values():
        sc = [p for p in v["passes"] if p is not None]
        if not sc:
            continue
        tot_pass += sum(sc); tot_runs += len(sc)
        if all(sc):
            bands["always-right"] += 1
        elif not any(sc):
            bands["always-wrong"] += 1
        else:
            bands["unstable"] += 1
    n = sum(bands.values())
    print(f"\n=== bands over {n} questions (k1={k1}, k={args.k}) ===")
    for b, c in bands.items():
        print(f"{b}: {c} ({c/n:.0%})")
    print(f"aggregate pass rate: {tot_pass/tot_runs:.0%}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
