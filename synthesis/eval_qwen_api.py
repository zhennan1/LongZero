# -*- coding: utf-8 -*-
"""
LC-Zero pilot difficulty probe — API version.
Hits an OpenAI-compatible endpoint (e.g. vLLM serve) instead of loading the model.

Usage (with your ssh -N -L 8000:127.0.0.1:8000 tunnel up):
  python eval_qwen_api.py                          # auto-detect model, thinking on
  python eval_qwen_api.py --no-thinking
  python eval_qwen_api.py --base http://localhost:8000/v1 --model Qwen/Qwen3-4B

Only stdlib is required (urllib). 28 requests total (14 questions x open/closed).
"""

import argparse, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OPEN_BOOK_TMPL = """You are given several documents, each wrapped in <CHUNK_i> tags. Answer the question using ONLY information from the documents.

{context}

Question: {question}

First identify which chunks contain the evidence needed, then give a short final answer. Respond in exactly this format:
<useful_chunks> CHUNK_i, CHUNK_j, ... </useful_chunks>
<answer> your short answer </answer>"""

CLOSED_BOOK_TMPL = """Answer the following question with a short answer. If you do not know, say "unknown".

Question: {question}
<answer> your short answer </answer>"""


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def norm(s):
    return re.sub(r"[\s,，、．.。'\"“”]+", "", s.lower())


def answer_correct(pred, gold, aliases):
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


def evidence_recall(pred_str, gold_chunks):
    pred = set(re.findall(r"CHUNK[_\s]?(\d+)", pred_str))
    gold = set(c.split("_")[1] for c in gold_chunks)
    return (len(pred & gold) / len(gold)) if gold else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="default: first model from /v1/models")
    ap.add_argument("--samples", default="samples.jsonl")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    thinking = not args.no_thinking

    bases = [b.strip() for b in args.base.split(",") if b.strip()]
    try:
        models = get(f"{bases[0]}/models")
    except Exception as e:
        sys.exit(f"Cannot reach {args.base} ({e}). Is the SSH tunnel up?")
    model = args.model or models["data"][0]["id"]
    think_only = "thinking" in model.lower()
    print(f"endpoint OK, using model: {model}, thinking={thinking}, thinking_only={think_only}")

    tasks = []
    for line in open(args.samples, encoding="utf-8"):
        s = json.loads(line)
        for q in s["questions"]:
            tasks.append(("open", s["sample_id"], q,
                          OPEN_BOOK_TMPL.format(context=s["context"], question=q["question"])))
            tasks.append(("closed", s["sample_id"], q,
                          CLOSED_BOOK_TMPL.format(question=q["question"])))

    import itertools, threading
    _rr = itertools.cycle(bases)
    _lock = threading.Lock()

    def run_task(task):
        mode, sid, q, prompt = task
        with _lock:
            base = next(_rr)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        resp = post(f"{base}/chat/completions", payload)
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        if think_only:
            # implicit-open reasoning: no </think> = truncated, no valid answer
            text = text.split("</think>")[-1] if "</think>" in text else ""
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return text, resp.get("usage", {})

    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            mode, sid, q, _ = futs[fut]
            t0 = time.time()
            try:
                text, usage = fut.result()
            except Exception as e:
                print(f"[{i}/{len(tasks)}] {sid}/{q['qid']} {mode} REQUEST FAILED: {e}")
                continue
            key = (sid, q["qid"])
            rows.setdefault(key, {"q": q})
            pred = extract("answer", text)
            if mode == "open":
                rows[key]["open_pred"] = pred
                rows[key]["open_ok"] = answer_correct(pred, q["answer"], q["aliases"])
                rows[key]["ev_recall"] = evidence_recall(extract("useful_chunks", text),
                                                         q["evidence_chunks"])
                rows[key]["prompt_tokens"] = usage.get("prompt_tokens")
            else:
                rows[key]["closed_pred"] = pred
                rows[key]["leak"] = answer_correct(pred, q["answer"], q["aliases"])
            print(f"[{i}/{len(tasks)}] {sid}/{q['qid']} {mode} done "
                  f"(prompt={usage.get('prompt_tokens')} tok)")

    print("\n=== RESULTS ===")
    n_ok = n_leak = 0
    for (sid, qid), r in sorted(rows.items()):
        q = r["q"]
        ok, leak = r.get("open_ok", False), r.get("leak", False)
        n_ok += ok; n_leak += leak
        print(f"{sid}/{qid} [{q['n_hops']} hops, ctx={r.get('prompt_tokens')} tok]"
              f"  open={'PASS' if ok else 'FAIL'}"
              f"  ev_recall={r.get('ev_recall', 0):.2f}"
              f"  closed_leak={'YES' if leak else 'no'}")
        print(f"    gold  : {q['answer']}")
        print(f"    open  : {r.get('open_pred','')[:150]}")
        print(f"    closed: {r.get('closed_pred','')[:80]}")
    total = len(rows)
    print(f"\nopen-book accuracy: {n_ok}/{total}   leakage: {n_leak}/{total}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({f"{sid}/{qid}": {k: v for k, v in r.items() if k != "q"}
                   for (sid, qid), r in rows.items()}, f, ensure_ascii=False, indent=2)
    print(f"raw results saved to {args.out}")


if __name__ == "__main__":
    main()
