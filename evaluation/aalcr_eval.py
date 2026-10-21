#!/usr/bin/env python3
"""AA-LCR (ArtificialAnalysis/AA-LCR v1.1, 100 questions, 72k-115k input tokens each): generation side.

Protocol: the official prompt template verbatim (BEGIN INPUT DOCUMENTS ... START QUESTION ...); documents are
concatenated in the order of data_source_filenames in the CSV. Thinking-model settings match frames_eval.py:
temperature 0.7 / top_p 0.95 / max_tokens 16384; the text after the last </think> is the prediction; without </think>
resample at most 2 more times, otherwise mark invalid. Each question is sampled --k times (default 2), one output row per sample.
Judging is done separately by aa_judge_api.py (official v1.1 equality-checker prompt).

File names inside the zip are UTF-8 stored as cp437 mojibake (e.g. ΓÇö = —) and some use NFD-decomposed characters;
they are restored and NFC-normalized before matching.
"""
import argparse, json, os, unicodedata, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from openai import OpenAI
import os as _os
# NO_THINK=1: the evaluated model does not think (instruct); the whole output is the prediction, no resampling
NO_THINK = _os.getenv("NO_THINK") == "1"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base_url", required=True)
ap.add_argument("--data_dir", required=True, help="AA-LCR snapshot directory (contains AA-LCR_Dataset.csv)")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()


def fix_name(s):
    try:
        s = s.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize("NFC", s)


df = pd.read_csv(os.path.join(args.data_dir, "AA-LCR_Dataset.csv"))
zf = zipfile.ZipFile(os.path.join(args.data_dir, "extracted_text", "AA-LCR_extracted-text.zip"))
files = {fix_name(n): n for n in zf.namelist() if not n.endswith("/")}


def build_prompt(r):
    docs = []
    for f in r["data_source_filenames"].split(";"):
        key = unicodedata.normalize("NFC", f"lcr/{r['document_category']}/{r['document_set_id']}/{f}")
        docs.append(zf.read(files[key]).decode("utf-8", errors="replace"))
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
    return (f"BEGIN INPUT DOCUMENTS\n\n{documents_text}\n\nEND INPUT DOCUMENTS\n\n"
            f"Answer the following question using the input documents provided above.\n\n"
            f"START QUESTION\n\n{r['question']}\n\nEND QUESTION\n")


rows = [r for _, r in df.iterrows()]
if args.limit:
    rows = rows[:args.limit]
prompts = {int(r["question_id"]): build_prompt(r) for r in rows}  # build all prompts first so a missing file fails immediately


def one(job):
    r, s = job
    qid = int(r["question_id"])
    rec = {"question_id": qid, "sample": s, "category": r["document_category"],
           "question": r["question"], "answer": r["answer"], "input_tokens": int(r["input_tokens"])}
    txt, tries, ctoks, fin = "", 0, 0, ""
    for attempt in range(3):
        tries = attempt + 1
        try:
            resp = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompts[qid]}],
                temperature=0.7, top_p=0.95, max_tokens=16384)
            txt = resp.choices[0].message.content or ""
            ctoks = getattr(resp.usage, "completion_tokens", 0)
            fin = resp.choices[0].finish_reason
        except Exception as e:
            rec.update(error=repr(e), valid=False, tries=tries, pred="")
            return rec
        if NO_THINK or "</think>" in txt:
            break
    valid = NO_THINK or "</think>" in txt
    rec.update(tries=tries, completion_tokens=ctoks, finish_reason=fin, valid=valid,
               pred=(txt.strip() if NO_THINK else txt.rsplit("</think>", 1)[1].strip()) if valid else "")
    return rec


client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
jobs = [(r, s) for s in range(args.k) for r in rows]
print(f"{len(rows)} questions × k={args.k} = {len(jobs)} generations", flush=True)
results = []
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for i, rec in enumerate(ex.map(one, jobs)):
        results.append(rec)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(jobs)}", flush=True)

with open(args.out, "w") as f:
    for rec in results:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
inval = sum(1 for r in results if not r.get("valid"))
err = sum(1 for r in results if "error" in r)
print(f"GEN_DONE n={len(results)} invalid={inval} errors={err} "
      f"retried={sum(1 for r in results if r.get('tries', 1) > 1)}", flush=True)
