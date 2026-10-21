#!/usr/bin/env python3
"""Training-data assembly for the fixed v2 recipe (2026-10-08). Run on the Linux training cluster (a Parquet file written on macOS was suspected of causing NaN at step 1).
- Takes unstable questions (both correct and incorrect within 4 trials) from batches/{worlds,repeat}_b*.jsonl|json; the prompt format
  is verbatim identical to the r2 / abl-qa runs (verified verbatim on the 1,152 abl_qa questions with --selftest).
- Split by world: iterate in (batch number, world number) order; worlds whose stable hash %10==0 go to val, the rest to train;
  val takes the first N_VAL questions and train the first N_TRAIN, so 128⊂256 and 1152⊂2304, and no world spans train/val.
- ground_truth = [answer] + aliases (aliases already cleaned to equivalent forms during v2 generation); reward is answer_line_exact_match_v2.
- Optionally concatenates the MRCR (v3, self-generated) train / val Parquet files, then writes train.parquet / val.parquet.
Usage: python build_training_data.py --batches DIR --n-train 1152 --n-val 128 --mrcr DIR/mix_subset --mrcr-train 1152 --out DIR
"""
import argparse, glob, hashlib, json, os, re
import pandas as pd

HEAD = ("Answer the question based on the given passages.\n\nReason step by step, making sure every step is backed by the passages. "
        "End your response with a final line: The answer is <answer>. Keep the answer precise and succinct.\n\n")
CH = re.compile(r"<CHUNK_(\d+)>\n(.*?)\n</CHUNK_\1>", re.S)


def make_prompt(context, question):
    body = "\n\n".join(t for _, t in CH.findall(context))
    return HEAD + "The following are the given passages.\n" + body + "\n\nQuestion: " + question


def bnum(path):
    return int(re.search(r"_b(\d+)\.", path).group(1))


def is_val(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0


def collect(batch_dir):
    rows = []
    for wf in sorted(glob.glob(f"{batch_dir}/worlds_b*.jsonl"), key=bnum):
        b = bnum(wf); rf = f"{batch_dir}/repeat_b{b}.json"
        if not os.path.exists(rf):
            continue
        rep = json.load(open(rf, encoding="utf-8"))
        worlds = sorted(map(json.loads, open(wf, encoding="utf-8")), key=lambda w: int(w["sample_id"].split("_")[1]))
        for w in worlds:
            wkey = f"b{b}_{w['sample_id']}"
            for q in w["questions"]:
                v = rep.get(f"{w['sample_id']}/{q['qid']}")
                if not v:
                    continue
                sc = [p for p in v["passes"] if p is not None]
                if not (sc and 0 < sum(sc) < len(sc)):
                    continue
                gts = [str(q["answer"])] + [str(a) for a in q.get("aliases", []) if str(a) != str(q["answer"])]
                rows.append({"wkey": wkey, "val": is_val(wkey), "row": {
                    "data_source": "custom_longcontextqa_synthesized",
                    "prompt": [{"role": "user", "content": make_prompt(w["context"], q["question"])}],
                    "ability": "longcontext_qa",
                    "reward_model": {"ground_truth": gts, "style": "rule"},
                    "extra_info": {"answer": str(q["answer"]), "domain": w.get("domain"), "evidence_chunks": q.get("evidence_chunks", []),
                                   "input_question": q["question"], "k_passes": float(sum(sc)), "k_total": float(len(sc)),
                                   "n_hops": float(q.get("n_hops") or 0), "qid": q["qid"], "sample_id": wkey,
                                   "salient_wrong_candidate": None}}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches"); ap.add_argument("--n-train", type=int); ap.add_argument("--n-val", type=int)
    ap.add_argument("--mrcr", default=""); ap.add_argument("--mrcr-train", type=int, default=0); ap.add_argument("--out")
    ap.add_argument("--selftest", default="", help="abl_qa train.parquet: rebuild its prompts from the old batches and compare verbatim")
    a = ap.parse_args()
    if a.selftest:
        d = pd.read_parquet(a.selftest); ok = bad = 0; cache = {}
        for _, r in d.iterrows():
            ei = r["extra_info"]; b, w = ei["sample_id"].split("_", 1)
            if b not in cache:
                cache[b] = {x["sample_id"]: x for x in map(json.loads, open(f"{a.batches}/worlds_{b}.jsonl", encoding="utf-8"))}
            same = make_prompt(cache[b][w]["context"], ei["input_question"]) == r["prompt"][0]["content"]
            ok += same; bad += not same
        print(f"SELFTEST prompt identical: {ok}/{ok + bad}"); return
    rows = collect(a.batches)
    val = [x["row"] for x in rows if x["val"]][: a.n_val]
    train = [x["row"] for x in rows if not x["val"]][: a.n_train]
    print(f"pool unstable={len(rows)} -> qa_train={len(train)}/{a.n_train} qa_val={len(val)}/{a.n_val}")
    assert len(train) == a.n_train and len(val) == a.n_val, "not enough unstable questions yet"
    for i, r in enumerate(train): r["extra_info"].update(split="train", index=i)
    for i, r in enumerate(val): r["extra_info"].update(split="val", index=i)
    tr, va = pd.DataFrame(train), pd.DataFrame(val)
    if a.mrcr:
        mt = pd.read_parquet(f"{a.mrcr}/train.parquet"); mv = pd.read_parquet(f"{a.mrcr}/val.parquet")
        if a.mrcr_train:
            mt = mt.iloc[: a.mrcr_train]
        # MRCR v3 ground_truth is a string while r2 training used a list; normalize to [str] (same type as r2, and avoids a list-column type clash with QA)
        for m in (mt, mv):
            m["reward_model"] = [dict(rm, ground_truth=[rm["ground_truth"]] if isinstance(rm["ground_truth"], str) else list(rm["ground_truth"]))
                                 for rm in m["reward_model"]]
        tr = pd.concat([tr, mt], ignore_index=True); va = pd.concat([va, mv], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    tr.sample(frac=1, random_state=0).reset_index(drop=True).to_parquet(f"{a.out}/train.parquet")
    va.to_parquet(f"{a.out}/val.parquet")
    print("WROTE", a.out, "train", len(tr), tr["data_source"].value_counts().to_dict(), "| val", len(va), va["data_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
