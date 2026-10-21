# -*- coding: utf-8 -*-
"""
LC-Zero data generation pipeline v0 — runs against your vLLM endpoint.

Stages per world:
  A. seed        : domain x genre x language (built-in lists)
  B. world spec  : entities + fact chains + distractor chains + QA  (strict JSON)
  C. self-verify : evidence-only answerability check (short context, cheap)
  D. expansion   : facts -> chunk plan -> per-chunk generation (parallel-safe)
  E. filters     : closed-book leak test + evidence-only test
Output: worlds.jsonl (same schema as samples.jsonl -> reuse eval_qwen_api.py)

Difficulty knobs from the probe are baked into the prompts:
  K1 de-lexicalization : key hops must use descriptive references, entities share surnames
  K2 arithmetic hops   : >=1 chain per world encodes a year/number as an offset
  K3 burial            : answer-bearing facts must sit inside same-type enumerations,
                         with a more salient wrong candidate present
  K4 provenance        : one "corrected old record" conflict per world (authority rule stated)

Usage:
  python gen_pipeline.py --n-worlds 10                     # smoke test
  python gen_pipeline.py --n-worlds 100 --workers 8        # yield measurement run
Then evaluate difficulty of the accepted worlds:
  python eval_qwen_api.py --samples worlds.jsonl
"""

import argparse, json, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- seeds -----------------------------------------
# Chinese-language ("zh") seeds and prompt text below are model inputs for --langs zh, kept verbatim;
# the reported data use --langs en only.
DOMAINS = [
    ("tech industry", "en"), ("biotech/pharma", "en"), ("maritime shipping", "en"),
    ("aerospace suppliers", "en"), ("mining & energy", "en"), ("football clubs", "en"),
    ("semiconductor equipment", "en"), ("satellite communications", "en"),
    ("rail freight operators", "en"), ("wind & tidal energy", "en"),
    ("hotel & resort chains", "en"), ("specialty chemicals", "en"),
    ("民国出版业", "zh"), ("地方铁路史", "zh"), ("近代纺织实业", "zh"),
    ("戏曲班社沿革", "zh"), ("侨批与钱庄", "zh"), ("大学院系变迁", "zh"),
    ("盐业公所与盐商", "zh"), ("电报电话局沿革", "zh"), ("内河航运公司", "zh"),
    ("教会医院院史", "zh"), ("茶庄商号谱系", "zh"), ("师范学堂变迁", "zh"),
]
NAME_FLAVORS_EN = ["Nordic", "Iberian", "Slavic", "Dutch/Flemish", "Scottish",
                   "Franco-Belgian", "Greek", "Turkish", "Hungarian", "Baltic",
                   "Irish", "Italian"]
REGION_FLAVORS_ZH = ["江浙", "闽粤", "川渝", "华北", "两湖", "山陕", "东北", "云贵"]
BANNED_NAMES = ("Voss, Carter, Smith, Vane, Vance, Hart, Lee, Elara, Marcus, James, "
                "NexTech, QuantumCore, TechNova, MarineTech, AeroCore, AeroTech, "
                "李维, 李华, 张伟, 陈伟, 徐维, 张维")
GENRES = {
    "en": ["magazine feature", "corporate history chapter", "analyst research note",
           "conference proceedings digest", "podcast interview transcript",
           "obituary", "regulatory filing summary", "trade-press news roundup"],
    "zh": ["方志摘录", "社史稿", "年谱简编", "回忆录", "行业月报纪事",
           "书信集附跋", "碑传集", "档案汇编按语"],
}

# ----------------------------- prompts ---------------------------------------
SPEC_PROMPT = """You are constructing a synthetic "world" for long-context QA training data. ALL entities must be fictional (realistic style, but names/companies/events must not correspond to real ones). World language: {lang_name}. Domain: {domain}.

Style card (mandatory):
{style_card}

Design requirements (all mandatory):
1. 8-14 named entities. At least two PEOPLE share a surname, and at least two ORGANIZATIONS have confusably similar names.
2. Create 4 QUESTION CHAINS of 3-5 hops each. A hop = one atomic fact linking two entities/attributes. Chains must partially share bridge entities.
3. Knob K2: at least one chain encodes a key year/number ONLY as an offset (e.g. "founded the firm N years after finishing the doctorate" + a separate fact stating the firm's founding year), so the answer requires arithmetic. Invent your own N and years.
4. For each chain, create 2-3 DISTRACTOR facts that share the bridge entity but differ in role/time/attribute (e.g. founder vs CEO, predecessor product, earlier residence, similarly-named org).
5. Knob K4: pick ONE fact and create a conflicting "old record" version, plus an explicit authority rule fact (e.g. a revised edition noting the old record is erroneous). The question's answer follows the authoritative version.
6. Knob K3: for each chain, the final answer-bearing attribute must have a SALIENT WRONG CANDIDATE — a same-type value that appears much more frequently in the world (mark it in the JSON).
7. Every fact gets an id F1..Fn (distractors D1..Dn). Questions reference supporting fact ids.
8. Questions must NOT name the answer entity directly when a descriptive reference is possible (Knob K1) — refer to "the founder of the magazine's parent press" rather than that person's name.
9. At most ONE question may have a bare year as its answer; prefer names, places, products, titles.

Output STRICT JSON only, no markdown fences, exactly this schema:
{{"entities": [{{"name": "...", "type": "person|org|place|product", "note": "..."}}],
 "facts": [{{"id": "F1", "text": "...", "entities": ["..."]}}],
 "distractors": [{{"id": "D1", "text": "...", "confuses_with": "F3", "type": "role|time|name|place"}}],
 "conflict": {{"old_record": "...", "authority_rule": "...", "authoritative_fact_id": "F7"}},
 "questions": [{{"qid": "Q1", "question": "...", "answer": "...", "aliases": ["..."],
   "supporting_facts": ["F1","F4","F7"], "n_hops": 4,
   "salient_wrong_candidate": "..."}}]}}"""

VERIFY_PROMPT = """Given ONLY the following facts, answer the question. If the facts are insufficient to determine the answer, reply exactly: INSUFFICIENT.

Facts:
{facts}

Question: {question}
Answer with the short answer only."""

VERIFY_ALL_PROMPT = """Given the following facts (some may be irrelevant or about similar-sounding entities), answer the question with a short answer only.

Facts:
{facts}

Question: {question}"""

PLAN_PROMPT = """You will distribute facts into {n_chunks} documents for a long-context QA task. World language: {lang_name}. Available document genres: {genres}.

Facts (must ALL be placed, each in exactly one document):
{facts}

Rules:
- Facts supporting the same question should be spread across DIFFERENT documents where possible.
- Each document gets a genre and a one-line topic.
- Knob K3: for each fact marked [ANSWER-BEARING], note that it must later be embedded inside an enumeration of same-type items in that document, and the document should mention the salient wrong candidate value more often than the true value.
- Distractor facts go in documents DIFFERENT from the fact they confuse with, where possible.

Output STRICT JSON only:
{{"chunks": [{{"chunk_id": 1, "genre": "...", "topic": "...", "fact_ids": ["F1","D2"]}}]}}"""

EXPAND_PROMPT = """Write a document in {lang_name}. Genre: {genre}. Topic: {topic}. Target length: {target_words}.

The document MUST naturally state each of the following facts exactly once, without altering any name, number, or relationship. Facts marked [PARAPHRASE] must be expressed indirectly (implied by narrative order, relative dating, or descriptive reference — never as a plain direct statement). Facts marked [BURY] must appear inside an enumeration/list of at least 4 same-type items, not highlighted in any way.

Facts to embed:
{facts}

Hard constraints:
- Do NOT mention, quote, or allude to these instructions, the bracketed markers ([BURY], [PARAPHRASE]), or your embedding strategy anywhere in the document. No meta-commentary about how facts are placed. Pure in-world prose only.
- Do NOT state any of these forbidden facts, even in passing: {forbidden}
- Do NOT summarize or foreshadow other documents.
- Match the genre's register convincingly; include realistic filler content of your own invention that does not contradict the facts.
- Output the document text only, no titles about the task, no explanations."""

# --simple recipe (2026-10-05 ablation): removes the difficulty knobs K2/K3/K4, the [BURY]/[PARAPHRASE] markers, self-verification and the closed-book check;
# keeps fictional entities/style cards, fact chains + distractor facts, descriptive references (the basic form of multi-hop questions), the faithfulness gate and the length fallback.
SPEC_PROMPT_SIMPLE = SPEC_PROMPT.replace(
    "3. Knob K2: at least one chain encodes a key year/number ONLY as an offset (e.g. \"founded the firm N years after finishing the doctorate\" + a separate fact stating the firm's founding year), so the answer requires arithmetic. Invent your own N and years.\n", ""
).replace(
    "5. Knob K4: pick ONE fact and create a conflicting \"old record\" version, plus an explicit authority rule fact (e.g. a revised edition noting the old record is erroneous). The question's answer follows the authoritative version.\n", ""
).replace(
    "6. Knob K3: for each chain, the final answer-bearing attribute must have a SALIENT WRONG CANDIDATE — a same-type value that appears much more frequently in the world (mark it in the JSON).\n", ""
).replace(
    "8. Questions must NOT name the answer entity directly when a descriptive reference is possible (Knob K1) — refer to \"the founder of the magazine's parent press\" rather than that person's name.",
    "8. In the QUESTION text, refer to intermediate/bridge entities descriptively rather than by name (e.g. \"the founder of the magazine's parent press\"). The ANSWER itself must be the concrete name, place, product, title or number exactly as it appears in a fact — NEVER a description such as \"the person who founded X\" or \"the lead engineer of Y\"."
).replace(
    ' "conflict": {{"old_record": "...", "authority_rule": "...", "authoritative_fact_id": "F7"}},\n', ""
).replace(',\n   "salient_wrong_candidate": "..."}}]}}', '}}]}}')

PLAN_PROMPT_SIMPLE = PLAN_PROMPT.replace(
    "- Knob K3: for each fact marked [ANSWER-BEARING], note that it must later be embedded inside an enumeration of same-type items in that document, and the document should mention the salient wrong candidate value more often than the true value.\n", "")

EXPAND_PROMPT_SIMPLE = EXPAND_PROMPT.replace(
    " Facts marked [PARAPHRASE] must be expressed indirectly (implied by narrative order, relative dating, or descriptive reference — never as a plain direct statement). Facts marked [BURY] must appear inside an enumeration/list of at least 4 same-type items, not highlighted in any way.", ""
).replace(", the bracketed markers ([BURY], [PARAPHRASE])", "")

ECHO_RE = re.compile(r"\[BURY\]|\[PARAPHRASE\]|same-type items|forbidden fact"
                     r"|salient wrong|Facts to embed|ANSWER-BEARING", re.I)

CLOSED_BOOK_PROMPT = """Answer the following question with a short answer. If you do not know, say "unknown".

Question: {question}"""


# ----------------------------- api helpers -----------------------------------
def post(base, payload, timeout=900, retries=2):
    req = urllib.request.Request(f"{base}/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (ConnectionResetError, urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))


THINKING_ONLY = False  # set in main(); True for Qwen3-*-Thinking-2507 models
                       # (no enable_thinking switch — every call reasons)


def chat(base, model, prompt, thinking=False, max_tokens=4096, temperature=None):
    if THINKING_ONLY:
        # model always thinks: no template kwarg; caller's max_tokens is the
        # TOTAL budget (reasoning + answer) — no hidden bump, so expand can
        # still control document length via truncation
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens,
                   "temperature": 0.7, "top_p": 0.95}
    else:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens,
                   "temperature": (0.8 if temperature is None else temperature) if not thinking else 0.6,
                   "top_p": 0.95,
                   "chat_template_kwargs": {"enable_thinking": thinking}}
    resp = post(base, payload)
    msg = resp["choices"][0]["message"]
    text = msg.get("content") or ""
    if THINKING_ONLY:
        # 2507 opens the think block implicitly: everything before the lone
        # </think> is reasoning; no </think> at all = truncated mid-thought
        if "</think>" not in text:
            raise ValueError("truncated inside thinking (no </think>)")
        return text.split("</think>")[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "<think>" in text:
        raise ValueError("truncated: thinking exhausted max_tokens")
    return text


def parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def norm(s):
    # 2026-10-07: also strip curly/back quotes (‘’`´) — text often has L’Estuaire while the gold has L'Estuaire
    return re.sub(r"[\s,，、．.。'\"“”‘’`´]+", "", str(s).lower())


# 2026-10-08 fixed v2 recipe: keep only aliases that are equivalent forms of the answer (same rules as clean/clean_stage12.py).
# In the old data, 54% of aliases were descriptive phrases restating the question ("the manager of X in 1900") that a substring reward would accept.
_NUMW = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
            "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand and".split())
_ARTS = {"the", "a", "an"}
_SUFFIX = {"ltd", "inc", "co", "corp", "corporation", "company", "plc", "s", "sa", "ag", "gmbh", "llc", "group"}
def _wnorm(t):
    t = str(t).translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})).lower()
    return re.sub(r"\s+", " ", re.sub(r"[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/]", " ", t)).strip()
def alias_ok(alias, answer):
    a, g = _wnorm(alias).split(), _wnorm(answer).split()
    if not a or not g: return False
    if set(a) <= _NUMW or a == g: return True
    core = [w for w in a if w not in _ARTS]
    if f" {' '.join(g)} " in f" {' '.join(a)} ":
        return all(w in _ARTS | _SUFFIX for w in a if w not in g)
    if f" {' '.join(core)} " in f" {' '.join(g)} ":
        return len(core) >= 2 or (len(core) == 1 and len(core[0]) >= 5 and core[0] not in _SUFFIX)
    return False


def match(pred, gold, aliases=()):
    p = norm(pred)
    for g in [gold] + list(aliases):
        parts = [norm(x) for x in re.split(r"[,，]", str(g)) if str(x).strip()]
        if parts and all(x in p for x in parts):
            return True
    return False


# ----------------------------- pipeline --------------------------------------
def build_world(idx, args):
    base = args.bases[idx % len(args.bases)]  # per-world endpoint affinity
    rng = random.Random(args.seed + idx)
    domain, lang = rng.choice([d for d in DOMAINS if d[1] in args.lang_set])
    lang_name = "English" if lang == "en" else "中文（民国至现代书面语，按体裁调整文风）"
    genres = rng.sample(GENRES[lang], k=min(args.n_chunks + 2, len(GENRES[lang])))
    log = {"idx": idx, "domain": domain, "lang": lang, "stage": "spec", "reject": None}

    # B. spec — style card injects per-world entropy (era, naming flavor,
    # ban-list) to break 4B's name/year mode collapse
    era = rng.randrange(1870, 2000, 10)
    if lang == "en":
        flavor = rng.choice(NAME_FLAVORS_EN)
        style_card = (f"- Set the story mainly in the {era}s-{era+40}s.\n"
                      f"- Person and company names should have a {flavor} flavor "
                      f"(fictional, varied, not stereotypical).\n"
                      f"- NEVER use any of these overused names or similar variants: {BANNED_NAMES}.")
    else:
        flavor = rng.choice(REGION_FLAVORS_ZH)
        style_card = (f"- 故事主要发生在{era}至{era+40}年间。\n"
                      f"- 人名、商号带{flavor}地域色彩（虚构、多样，勿用套路名）。\n"
                      f"- 严禁使用下列高频名及其近似变体：{BANNED_NAMES}。")
    spec, err = None, None
    for _ in range(2):
        try:
            sp_tmpl = SPEC_PROMPT_SIMPLE if args.simple else SPEC_PROMPT
            if args.n_chains != 4:  # 2026-10-07 ablation: more questions per world (document cost unchanged)
                sp_tmpl = sp_tmpl.replace("Create 4 QUESTION CHAINS", f"Create {args.n_chains} QUESTION CHAINS").replace(
                    "1. 8-14 named entities.", f"1. {8 + 2 * (args.n_chains - 4)}-{14 + 3 * (args.n_chains - 4)} named entities.")
            spec = parse_json(chat(base, args.model,
                                   sp_tmpl.format(lang_name=lang_name, domain=domain,
                                                      style_card=style_card),
                                   thinking=True, max_tokens=28672))
            assert spec.get("questions") and spec.get("facts")
            assert all(isinstance(f, dict) and f.get("id") and f.get("text")
                       for f in spec["facts"] + spec.get("distractors", [])), \
                "malformed fact objects"
            assert all(isinstance(q, dict) and q.get("question") and
                       q.get("answer") is not None
                       for q in spec["questions"]), "malformed question objects"
            break
        except Exception as e:
            spec, err = None, e
    if spec is None:
        log["reject"] = f"spec_parse: {err}"
        return None, log

    facts_by_id = {f["id"]: f["text"] for f in spec["facts"]}
    facts_by_id.update({d["id"]: d["text"] for d in spec.get("distractors", [])})

    # C. self-verify (evidence-only, and all-facts robustness)
    log["stage"] = "verify"
    kept_qs = []
    if args.simple:  # no self-verify: only require that all supporting fact ids exist
        kept_qs = [q for q in spec["questions"]
                   if q.get("supporting_facts") and all(facts_by_id.get(f) for f in q["supporting_facts"])]
        for q in kept_qs:  # fixed v2 recipe: keep only equivalent aliases
            q["aliases"] = [a for a in q.get("aliases", []) if alias_ok(a, q["answer"])]
    for q in ([] if args.simple else spec["questions"]):
        sup = [facts_by_id.get(fid, "") for fid in q.get("supporting_facts", [])]
        if not all(sup):
            continue
        try:
            a1 = chat(base, args.model,
                      VERIFY_PROMPT.format(facts="\n".join(f"- {s}" for s in sup),
                                           question=q["question"]),
                      thinking=True, max_tokens=12288)
            if "INSUFFICIENT" in a1 or not match(a1, q["answer"], q.get("aliases", [])):
                continue
            allf = "\n".join(f"- {t}" for t in facts_by_id.values())
            a2 = chat(base, args.model,
                      VERIFY_ALL_PROMPT.format(facts=allf, question=q["question"]),
                      thinking=True, max_tokens=12288)
            if not match(a2, q["answer"], q.get("aliases", [])):
                continue
            kept_qs.append(q)
        except Exception:
            continue
    if not kept_qs:
        log["reject"] = "verify: no question survived"
        return None, log

    # D. plan + expand
    log["stage"] = "expand"
    answer_bearing = set()
    swc = {}
    for q in kept_qs:
        if q.get("supporting_facts"):
            answer_bearing.add(q["supporting_facts"][-1])
            swc[q["supporting_facts"][-1]] = q.get("salient_wrong_candidate", "")
    fact_lines = []
    for fid, text in facts_by_id.items():
        tag = " [ANSWER-BEARING]" if (fid in answer_bearing and not args.simple) else ""
        fact_lines.append(f"{fid}: {text}{tag}")
    plan, perr = None, None
    for _ in range(2):
        try:
            plan = parse_json(chat(base, args.model,
                                   (PLAN_PROMPT_SIMPLE if args.simple else PLAN_PROMPT).format(n_chunks=args.n_chunks, lang_name=lang_name,
                                                      genres=", ".join(genres),
                                                      facts="\n".join(fact_lines)),
                                   thinking=False, max_tokens=16384))
            break
        except Exception as e:
            plan, perr = None, e
    try:
        if plan is None:
            raise perr
        chunks_plan = plan["chunks"]
        assert chunks_plan, "empty chunk plan"
        # repair: drop hallucinated/duplicate ids, then place missing facts
        # into the currently smallest chunk (4B routinely drops a few)
        placed = set()
        for c in chunks_plan:
            ids = [fid for fid in c.get("fact_ids", [])
                   if fid in facts_by_id and fid not in placed]
            c["fact_ids"] = ids
            placed.update(ids)
        for fid in facts_by_id:
            if fid not in placed:
                min(chunks_plan, key=lambda c: len(c["fact_ids"]))["fact_ids"].append(fid)
    except Exception as e:
        log["reject"] = f"plan: {e}"
        return None, log

    chunk_texts = []
    for c in chunks_plan:
        lines, forbidden = [], []
        for fid, text in facts_by_id.items():
            if fid in c.get("fact_ids", []):
                marks = []
                if fid in answer_bearing and not args.simple:
                    marks.append("[BURY]")
                    if swc.get(fid):
                        text = text + f" (salient wrong candidate to over-mention: {swc[fid]})"
                if not args.simple and random.Random(args.seed + idx + hash(fid) % 997).random() < args.paraphrase_p:
                    marks.append("[PARAPHRASE]")
                lines.append(f"- {' '.join(marks)} {text}".strip())
            else:
                forbidden.append(text)
        try:
            # total budget (reasoning + document); the length guard downstream
            # rejects worlds whose contexts still come out too long
            budget = args.expand_max_tokens
            for _attempt in range(2):
                doc = chat(base, args.model,
                           (EXPAND_PROMPT_SIMPLE if args.simple else EXPAND_PROMPT).format(lang_name=lang_name, genre=c.get("genre", genres[0]),
                                                topic=c.get("topic", ""),
                                                target_words=args.target_words,
                                                facts="\n".join(lines),
                                                forbidden=" | ".join(forbidden[:15])),
                           thinking=False, max_tokens=budget, temperature=0.8)
                # 2507 ignores word targets and fills its whole budget: hard-trim
                # at a paragraph boundary so 8 en chunks stay ~26.7k tokens total
                if lang == "en" and len(doc) > 14000:
                    cut = doc.rfind("\n\n", 0, 14000)
                    doc = doc[:cut if cut > 8000 else 14000]
                if not ECHO_RE.search(doc):
                    break
            assert not ECHO_RE.search(doc), "instruction echo in document"
            assert len(doc) > 200, "chunk too short"
            chunk_texts.append(doc)
        except Exception as e:
            log["reject"] = f"expand chunk {c.get('chunk_id')}: {e}"
            return None, log

    # E. filters
    log["stage"] = "filter"
    order = list(range(len(chunk_texts)))
    rng.shuffle(order)
    ctx = "\n\n".join(f"<CHUNK_{i+1}>\n{chunk_texts[o]}\n</CHUNK_{i+1}>"
                      for i, o in enumerate(order))
    # length guard: eval total (ctx + question + 4096 thinking) must stay
    # inside Qwen3's NATIVE 32768 window — beyond it RoPE degradation fakes
    # difficulty. (ratios measured: en ~4.2+ chars/tok, zh ~1.5)
    est_tokens = len(ctx) / (1.5 if lang == "zh" else 4.2)
    if est_tokens > 27500:
        log["reject"] = f"filter: context too long (~{est_tokens/1000:.0f}k tok)"
        return None, log
    fid_to_chunk = {}
    for pos, o in enumerate(order):
        for fid in chunks_plan[o].get("fact_ids", []):
            fid_to_chunk[fid] = f"CHUNK_{pos+1}"

    all_text = norm(" ".join(chunk_texts))
    final_qs = []
    for q in kept_qs:
        # faithfulness gate: non-year answers must appear verbatim in the
        # expanded text (expansion can silently drop/alter facts; such
        # questions are unanswerable and pollute the always-wrong band)
        if not re.fullmatch(r"(19|20)\d\d(年)?", str(q["answer"]).strip()):
            golds = [q["answer"]] + list(q.get("aliases", []))
            hit = any(all(norm(p) in all_text
                          for p in re.split(r"[,，]", str(g)) if p.strip())
                      for g in golds)
            if not hit:
                continue
        # closed-book leak check. Default (2507 recipe): two no-think calls (greedy + T0.8);
        # for a thinking-only model every call reasons anyway. --closedbook-thinking (hybrid
        # models, 2026-10-05): two thinking calls instead, matching 2507's filtering strength.
        # --closedbook-compare: run BOTH variants, record both verdicts in the log, filter by
        # the variant selected via --closedbook-thinking (to measure whether thinking matters).
        def cb_leak(thinking):
            for temp in (0.0, 0.8):  # greedy + sampled: also catches guessable answers
                cb = chat(base, args.model,
                          CLOSED_BOOK_PROMPT.format(question=q["question"]),
                          thinking=thinking, max_tokens=8192 if thinking else 6144,
                          temperature=temp)
                if match(cb, q["answer"], q.get("aliases", [])):
                    return True
            return False
        try:
            if args.simple:
                raise StopIteration  # --simple: no closed-book check
            if args.closedbook_compare:
                v_no, v_th = cb_leak(False), cb_leak(True)
                log.setdefault("cb_compare", []).append([int(v_no), int(v_th)])
                leaked = v_th if args.closedbook_thinking else v_no
            else:
                leaked = cb_leak(args.closedbook_thinking)
            if leaked:
                continue  # leakage/guessable -> drop question
        except StopIteration:
            pass
        except Exception:
            pass
        ev = sorted({fid_to_chunk[f] for f in q["supporting_facts"] if f in fid_to_chunk},
                    key=lambda t: int(t.split("_")[1]))
        final_qs.append({"qid": f"W{idx}_{q['qid']}", "question": q["question"],
                         "answer": q["answer"], "aliases": q.get("aliases", []),
                         "n_hops": q.get("n_hops", len(q["supporting_facts"])),
                         "evidence_chunks": ev})
    if not final_qs:
        log["reject"] = "filter: all questions leaked/failed"
        return None, log

    log["stage"] = "done"
    return {"sample_id": f"world_{idx}", "domain": f"{domain} ({lang})",
            "context": ctx, "questions": final_qs,
            "_spec": spec, "_plan": chunks_plan}, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-worlds", type=int, default=10)
    ap.add_argument("--n-chunks", type=int, default=6)
    ap.add_argument("--target-words", default="900-1200 words (中文则2000-2600字)")
    ap.add_argument("--expand-max-tokens", type=int, default=3500)
    ap.add_argument("--langs", default="en,zh",
                    help="comma list of world languages to generate (en, zh)")
    ap.add_argument("--paraphrase-p", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="worlds.jsonl")
    ap.add_argument("--closedbook-thinking", action="store_true",
                    help="hybrid models: run the closed-book leak check with thinking on")
    ap.add_argument("--n-chains", type=int, default=4, help="question chains (= questions) per world in the spec")
    ap.add_argument("--simple", action="store_true",
                    help="simplified recipe: no difficulty knobs K2-K4/[BURY]/[PARAPHRASE], no self-verify, no closed-book check")
    ap.add_argument("--force-thinking-only", action="store_true",
                    help="hybrid models: treat as thinking-only (every call reasons, max_tokens = total budget), "
                         "i.e. exactly the 2507 production regime")
    ap.add_argument("--closedbook-compare", action="store_true",
                    help="run closed-book check both with and without thinking; log both verdicts")
    args = ap.parse_args()

    args.bases = [b.strip() for b in args.base.split(",") if b.strip()]
    args.lang_set = {l.strip() for l in args.langs.split(",") if l.strip()}
    if args.model is None:
        with urllib.request.urlopen(f"{args.bases[0]}/models", timeout=30) as r:
            args.model = json.loads(r.read())["data"][0]["id"]
    global THINKING_ONLY
    THINKING_ONLY = "thinking" in args.model.lower() or args.force_thinking_only
    if THINKING_ONLY:
        print("thinking-only model detected: no-thinking call sites get +4096 budget")
    print(f"model: {args.model}, generating {args.n_worlds} worlds "
          f"x {args.n_chunks} chunks, {args.workers} workers")

    t0, ok, logs = time.time(), 0, []
    with open(args.out, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_world, i, args): i for i in range(args.n_worlds)}
        for fut in as_completed(futs):
            try:
                world, log = fut.result()
            except Exception as e:
                world, log = None, {"idx": futs[fut], "domain": "?", "lang": "?",
                                    "stage": "crash", "reject": f"unhandled: {e}"}
            logs.append(log)
            if world:
                ok += 1
                fout.write(json.dumps(world, ensure_ascii=False) + "\n")
                fout.flush()
                nq = len(world["questions"])
                print(f"[{ok}] world_{log['idx']} OK ({log['domain']}, {nq} qs, "
                      f"{len(world['context'])} chars)")
            else:
                print(f"[-] world_{log['idx']} REJECTED at {log['stage']}: {log['reject']}")

    with open("gen_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"\nyield: {ok}/{args.n_worlds} worlds in {(time.time()-t0)/60:.1f} min")
    print(f"next: python eval_qwen_api.py --samples {args.out}   "
          f"(target: open-book accuracy in 30-70%)")


if __name__ == "__main__":
    main()
