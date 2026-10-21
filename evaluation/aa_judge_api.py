"""AA-LCR equality judge using environment-based API credentials."""
import os
from openai import OpenAI
import argparse, ast, concurrent.futures, fcntl, hashlib, json, pathlib, time


def assignments(path, names):
    tree = ast.parse(pathlib.Path(path).read_text())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and names == {'client'}:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            body.append(node)
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'selected configuration', 'exec'), ns)
    return {name: ns[name] for name in names}


def validate(rows):
    if len(rows) != 200:
        raise ValueError('AA input must contain exactly 200 rows')
    keys = {(r['question_id'], r['sample']) for r in rows}
    rounds = {r['sample'] for r in rows}
    if len(keys) != 200 or len(rounds) != 2:
        raise ValueError('AA needs two unique 100-question rounds')
    ids = [{r['question_id'] for r in rows if r['sample'] == s} for s in sorted(rounds)]
    if len(ids[0]) != 100 or ids[0] != ids[1]:
        raise ValueError('AA question identities differ between rounds')
    for row in rows:
        for field in ('question', 'answer', 'pred', 'valid'):
            if field not in row:
                raise ValueError('Missing required AA field: ' + field)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--prompts', default=str(pathlib.Path(__file__).with_name('aa_judge_prompt.py')))
    ap.add_argument('--workers', type=int, default=25)
    args = ap.parse_args()
    path = pathlib.Path(args.input)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    validate(rows)
    source_sha = hashlib.sha256(raw).hexdigest()
    out = path.with_name(path.stem + '_judged_gpt56luna.jsonl')
    receipt = out.with_suffix('.receipt.json')
    lock = out.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['input_sha256'] != source_sha:
            raise ValueError('Input changed after judging')
        if hashlib.sha256(out.read_bytes()).hexdigest() == old['output_sha256']:
            print('ALREADY_COMPLETE', path.name, flush=True)
            return
        raise ValueError('Existing completed output SHA mismatch')
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ.get('OPENAI_BASE_URL')).with_options(timeout=120, max_retries=0)
    prompts = assignments(args.prompts, {'SYSTEM', 'USER'})
    prompt_sha = hashlib.sha256((prompts['SYSTEM'] + prompts['USER']).encode()).hexdigest()
    cache = path.parent / (path.stem + '_gpt56luna_cache')
    cache.mkdir(exist_ok=True)

    def judge(row):
        key = hashlib.sha256((source_sha + prompt_sha + json.dumps(row, sort_keys=True)).encode()).hexdigest()
        checkpoint = cache / (key + '.json')
        if checkpoint.exists():
            return json.loads(checkpoint.read_text())
        result = dict(row)
        result.update(judge_model='GPT-5.6-Luna', judge_reasoning_effort='medium', judge_prompt_sha256=prompt_sha)
        if not row['valid']:
            result.update(score=0.0, judge_raw='invalid generation')
        else:
            message = prompts['USER'].format(question=row['question'], official_answer=row['answer'], candidate_answer=row['pred'])
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model='GPT-5.6-Luna', reasoning_effort='medium', max_completion_tokens=4096,
                        messages=[{'role': 'system', 'content': prompts['SYSTEM']}, {'role': 'user', 'content': message}],
                    )
                    if completion.choices[0].finish_reason != 'stop':
                        raise ValueError('Judge completion unfinished')
                    text = completion.choices[0].message.content or ''
                    parsed = json.loads(text)
                    verdict = parsed.get('verdict')
                    if verdict not in ('CORRECT', 'INCORRECT'):
                        raise ValueError('Judge verdict not recognized')
                    result.update(score=float(verdict == 'CORRECT'), judge_raw=text, judge_usage=completion.usage.model_dump() if completion.usage else None)
                    break
                except Exception as exc:
                    if attempt == 2:
                        # An API failure is never scored as an incorrect model answer.
                        raise RuntimeError('API judge failed: ' + type(exc).__name__) from None
                    time.sleep(2 ** (attempt + 1))
        tmp = checkpoint.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False))
        tmp.replace(checkpoint)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(judge, rows))
    tmp = out.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    tmp.replace(out)
    summary = {'input_sha256': source_sha, 'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
               'judge': 'GPT-5.6-Luna', 'reasoning_effort': 'medium', 'prompt_sha256': prompt_sha,
               'rows': 200, 'k': 2, 'n_per_round': 100, 'score': 100 * sum(r['score'] for r in results) / 200,
               'invalid': sum(not r['valid'] for r in results), 'original_answers_preserved': True}
    receipt.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
