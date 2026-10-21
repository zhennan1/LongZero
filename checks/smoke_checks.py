import ast,os,pathlib,runpy
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['REWARD_CALC_TYPE']='answer_line_exact_match_v2'
os.environ['ANSWER_OVER_FLOW_LIMIT']='128'
for size in ['4b','8b']:
 b=ROOT/'training'/('verl_'+size)/'verl/utils/reward_score'
 qa=runpy.run_path(str(b/'longcontext_qa.py'))['compute_score']
 assert qa('Reasoning. The answer is Ingrid Sohl.', ['Ingrid Sohl'])==1
 assert qa('The answer is 15.', ['5'])==0
 assert qa('The answer is Baltic Institute.', ['Baltic Institute','Baltic Inst.'])==1
 assert qa('The answer is a wrong value.', ['Ingrid Sohl'])==0
 mrcr=runpy.run_path(str(b/'mrcr.py'))['compute_score']
 gt='abc123A poem about a tiger';info={'random_string_to_prepend':'abc123'}
 assert mrcr('<think>reasoning</think>'+gt,gt,info)==1
 assert mrcr('wrongprefixA poem about a tiger',gt,info)==0
for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
print('Reward checks and Python syntax checks passed')
