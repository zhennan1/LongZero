import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--input-tokens',type=int,required=True);a=p.parse_args()
d=pathlib.Path(__file__).parent/'longbench_v2/config';d.mkdir(exist_ok=True)
(d/'model2path.json').write_text(json.dumps({'longzero':a.model_path}))
(d/'model2maxlen.json').write_text(json.dumps({'longzero':a.input_tokens}))
