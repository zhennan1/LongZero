import csv,collections,pathlib
root=pathlib.Path(__file__).parent
for name in ['main','ablation']:
 d=root/name;groups=collections.defaultdict(list)
 for r in csv.DictReader((d/'details.csv').open()):groups[(r['model'],r['benchmark'],int(r['run']))].append(float(r['score']))
 for r in csv.DictReader((d/'summary.csv').open()):
  values=[sum(groups[(r['model'],r['benchmark'],i)])/len(groups[(r['model'],r['benchmark'],i)]) for i in [1,2]]
  assert abs(sum(values)/2-float(r['reference_mean']))<1e-8
 print(name.capitalize(),'results: all summary scores match per-item scores')
