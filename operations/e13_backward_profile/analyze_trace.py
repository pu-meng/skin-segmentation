import json,collections,bisect
from pathlib import Path
P=Path('/home/PuMengYu/experiments/skin_segmentation/operations/e13_backward_profile')
events=json.loads((P/'trace.json').read_text())['traceEvents']
# Recover nesting independently per CPU thread; autograd executes on another thread.
cpu=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('cpu_op','user_annotation','cuda_runtime','cuda_driver')]
threads=collections.defaultdict(list)
for e in cpu:threads[e['tid']].append(e)
seqmod={};ext={};parent={}
for tid,ev in threads.items():
 stack=[]
 for e in sorted(ev,key=lambda e:(e['ts'],-e['dur'])):
  while stack and stack[-1]['ts']+stack[-1]['dur']<e['ts']+e['dur']:stack.pop()
  eid=id(e)
  if e.get('cat') in ('cuda_runtime','cuda_driver'):ext[e.get('args',{}).get('correlation')]=e
  parent[eid]=list(stack)
  seq=e.get('args',{}).get('Sequence number')
  mods=[a['name'][8:] for a in stack if a['name'].startswith('MODULE::')]
  if seq is not None and mods:seqmod[seq]=mods[-1]
  stack.append(e)
bw=[e for e in cpu if e['name']=='BACKWARD']
ops=collections.defaultdict(lambda:[0.,0]);mods=collections.defaultdict(float);cats=collections.defaultdict(float);kernels=collections.defaultdict(float);convmods=collections.defaultdict(float)
for k in events:
 if k.get('ph')!='X' or k.get('cat') not in ('kernel','gpu_memset','gpu_memcpy'):continue
 e=ext.get(k.get('args',{}).get('correlation'))
 if e is None or not any(b['ts']<=e['ts']<=b['ts']+b['dur'] for b in bw):continue
 chain=parent.get(id(e),[])
 e=next((a for a in reversed(chain) if a.get('cat')=='cpu_op'),e)
 us=k['dur'];ops[e['name']][0]+=us;ops[e['name']][1]+=1;kernels[k['name']]+=us
 module=next((seqmod[a['args']['Sequence number']] for a in reversed(chain) if a.get('args',{}).get('Sequence number') in seqmod),'unattributed');mods[module]+=us
 names=' '.join(a['name'].lower() for a in chain)
 cat='convolution' if 'convolutionbackward' in names or 'convolution_backward' in names else 'selective_scan' if 'selectivescanbackward' in names or 'selective_scan_bwd' in k['name'].lower() else 'batchnorm' if 'batchnormbackward' in names else 'matrix_ffn' if any(n in names for n in ('mmbackward','bmmbackward','addmmbackward')) else 'copy_reduction' if any(n in e['name'].lower() for n in ('copy','sum','mean','clone','contiguous')) else 'other'
 cats[cat]+=us
 if cat=='convolution':convmods[module]+=us
p=json.loads((P/'profile.json').read_text());total=sum(v[0] for v in ops.values())
p.update(backward_self_cuda_total_ms=total/1000,top15=[dict(op=n,self_cuda_ms=v[0]/1000,ms_per_step=v[0]/3000,kernel_calls=v[1],percent=v[0]/total*100) for n,v in sorted(ops.items(),key=lambda t:-t[1][0])[:15]],module_top20=[dict(module=n,self_cuda_ms=v/1000,ms_per_step=v/3000) for n,v in sorted(mods.items(),key=lambda t:-t[1])[:20]],categories_ms={n:v/1000 for n,v in cats.items()},kernel_top15=[dict(kernel=n,self_cuda_ms=v/1000,ms_per_step=v/3000) for n,v in sorted(kernels.items(),key=lambda t:-t[1])[:15]],convolution_module_top10=[dict(module=n,ms_per_step=v/3000) for n,v in sorted(convmods.items(),key=lambda t:-t[1])[:10]],module_attribution='Trace GPU External id -> CPU launch op -> backward sequence number -> forward leaf module. Sum actual GPU kernel/memset/memcpy durations once; not parent inclusive times. Backward identified by CPU launch timestamp within BACKWARD scope because autograd runs on separate CPU thread.')
(P/'profile.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
t=json.loads((P/'trial.json').read_text());agg={}
for mode in ('baseline','workspace'):
 rows=[r for r in t['timings'] if r['mode']==mode]
 f=[v for r in rows for v in r['forward_ms']];b=[v for r in rows for v in r['backward_ms']]
 agg[mode]=dict(forward_ms=sum(f)/len(f),backward_ms=sum(b)/len(b),total_ms=(sum(f)+sum(b))/len(b))
agg['backward_reduction_percent']=(1-agg['workspace']['backward_ms']/agg['baseline']['backward_ms'])*100
agg['total_reduction_percent']=(1-agg['workspace']['total_ms']/agg['baseline']['total_ms'])*100
t['aggregate']=agg;(P/'trial.json').write_text(json.dumps(t,indent=2));print('AGGREGATE',agg)
