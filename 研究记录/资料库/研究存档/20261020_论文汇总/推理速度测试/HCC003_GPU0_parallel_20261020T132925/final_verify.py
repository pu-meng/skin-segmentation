import json,torch,importlib.util
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());torch.set_num_threads(1)
for worker in ['A','B']:
 base=torch.load(P/f'1_serial_{worker}_logits_cpu.pt',map_location='cpu',weights_only=False)
 for row in [x for x in r['rows'] if x['worker']==worker]:
  x=torch.load(row['logits_path'],map_location='cpu',weights_only=False);d=(x.float()-base.float()).abs()
  row['difference_vs_same_worker_first_serial']=dict(max_abs=d.max().item(),mean_abs=d.mean().item(),argmax_mismatch_ratio=(x.argmax(1)!=base.argmax(1)).double().mean().item())
  del x,d
 del base
spec=importlib.util.spec_from_file_location('b',P/'benchmark.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
r['formal_final_verification']=b.old.formal();r['gpu_final_verification']=b.old.gpu()
(P/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
with (P/'report.md').open('a') as f:
 f.write('\n## 补充核验\n')
 f.write('- 各worker以自己首轮串行为参照，后续串行/并行的全部logits逐元素比较：'+json.dumps([dict(label=x['label'],worker=x['worker'],diff=x['difference_vs_same_worker_first_serial']) for x in r['rows']],ensure_ascii=False)+'\n')
 f.write('- 最后只读核验：'+json.dumps(r['formal_final_verification'],ensure_ascii=False)+'\n')
print(json.dumps(dict(same_worker_differences=[x['difference_vs_same_worker_first_serial'] for x in r['rows']],final_formal=r['formal_final_verification'],gpu=r['gpu_final_verification'],peak_allocator={w:dict(allocated_GiB=max(x['peak_allocated_bytes'] for x in r['rows'] if x['worker']==w)/2**30,reserved_GiB=max(x['peak_reserved_bytes'] for x in r['rows'] if x['worker']==w)/2**30) for w in ['A','B']},previous_baseline=r['serial_A_vs_previous_baseline'],paired_ratios=r['paired_ratios'],serial_drift=r['serial_drift_last_over_first']),ensure_ascii=False,indent=2))
