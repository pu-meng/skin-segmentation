from pathlib import Path
import json, subprocess, time
r=Path('/home/PuMengYu/experiments/skin_segmentation')
out=r/'operations/e13_slow_isolation_20261105'
out.mkdir(exist_ok=False)
s=(r/'operations/engineering_speed_lab/speed_lab.py').read_text()
s=s.replace("ROOT/'log/TinyViM_E3.1_FDConvSkip_HAM_20261105/protocol.json'","ROOT/'log/TinyViM_E1E3_HAM_20261105/protocol.json'")
s=s.replace("glob('*/run/history.csv')","glob('**/history.csv')")
s=s.replace('budget_seconds=1800','budget_seconds=900').replace('time.monotonic()+1800','time.monotonic()+900')
s=s.replace('gpu=1','gpu=0').replace("CUDA_VISIBLE_DEVICES='1'","CUDA_VISIBLE_DEVICES='0'")
s=s.replace("OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'","OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU'")
s=s.replace('from Models.E3DynamicSkip.model import build_candidate','from source.e1_e3_combo.candidate import build_candidate')
s=s.replace('torch.set_num_threads(2)','torch.set_num_threads(4)').replace('memory_fraction(0.28','memory_fraction(0.20')
s=s.replace('model=build_candidate(True)',"model=build_candidate(pretrained=True,arm='E1_E3')")
s=s.replace("from speed_contract import valid_budget,within_root",'from speed_contract import valid_budget')
s=s.replace('assert within_root(str(OUT));','')
s=s.replace("model='E3.1 real build_candidate'","model='E1_E3 real build_candidate'").replace('physical_gpu=1','physical_gpu=0').replace('nvidia-smi -i 1','nvidia-smi -i 0')
s=s.replace('for workers in (2,4):','for threads in (4,1):\n  workers=2;torch.set_num_threads(threads)')
s=s.replace('if workers==2:','if threads==4:')
s=s.replace("dict(workers=workers,steps=diag)","dict(threads=threads,workers=workers,steps=diag)")
s=s.replace("armout=OUT/f'workers{workers}'","armout=OUT/f'threads{threads}'")
s=s.replace('arm=dict(workers=workers,','arm=dict(threads=threads,workers=workers,')
s=s.replace("write(f'workers{workers}/summary.json',arm)","write(f'threads{threads}/summary.json',arm)")
s=s.replace("print('EPOCH COMPLETE',json.dumps(arm),flush=True)","print('EPOCH COMPLETE',json.dumps(arm),flush=True)\n   if threads==4 and wall<=70:\n    result['complete']=True;result['stop_reason']='Slow baseline not reproduced (<=70s); no optimization tested';write('report.json',result);return")
start=s.index(' # Local pointwise/Gate compilation')
s=s[:start]+''' result['complete']=len(result['arms'])==2
 result['ratio']=result['arms'][0]['wall_seconds']/result['arms'][1]['wall_seconds']
 s4=torch.load(OUT/'threads4/best.pth',map_location='cpu',weights_only=True);s1=torch.load(OUT/'threads1/best.pth',map_location='cpu',weights_only=True)
 result['epoch_weight_buffer_difference']=compare(s4,s1)
 write('report.json',result)
 print('LAB COMPLETE',result['ratio'],flush=True)
'''+s[s.index('def sha_bytes'):]
# Baseline self drift: replay the three diagnostic steps, not another timed epoch.
needle="   # Reset all state after diagnostics; no diagnostic step is reused as training."
insert='''   if threads==4:
    reset();opt=optimizer();selfdiag=[];it=iter(train)
    for step in range(3):
     x,y,ids,_=next(it);inp=dict(ids=list(ids),x=sha_bytes(x.numpy().tobytes()),y=sha_bytes(y.numpy().tobytes()))
     opt.zero_grad(set_to_none=True);p=model(x.cuda(non_blocking=True));loss=crit(p,y.cuda(non_blocking=True));loss.backward()
     grads={n:(v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu')) for n,v in model.named_parameters()};opt.step()
     weights={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};ref=torch.load(OUT/f'diagnostic_step{step+1}.pth',map_location='cpu')
     selfdiag.append(dict(step=step+1,input_equal=ref['input']==inp,loss_abs=abs(float(loss.detach())-ref['loss']),prediction=compare({'prediction':ref['prediction']},{'prediction':p.detach().cpu()}),gradients=compare(ref['gradients'],grads),weights_buffers=compare(ref['weights'],weights)))
     del grads,weights,ref
    result['baseline_self_replay']=selfdiag;write('report.json',result)
'''
s=s.replace(needle,insert+needle)
(out/'speed_lab.py').write_text(s)
(out/'speed_contract.py').write_text((r/'operations/engineering_speed_lab/speed_contract.py').read_text())
compile(s,str(out/'speed_lab.py'),'exec')
script='#!/bin/bash\nset -euo pipefail\ncd '+str(r)+'\nexport PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU\nexec /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u '+str(out/'speed_lab.py')+'\n'
(out/'run.sh').write_text(script)
subprocess.run(['bash','-n',str(out/'run.sh')],check=True)
log=(out/'console.log').open('w')
p=subprocess.Popen(['bash',str(out/'run.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(out=str(out),controller_pid=p.pid)))
