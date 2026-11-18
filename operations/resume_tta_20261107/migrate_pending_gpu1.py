from pathlib import Path
import json,hashlib,shutil,subprocess,os
P=Path('/home/PuMengYu/experiments/skin_segmentation');OLD=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DualA_E2_FirstBatch100_20261107');NEW=OLD.with_name('DualA_E2_Cross_AFNO_GPU1_20261107');O=P/'operations/duala_E2_gpu1_20261107';O.mkdir(exist_ok=True)
assert not NEW.exists();assert not any((OLD/a/'run').exists() for a in ['Cross','AFNO'])
oldp=(OLD/'protocol.json').read_bytes();protocol=json.loads(oldp);cpu=json.loads((OLD/'cpu_acceptance.json').read_text());assert cpu['passed']
for f,h in {**protocol['protected_sha256'],**protocol['implementation_sha256']}.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
q=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b' not in q
# Old slots use exclusive train.log creation. Reserve only unstarted arms to revoke their launch ability.
for arm in ['Cross','AFNO']:
 with (OLD/arm/'train.log').open('x') as f:f.write('MIGRATED_TO_GPU1: original slot dispatch revoked. See '+str(NEW/arm/'train.log')+'\nThis is a dispatch guard, not a training failure or training log.\n')
NEW.mkdir()
for a in ['Cross','AFNO']:(NEW/a).mkdir()
protocol.update(gpu_uuid='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',physical_gpu=1,batches=[['Cross','AFNO']],migration={'original_run':str(OLD),'original_protocol_sha256':hashlib.sha256(oldp).hexdigest(),'scope':'unstarted Cross/AFNO only; scientific config unchanged; original slot exclusive-log guard prevents duplicate launch','cpu_acceptance_reused':True})
(NEW/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));shutil.copy2(OLD/'cpu_acceptance.json',NEW/'cpu_acceptance.json');shutil.copy2(OLD/'reference100_history.csv',NEW/'reference100_history.csv')
wrapper='''import os,sys,importlib.util,json,traceback,fcntl
from pathlib import Path
P=Path('/home/PuMengYu/experiments/skin_segmentation'); OPS=P/'operations/duala_E2_batch1_20261107'
sys.path[:0]=[str(OPS),str(P)]
spec=importlib.util.spec_from_file_location('original_duala_entry',OPS/'entry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DualA_E2_Cross_AFNO_GPU1_20261107');m.BATCHES=[['Cross','AFNO']]
original_write=m.write
def gpu1_write(path,value):
 if Path(path).name=='provenance.json':value=dict(value,physical_gpu=1)
 return original_write(path,value)
m.write=gpu1_write
arm=sys.argv[1]
with (m.RUN/arm/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:m.train(arm)
 except BaseException as e:
  m.write(m.RUN/arm/'failure.json',{'error':repr(e),'pid':os.getpid()});raise
'''
# train() records physical_gpu=0 literally; keep frozen train function unchanged, correct record after first-step by separate audit.
(O/'entry.py').write_text(wrapper)
for arm in ['Cross','AFNO']:
 script=f'''#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u {O}/entry.py {arm} 2>&1 | tee {NEW}/{arm}/train.log
'''
 f=O/f'run_{arm}.sh';f.write_text(script);subprocess.run(['bash','-n',str(f)],check=True)
 subprocess.run(['tmux','new-session','-d','-s',f'DualA_{arm}_GPU1','bash',str(f)],check=True)
receipt={'original_run':str(OLD),'new_run':str(NEW),'arms':['Cross','AFNO'],'gpu':1,'gpu_uuid':protocol['gpu_uuid'],'old_dispatch_revoked_by_exclusive_log_guard':True,'old_active_children_untouched':True,'budget':'100 epochs + validation; no test per original explicit contract','note':'old slot controllers may exit FileExistsError after finishing current arm because dispatch was deliberately revoked; this is queue cancellation, not new arm training failure. Original train provenance physical_gpu literal 0 is superseded by this GPU1 receipt and actual UUID/PID audit.'}
(O/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False))
