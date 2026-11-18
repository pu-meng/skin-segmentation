import os,sys,json,time,fcntl,hashlib,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');HERE=ROOT/'source/three_module_e23_20261104'
sys.argv=[str(HERE/'wait_queue.py'),'E3']
g={'__name__':'reuse_original_queue_helpers','__file__':str(HERE/'wait_queue.py')}
exec(compile((HERE/'wait_queue.py').read_text(),str(HERE/'wait_queue.py'),'exec'),g)
g['GPU']=0;g['ENV'].update(CUDA_VISIBLE_DEVICES='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',EXPERIMENT_PHYSICAL_GPU='0')
g['state'].update(gpu=0,shared_authorized=True,upstream_dependency=None,migration_record=str(D/'migration.json'))
RUN=g['RUN'];save=g['save'];phase=g['phase']
def verify_sources():
 for rel,digest in json.loads((D/'migration.json').read_text())['protected_sha256'].items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,'Protected source drift '+rel
with (RUN/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (RUN/'controller_status.json').exists(),'No implicit retry'
 assert not (RUN/'run').exists() and not (RUN/'RUNNING.lock').exists()
 (RUN/'controller.pid').write_text(str(os.getpid()))
 try:
  verify_sources()
  cp=json.loads((HERE/'cpu_E3_acceptance.json').read_text());assert cp['passed'] and not cp['cuda_initialized'] and cp['scan_restored']
  memory=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip()
  compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
  assert memory.split(',')[0].strip()=='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
  assert int(memory.split(',')[1])>=6000,'Insufficient free VRAM; block, not queue'
  save('starting_shared_gpu0',memory_before=memory,compute_before=compute)
  phase('gpu_acceptance',[str(HERE/'acceptance.py'),'gpu','E3'],3600)
  assert json.loads((HERE/'gpu_E3_acceptance.json').read_text())['passed']
  verify_sources()
  phase('prepare',[str(HERE/'entry.py'),'prepare'],600)
  phase('preflight',[str(HERE/'entry.py'),'preflight'],3600)
  phase('train',[str(HERE/'entry.py'),'train'])
  phase('test',[str(HERE/'entry.py'),'test'],12*3600)
  a=json.loads((RUN/'final_test/E3/artifact_audit.json').read_text());assert a['complete'] and a['prediction_count']==2020 and a['metric_count']==2020 and a['visualizations']==12
  verify_sources();save('complete',test_artifact_audit=a);(RUN/'controller.exitcode').write_text('0')
 except BaseException as e:
  save('failed',error=repr(e),old_queue_restored=False,automatically_waiting=False);(RUN/'controller.exitcode').write_text('1');raise
