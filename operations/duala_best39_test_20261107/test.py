import os,sys,json,csv,shutil,fcntl,time,subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106')
OUT=RUN/'screen100_best39_test_20261107'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports,sha256_file as sha
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import torch
from source.decoder_dual_a_20261106.candidate import build_candidate
import final_test_pipeline as p
UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
def verify(m):
 for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,'source drift '+name
 assert sha(m['config']['data']['split_file'])==m['config']['data']['split_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30

def main():
 OUT.mkdir(exist_ok=True)
 lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'snapshot.json').exists(),'No duplicate test'
 started=time.monotonic();m=json.loads((RUN/'protocol.json').read_text());verify(m)
 train=RUN/'DualA/train';status=json.loads((train/'status.json').read_text())
 assert status['status']=='paused_at_screen_boundary' and status['epoch_completed']==100 and status['best_epoch']==39
 rows=list(csv.DictReader((train/'history.csv').open()))
 assert len(rows)==100 and [int(r['epoch']) for r in rows]==list(range(1,101))
 winner=min(rows,key=lambda r:float(r['val_loss']))
 assert int(winner['epoch'])==39 and float(winner['val_loss'])==status['best_loss']
 source=train/'checkpoints/best.pth';snap=OUT/'best_epoch039.pth'
 before=sha(source);shutil.copyfile(source,snap);assert sha(source)==before==sha(snap)
 fixed=dict(checkpoint=str(snap),checkpoint_sha256=before,selected_epoch=39,split_sha256=m['config']['data']['split_sha256'],seed=42,history_rows=100,parameters=9661072,selection='earliest minimum validation loss within completed screen100; test never selects weights',validation_summary=None)
 record=dict(**fixed,source_checkpoint=str(source),source_sha_before=before,source_sha_after=sha(source),training_complete=False,screening_complete=True,epoch_completed=100,val_loss=float(winner['val_loss']),val_global_dice=float(winner['val_global_dice']),tta=False,postprocessing='none; official fixed evaluator',protocol_sha256=sha(RUN/'protocol.json'),created_at=datetime.now().astimezone().isoformat())
 p.write_json(OUT/'snapshot.json',record)
 print('FIXED_SCREEN100_BEST39',json.dumps(record),flush=True)
 # Separate hidden-CUDA process proves complete true constructor strict load.
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 subprocess.run([sys.executable,'-B',__file__,'cpu'],env=env,check=True,timeout=300)
 source_code=Path(p.__file__).read_text()
 old="if fixed['history_rows']!=300:raise ValueError('Expected exactly 300 completed training epochs')"
 assert source_code.count(old)==1
 source_code=source_code.replace(old,"assert fixed['history_rows']==100")
 old_loader="num_workers=2,\n                          pin_memory=True,multiprocessing_context='spawn'"
 assert source_code.count(old_loader)==1
 source_code=source_code.replace(old_loader,'num_workers=0, pin_memory=True')
 source_code=source_code.replace('complete normal pipeline: fixed validation-best -> final test','authorized screen100 best39 complete test; not final300; no TTA')
 exec(compile(source_code,p.__file__,'exec'),p.__dict__)
 p.ARMS={'DualA':('DualA 筛选100轮验证best39（非300轮最终）',train)}
 p.training_ready=lambda run:True
 p.fixed_best_record=lambda run:dict(fixed)
 p.build_model=lambda arm:build_candidate(pretrained=False)
 import ham_fast_runner as runner
 api=runner.baseline_api();api.seed_all(42)
 original_evaluate=api.evaluate
 def evaluated(model,loader,criterion,device,output):
  mapping=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
  assert any(UUID in line and f', {os.getpid()},' in line for line in mapping.splitlines()),mapping
  p.write_json(OUT/'gpu_mapping.json',dict(pid=os.getpid(),gpu_uuid=UUID,compute_apps=mapping,shared_gpu_authorized=True))
  return original_evaluate(model,loader,criterion,device,output)
 api.evaluate=evaluated
 sys.argv=[__file__,'--arm','DualA','--output-root',str(OUT),'--gpu','0']
 p.main();verify(m);assert sha(source)==sha(snap)==before
 summary=json.loads((OUT/'DualA/test/summary.json').read_text())
 p.write_json(OUT/'completion.json',dict(complete=True,pid=os.getpid(),checkpoint_sha256=before,source_unchanged=True,full_pipeline_elapsed_seconds=time.monotonic()-started,summary=summary))
 print('COMPLETE_SCREEN100_TEST',json.dumps(summary),flush=True)

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='cpu':
  assert os.environ['CUDA_VISIBLE_DEVICES']==''
  torch.set_num_threads(2);snap=OUT/'best_epoch039.pth';model=build_candidate(False)
  model.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True)
  assert not torch.cuda.is_initialized()
  p.write_json(OUT/'cpu_strict_load.json',dict(passed=True,strict=True,full_constructor='source.decoder_dual_a_20261106.candidate.build_candidate(False)',checkpoint_sha256=sha(snap),parameters=sum(v.numel() for v in model.parameters()),cuda_initialized=False))
  print('HIDDEN_CUDA_CPU_STRICT_LOAD_PASSED',flush=True)
 else:
  try:main()
  except BaseException as e:
   p.write_json(OUT/'failure.json',dict(error=repr(e),pid=os.getpid()));raise
