import os,sys,json,csv,hashlib,shutil,fcntl,subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=ROOT/'log/TinyViM_E123.1_联合新版_HAM_20261105'
OUT=RUN/'early_test/current_best_20261105'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'snapshot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'snapshot.json').exists(),'No duplicate early test'
 status=json.loads((RUN/'run/status.json').read_text())
 assert not (RUN/'final_test/E123.1/status.json').exists(),'Formal test already started; inspect/reuse instead'
 cp=RUN/'run/checkpoints/best.pth';snap=OUT/'best.pth'
 before=sha(cp);shutil.copyfile(cp,snap);after=sha(cp)
 assert before==after==sha(snap),'Checkpoint changed during copy'
 with (RUN/'run/history.csv').open() as f:rows=list(csv.DictReader(f))
 winner=min(rows,key=lambda r:float(r['val_loss']))
 assert int(winner['epoch'])==status['best_epoch']
 assert float(winner['val_loss'])==status['best_val_loss']
 prov=json.loads((RUN/'run/provenance.json').read_text())
 fixed=dict(checkpoint=str(snap),checkpoint_sha256=before,selected_epoch=int(winner['epoch']),split_sha256=prov['split_sha256'],seed=prov['seed'],history_rows=len(rows),parameters=prov['parameters'],selection='current earliest minimum validation loss; early snapshot; test never selects weights',validation_summary=None)
 record=dict(**fixed,source_checkpoint=str(cp),source_sha_before=before,source_sha_after=after,training_complete_at_snapshot=status.get('epoch_completed')==300,epoch_completed_at_snapshot=status.get('epoch_completed'),val_loss=float(winner['val_loss']),created_at=datetime.now().astimezone().isoformat(),scope='E1 original + latest E2 + latest dual-source E3; isolated early HAM test; no TTA; training unchanged')
 (OUT/'snapshot.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
 print('FIXED_EARLY_SNAPSHOT',json.dumps(record,ensure_ascii=False),flush=True)
 sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
 from source.e123_joint_20261105.candidate import build_candidate
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 import final_test_pipeline as pipeline
 source=Path(pipeline.__file__).read_text()
 old="if fixed['history_rows']!=300:raise ValueError('Expected exactly 300 completed training epochs')"
 assert source.count(old)==1
 source=source.replace(old,"assert fixed['history_rows']=="+str(len(rows)))
 source=source.replace('complete normal pipeline: fixed validation-best -> final test','authorized isolated early test: current validation-best snapshot; training not yet complete')
 exec(compile(source,pipeline.__file__,'exec'),pipeline.__dict__)
 pipeline.ARMS={'E123.1':('E123.1 三模块联合【提前测试，非最终best】',RUN/'run')}
 pipeline.DEFAULT_ROOT=OUT
 pipeline.training_ready=lambda run:True
 pipeline.fixed_best_record=lambda run:dict(fixed)
 pipeline.build_model=lambda arm:build_candidate(pretrained=False)
 sys.argv=[__file__,'--arm','E123.1','--output-root',str(OUT),'--gpu','0']
 pipeline.main()
 assert sha(snap)==before
 print('EARLY_TEST_COMPLETE_SNAPSHOT_SHA',before,flush=True)
if __name__=='__main__':main()
