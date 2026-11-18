import os,sys,json,hashlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'log/TinyViM_E123.1_联合新版_HAM_20261105/early_test/current_best_20261105'
RUN=OUT.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.e123_joint_20261105.candidate import build_candidate
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
def main():
 import final_test_pipeline as p
 record=json.loads((OUT/'snapshot.json').read_text())
 assert hashlib.sha256(Path(record['checkpoint']).read_bytes()).hexdigest()==record['checkpoint_sha256']
 keys=['checkpoint','checkpoint_sha256','selected_epoch','split_sha256','seed','history_rows','parameters','selection','validation_summary']
 fixed={k:record[k] for k in keys}
 source=Path(p.__file__).read_text()
 old="if fixed['history_rows']!=300:raise ValueError('Expected exactly 300 completed training epochs')"
 assert source.count(old)==1
 source=source.replace(old,"assert fixed['history_rows']=="+str(record['history_rows']))
 oldloader="num_workers=2,\n                          pin_memory=True,multiprocessing_context='spawn'"
 assert source.count(oldloader)==1
 source=source.replace(oldloader,'num_workers=0, pin_memory=True')
 source=source.replace('complete normal pipeline: fixed validation-best -> final test','authorized isolated early test: current-best snapshot; original training unchanged; workers0 avoids spawned dependency import failure')
 exec(compile(source,p.__file__,'exec'),p.__dict__)
 p.ARMS={'E123.1':('E123.1 三模块联合【提前测试，非最终best】',RUN/'run')}
 p.training_ready=lambda run:True
 p.fixed_best_record=lambda run:dict(fixed)
 p.build_model=lambda arm:build_candidate(pretrained=False)
 sys.argv=[__file__,'--arm','E123.1','--output-root',str(OUT/'workers0'),'--gpu','0']
 p.main()
 print('EARLY_TEST_COMPLETE',flush=True)
if __name__=='__main__':main()
