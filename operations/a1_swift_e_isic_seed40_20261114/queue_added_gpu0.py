import json, subprocess, sys, time
from pathlib import Path
OPS=Path('/home/PuMengYu/experiments/skin_segmentation/operations')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
EXPCTL=OPS/'expctl.py'
ES_DONE=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ES_ISIC_SEED40_FRESH300_20261114/ISIC2018/E+S_no_bottle_no_d2d3/completion.json')

def update(exp_id, **fields):
 cmd=[PY,str(EXPCTL),'update','--id',exp_id]
 for k,v in fields.items():
  if v is not None: cmd += ['--'+k.replace('_','-'),str(v)]
 subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)

def wait_complete(path, label):
 while not path.is_file() or not json.loads(path.read_text()).get('complete'):
  time.sleep(60)

wait_complete(ES_DONE,'E+S seed40')
d2_id='A1_ESD_SINGLE_HAM_ISIC_SEED40_20261114'
d2_entry=OPS/'a1_esd_single_seed40_20261114'/'entry.py'
update(d2_id,status='waiting_gpu',wait_reason='E+S seed40完成；等待GPU0准入后按ISIC、HAM顺序执行')
rc=subprocess.run([PY,'-B',str(d2_entry),'ordered_queue']).returncode
if rc:
 update(d2_id,status='failed',wait_reason='seed40 D2队列失败；检查运行目录failure.json')
 sys.exit(rc)
update(d2_id,status='complete',epochs_done=300,epochs=300,gpu=0,wait_reason='',note='seed40；ISIC与HAM fresh300训练及无TTA/TTA4审计完成。')

e_id='A1_SWIFTFORMER_E_ISIC_SEED40_20261114'
e_entry=OPS/'a1_swift_e_isic_seed40_20261114'/'entry.py'
update(e_id,status='waiting_gpu',wait_reason='D2 seed40 ISIC/HAM完成；等待GPU0准入后执行ISIC')
rc=subprocess.run([PY,'-B',str(e_entry),'queue','ISIC2018','0']).returncode
if rc:
 update(e_id,status='failed',wait_reason='E seed40队列失败；检查运行目录failure.json')
 sys.exit(rc)
update(e_id,status='complete',epochs_done=300,epochs=300,gpu=0,wait_reason='',note='E-only ISIC seed40 fresh300；无TTA/TTA4测试审计完成。')
