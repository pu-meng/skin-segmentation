import json,os,sys,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path.insert(0,str(R/'operations'));import registry
O=R/'operations/isic2017_table1_20261116';OUT=R/'log/ISIC2017_表1_去重_20261116';Q=OUT/'queue_status.json'
entries={'A1-SwiftFormer-S':('A1',300),'MambaLiteUNet':('MambaLiteUNet',300),'UNet2D':('UNet2D',300),'CMUNeXt':('CMUNeXt',300),'SkinFormer-local':('SkinFormer_local',300),'S3H-Net':('S3HNet',200),'FocalTransNet':('FocalTransNet',100),'FuseUNet':('FuseUNet',1000),'AD-GBC':('ADGBC',400)}
ids={name:'ISIC2017_DEDUP_'+slug+'_20261116' for name,(slug,_) in entries.items()}
prior=registry.load().get('experiments',{})
for name,(slug,epochs) in entries.items():
 if ids[name] not in prior:registry.register({'id':ids[name],'name':'ISIC2017 '+name,'run_dir':str(OUT/name),'datasets':['ISIC2017'],'epochs_total':epochs,'epochs_done':0,'gpu':1,'status':'queued','wait_reason':'等待 GPU1 两槽位及真实预检','controller_pid':os.getpid(),'source':'submit','note':'去重划分1997/149/600；队列状态见 operations/isic2017_table1_20261116'})
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def mapped(v):
 if v in ('train','training'):return 'training'
 if v in ('test','testing','ready_to_test'):return 'testing' if v in ('test','testing') else 'queued'
 if v=='preflight':return 'preflight'
 if v=='complete':return 'complete'
 if 'failed' in v:return 'failed'
 return 'queued'
try:
 while True:
  primary=read(Q);extra=read(O/'extras_status.json')
  for name in entries:
   if name in ('A1-SwiftFormer-S','MambaLiteUNet'):j=primary.get('jobs',{}).get(name,{});raw=j.get('state','unknown');pid=j.get('pid',primary.get('controller_pid'))
   elif name in ('UNet2D','CMUNeXt'):j=read(OUT/name/'queue_status.json');raw=j.get('status','unknown');pid=j.get('child_pid',j.get('controller_pid'))
   else:j=extra.get('jobs',{}).get(name,{});raw=j.get('status','unknown');pid=j.get('pid',extra.get('pid'))
   state=mapped(raw);fields={'status':state,'controller_pid':pid,'gpu':1,'wait_reason':'等待 GPU1 两槽位及实际显存门槛' if state=='queued' else '', 'note':f'ISIC2017 去重；队列原始状态 {raw}'}
   registry.update(ids[name],**fields)
  time.sleep(30)
except BaseException:
 (O/'registry_bridge_error.log').write_text(traceback.format_exc());raise
