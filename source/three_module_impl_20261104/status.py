from pathlib import Path
import subprocess,json,datetime
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=Path(__file__).parent
print(datetime.datetime.now().astimezone().isoformat())
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
for name,root in [('E0',R/'log/TinyViM_四层对应Decoder_基础_HAM_20261104'),('E1',R/'log/TinyViM_三模块_E1_DWPO_HAM_20261104')]:
 print(name,str(root))
 s=root/'run/status.json'
 if s.exists():
  state=json.loads(s.read_text());print({k:state.get(k) for k in ['status','pid','epoch_completed','epoch','step','updated_at']})
  pid=state.get('pid')
  if pid:subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,etime,args'])
 for rel in ['protocol.json','controller.pid','controller.exitcode','preflight/preflight_status.json','run/checkpoints/best.pth','run/checkpoints/final.pth','run/artifact_audit.json','final_test/E1/artifact_audit.json']:
  p=root/rel
  if p.exists():print(rel,'bytes',p.stat().st_size,'mtime',datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat())
 for rel in ['preflight_console.log','train_console.log','test_console.log','gpu_train_console.log']:
  p=root/rel
  if p.exists():print(rel,'mtime',p.stat().st_mtime,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-4:]))
print('E2/E3 NOT STARTED: core gate/SS2D definition unresolved; no automatic queue')
