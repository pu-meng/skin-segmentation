from pathlib import Path
import shutil,json,hashlib,subprocess,os
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/resume300_20261106';OPS.mkdir(exist_ok=True)
base=Path('/media/8T/PuMengYu/experiments/skin_segmentation')
assert shutil.disk_usage(base).free>12*2**30
for arm in ['R0','R1','R2','R3','DualA']:
 old=base/'DecoderDualA_SCREEN100of300_20261106' if arm=='DualA' else ROOT/'log/TinyViM_E1232_FourArm_SCREEN100of300_20261106'
 run=base/('DecoderDualA_RESUME100to300_20261106' if arm=='DualA' else 'TinyViM_E1232_FourArm_RESUME100to300_20261106')
 work=OPS/arm;work.mkdir(exist_ok=True)
 src=old/arm/'train';dst=run/arm/'train';dst.mkdir(parents=True,exist_ok=True);(dst/'checkpoints').mkdir(exist_ok=True)
 assert not (dst/'history.csv').exists(),'refuse redeploy'
 for f in ['protocol.json','cpu_acceptance.json']:
  if not (run/f).exists():shutil.copy2(old/f,run/f)
 for f in ['history.csv','status.json','provenance.json']:
  shutil.copy2(src/f,dst/f)
 for f in ['latest.pth']:(dst/'checkpoints'/f).symlink_to(src/'checkpoints'/f)
 shutil.copy2(src/'checkpoints/best.pth',dst/'checkpoints/best.pth')
 gpu='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b' if arm=='DualA' else 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
 contract=dict(arm=arm,original_run=str(old),resume_run=str(run),start_checkpoint=str(src/'checkpoints/latest.pth'),start_epoch=100,total_epochs=300,scheduler_T_max=50,shared_gpu=True,gpu_uuid=gpu,original_results_untouched=True,reason_new_result_root='home9.26GB free insufficient for retained100 plus concurrent checkpoint temp plus8GiB guard; no relocation/deletion',final_test='after successful300: earliest val minloss best; noTTA HAM2020 predictions metrics visualizations audit',failure_policy='block; no automatic retry',deadline_seconds=259200)
 (work/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 script=f'''#!/bin/bash
set -euo pipefail
cd {ROOT}
exec 9>{work}/controller.lock
flock -n 9 || exit 73
export RESUME_ARM={arm} CUDA_VISIBLE_DEVICES={gpu} EXPERIMENT_PHYSICAL_GPU={'1' if arm=='DualA' else '0'}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
printf '%s\\n' $$ > {work}/controller.pid
phase=preflight
finish() {{ rc=$?; printf '%s\\n' "$rc" > {work}/controller.exitcode; printf '{{"phase":"%s","exitcode":%d}}\\n' "$phase" "$rc" > {work}/controller_status.json; }}
trap finish EXIT
for phase in preflight train test; do
 printf '{{"phase":"%s"}}\\n' "$phase" > {work}/controller_status.json
 timeout --foreground --signal=TERM 259200 "$PY" -u {OPS}/entry.py "$phase" 2>&1 | tee {work}/"$phase".log
 echo 0 > {work}/"$phase".exitcode
done
phase=complete
'''
 (work/'launch.sh').write_text(script);subprocess.run(['bash','-n',str(work/'launch.sh')],check=True)
status='''#!/bin/bash
set -u
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
for arm in R0 R1 R2 R3 DualA; do
 d=/home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/$arm
 echo "=== $arm ==="
 [ ! -f "$d/controller.pid" ] || ps -p "$(< "$d/controller.pid")" -o pid,ppid,sid,etime,args
 for f in controller_status.json preflight_passed.json train_identity.json first_resumed_optimizer_update.json failure.json completion.json; do [ ! -f "$d/$f" ] || { echo "$f"; cat "$d/$f"; }; done
 [ ! -f "$d/train.log" ] || { stat -c '%y %s %n' "$d/train.log"; tail -n 3 "$d/train.log"; }
done
'''
(OPS/'status.sh').write_text(status);subprocess.run(['bash','-n',str(OPS/'status.sh')],check=True)
print('DEPLOYED',OPS)
