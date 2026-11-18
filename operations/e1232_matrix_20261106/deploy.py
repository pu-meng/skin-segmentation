from pathlib import Path
import json,subprocess,hashlib,shutil
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/e1232_matrix_20261106';RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106');RUN.mkdir(exist_ok=False)
old=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/TinyViM_E1232_FourArm_RESUME100to300_20261106/protocol.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources=old['source_sha256'].copy()
for p in [ROOT/'source/e1232_matrix_20261106/candidate.py',OPS/'worker.py',OPS/'deploy.py',ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py']:
 sources[str(p.relative_to(ROOT))]=sha(p)
for p,h in old['source_sha256'].items():assert sha(ROOT/p)==h,p
protocol=dict(version='E1232-2x3-gate-matrix-fresh300',config=old['config'],source_sha256=sources,reference_protocol_sha256=sha('/media/8T/PuMengYu/experiments/skin_segmentation/TinyViM_E1232_FourArm_RESUME100to300_20261106/protocol.json'),matrix={'AT':'reuse R1','AS':'reuse R3','BT':'E1232-BT-HAM300-S42','BS':'E1232-BS-HAM300-S42','AU':'E1232-AU-HAM300-S42','BU':'E1232-BU-HAM300-S42'},source_B='independent LN512/DW3/GELU/GAP for full M and full W; contexts independently trainable, initialized identical to A; ordered concat1024; independent1024->128->512 sigmoid heads; last zero/biaslogit(.1)',source_A='original X controller unchanged',skip_T='k controlled v; v controlled k',skip_S='k controlled v; v controlled v',skip_U='k controlled v; original S finish; D returned exact input; unused D finish and gd removed',initialization='original ImageNet and decoder/align seed42; bottleneck45; new B expanded heads46; common tensors exact paired; no trained best loaded',capacity_note='B adds independent context5632 plus two expanded first heads131072 =136704 parameters vs A same skip; not equal capacity or unique causal attribution',batch=8,precision='FP32',epochs=300,scheduler_Tmax=50,workers=2,validation_fast_batch=8,potential_best_recheck_batch=1,test_contract='successful 300 epochs/261900 updates -> earliest min val_loss -> immutable best snapshot SHA -> CPU strict -> HAM2020 none AND fixed mirror4 logit mean -> predictions/metrics/12 visualizations each/audits',resource_policy={'shared_authorized':True,'preferred_gpu0':'only about7GiB free insufficient four-resident formalbatch; use GPU1','physical_gpu':1,'uuid':'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b','existing_tasks':'never interrupt or modify','admission_free_gpu_mib':15000,'disk_admission_gib':18,'disk_guard_gib':8,'new_outputs_budget_gib':6,'other_active_writes_allowance_gib':4,'deadline_hours':120},test_used_for_design=False)
protocol['config']['evaluation']['test_evaluation']='automatic successful300 fixed-best none and predefined mirror4; no test selection'
(RUN/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
for arm in ['BT','BS','AU','BU']:
 d=OPS/arm;d.mkdir(exist_ok=True)
 text=f'''#!/bin/bash
set -euo pipefail
cd {ROOT}
exec 9>{d}/controller.lock
flock -n 9 || exit 73
export MATRIX_ARM={arm} CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b EXPERIMENT_PHYSICAL_GPU=1
export MATRIX_COHORT="{'BU' if arm=='BU' else 'BT BS AU'}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
printf '%s\\n' $$ > {d}/controller.pid
phase=waiting_resources
finish() {{ rc=$?; printf '%s\\n' "$rc" > {d}/controller.exitcode; printf '{{"phase":"%s","exitcode":%d}}\\n' "$phase" "$rc" > {d}/controller_status.json; }}
trap finish EXIT
"$PY" -u {OPS}/admission.py 2>&1 | tee {d}/admission.log
for phase in preflight train test; do
 if [ "$phase" = train ]; then
  for peer in ${{MATRIX_COHORT}}; do
   until [ -f {OPS}/$peer/preflight.exitcode ]; do
    [ ! -f {OPS}/$peer/failure.json ] || exit 75
   sleep 2
   done
  done
 fi
 printf '{{"phase":"%s"}}\\n' "$phase" > {d}/controller_status.json
 timeout --foreground --signal=TERM 432000 "$PY" -u {OPS}/worker.py "$phase" 2>&1 | tee {d}/"$phase".log
 echo 0 > {d}/"$phase".exitcode
done
phase=complete
'''
 (d/'launch.sh').write_text(text);subprocess.run(['bash','-n',str(d/'launch.sh')],check=True)
print('DEPLOYED',RUN,'freeGiB',shutil.disk_usage(RUN).free/2**30)
