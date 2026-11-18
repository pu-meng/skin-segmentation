import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,gc,statistics,subprocess,signal
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as m
from source.e1232_matrix_20261106.candidate import build_candidate as matrix
from source.e123_candidate_a_20261106.candidate import build_candidate as original
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
from ham_fast_pipeline import PlannedDataset
OUT=ROOT/'operations/engineering_speed_lab/paired_model_cost_20261106'
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.16)
 config=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106/protocol.json').read_text())['config'];ds=PlannedDataset(Path(config['data']['split_file']).parent,'train');items=[ds[i] for i in range(8)];x=torch.stack([z[0] for z in items]).cuda();y=torch.stack([z[1] for z in items]).cuda();api=m.runner.baseline_api();criterion=api.BceDiceLoss().cuda()
 report={'scope':'same real 8 training images without random augmentation, paired common initialization, FP32 batch8; 2warmup+4timed training steps; reverse-order repeat; shared GPU; not epoch speed or accuracy','runs':[],'input_ids':[z[2] for z in items]}
 def save():(OUT/'result.json').write_text(json.dumps(report,indent=2))
 order=['E0','CandidateA','BT','BT_E2off','BT_E3off'];order=order+order[::-1]
 for name in order:
  api.seed_all(42);handles=[];model=None;opt=None
  try:
   model=(build_four_stage_baseline(False) if name=='E0' else original(False) if name=='CandidateA' else matrix('BT',False)).cuda().train()
   if name=='BT_E2off':handles.append(model.decoder.e2.register_forward_hook(lambda mod,args,out:args[0]))
   if name=='BT_E3off':handles.append(model.decoder.e3.register_forward_hook(lambda mod,args,out:(args[0],args[1])))
   # Output hooks above would still compute the branch: override the isolated instance's forward instead.
   for h in handles:h.remove()
   if name=='BT_E2off':model.decoder.e2.forward=lambda value:value
   if name=='BT_E3off':model.decoder.e3.forward=lambda s,d:(s,d)
   params=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
   spans=[];walls=[];losses=[];torch.cuda.reset_peak_memory_stats()
   for i in range(6):
    opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)];ev[0].record();pred=model(x);loss=criterion(pred,y);ev[1].record();loss.backward();ev[2].record();opt.step();ev[3].record();torch.cuda.synchronize();wall=time.perf_counter()-start
    if i>=2:spans.append([ev[j].elapsed_time(ev[j+1]) for j in range(3)]);walls.append(wall*1000);losses.append(float(loss.detach()))
   model.eval();val=[]
   with torch.no_grad():
    model(x);torch.cuda.synchronize()
    for i in range(4):
     a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();pred=model(x);loss=criterion(pred,y);b.record();torch.cuda.synchronize();val.append(a.elapsed_time(b))
   row={'name':name,'parameters':params,'forward_ms':statistics.mean(z[0] for z in spans),'backward_ms':statistics.mean(z[1] for z in spans),'optimizer_ms':statistics.mean(z[2] for z in spans),'wall_ms':statistics.mean(walls),'step_wall_ms':walls,'validation_batch8_ms':statistics.mean(val),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'passed':True}
   report['runs'].append(row);print('PAIR_RESULT',json.dumps(row),flush=True)
  except Exception as e:report['runs'].append({'name':name,'passed':False,'error':repr(e)});print('ERROR',name,repr(e),flush=True)
  finally:
   del model,opt;gc.collect();torch.cuda.empty_cache();save()
 report['complete']=len(report['runs'])==10 and all(z.get('passed') for z in report['runs']);report['gpu_after']=subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader');save()
if __name__=='__main__':main()
