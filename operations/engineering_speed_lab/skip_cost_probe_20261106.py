import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,statistics,copy,gc
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as entry
from source.e123_joint_20261105.candidate import E3
OUT=ROOT/'operations/engineering_speed_lab/skip_cost_parts_20261106'
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
 # Real aligned S,D captured from a separate unchanged full model, not synthetic shapes.
 from source.e1232_matrix_20261106.candidate import build_candidate
 from ham_fast_pipeline import PlannedDataset
 cfg=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106/protocol.json').read_text())['config'];ds=PlannedDataset(Path(cfg['data']['split_file']).parent,'train');x=torch.stack([ds[i][0] for i in range(8)]).cuda();model=build_candidate('BT',False).cuda().eval();captured={}
 def take(mod,args):captured['s']=args[0].detach().clone();captured['d']=args[1].detach().clone()
 h=model.decoder.e3.register_forward_pre_hook(take)
 with torch.no_grad():model(x)
 h.remove();base=copy.deepcopy(model.decoder.e3).cpu();del model,x;gc.collect();torch.cuda.empty_cache()
 s=captured['s'];d=captured['d'];report={'scope':'isolated real E3 train-mode component cost batch8 C64 H64 W64; no module deletion/accuracy claim; sharedGPU; each component5warmup8timed; repeat reversed order','shapes':[list(s.shape),list(d.shape)],'runs':[]}
 cases=['pre','frequency_stage','spatial_stage_select','ffn','post','gate','full_E3']
 def run(name):
  mod=copy.deepcopy(base).cuda().train();side=mod.s
  # Use each component actual interface tensors produced by original module.
  with torch.no_grad():
   u=side.pre(s);z=side.n1(u);a,b,c=z.split([24,24,16],1);joined=side.gen.n(side.gen.fuse(torch.cat([side.gen.lap(a),side.gen.haar(b),side.gen.fourier(c)],1)))
  if name=='pre':fn=lambda q:side.pre(q);inp=s
  elif name=='frequency_stage':
   def fn(q):
    a,b,c=q.split([24,24,16],1);return side.gen.n(side.gen.fuse(torch.cat([side.gen.lap(a),side.gen.haar(b),side.gen.fourier(c)],1)))
   inp=z
  elif name=='spatial_stage_select':
   def fn(q):
    a,b,c=q.split([24,24,16],1);a=a+side.gen.rep(a);b=side.gen.axis(b);c=c+torch.nn.functional.interpolate(side.gen.coarse(torch.nn.functional.avg_pool2d(c,4,4)),size=c.shape[-2:],mode='bilinear',align_corners=False);j=torch.cat([a,b,c],1);w=side.gen.select(j.mean((2,3),keepdim=True)).softmax(1);return side.gen.out(torch.cat([3*w[:,0:1]*a,3*w[:,1:2]*b,3*w[:,2:3]*c],1))
   inp=joined
  elif name=='ffn':fn=lambda q:side.ffn(side.n2(q));inp=u
  elif name=='post':fn=side.post;inp=u
  elif name=='gate':fn=lambda q:2*mod.gs(q.mean((2,3),keepdim=True)).sigmoid();inp=u
  else:
   fn=lambda q:torch.cat(mod(q,d),1);inp=s
  inp=inp.detach().clone().requires_grad_();rows=[]
  for k in range(13):
   mod.zero_grad(set_to_none=True);inp.grad=None;ev=[torch.cuda.Event(enable_timing=True) for _ in range(3)];ev[0].record();out=fn(inp);loss=out.float().square().mean();ev[1].record();loss.backward();ev[2].record();torch.cuda.synchronize()
   if k>=5:rows.append([ev[0].elapsed_time(ev[1]),ev[1].elapsed_time(ev[2])])
  row={'name':name,'forward_ms':statistics.mean(t[0] for t in rows),'backward_ms':statistics.mean(t[1] for t in rows),'total_ms':statistics.mean(sum(t) for t in rows),'note':'one S side only' if name!='full_E3' else 'both S/D sides'};report['runs'].append(row);print('PART',json.dumps(row),flush=True);(OUT/'result.json').write_text(json.dumps(report,indent=2));del mod,inp,out,loss;gc.collect();torch.cuda.empty_cache()
 for name in cases+cases[::-1]:run(name)
 report['complete']=True;(OUT/'result.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
