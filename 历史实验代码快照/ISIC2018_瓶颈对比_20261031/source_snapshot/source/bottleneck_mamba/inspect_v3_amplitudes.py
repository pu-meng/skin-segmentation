"""Read-only validation activation diagnostics for fixed V3 best checkpoints."""
from pathlib import Path
import sys,json,csv,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
from experiment_v3 import build_model,bottleneck_of
import ham_fast_runner as existing
from project_paths import PATHS
from pretrained_unet2d import sha256_file

def main():
 torch.set_num_threads(4);torch.cuda.set_device(1)
 out=ROOT/'log/瓶颈Mamba_第三版幅度检查_20261030_完整';out.mkdir(exist_ok=False)
 api=existing.baseline_api();data=Path(PATHS['active_split']).parent;ds=api.HAMDataset(data,'val')
 assert not ds.training and len(ds)==1010
 loader=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
 results={}
 for arm,folder in [('tinyvim_v3','瓶颈Mamba_版本3'),('unet_v3','UNet2D_瓶颈Mamba_版本3')]:
  run=ROOT/'log'/folder/'run';cp=run/'checkpoints/best.pth';val=json.loads((run/'validation/summary.json').read_text())
  digest=sha256_file(cp);assert digest==val['checkpoint_sha256'];assert sha256_file(data/'split.csv')==val['split_sha256']
  prov=json.loads((run/'provenance.json').read_text());code='Models/TinyViM/mamba_bottleneck_v3.py'
  assert sha256_file(ROOT/code)==prov['source_sha256'][code]
  model=build_model(arm,pretrained=False);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True);model=model.cuda().eval()
  module=bottleneck_of(model,arm);m=module.mixer;rows=[];current_ids=[];max_error=0.
  def observe(m,inputs,output):
   nonlocal max_error
   hidden,shape=inputs;b,l,_=hidden.shape;h,w=shape
   f,z=m.in_proj(hidden).transpose(1,2).chunk(2,dim=1)
   t=m.act(m.conv1d(f)[...,:l]);delta=F.softplus(m.dt_proj(m.x_proj(t.transpose(1,2)))).transpose(1,2)
   a=-torch.exp(m.A_log.float());gu=torch.exp(delta.float().unsqueeze(-1)*a[None,:,None,:]).sum(-1)
   u=m.spatial_conv(t.reshape(b,m.d_inner,h,w)).flatten(2)
   p=gu*u;q=m.D[None,:,None]*delta*t;gate=F.silu(z)
   reconstructed=m.out_proj(((p+q)*gate).transpose(1,2))
   err=(reconstructed-output).abs().max().item();max_error=max(max_error,err);assert err<1e-6
   tensors={'gu':gu,'delta':delta,'T':t,'U':u,'P':p,'Q':q,'P_after_z':p*gate,'Q_after_z':q*gate,'Y1':p+q,'mixer_output':output,'normalized_input':hidden}
   stats={}
   for name,x in tensors.items():
    flat=x.float().reshape(b,-1)
    for stat,v in [('absmean',flat.abs().mean(1)),('rms',flat.square().mean(1).sqrt()),('min',flat.min(1).values),('max',flat.max(1).values)]:stats[name+'_'+stat]=v.cpu().tolist()
   for name,condition in [('gu_below_001',gu<.01),('gu_above_159',gu>15.9),('delta_below_1e6',delta<1e-6)]:
    stats[name+'_fraction']=condition.float().reshape(b,-1).mean(1).cpu().tolist()
   energy=p.float().reshape(b,-1).square()
   stats['P_top1percent_energy_fraction']=(energy.topk(max(1,int(energy.shape[1]*.01)),dim=1).values.sum(1)/energy.sum(1).clamp_min(1e-30)).cpu().tolist()
   for j,id in enumerate(current_ids):
    row={'image_id':id,**{k:v[j] for k,v in stats.items()}}
    row['P_Q_rms_ratio']=row['P_rms']/max(row['Q_rms'],1e-30)
    row['P_Q_after_z_rms_ratio']=row['P_after_z_rms']/max(row['Q_after_z_rms'],1e-30)
    row['P_Q_cosine']=F.cosine_similarity(p[j].flatten(),q[j].flatten(),dim=0).item()
    rows.append(row)
  first=ds[0][0].unsqueeze(0).cuda()
  with torch.inference_mode():without=model(first)
  hook=m.register_forward_hook(observe);current_ids=[ds.rows[0]['image_id']]
  with torch.inference_mode():with_hook=model(first)
  assert torch.equal(without,with_hook);rows.clear()
  with torch.inference_mode():
   for x,_,ids,_ in loader:
    current_ids=list(ids);model(x.cuda())
    if len(rows)%200<8:print(arm,len(rows),flush=True)
  hook.remove();assert len(rows)==1010 and len({x['image_id'] for x in rows})==1010
  keys=list(rows[0]);fpath=out/f'{arm}_per_image.csv'
  with fpath.open('w') as f:
   writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
  summary={}
  for k in keys[1:]:
   values=np.array([r[k] for r in rows]);summary[k]={'mean':float(values.mean()),'p05':float(np.quantile(values,.05)),'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'min':float(values.min()),'max':float(values.max())}
  weights={}
  for name,x in [('D',m.D),('A_log',m.A_log),('A',-m.A_log.exp())]:
   arr=x.detach().float().cpu().numpy().ravel();weights[name]={'shape':list(x.shape),'min':float(arr.min()),'p05':float(np.quantile(arr,.05)),'median':float(np.median(arr)),'p95':float(np.quantile(arr,.95)),'max':float(arr.max()),'absmean':float(np.abs(arr).mean())}
  results[arm]={'checkpoint':str(cp),'checkpoint_sha256':digest,'selected_epoch':val['selected_epoch'],'images':len(rows),'prediction_identical_with_hook':True,'max_reconstruction_error':max_error,'statistics':summary,'weights':weights}
  (out/'results.json').write_text(json.dumps({'split':'val','test_accessed':False,'training_performed':False,'gpu':1,'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'results':results},ensure_ascii=False,indent=2)+'\n')
  print(arm,'RATIO',summary['P_Q_rms_ratio'],'GU',summary['gu_absmean'],'DELTA',summary['delta_absmean'],flush=True)
  del model,module,m;torch.cuda.empty_cache()
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
