"""Read trained TinyViM features; ICLR FFT; no architecture edit or training."""
import argparse,csv,hashlib,importlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from pretrained_unet2d import PretrainedUNet2D,sha256_file
from ham_fast_runner import validate_protocol
RUN=ROOT/'runs/protocol_v5_dedup_seed42/tinyvim_s/seed42'

def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def state_hash(model):
 h=hashlib.sha256()
 for n,x in model.state_dict().items():h.update(n.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class Validation(Dataset):
 def __init__(self,path):
  self.rows=sorted([r for r in csv.DictReader((path/'split.csv').open()) if r['split']=='val'],key=lambda r:r['image_id'])
  self.images=np.load(path/'images.npy',mmap_mode='r');self.masks=np.load(path/'masks.npy',mmap_mode='r')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  idx=int(self.rows[i]['cache_index'])
  return torch.tensor(self.images[idx].astype(np.float64)/255,dtype=torch.float32).permute(2,0,1),torch.tensor((self.masks[idx]>0).copy(),dtype=torch.bool)[None]

class Observer:
 def __init__(self,model):
  self.acc={};self.handles=[];self.blocks=[];self.active=None
  backbone=model.encoder.backbone
  for stage,net_idx in enumerate((0,2,4,6),1):
   for index,block in enumerate(backbone.network[net_idx]):
    name=f'stage{stage}/block{index:02d}';kind=type(block).__name__
    self.blocks.append(dict(name=name,stage=stage,index=index,kind=kind))
    self.pre(block,name+'/block_input');self.post(block,name+'/block_output')
    self.post(block.mlp,name+'/ffn_output')
    if kind=='LocalBlock':self.post(block.dwconv,name+'/conv_output')
    else:
     assert kind=='TViMBlock'
     self.handles.append(block.op.register_forward_pre_hook(self.activate(name)))
     self.handles.append(block.op.register_forward_hook(self.finish(name)))
     self.pre(block.op.local_conv,name+'/high_branch_input');self.post(block.op.local_conv,name+'/high_branch_output')
     self.handles.append(block.op.out_norm.register_forward_pre_hook(self.merge_hook(name)))
     self.pre(block.mlp,name+'/after_mixer_residual')
  self.scan_module=importlib.import_module(type(backbone.network[0][-1].op).__module__)
  self.original_scan=self.scan_module.cross_selective_scan
  def scan(*args,**kwargs):
   x=args[0] if args else kwargs['x'];name=self.active;assert name is not None
   self.add(name+'/scan_input',x);y=self.original_scan(*args,**kwargs)
   self.add(name+'/scan_output',y.permute(0,3,1,2));return y
  self.scan_module.cross_selective_scan=scan
 def pre(self,m,key):self.handles.append(m.register_forward_pre_hook(lambda m,a:self.add(key,a[0])))
 def post(self,m,key):self.handles.append(m.register_forward_hook(lambda m,a,y:self.add(key,y)))
 def activate(self,name):
  def hook(m,a):self.active=name
  return hook
 def finish(self,name):
  def hook(m,a,y):self.add(name+'/operator_output',y);self.active=None
  return hook
 def merge_hook(self,name):
  def hook(m,a):
   x=a[0];b,n,c=x.shape;h=math.isqrt(n);assert h*h==n
   self.add(name+'/scan_merge_pre_norm',x.transpose(1,2).reshape(b,c,h,h))
  return hook
 def add(self,key,x):
  assert x.ndim==4 and x.shape[-1]==x.shape[-2],(key,x.shape)
  x=x.detach().float();b,c,h,w=x.shape
  logamp=torch.fft.fftshift((torch.fft.fft2(x).abs()+1e-6).log(),dim=(-2,-1))
  s=logamp.double().sum((0,1)).cpu();ch=logamp[:,:8].double().sum(0).cpu()
  if key not in self.acc:self.acc[key]=dict(samples=0,channels=c,height=h,width=w,logsum=torch.zeros_like(s),channel_sum=torch.zeros_like(ch))
  d=self.acc[key];assert (d['channels'],d['height'],d['width'])==(c,h,w)
  d['samples']+=b;d['logsum']+=s;d['channel_sum']+=ch
 def close(self):
  self.scan_module.cross_selective_scan=self.original_scan
  for h in self.handles:h.remove()
 def results(self):
  out={}
  for key,d in self.acc.items():
   mean=d['logsum']/(d['samples']*d['channels']);curve=mean.diag()[d['height']//2:];curve=curve-curve[0]
   out[key]=dict(samples=d['samples'],shape=[d['channels'],d['height'],d['width']],relative_log_amplitude=curve.tolist(),high_end_relative_log_amplitude=float(curve[-1]),first8_mean_log_amplitude_maps=(d['channel_sum']/d['samples']).tolist())
  return out

def plots(out,stats,blocks):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(4,3,figsize=(13,12));chosen=[]
 for row,stage in enumerate(range(1,5)):
  local=[b for b in blocks if b['stage']==stage and b['kind']=='LocalBlock'][-1]['name']
  tv=[b for b in blocks if b['stage']==stage and b['kind']=='TViMBlock'][-1]['name'];chosen.append(dict(stage=stage,local=local,tinyvim=tv))
  groups=[(local,['block_input','conv_output'],'native Conv path'),(tv,['block_input','operator_output','block_output'],'native TinyViM block'),(tv,['scan_input','scan_merge_pre_norm','scan_output'],'native scan path')]
  for col,(name,points,title) in enumerate(groups):
   ax=axes[row,col]
   for point in points:
    d=stats[name+'/'+point];y=d['relative_log_amplitude'];ax.plot(np.linspace(0,1,len(y)),y,label=point)
   shape=stats[name+'/'+points[0]]['shape'];ax.set_title(f'Stage {stage}: {title}\n{name}, input {shape[0]}x{shape[1]}x{shape[2]}',fontsize=10)
   ax.set_xlabel('Normalized half-diagonal frequency');ax.set_ylabel('Relative log amplitude');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'native_frequency_curves.png',dpi=160);fig.savefig(out/'native_frequency_curves.svg');plt.close(fig)
 fig,axes=plt.subplots(1,4,figsize=(15,4))
 for col,stage in enumerate(range(1,5)):
  arm=[b for b in blocks if b['stage']==stage]
  vals=[stats[b['name']+'/block_output']['high_end_relative_log_amplitude']-stats[b['name']+'/block_input']['high_end_relative_log_amplitude'] for b in arm]
  axes[col].bar([b['index'] for b in arm],vals,color=['#cc8833' if b['kind']=='TViMBlock' else '#4477aa' for b in arm]);axes[col].axhline(0,color='gray',lw=.8);axes[col].set_title(f'Stage {stage}');axes[col].set_xlabel('Block index');axes[col].set_ylabel('Change in high-end relative log amplitude')
 fig.tight_layout();fig.savefig(out/'block_relative_changes.png',dpi=160);plt.close(fig);write(out/'displayed_blocks.json',chosen)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',choices=['best','final'],required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=4);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);device=torch.device(args.device);torch.cuda.set_device(device)
 config=validate_protocol(ROOT/'protocol_v5_dedup_seed42.json','tinyvim_s',check_cache=False)
 dataset=Validation(Path(config['data']['split_file']).parent);assert len(dataset)==1010
 checkpoint=RUN/'checkpoints'/(args.checkpoint+'.pth');digest=sha256_file(checkpoint);formal=json.loads((RUN/'validation/summary.json').read_text())
 if args.checkpoint=='best':assert digest==formal['checkpoint_sha256']
 epoch=formal['selected_epoch'] if args.checkpoint=='best' else 300
 model=PretrainedUNet2D('tinyvim_s',pretrained=False).to(device);model.load_state_dict(torch.load(checkpoint,map_location=device),strict=True);model.eval();model.requires_grad_(False)
 before=state_hash(model);sample=next(iter(DataLoader(dataset,batch_size=2)))[0].to(device)
 with torch.no_grad():reference=model(sample)
 observer=Observer(model)
 try:
  with torch.no_grad():observed=model(sample)
  error=(reference-observed).abs().max().item();assert error<1e-6,error
  observer.acc.clear()
  write(out/'provenance.json',dict(model='trained TinyViM-S + UNet decoder',checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_epoch=epoch,formal_training_epochs=300,split='val',sample_count=1010,input=[3,256,256],mode='eval + no_grad',source_sha256=sha256_file(Path(__file__)),external_source_locks=config['external_source_lock'],test_samples_accessed=False,training_performed=False,optimizer_created=False,hook_prediction_max_abs_error=error,fft='log(abs(fft2(x))+1e-6); spatial fftshift; sample/channel mean; positive half-diagonal minus DC'))
  write(out/'sample_ids.json',[r['image_id'] for r in dataset.rows]);loader=DataLoader(dataset,batch_size=args.batch,shuffle=False,num_workers=0)
  n=0;ds=0.;js=0.;started=time.monotonic()
  with torch.no_grad():
   for x,y in loader:
    x,y=x.to(device),y.to(device);pred=model(x)>=.5;inter=(pred&y).sum((1,2,3));pp=pred.sum((1,2,3));tt=y.sum((1,2,3));u=pp+tt-inter
    ds+=torch.where(pp+tt>0,2*inter/(pp+tt),1.).sum().item();js+=torch.where(u>0,inter/u,1.).sum().item();n+=len(x)
    write(out/'status.json',dict(status='extracting_features',checkpoint_epoch=epoch,images_completed=n,images_expected=1010,seconds=time.monotonic()-started))
    if n%100==0:print(f'{args.checkpoint}: {n}/1010',flush=True)
  assert before==state_hash(model)
  results=observer.results();assert all(d['samples']==1010 for d in results.values());observer.close()
  write(out/'frequency_statistics.json',results);write(out/'blocks.json',observer.blocks);rows=[]
  for b in observer.blocks:
   a=results[b['name']+'/block_input'];z=results[b['name']+'/block_output'];aa=a['high_end_relative_log_amplitude'];zz=z['high_end_relative_log_amplitude']
   rows.append(dict(**b,input_shape=str(a['shape']),input_high_end=aa,output_high_end=zz,change=zz-aa))
  with (out/'all_block_changes.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  plots(out,results,observer.blocks);dice=ds/n;iou=js/n
  batch1_verification=None
  if args.checkpoint=='best':
   dd=jj=0.;nn=0
   with torch.no_grad():
    for xx,yy in DataLoader(dataset,batch_size=1,shuffle=False):
     pr=model(xx.to(device))>=.5;yy=yy.to(device);ii=(pr&yy).sum((1,2,3));pp=pr.sum((1,2,3));tt=yy.sum((1,2,3));uu=pp+tt-ii
     dd+=torch.where(pp+tt>0,2*ii/(pp+tt),1.).sum().item();jj+=torch.where(uu>0,ii/uu,1.).sum().item();nn+=1
   batch1_verification=dict(dice=dd/nn,iou=jj/nn)
   assert abs(dd/nn-formal['per_image_mean']['dice'])<1e-6 and abs(jj/nn-formal['per_image_mean']['iou'])<1e-6
  write(out/'status.json',dict(status='complete',checkpoint_epoch=epoch,images_completed=n,feature_points=len(results),model_state_unchanged=True,hook_prediction_max_abs_error=error,per_image_dice=dice,per_image_iou=iou,best_metrics_match_formal_summary=args.checkpoint=='best',batch1_verification=batch1_verification,analysis_batch_size=args.batch,seconds=time.monotonic()-started,interpretation='native scan already receives frequency-decoupled input; not a vanilla Mamba preference experiment'))
  print((out/'status.json').read_text(),flush=True)
 finally:observer.close()
if __name__=='__main__':main()
