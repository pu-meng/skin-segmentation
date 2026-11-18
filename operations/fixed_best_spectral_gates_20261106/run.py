import os,sys,json,csv,time,hashlib,shutil,fcntl,signal,traceback,subprocess
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation')
O=Path('/media/8T/PuMengYu/diagnostics/fixed_best_spectral_gates_20261106')
OLD=Path('/media/8T/PuMengYu/diagnostics/new_completed_bypass_20261106')
os.environ['BD_ARM']='D4-B3D'
sys.path[:0]=[str(R),str(R/'source/comparison'),str(R/'source/bottleneck_mamba'),str(R/'source/three_module_impl_20261104')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ham_fast_runner as existing
api=existing.baseline_api()
from project_paths import PATHS
from source.e123_candidate_a_20261106.candidate import build_candidate as ba
from source.b3_e1_e31_20261105.candidate import build_candidate as bb
from source.bd_controls_20261105.candidate import build_candidate as bd
ARMS=['CandidateA','B3-E1-E3.1','D4-B3D-R2']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def build(a):return ba(pretrained=False) if a=='CandidateA' else bb(pretrained=False) if a=='B3-E1-E3.1' else bd(pretrained=False,count=3,med=True)
def sources():
 files={Path(__file__),Path(PATHS['active_split'])}
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(R)) and str(f).endswith('.py'):files.add(Path(f))
 return {str(p):sha(p) for p in sorted(files)}
def prepare():
 assert not O.exists(),'refuse overwrite';O.mkdir(parents=True)
 old=json.loads((OLD/'protocol.json').read_text());records={}
 for a in ARMS:
  if a=='CandidateA':
   s=R/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106/final_test/none/summary.json';v=json.loads(s.read_text());ck=Path(v['checkpoint']);expected=v['checkpoint_sha256'];ep=v['selected_epoch'];assert ep==173
   dest=O/a/'fixed_best.pth';dest.parent.mkdir();before=sha(ck);assert before==expected;shutil.copyfile(ck,dest);assert sha(ck)==sha(dest)==before
  else:
   v=old['checkpoints'][a];ck=Path(v['original_checkpoint']);dest=Path(v['snapshot']);expected=v['checkpoint_sha256'];ep=v['selected_epoch'];s=Path(v['summary']);assert sha(ck)==sha(dest)==expected
  m=build(a);msg=m.load_state_dict(torch.load(dest,map_location='cpu'),strict=True);assert not torch.cuda.is_initialized()
  records[a]={'original':str(ck),'snapshot':str(dest),'sha256':expected,'epoch':ep,'summary':str(s),'strict_load':str(msg)};del m
 protocol={'arms':records,'split':str(PATHS['active_split']),'split_sha256':sha(PATHS['active_split']),'images':1010,'batch_size':1,'mode':'eval no_grad FP32 none/noTTA','bands':{'radius':'sqrt(fx^2+fy^2)/sqrt(0.5^2+0.5^2), cycles/featurepixel','low':'0 <= r < 1/3','mid':'1/3 <= r < 2/3','high':'2/3 <= r <= 1'},'FFT':'full fft2 norm=ortho, per image/channel, float64 CPU; raw and per-channel spatial-demeaned','zero':'exact total energy == 0 excluded from ratios, counted separately; zero shell bins null','aggregation':'image equal-weight ratio distribution distinct from total-energy-weighted pooled ratio; channel ratios also summarized per image','shells':24,'physical_frequency_warning':'8x8 and 64x64 normalized featurepixel frequency are NOT same original-image physical frequency','attention':'not present in CandidateA; optional old HSA not included, no new training','shared_GPU':1,'protected':'never alter other jobs','deadline_seconds':86400,'sources_before':sources()}
 write(O/'protocol.json',protocol);print('CPU_STRICT_PASS',flush=True)
class Capture:
 def __init__(self,m,a):
  self.m=m;self.a=a;self.t={};self.g={};self.h=[];self.man={};self.install()
 def save(self,n,x,layout='NCHW'):
  if layout=='NLC':
   assert x.ndim==3 and x.shape[1]==64;x=x.transpose(1,2).reshape(x.shape[0],x.shape[2],8,8)
  assert x.ndim==4 and x.shape[0]==1,n
  self.t[n]=x.detach().float().cpu().numpy()[0].copy();self.man[n]={'shape':list(x.shape),'original_layout':layout,'stored_layout':'CHW','kind':'residual' if 'delta' in n or 'operator' in n else 'feature'}
 def gate(self,n,x,lo,hi):self.g[n]=(x.detach().float().cpu().numpy().reshape(-1).copy(),lo,hi)
 def hook(self,mod,fn):
  def observe(m,i,o):fn(m,i,o);return None
  self.h.append(mod.register_forward_hook(observe))
 def boundary(self,mod,n,layout='NCHW',res=False):
  def f(m,i,o):
   self.save(n+'.in',i[0],layout);self.save(n+'.out',o,layout)
   if res:self.save(n+'.delta',o-i[0],layout)
  self.hook(mod,f)
 def install(self):
  d=self.m.decoder;a=self.a
  self.hook(d.head,lambda m,i,o:setattr(self,'head',o.detach()))
  self.boundary(d.stages[0],'D0_fusion')
  if a=='CandidateA':
   e=d.e2;self.boundary(e,'E2',res=True);self.boundary(e.pre,'E2_pre');self.boundary(e.spatial,'Mamba_path',res=True)
   for j,b in enumerate(e.spatial):
    self.boundary(b,'Mamba_block'+str(j),res=True);self.boundary(b.mixer,'Mamba_operator'+str(j),'NLC')
   self.hook(e.channel,lambda m,i,o:self.gate('E2_channel_factor',2*o.sigmoid(),0,2))
   self.hook(e.select,lambda m,i,o:self.gate('E2_branch_weights',o.reshape(-1,2,128,1,1).softmax(1),0,1))
   self.boundary(e.fuse,'E2_weighted_fuse');self.boundary(e.output,'E2_correction_unscaled')
   self.hook(e.pre,lambda m,i,o:(self.save('E2_spatial_input',o.chunk(2,1)[0]),self.save('E2_channel_before',o.chunk(2,1)[1])))
   self.hook(e.channel,lambda m,i,o:self.save('E2_spectral_statistics',i[0].transpose(1,2).unsqueeze(-1)))
   def ch(m,i,o):
    v=torch.from_numpy(self.t['E2_channel_before']).to(o.device).unsqueeze(0);self.save('E2_channel_after',v*(2*o.sigmoid()).squeeze(-1)[:,:,None,None])
   self.hook(e.channel,ch)
   self.boundary(d.stages[3],'WPO',res=True);self.boundary(d.stages[3].wave,'WPO_operator')
   self.hook(d.e3,lambda m,i,o:(self.save('Skip_S',i[0]),self.save('Skip_D',i[1]),self.save('Skip_finalS',o[0]),self.save('Skip_finalD',o[1]),self.save('Skip_deltaS',o[0]-i[0]),self.save('Skip_deltaD',o[1]-i[1])))
   for side in ['s','d']:
    v=getattr(d.e3,side);self.boundary(v.pre,'Skip_'+side+'_pre');self.boundary(v.gen,'Skip_'+side+'_content');self.boundary(v.mix,'Skip_'+side+'_gated_content')
    self.hook(v.gen.select,lambda m,i,o,side=side:self.gate('Skip_'+side+'_branch_weights',o.softmax(1),0,1))
   for name in ['gs','gd']:self.hook(getattr(d.e3,name),lambda m,i,o,name=name:self.gate('Skip_'+name,2*o.sigmoid(),0,2))
  else:
   self.boundary(d.enhance,'B3',res=True)
   for j,b in enumerate(d.enhance):self.boundary(b,'B3_block'+str(j),res=True);self.boundary(b.mixer,'B3_operator'+str(j),'NLC')
   if a=='B3-E1-E3.1':
    self.boundary(d.stages[3],'WPO',res=True);self.boundary(d.stages[3].wave,'WPO_operator')
    e=d.refiner
    self.hook(e,lambda m,i,o:(self.save('Skip_S',i[0]),self.save('Skip_D',i[1]),self.save('Skip_finalS',o),self.save('Skip_finalD',i[1]),self.save('Skip_deltaS',o-i[0])))
    for name in ['global_path','local_fuse','g_global','g_local','decoder_project']:self.boundary(getattr(e,name),'Skip_'+name)
    for name in ['g_global','g_local']:
     self.hook(getattr(e,name),lambda m,i,o,name=name:self.gate('Skip_'+name+'_factor',1+torch.tanh(m.raw)*torch.sigmoid(m.proj(i[1])),0,2))
    self.hook(e.select,lambda m,i,o:self.gate('Skip_branch_weights',o.softmax(1),0,1))
   else:
    for j,stage in enumerate(d.stages):
     self.boundary(stage,'MedStage'+str(j))
     for k in [1,2]:self.boundary(stage[k],'Med'+str(j)+'_'+str(k),res=True)
 def clear(self):self.t={};self.g={};self.head=None
 def close(self):
  for h in self.h:h.remove()
def dist(v):
 v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
 return {'n':len(v),'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'p05':float(np.quantile(v,.05)),'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'max':float(v.max())} if len(v) else {'n':0}
def norm(x):return float(np.linalg.norm(x.ravel().astype(np.float64)))
def pair(x,y):
 nx,ny=norm(x),norm(y);return {'cosine':float(np.dot(x.ravel().astype(float),y.ravel().astype(float))/(nx*ny)) if nx and ny else None,'right_left_norm_ratio':ny/nx if nx else None,'delta_left_norm_ratio':norm(y-x)/nx if nx else None}
def spectral(x):
 x=x.astype(np.float64);h,w=x.shape[-2:];r=np.sqrt(np.fft.fftfreq(h)[:,None]**2+np.fft.fftfreq(w)[None,:]**2)/np.sqrt(.5)
 masks=[r<1/3,(r>=1/3)&(r<2/3),r>=2/3];idx=np.minimum((r*24).astype(int),23);counts=np.bincount(idx.ravel(),minlength=24)
 out={}
 for mode in ['raw','demean']:
  z=x if mode=='raw' else x-x.mean((-2,-1),keepdims=True);p=np.abs(np.fft.fft2(z,norm='ortho'))**2;en=p.sum((-2,-1));total=float(en.sum());zero=en==0
  band=np.stack([p[:,mask].sum(-1) for mask in masks],-1);rat=np.divide(band,en[:,None],out=np.full_like(band,np.nan),where=en[:,None]!=0)
  shell=np.array([float(p[:,idx==j].sum()) for j in range(24)])
  space=float((z*z).sum());pe=abs(total-space)/max(space,1e-300)
  out[mode]={'total_energy':total,'low_energy':float(band[:,0].sum()),'mid_energy':float(band[:,1].sum()),'high_energy':float(band[:,2].sum()),'low_ratio':float(band[:,0].sum()/total) if total else None,'mid_ratio':float(band[:,1].sum()/total) if total else None,'hf_ratio':float(band[:,2].sum()/total) if total else None,'channel_hf_mean':float(np.nanmean(rat[:,2])) if (~zero).any() else None,'channel_hf_p05':float(np.nanquantile(rat[:,2],.05)) if (~zero).any() else None,'channel_hf_p95':float(np.nanquantile(rat[:,2],.95)) if (~zero).any() else None,'zero_channels':int(zero.sum()),'channels':len(en),'parseval_relative_error':pe,'shell_energy':shell,'shell_frequency_counts':counts,'shell_mean_power':np.divide(shell,counts*x.shape[0],out=np.full(24,np.nan),where=counts!=0)}
  assert pe<1e-10 and np.isfinite(p).all()
 return out
class CSV:
 def __init__(self,p,fields):self.f=Path(p).open('w',newline='');self.w=csv.DictWriter(self.f,fieldnames=fields);self.w.writeheader()
 def row(self,r):self.w.writerow(r)
 def close(self):self.f.close()
def run():
 p=json.loads((O/'protocol_r3.json').read_text());assert {f:sha(f) for f in p['sources_before']}==p['sources_before']
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('deadline')));signal.alarm(86400)
 torch.set_num_threads(2);api.seed_all(42);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.14,0)
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'val');assert len(data)==1010 and not data.training
 ids=[v['image_id'] for v in data.rows];assert len(set(ids))==1010;write(O/'validation_identity.json',{'ids':ids,'split_sha256':sha(PATHS['active_split'])})
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 write(O/'gpu_before.json',{'pid':os.getpid(),'physical_gpu':1,'snapshot':subprocess.check_output(['nvidia-smi'],text=True)})
 for a in ARMS:
  if (O/a/'summary.json').exists():continue
  target=O/a;target.mkdir(exist_ok=True);m=build(a);m.load_state_dict(torch.load(p['arms'][a]['snapshot'],map_location='cpu'),strict=True);m.eval().requires_grad_(False).cuda();buf={n:v.cpu().clone() for n,v in m.named_buffers()}
  x=data[0][0].unsqueeze(0).cuda()
  with torch.no_grad():base=m(x).clone()
  cap=Capture(m,a)
  with torch.no_grad():hooked=m(x)
  assert torch.equal(base,hooked);assert torch.equal(torch.sigmoid(F.interpolate(cap.head,size=(256,256),mode='bilinear',align_corners=False)),hooked)
  assert all(np.isfinite(v).all() for v in cap.t.values());write(target/'node_manifest.json',cap.man)
  write(target/'first_image_preflight.json',{'passed':True,'image_id':ids[0],'original_probability_exact':True,'real_head_logits_sigmoid_exact':True,'nodes':len(cap.man),'layouts':'NLC explicitly B,L,C -> B,C,8,8, L=64; NCHW untouched','gates':list(cap.g),'pid':os.getpid()});print('FIRST_IMAGE_PASS',a,len(cap.man),flush=True)
  featurefields=['image_id','node','mode','total_energy','low_energy','mid_energy','high_energy','low_ratio','mid_ratio','hf_ratio','channel_hf_mean','channel_hf_p05','channel_hf_p95','zero_channels','channels','parseval_relative_error']
  fc=CSV(target/'per_image_spectrum.csv',featurefields);gc=CSV(target/'per_image_gates.csv',['image_id','gate','mean','std','min','max','p05','p95','lower_saturated_fraction','upper_saturated_fraction']);pc=CSV(target/'per_image_pairs.csv',['image_id','pair','cosine','right_left_norm_ratio','delta_left_norm_ratio']);ec=CSV(target/'implementation_rectangular_hf.csv',['image_id','channel','hf_ratio']);metrics=CSV(target/'per_image_dice.csv',['image_id','dice'])
  acc={};gacc={};pairs={};maxpe=0
  for i,(x,y,*extra) in enumerate(loader):
   cap.clear()
   with torch.no_grad():prob=m(x.cuda())
   pred=prob.detach().cpu().numpy()>=.5;gt=y.numpy()>=.5;tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum());dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.;metrics.row({'image_id':ids[i],'dice':dice})
   assert set(cap.t)==set(cap.man)
   for n,v in cap.t.items():
    assert np.isfinite(v).all(),n
    if n=='E2_spectral_statistics':continue
    for mode,s in spectral(v).items():
     key=n+'/'+mode
     if key not in acc:acc[key]={'rows':0,'hf':[],'low':[],'mid':[],'channel_hf':[],'energies':np.zeros(4),'shell_energy':np.zeros(24),'shell_mean_sum':np.zeros(24),'shell_counts':s['shell_frequency_counts'],'zero_channels':0,'zero_images':0,'channels_seen':0}
     b=acc[key];b['rows']+=1
     for k in ['hf','low','mid']:
      val=s['hf_ratio' if k=='hf' else k+'_ratio']
      if val is not None:b[k].append(val)
     if s['channel_hf_mean'] is not None:b['channel_hf'].append(s['channel_hf_mean'])
     b['energies']+=np.array([s['total_energy'],s['low_energy'],s['mid_energy'],s['high_energy']]);b['shell_energy']+=s['shell_energy'];b['shell_mean_sum']+=np.nan_to_num(s['shell_mean_power']);b['zero_channels']+=s['zero_channels'];b['zero_images']+=s['total_energy']==0;b['channels_seen']+=s['channels'];maxpe=max(maxpe,s['parseval_relative_error'])
     fc.row(dict(image_id=ids[i],node=n,mode=mode,**{k:s[k] for k in featurefields[3:]}))
   if a=='CandidateA':
    v=cap.t['E2_channel_before'].astype(float);power=np.abs(np.fft.fft2(v,norm='ortho'))**2;h,w=v.shape[-2:];mask=(abs(np.fft.fftfreq(h))[:,None]<=.25)&(abs(np.fft.fftfreq(w))[None,:]<=.25);eh=power[:,~mask].sum(-1);ee=power.sum((-2,-1));rr=eh/(ee+1e-8)
    for j,r in enumerate(rr):ec.row({'image_id':ids[i],'channel':j,'hf_ratio':r})
   for n,(v,lo,hi) in cap.g.items():
    assert np.isfinite(v).all();gacc.setdefault(n,[]).append(v);tol=.01*(hi-lo);s=dist(v);gc.row({'image_id':ids[i],'gate':n,**{k:s[k] for k in ['mean','std','min','max','p05','p95']},'lower_saturated_fraction':float((v<=lo+tol).mean()),'upper_saturated_fraction':float((v>=hi-tol).mean())})
   ps={}
   for n,v in cap.t.items():
    if n.endswith('.delta') and n[:-6]+'.in' in cap.t:ps[n]=pair(cap.t[n[:-6]+'.in'],v)
   if a=='CandidateA':
    ps['E2_spatial_vs_channel']=pair(cap.t['Mamba_path.out'],cap.t['E2_channel_after'])
    weighted=cap.t['E2_weighted_fuse.in'];ps['E2_weighted_spatial_vs_channel']=pair(weighted[:128],weighted[128:]);ps['E2_weighted_spatial_inputnorm']=pair(cap.t['E2_spatial_input'],weighted[:128]);ps['E2_weighted_channel_inputnorm']=pair(cap.t['E2_channel_before'],weighted[128:])
    for side,raw in [('s','Skip_S'),('d','Skip_D')]:
     ps['Skip_'+side+'_gated_inputnorm']=pair(cap.t[raw],cap.t['Skip_'+side+'_gated_content.in'])
     ps['Skip_'+side+'_final']=pair(cap.t[raw],cap.t['Skip_final'+('S' if side=='s' else 'D')])
    ps['Skip_K_vs_V']=pair(cap.t['Skip_s_content.out'],cap.t['Skip_d_content.out'])
   elif a=='B3-E1-E3.1':
    ps['Skip_global_vs_local']=pair(cap.t['Skip_global_path.out'],cap.t['Skip_local_fuse.out']);ps['Skip_gated_global_vs_local']=pair(cap.t['Skip_g_global.out'],cap.t['Skip_g_local.out']);ps['Skip_final']=pair(cap.t['Skip_S'],cap.t['Skip_finalS'])
    weights=cap.g['Skip_branch_weights'][0].reshape(2,64,64)
    ps['Skip_weighted_global_inputnorm']=pair(cap.t['Skip_S'],cap.t['Skip_g_global.out']*weights[0]);ps['Skip_weighted_local_inputnorm']=pair(cap.t['Skip_S'],cap.t['Skip_g_local.out']*weights[1])
   for n,s in ps.items():pairs.setdefault(n,[]).append(s);pc.row({'image_id':ids[i],'pair':n,**s})
   if i%50==0:
    for c in [fc,gc,pc,ec,metrics]:c.f.flush()
    write(O/'status.json',{'status':'analyzing','arm':a,'completed_images':i+1,'expected':1010,'pid':os.getpid(),'updated':time.time()});print('PROGRESS',a,i+1,flush=True)
  for c in [fc,gc,pc,ec,metrics]:c.close()
  summary={}
  for k,b in acc.items():
   assert b['rows']==1010;kcounts=b['shell_counts'];ee=b['energies'];summary[k]={'images':b['rows'],'image_equal_weight_hf_ratio':dist(b['hf']),'image_equal_weight_low_ratio':dist(b['low']),'image_equal_weight_mid_ratio':dist(b['mid']),'image_equal_weight_channel_mean_hf_ratio':dist(b['channel_hf']),'total_energy_sum':float(ee[0]),'band_energy_sum':ee[1:].tolist(),'energy_weighted_band_ratio':(ee[1:]/ee[0]).tolist() if ee[0] else None,'zero_channels':b['zero_channels'],'channels_seen':b['channels_seen'],'zero_images':b['zero_images'],'shell_frequency_counts':kcounts.tolist(),'shell_mean_power_image_mean':[float(v) if c else None for v,c in zip(b['shell_mean_sum']/1010,kcounts)],'shell_energy_sum':b['shell_energy'].tolist()}
  gates={}
  for n,ls in gacc.items():
   v=np.stack(ls);lo,hi=cap.g[n][1:];gates[n]={'all_sample_channel_distribution':dist(v.ravel()),'image_mean_distribution':dist(v.mean(1)),'per_channel_mean':v.mean(0).tolist(),'per_channel_std_across_images':v.std(0).tolist(),'lower_saturated_fraction':float((v<=lo+.01*(hi-lo)).mean()),'upper_saturated_fraction':float((v>=hi-.01*(hi-lo)).mean()),'range':[lo,hi]}
  pairsummary={n:{k:dist([s[k] for s in ls if s[k] is not None]) for k in ['cosine','right_left_norm_ratio','delta_left_norm_ratio']} for n,ls in pairs.items()}
  assert all(torch.equal(v.detach().cpu(),buf[n]) for n,v in m.named_buffers());cap.close();del m;torch.cuda.empty_cache()
  write(target/'summary.json',{'complete':True,'arm':a,'images':1010,'spectrum':summary,'gates':gates,'pairs':pairsummary,'maximum_parseval_relative_error':maxpe,'buffers_unchanged':True})
  print('ARM_COMPLETE',a,flush=True)
 after=sources();assert after==p['sources_before'];assert sha(PATHS['active_split'])==p['split_sha256']
 for v in p['arms'].values():assert sha(v['snapshot'])==sha(v['original'])==v['sha256']
 write(O/'sources_after.json',after);finalize()
def finalize():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 summaries={a:json.loads((O/a/'summary.json').read_text()) for a in ARMS};p=json.loads((O/'protocol_r3.json').read_text());audit={};correlations={}
 for a,s in summaries.items():
  with (O/a/'per_image_spectrum.csv').open() as f:rows=list(csv.DictReader(f))
  groups={}
  for r in rows:groups.setdefault((r['node'],r['mode']),set()).add(r['image_id'])
  ids=set(json.loads((O/'validation_identity.json').read_text())['ids']);assert all(v==ids and len(v)==1010 for v in groups.values());assert all(v['images']==1010 for v in s['spectrum'].values());audit[a]={'node_mode_groups':len(groups),'all_groups_exact_1010_unique_ids':True,'max_parseval_relative_error':s['maximum_parseval_relative_error'],'finite':True}
  fig,ax=plt.subplots(figsize=(12, max(6,len(s['spectrum'])/4)))
  raw=[k for k in s['spectrum'] if k.endswith('/raw')];names=[k[:-4] for k in raw];vals=np.array([s['spectrum'][k]['energy_weighted_band_ratio'] or [0,0,0] for k in raw]);left=np.zeros(len(raw))
  for j,label in enumerate(['Low','Mid','High']):ax.barh(names,vals[:,j],left=left,label=label);left+=vals[:,j]
  ax.legend();ax.set_xlabel('Pooled energy-weighted radial band ratio (raw)');ax.set_title(a+' | HAM val1010 | fixed best | noTTA');fig.tight_layout();fig.savefig(O/a/'bands.png',dpi=150);plt.close(fig)
  focus=['E2.in','E2.out','E2.delta','WPO.in','WPO.out','WPO.delta','B3.in','B3.out','B3.delta','Skip_S','Skip_finalS','Skip_deltaS','MedStage0.in','MedStage0.out']
  fig,axs=plt.subplots(1,2,figsize=(14,5))
  for j,mode in enumerate(['raw','demean']):
   for n in focus:
    v=s['spectrum'].get(n+'/'+mode)
    if v:axs[j].plot((np.arange(24)+.5)/24,v['shell_mean_power_image_mean'],marker='.',label=n)
   axs[j].set_yscale('symlog',linthresh=1e-10);axs[j].set_title(mode+' | per-shell frequency-point mean power');axs[j].set_xlabel('Normalized cycles/featurepixel radius');axs[j].legend(fontsize=7)
  fig.tight_layout();fig.savefig(O/a/'radial_curves.png',dpi=160);plt.close(fig)
  fig,ax=plt.subplots(figsize=(12,5));nn=[n for n in focus if n+'/raw' in s['spectrum']]
  for mode in ['raw','demean']:ax.plot(nn,[s['spectrum'][n+'/'+mode]['image_equal_weight_hf_ratio'].get('mean',0) for n in nn],marker='o',label=mode)
  ax.legend();ax.set_ylabel('Image equal-weight HF-RATIO mean');ax.tick_params(axis='x',rotation=60);fig.tight_layout();fig.savefig(O/a/'hf_ratio.png',dpi=160);plt.close(fig)
  if s['gates']:
   fig,axs=plt.subplots(len(s['gates']),1,figsize=(12,3*len(s['gates'])),squeeze=False)
   for ax,(n,v) in zip(axs[:,0],s['gates'].items()):ax.plot(v['per_channel_mean']);ax.fill_between(np.arange(len(v['per_channel_mean'])),np.array(v['per_channel_mean'])-np.array(v['per_channel_std_across_images']),np.array(v['per_channel_mean'])+np.array(v['per_channel_std_across_images']),alpha=.3);ax.set_title(n+' channel/position mean +/- across-image std');ax.set_ylim(v['range'])
   fig.tight_layout();fig.savefig(O/a/'gates.png',dpi=130);plt.close(fig)
  fig,ax=plt.subplots(figsize=(12,max(5,len(s['pairs'])/3)));names=list(s['pairs']);values=[s['pairs'][n]['right_left_norm_ratio'].get('median',0) for n in names];ax.barh(names,values);ax.set_xscale('symlog',linthresh=.001);ax.set_xlabel('Median weighted correction/branch norm divided by actual reference input norm');fig.tight_layout();fig.savefig(O/a/'branch_amplitudes.png',dpi=150);plt.close(fig)
  if a!='CandidateA':
   full=OLD/a/'none/full/per_image_metrics.csv'
   with full.open() as f:fr={r['image_id']:float(r['dice']) for r in csv.DictReader(f)}
   with (O/a/'per_image_pairs.csv').open() as f:pr=list(csv.DictReader(f))
   groups=['B3_off','E1_WPO_off','E31_off'] if a=='B3-E1-E3.1' else ['B3_off','Med_residual_off_keep_projection']
   correlations[a]={}
   for g in groups:
    with (OLD/a/'none'/g/'per_image_metrics.csv').open() as f:dr={r['image_id']:float(r['dice'])-fr[r['image_id']] for r in csv.DictReader(f)}
    correlations[a][g]={}
    for n in s['pairs']:
     rr=[r for r in pr if r['pair']==n and r['right_left_norm_ratio']!=''];xx=np.array([float(r['right_left_norm_ratio']) for r in rr]);yy=np.array([dr[r['image_id']] for r in rr]);assert len(rr)==1010
     correlations[a][g][n]={'n':1010,'pearson':float(np.corrcoef(xx,yy)[0,1]) if xx.std() and yy.std() else None,'bypass_minus_same_round_full_dice':dist(yy),'description':'descriptive association only; not causal, many comparisons; existing bypass full reused'}
   with (O/a/'per_image_dice.csv').open() as f:nr={r['image_id']:float(r['dice']) for r in csv.DictReader(f)}
   audit[a]['max_dice_difference_vs_existing_full']=max(abs(nr[k]-fr[k]) for k in ids)
 write(O/'bypass_correlations.json',correlations);write(O/'audit.json',audit)
 lines=['# 固定 best：频带、门控与分支幅度诊断','', 'HAM 验证集1010张；batch1、FP32、eval/no_grad、无TTA；没有新增训练。','', '|对象|best轮次|节点×口径|最大Parseval相对误差|','|---|---:|---:|---:|']
 for a,v in audit.items():lines.append(f"|{a}|{p['arms'][a]['epoch']}|{v['node_mode_groups']}|{v['max_parseval_relative_error']:.3g}|")
 lines+=['','## 读图入口','各对象目录：bands.png（能量加权频带）、hf_ratio.png（样本等权HF比率）、radial_curves.png（每环频点平均功率）、gates.png（存在动态门控时）、branch_amplitudes.png（幅度）。summary.json保留全部未舍入读数。','', '## 定义与限制','完整fft2、norm=ortho，逐图逐通道计算；半径=sqrt(fx²+fy²)/sqrt(.5²+.5²)，低[0,1/3)、中[1/3,2/3)、高[2/3,1]。这是预先固定的操作性频带，不代表语义高频或病灶边界恢复。','raw与逐通道去空间均值demean并列；零能量通道从比率排除并单独计数；空频率环输出null。shell_mean_power按每环频点数及通道数归一。','样本等权统计与总能量加权比率分列。8×8与64×64的featurepixel频率不可当作原图相同物理频率。算子输出、完整残差输出、加权修正分别保存。','门控范围/1%边缘饱和比例、跨图每通道标准差及加权分支幅度只是描述；gate大不等于有益，接近零不自动死支，相似不自动冗余。','既有旁路Dice关联仅描述，不据此决定结构有无更好；未新增attention参照，也没有虚构CandidateA里不存在的attention。','', '## 主要接口读数','|对象/接口|raw HF样本均值|demean HF样本均值|raw总能量加权HF|','|---|---:|---:|---:|']
 for a,s in summaries.items():
  for n in ['E2.in','E2.out','E2.delta','B3.in','B3.out','B3.delta','WPO.delta','Skip_deltaS','MedStage0.out']:
   if n+'/raw' not in s['spectrum']:continue
   r=s['spectrum'][n+'/raw'];d=s['spectrum'][n+'/demean'];lines.append(f"|{a}/{n}|{r['image_equal_weight_hf_ratio'].get('mean',0):.6g}|{d['image_equal_weight_hf_ratio'].get('mean',0):.6g}|{(r['energy_weighted_band_ratio'] or [0,0,0])[2]:.6g}|")
 lines+=['','结构化证据：protocol.json、validation_identity.json、每对象node_manifest.json/first_image_preflight.json/per_image_spectrum.csv/per_image_gates.csv/per_image_pairs.csv/summary.json、bypass_correlations.json、audit.json。实现对应的原矩形mask HF另见CandidateA/implementation_rectangular_hf.csv，不能与径向HF混用。','源与权重前后SHA、BN及其他buffers不变、首图hook输出与原forward和实际head logits一致均已验收。无全量特征落盘。']
 (O/'报告.md').write_text('\n'.join(lines)+'\n');write(O/'results.json',{'complete':True,'arms':ARMS,'images_per_arm':1010,'audit':audit,'sources_and_checkpoints_unchanged':True});write(O/'status.json',{'status':'complete','arms':ARMS,'updated':time.time(),'pid':os.getpid()});print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   with (O/'RUNNING.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(O/'pid').write_text(str(os.getpid()));run()
 except BaseException as e:
  if O.exists():write(O/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});write(O/'status.json',{'status':'failed_blocked','error':repr(e),'pid':os.getpid()})
  raise
