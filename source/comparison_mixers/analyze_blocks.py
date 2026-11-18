"""Analyze trained sequential blocks at explicitly named boundaries; never train."""
import argparse,csv,json,math,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from models import PureMixerUNet
from experiment import Images,amp,sha,write,check_protocol

class Observer:
    def __init__(self,model):
        self.handles=[];self.data={};self.active=True
        for si,stage in enumerate(model.stages,1):
            for bi,block in enumerate(stage,1):
                name=f'stage{si}/block{bi:02d}'
                self.pre(block,name+'/block_input')
                self.pre(block.mixer,name+'/mixer_input')
                self.post(block.mixer,name+'/mixer_output')
                self.pre(block.norm2,name+'/after_mixer_residual')
                self.post(block.ffn,name+'/ffn_output')
                self.post(block,name+'/block_output')
    def pre(self,module,key):
        def hook(m,args):self.add(key,args[0])
        self.handles.append(module.register_forward_pre_hook(hook))
    def post(self,module,key):
        def hook(m,args,y):self.add(key,y)
        self.handles.append(module.register_forward_hook(hook))
    @torch.no_grad()
    def add(self,key,x):
        if not self.active:return
        x=x.detach().float();b,c,h,w=x.shape
        fft=torch.fft.fft2(x,norm='ortho')
        log=torch.fft.fftshift(fft.abs().clamp_min(1e-12).log(),dim=(-2,-1))
        curve=log.mean((0,1)).diagonal()[h//2:];curve=curve-curve[0]
        centered=x-x.mean((-2,-1),keepdim=True)
        power=torch.fft.fft2(centered,norm='ortho').abs().square()
        f=torch.fft.fftfreq(h,device=x.device);radius=(f[:,None].square()+f[None,:].square()).sqrt()
        total=power.sum((1,2,3));valid=total>1e-20
        low=power[:,:,(radius>0)&(radius<=.125)].sum((1,2))
        high=power[:,:,radius>=.25].sum((1,2))
        # Invalid constant images contribute no ratio and retain an explicit valid count.
        values=torch.stack([(low/total.clamp_min(1e-20)*valid).sum(),(high/total.clamp_min(1e-20)*valid).sum(),valid.sum(),total.sum()]).double()
        if key not in self.data:self.data[key]={'shape':[c,h,w],'samples':0,'sums':torch.zeros_like(values),'curve':torch.zeros_like(curve,dtype=torch.float64)}
        d=self.data[key];d['samples']+=b;d['sums']+=values;d['curve']+=curve.double()*b
    def result(self):
        out={}
        for key,d in self.data.items():
            lo,hi,valid,energy=d['sums'].cpu().tolist();assert valid>0
            out[key]=dict(shape=d['shape'],samples=d['samples'],valid_images=int(valid),low_share=lo/valid,high_share=hi/valid,non_dc_power_mean=energy/d['samples'],relative_log_amplitude=(d['curve']/d['samples']).cpu().tolist())
        return out
    def close(self):
        for h in self.handles:h.remove()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',required=True,choices=['conv','transformer','mamba']);p.add_argument('--device',default='cuda:1');p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    protocol=check_protocol();root=Path(__file__).resolve().parents[2];start=time.monotonic();torch.set_num_threads(4)
    ckpath=root/'log/Conv_Attention_Mamba_整网对比/runs'/a.kind/'seed42/final.pth'
    ck=torch.load(ckpath,map_location='cpu');assert ck['formal_training'] and ck['epoch']==300 and ck['seed']==42 and ck['kind']==a.kind
    assert ck['protocol_sha256']==sha(root/'log/Conv_Attention_Mamba_整网对比/protocol.json')
    a.output.mkdir(parents=True,exist_ok=False)
    device=torch.device(a.device);torch.cuda.set_device(device)
    model=PureMixerUNet(a.kind,42).to(device).eval();model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False)
    ids=json.loads((root/'log/UNet_冻结模块频谱对比/runs_trajectory/seed42/trajectory/epoch030/spectral_sample_ids.json').read_text())
    if a.limit:ids=ids[:a.limit]
    ds=Images('val');mapping={r['image_id']:i for i,r in enumerate(ds.rows)};subset=Subset(ds,[mapping[i] for i in ids]);loader=DataLoader(subset,batch_size=1,shuffle=False)
    x=ds[mapping[ids[0]]][0][None].to(device)
    with torch.no_grad(),amp(device):reference=model(x)
    observer=Observer(model);observer.active=False
    with torch.no_grad(),amp(device):hooked=model(x)
    assert torch.equal(reference,hooked);observer.active=True
    sample_output=None
    with torch.no_grad():
        for index,(x,_) in enumerate(loader,1):
            with amp(device):pred=model(x.to(device))
            if index==1:sample_output=pred.cpu()
            if index==1 or index%32==0:
                write(a.output/'status.json',dict(status='extracting',kind=a.kind,images_completed=index,images_total=len(ids),elapsed_seconds=time.monotonic()-start))
                print(a.kind,index,len(ids),round(time.monotonic()-start,1),flush=True)
    observer.close();stats=observer.result();assert len(stats)==126 and all(d['samples']==len(ids) for d in stats.values())
    write(a.output/'frequency_statistics.json',stats);write(a.output/'sample_ids.json',ids)
    rows=[]
    for stage,depth in enumerate([3,3,9,6],1):
        for block in range(1,depth+1):
            prefix=f'stage{stage}/block{block:02d}'
            for boundary,d in [('mixer',('mixer_input','mixer_output')),('whole_block',('block_input','block_output'))]:
                x,y=[stats[prefix+'/'+point] for point in d]
                rows.append(dict(kind=a.kind,stage=stage,block=block,boundary=boundary,input_low=100*x['low_share'],output_low=100*y['low_share'],delta_low_pp=100*(y['low_share']-x['low_share']),input_high=100*x['high_share'],output_high=100*y['high_share'],delta_high_pp=100*(y['high_share']-x['high_share'])))
    with (a.output/'block_changes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(a.output/'status.json',dict(status='complete',kind=a.kind,seed=42,checkpoint_epoch=300,checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),samples=len(ids),feature_points=len(stats),hook_predictions_identical=True,training_performed=False,test_samples_accessed=False,elapsed_seconds=time.monotonic()-start))
    write(a.output/'provenance.json',dict(analysis_source_sha256=sha(Path(__file__)),model_source_sha256=sha(Path(__file__).parent/'models.py'),checkpoint_sha256=sha(ckpath),sample_rule='same ordered 384 validation image IDs as prior frozen-UNet probe; distinct models and input features',precision='original BF16 forward and FP32 spectral analysis',thresholds_cycles_per_feature_pixel={'low':'.0 < r <= .125','high':'r >= .25'},zero_non_dc_images_excluded_per_feature=True))

if __name__=='__main__':main()
