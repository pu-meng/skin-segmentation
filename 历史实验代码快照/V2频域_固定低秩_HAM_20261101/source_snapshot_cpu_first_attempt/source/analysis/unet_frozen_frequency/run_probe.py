"""Paired Conv/attention/Mamba diagnostic on frozen V5 UNet-2D features."""
import argparse
import csv
import hashlib
import json
import random
import os
import time
from pathlib import Path
import sys as _project_sys
_PROJECT = Path(__file__).resolve().parents[3]
_project_sys.path.insert(0, str(_PROJECT / 'source/comparison'))
from project_paths import CACHE, original_source, verify_runtime, current_protocol
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from controlled_probe import ROOT,FrozenUNet,Probe,loss_fn
from spectral_metrics import measure,self_check,band_response

HERE=_PROJECT / 'log/UNet_冻结模块频谱对比'
KINDS=('conv','attention','mamba')
BASE=_PROJECT/'log/八模型对比/runs/protocol_v5_dedup_seed42/unet2d/seed42'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


class Images(Dataset):
    def __init__(self,split,limit=None):
        protocol=current_protocol(_PROJECT/'source/config/protocol_v5_dedup_seed42.json')
        path=Path(protocol['data']['split_file'])
        assert sha(path)==protocol['data']['split_sha256']
        rows=list(csv.DictReader(path.open()))
        self.rows=sorted([r for r in rows if r['split']==split],key=lambda r:r['image_id'])
        if limit:self.rows=self.rows[:limit]
        self.images=np.load(CACHE/'images.npy',mmap_mode='r')
        self.masks=np.load(CACHE/'masks.npy',mmap_mode='r')

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        r=self.rows[i];idx=int(r['cache_index'])
        x=torch.tensor(self.images[idx].astype(np.float64)/255,dtype=torch.float32).permute(2,0,1)
        y=torch.tensor((self.masks[idx]>0).copy(),dtype=torch.float32)[None]
        return x,y


@torch.no_grad()
def validate(base,probes,loader,device):
    sums={k:dict(loss=0.,dice=0.,iou=0.,samples=0) for k in ('identity',)+KINDS}
    for x,y in loader:
        x,y=x.to(device),y.to(device);f=base.encode(x)
        for k in sums:
            b=f[-1] if k=='identity' else probes[k](f[-1])
            pred=base.decode(b,f);p=pred>=.5;t=y>=.5
            inter=(p&t).sum((1,2,3));ps=p.sum((1,2,3));ts=t.sum((1,2,3));union=ps+ts-inter
            dice=torch.where(ps+ts>0,2*inter/(ps+ts),1.)
            iou=torch.where(union>0,inter/union,1.)
            d=sums[k];d['loss']+=loss_fn(pred,y).item()*len(x);d['dice']+=dice.sum().item();d['iou']+=iou.sum().item();d['samples']+=len(x)
    return {k:{n:v/d['samples'] for n,v in d.items() if n!='samples'} for k,d in sums.items()}


def spectral_analysis(base,probes,dataset,device,out):
    # Same fixed random 384 validation images; independent from checkpoint choice.
    rng=np.random.default_rng(20261025);indices=rng.choice(len(dataset),min(384,len(dataset)),replace=False).tolist()
    write(out/'spectral_sample_ids.json',[dataset.rows[i]['image_id'] for i in indices])
    loader=DataLoader(torch.utils.data.Subset(dataset,indices),batch_size=8,shuffle=False)
    totals={};maps={};count=0;responses={k:{} for k in KINDS}
    with torch.no_grad():
        for batch_index,(x,_) in enumerate(loader):
            f=base.encode(x.to(device));count+=len(x)
            for kind,probe in probes.items():
                _,features=probe(f[-1],capture=True)
                sensitivity=band_response(probe.mixer,features['mixer_input'],seed=20261025+batch_index)
                for metric,value in sensitivity.items():
                    responses[kind][metric]=responses[kind].get(metric,0)+value*len(x)
                for name,x in features.items():
                    key=kind+'/'+name;result=measure(x)
                    if key not in totals:totals[key]={}
                    for metric,value in result.items():
                        if metric in ('samples','zero_non_dc_images') or value is None:continue
                        a=np.asarray(value)*len(x)
                        totals[key][metric]=totals[key].get(metric,0)+a
                    logmap=torch.fft.fftshift(torch.fft.fft2(x.float(),norm='ortho').abs().clamp_min(1e-12).log(),dim=(-2,-1))
                    maps[key]=maps.get(key,0)+logmap[:,:8].sum(0).cpu()
    for key in totals:
        totals[key]={k:(v/count).tolist() for k,v in totals[key].items()}
    write(out/'spectra.json',dict(samples=count,statistics=totals,trained_probe_checkpoints=True))
    write(out/'band_response.json',dict(samples=count,relative_rms=.01,interpretation='local mixer sensitivity to equal-RMS feature perturbations, not task usefulness',results={k:{n:v/count for n,v in d.items()} for k,d in responses.items()}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(10,10))
    for row,kind in enumerate(KINDS):
        for col,names in enumerate([('mixer_input','mixer_output'),('raw_input','residual_output')]):
            for name in names:
                curve=totals[kind+'/'+name]['diagonal_relative_log_amplitude'];axes[row,col].plot(np.linspace(0,1,len(curve)),curve,label=name)
            axes[row,col].set_title(kind+(' / mixer' if col==0 else ' / residual'))
            axes[row,col].set_xlabel('Positive diagonal frequency (normalized)');axes[row,col].set_ylabel('Relative log amplitude');axes[row,col].legend()
    fig.tight_layout();fig.savefig(out/'spectral_curves.png',dpi=180);plt.close(fig)
    keys=[k+'/'+name for k in KINDS for name in ('mixer_input','mixer_output')]
    data=torch.stack([maps[k]/count for k in keys]);lo,hi=torch.quantile(data,torch.tensor([.02,.98])).tolist()
    fig,axes=plt.subplots(6,8,figsize=(14,10))
    for row,key in enumerate(keys):
        for ch in range(8):
            axes[row,ch].imshow(data[row,ch],vmin=lo,vmax=hi,cmap='viridis');axes[row,ch].set_xticks([]);axes[row,ch].set_yticks([])
        axes[row,0].set_ylabel(key,fontsize=8)
    fig.tight_layout();fig.savefig(out/'spectral_maps.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','train'],default='preflight')
    p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    verify_runtime()
    protocol_path=HERE/'protocol.json'
    assert sha(protocol_path)==(HERE/'protocol.sha256').read_text().strip()
    protocol=json.loads(protocol_path.read_text())
    assert all(sha(original_source('/home/PuMengYu/experiments/MambaLiteUNet/reproduction/频谱对照试验/'+name))==expected for name,expected in protocol['source_sha256'].items())
    assert sha(original_source('/home/PuMengYu/experiments/MambaLiteUNet/reproduction/unet2d.py'))==protocol['unet_source_sha256']
    if args.mode=='train':
        assert args.epochs==protocol['training']['epochs'] and args.seed in protocol['training']['seeds']
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'status.json',dict(status='starting',mode=args.mode,pid=os.getpid(),epoch_completed=0,epochs_requested=args.epochs if args.mode=='train' else 1))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    checkpoint=BASE/'checkpoints/best.pth';expected=json.loads((BASE/'validation/summary.json').read_text())['checkpoint_sha256']
    assert sha(checkpoint)==expected
    base=FrozenUNet(checkpoint).to(device);probes={k:Probe(k,args.seed).to(device) for k in KINDS}
    optimizers={k:torch.optim.AdamW([p for p in v.parameters() if p.requires_grad],lr=1e-3,weight_decay=.01) for k,v in probes.items()}
    initial={n:v.detach().clone() for n,v in base.base.state_dict().items()}
    common={k:{n:v.detach().cpu() for n,v in probe.state_dict().items() if n.startswith(('down.','up.','ffn.'))} for k,probe in probes.items()}
    assert all(torch.equal(v,common['conv'][n]) for k,d in common.items() for n,v in d.items())
    write(args.output/'provenance.json',dict(mode=args.mode,seed=args.seed,device=str(device),torch=torch.__version__,base_checkpoint=str(checkpoint),base_sha256=expected,
        protocol_sha256=sha(protocol_path),
        source_sha256={f:sha(Path(__file__).parent/f) for f in ['controlled_probe.py','spectral_metrics.py','run_probe.py']},
        trainable_parameters={k:sum(p.numel() for p in v.parameters() if p.requires_grad) for k,v in probes.items()},
        full_parameters={k:sum(p.numel() for p in base.parameters())+sum(p.numel() for p in v.parameters()) for k,v in probes.items()},
        pretrained_backbone_validation_selected=True,test_samples_accessed=False,augmentation='none for all arms',spectral_metric_tests=self_check()))
    train=Images('train',8 if args.mode=='preflight' else None)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(train,batch_size=2 if args.mode=='preflight' else 8,shuffle=True,generator=generator,num_workers=0)
    validation=Images('val',8 if args.mode=='preflight' else None)
    vloader=DataLoader(validation,batch_size=8,shuffle=False)
    before=validate(base,probes,vloader,device)
    history=[];best={k:float('inf') for k in KINDS}
    epochs=1 if args.mode=='preflight' else args.epochs
    for epoch in range(1,epochs+1):
        start=time.monotonic()
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device);features=base.encode(x)
            core_input=None
            for kind,probe in probes.items():
                optimizers[kind].zero_grad(set_to_none=True)
                z,cap=probe(features[-1],capture=True)
                if core_input is None:core_input=cap['mixer_input'].detach().clone()
                else:assert torch.equal(core_input,cap['mixer_input'])
                loss=loss_fn(base.decode(z,features),y);assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters() if p.requires_grad)
                optimizers[kind].step()
            if args.mode=='preflight' and step==1:break
        assert all(torch.equal(initial[n],v) for n,v in base.base.state_dict().items())
        metrics=validate(base,probes,vloader,device)
        history.append(dict(epoch=epoch,seconds_all_three_arms=time.monotonic()-start,validation=metrics))
        for kind in KINDS:
            if metrics[kind]['loss']<best[kind]:
                best[kind]=metrics[kind]['loss'];torch.save(dict(probe=probes[kind].state_dict(),epoch=epoch,base_sha256=expected),args.output/(kind+'_best.pth'))
        write(args.output/'history.json',history)
        write(args.output/'status.json',dict(status='training' if epoch<epochs else 'final_analysis_pending',pid=os.getpid(),epoch_completed=epoch,epochs_requested=epochs,test_samples_accessed=False))
        print(json.dumps(history[-1]),flush=True)
    for kind in KINDS:
        saved=torch.load(args.output/(kind+'_best.pth'),map_location=device);fresh=Probe(kind,args.seed).to(device);fresh.load_state_dict(saved['probe'],strict=True)
        with torch.no_grad():assert torch.equal(fresh(features[-1]),probes[kind](features[-1])) if args.mode=='preflight' else True
        probes[kind]=fresh.eval()
    if args.mode=='train':
        metrics=validate(base,probes,vloader,device)
        spectral_analysis(base,probes,validation,device,args.output)
    write(args.output/'status.json',dict(status='preflight_passed' if args.mode=='preflight' else 'trained_diagnostic_complete',frozen_backbone_unchanged=True,shared_mixer_inputs_equal=True,
         gradients_finite=True,strict_reload=True,steps_per_arm=2 if args.mode=='preflight' else len(loader)*epochs,
         validation_before=before,validation_after=metrics,spectral_conclusion_available=args.mode=='train',not_a_causal_proof_of_all_architectures=True))

if __name__=='__main__':main()
