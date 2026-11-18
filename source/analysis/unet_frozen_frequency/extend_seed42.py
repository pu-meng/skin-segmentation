"""Continue the frozen-UNet seed42 probes, retaining the original 30-epoch study."""
import argparse, copy, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from run_probe_trajectory import (HERE, BASE, KINDS, Images, FrozenUNet, Probe, loss_fn,
                                  validate, spectral_analysis, sha, write)
from project_paths import verify_runtime


def plot_progress(out, original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':16,'axes.unicode_minus':False})
    points={}
    for directory in (original/'trajectory',out/'trajectory'):
        for f in directory.glob('epoch*/spectra.json'):
            d=json.loads(f.read_text());points[d['epoch']]=d['statistics']
    fig,axes=plt.subplots(1,2,figsize=(14,7.4))
    for ax,metric,band in zip(axes,['low_non_dc_power_share','high_non_dc_power_share'],['低频','高频']):
        for kind,color in [('conv','#4477aa'),('attention','#cc6677'),('mamba','#228833')]:
            epochs=sorted(points)
            ax.plot(epochs,[100*points[e][kind+'/mixer_output'][metric] for e in epochs],'o-',label=kind,color=color)
        ax.axhline(100*points[0]['conv/mixer_input'][metric],color='gray',ls='--',label='模块处理前的输入')
        ax.set(xlabel='训练轮次',ylabel=band+'能量占比（%）',title=band+'能量变化');ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    fig.suptitle('冻结UNet：seed42延长训练（单次运行，不是三seed均值）',fontsize=21,y=.98)
    fig.tight_layout(rect=(0,.04,1,.81))
    fig.savefig(out/'frequency_seed42.png',dpi=200);fig.savefig(out/'frequency_seed42.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--preflight',action='store_true')
    a=parser.parse_args();verify_runtime();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device=torch.device(a.device);torch.cuda.set_device(device)
    protocol_path=HERE/'protocol_seed42_100epochs.json'
    protocol=json.loads(protocol_path.read_text());original=HERE/'runs_trajectory/seed42'
    checkpoint=BASE/'checkpoints/best.pth'
    assert sha(checkpoint)==protocol['base_checkpoint_sha256']
    base=FrozenUNet(checkpoint).to(device)
    probes={k:Probe(k,42).to(device) for k in KINDS}
    optimizers={k:torch.optim.AdamW([p for p in v.parameters() if p.requires_grad],lr=.001,weight_decay=.01) for k,v in probes.items()}
    source={}
    for k in KINDS:
        p=original/'checkpoints/epoch030'/(k+'.pth')
        assert sha(p)==protocol['resume_checkpoint_sha256'][k]
        d=torch.load(p,map_location=device);assert d['epoch']==30 and d['base_sha256']==sha(checkpoint)
        probes[k].load_state_dict(d['probe'],strict=True);optimizers[k].load_state_dict(d['optimizer'])
        assert all(g['lr']==.001 and g['weight_decay']==.01 for g in optimizers[k].param_groups)
        source[k]={'path':str(p),'sha256':sha(p)}
    baseline={n:v.detach().clone() for n,v in base.base.state_dict().items()}
    write(a.output/'provenance.json',dict(protocol_sha256=sha(protocol_path),source_checkpoints=source,
        source_sha256={name:sha(Path(__file__).parent/name) for name in ['extend_seed42.py','run_probe_trajectory.py','controlled_probe.py','spectral_metrics.py']},
        seed=42,start_epoch=30,target_epoch=100,device=str(device),shared_gpu=True,
        exact_uninterrupted_rng_equivalence=False,reason='Parent checkpoints did not save global or loader RNG states; loader order is reconstructed by advancing the original sampler for 30 epochs.',
        optimizer_restored=True,test_samples_accessed=False))
    train=Images('train',8 if a.preflight else None);val=Images('val',8 if a.preflight else None)
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(train,batch_size=8,shuffle=True,generator=generator,num_workers=0)
    if not a.preflight:
        # DataLoader consumes its base seed before its sampler, even with no workers.
        for _ in range(30):
            it=iter(loader);list(it._sampler_iter)
    vloader=DataLoader(val,batch_size=8,shuffle=False)
    before=validate(base,probes,vloader,device)
    for k in KINDS:
        probes[k].train()
    if not a.preflight:
        plot_progress(a.output,original)
    history=[];target=31 if a.preflight else 100
    for epoch in range(31,target+1):
        start=time.monotonic();losses={k:[] for k in KINDS}
        for step,(x,y) in enumerate(loader,1):
            x,y=x.to(device),y.to(device);features=base.encode(x);common=None
            for k in KINDS:
                opt=optimizers[k];opt.zero_grad(set_to_none=True)
                z,capture=probes[k](features[-1],capture=True)
                if common is None:common=capture['mixer_input'].detach().clone()
                else:assert torch.equal(common,capture['mixer_input'])
                loss=loss_fn(base.decode(z,features),y);assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probes[k].parameters() if p.requires_grad)
                opt.step();losses[k].append(loss.item())
            if step==1 or step%50==0:
                write(a.output/'status.json',dict(status='preflight' if a.preflight else 'training',pid=os.getpid(),epoch=epoch,step=step,steps_per_epoch=len(loader),epochs_completed=epoch-1,target_epoch=100))
        assert all(torch.equal(baseline[n],v) for n,v in base.base.state_dict().items())
        for p in probes.values():p.eval()
        metrics=validate(base,probes,vloader,device)
        row=dict(epoch=epoch,train_loss={k:float(np.mean(v)) for k,v in losses.items()},validation=metrics,seconds_all_three_arms=time.monotonic()-start)
        history.append(row);write(a.output/'history.json',history)
        folder=a.output/'checkpoints'/f'epoch{epoch:03d}';folder.mkdir(parents=True,exist_ok=False)
        for k in KINDS:
            payload=dict(probe=probes[k].state_dict(),optimizer=optimizers[k].state_dict(),epoch=epoch,base_sha256=sha(checkpoint))
            torch.save(payload,folder/(k+'.pth'))
            if a.preflight:
                fresh=Probe(k,42).to(device).eval();fresh.load_state_dict(torch.load(folder/(k+'.pth'),map_location=device)['probe'],strict=True)
                with torch.no_grad():assert torch.equal(fresh(features[-1]),probes[k](features[-1]))
                assert any(not torch.equal(payload['probe'][n],torch.load(original/'checkpoints/epoch030'/(k+'.pth'),map_location=device)['probe'][n]) for n in payload['probe'])
        torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=generator.get_state()),folder/'rng.pth')
        if not a.preflight and epoch in (40,50,60,80,100):
            dest=a.output/'trajectory'/f'epoch{epoch:03d}';dest.mkdir(parents=True,exist_ok=False)
            spectral_analysis(base,probes,val,device,dest)
            data=json.loads((dest/'spectra.json').read_text());data.update(epoch=epoch,seed=42,weight_rule='fixed epoch, continued from epoch30');write(dest/'spectra.json',data)
            write(dest/'checkpoint_sources.json',{k:{'path':str(folder/(k+'.pth')),'sha256':sha(folder/(k+'.pth')),'epoch':epoch} for k in KINDS})
            plot_progress(a.output,original)
        write(a.output/'status.json',dict(status='preflight_passed' if a.preflight else ('training' if epoch<target else 'training_and_frequency_complete'),pid=os.getpid(),epochs_completed=epoch,target_epoch=100,frozen_backbone_unchanged=True,shared_inputs_equal=True,finite_gradients=True,optimizer_restored=True,test_samples_accessed=False))
        print(json.dumps(row),flush=True)
        for p in probes.values():p.train()

if __name__=='__main__':main()
