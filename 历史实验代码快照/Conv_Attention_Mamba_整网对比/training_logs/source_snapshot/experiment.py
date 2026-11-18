"""Train/evaluate a homogeneous-mixer segmentation model on the locked V5 split.

Defaults only run preflight. Formal training requires --mode train explicitly.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models import ROOT, KINDS, PureMixerUNet

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


class Images(Dataset):
    def __init__(self, split, limit=None):
        if split not in ('train','val'):
            raise ValueError('Test data is outside this experiment')
        protocol = json.loads((ROOT/'protocol_v5_dedup_seed42.json').read_text())
        path = Path(protocol['data']['split_file'])
        assert sha(path) == protocol['data']['split_sha256']
        self.rows = sorted([r for r in csv.DictReader(path.open()) if r['split']==split],
                           key=lambda r:r['image_id'])
        if limit:
            self.rows = self.rows[:limit]
        self.images = np.load(path.parent/'images.npy', mmap_mode='r')
        self.masks = np.load(path.parent/'masks.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        n = int(self.rows[i]['cache_index'])
        x = torch.tensor(self.images[n].astype(np.float64)/255, dtype=torch.float32).permute(2,0,1)
        y = torch.tensor((self.masks[n]>0).copy(), dtype=torch.float32)[None]
        return x,y


def loss_fn(p, y):
    dims=(1,2,3)
    dice=(2*(p*y).sum(dims)+1e-5)/(p.sum(dims)+y.sum(dims)+1e-5)
    # Explicit float32 loss outside autocast; network returns probabilities.
    return torch.nn.functional.binary_cross_entropy(p.float(),y.float()) + 1-dice.mean()


def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval();total=0;loss=0.;dice=0.;iou=0.
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with amp(device):
            p=model(x)
        loss+=loss_fn(p.float(),y).item()*len(x)
        pred=p>=.5;t=y>=.5
        inter=(pred&t).sum((1,2,3));s=pred.sum((1,2,3))+t.sum((1,2,3));u=s-inter
        dice+=torch.where(s>0,2*inter/s,1.).sum().item()
        iou+=torch.where(u>0,inter/u,1.).sum().item();total+=len(x)
    return dict(loss=loss/total,dice=dice/total,iou=iou/total,samples=total)


def check_protocol():
    p=json.loads((HERE/'protocol.json').read_text())
    assert sha(ROOT/'protocol_v5_dedup_seed42.json')==p['parent_protocol_sha256']
    for path,digest in p['source_sha256'].items():
        assert sha(path)==digest,path
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=KINDS, required=True)
    parser.add_argument('--mode', choices=['preflight','train'], default='preflight')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();protocol=check_protocol()
    assert a.seed in protocol['seeds']
    if a.output.exists():
        raise FileExistsError('Use a new output directory; existing results are preserved')
    a.output.mkdir(parents=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        assert torch.cuda.is_bf16_supported()
        torch.cuda.reset_peak_memory_stats(device)
    write(a.output/'status.json',dict(status='preflight_running' if a.mode=='preflight' else 'training',pid=os.getpid(),epoch=0))
    model=PureMixerUNet(a.kind,a.seed).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    epochs=1 if a.mode=='preflight' else protocol['epochs']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'])
    generator=torch.Generator().manual_seed(a.seed)
    train=Images('train',2 if a.mode=='preflight' else None)
    validation=Images('val',2 if a.mode=='preflight' else None)
    train_loader=DataLoader(train,batch_size=1 if a.mode=='preflight' else protocol['batch_size'],shuffle=True,generator=generator)
    val_loader=DataLoader(validation,batch_size=1,shuffle=False)
    write(a.output/'provenance.json',dict(kind=a.kind,seed=a.seed,mode=a.mode,device=str(device),torch=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        protocol_sha256=sha(HERE/'protocol.json'),pretraining=False,frozen_parameters=False,test_samples_accessed=False,
        precision='BF16 autocast with FP32 scan and loss' if device.type=='cuda' else 'FP32',source_sha256=protocol['source_sha256']))
    best=float('inf');history=[]
    if a.mode=='train':
        torch.save(dict(model=model.state_dict(),epoch=0,kind=a.kind,seed=a.seed,formal_training=True,protocol_sha256=sha(HERE/'protocol.json')),a.output/'epoch000.pth')
    changed=False
    for epoch in range(1,epochs+1):
        model.train();start=time.monotonic();train_loss=0.
        for step,(x,y) in enumerate(train_loader,1):
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            with amp(device):
                p=model(x)
            assert p.shape==y.shape and torch.isfinite(p).all()
            loss=loss_fn(p.float(),y);assert torch.isfinite(loss)
            loss.backward()
            if a.mode=='preflight':
                for name,param in model.named_parameters():
                    assert param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all(),name
                parameter=next(model.parameters());before=parameter.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if a.mode=='preflight':
                changed |= not torch.equal(before,parameter)
            train_loss+=loss.item()*len(x)
            if a.mode=='train' and (step==1 or step%50==0 or step==len(train_loader)):
                write(a.output/'progress.json',dict(pid=os.getpid(),epoch=epoch,step=step,
                    steps_per_epoch=len(train_loader),optimizer_updates=(epoch-1)*len(train_loader)+step,
                    loss=loss.item(),epoch_elapsed_seconds=time.monotonic()-start))
                print(json.dumps(dict(epoch=epoch,step=step,steps_per_epoch=len(train_loader),loss=loss.item())),flush=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        train_seconds=time.monotonic()-start
        vstart=time.monotonic();metrics=evaluate(model,val_loader,device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        history.append(dict(epoch=epoch,train_loss=train_loss/len(train),validation=metrics,train_seconds=train_seconds,
                            validation_seconds=time.monotonic()-vstart,learning_rate=optimizer.param_groups[0]['lr']))
        scheduler.step()
        common=dict(model=model.state_dict(),kind=a.kind,seed=a.seed,epoch=epoch,formal_training=a.mode=='train',protocol_sha256=sha(HERE/'protocol.json'))
        if metrics['loss']<best:
            best=metrics['loss'];torch.save(common,a.output/'best.pth')
        if a.mode=='train':
            if epoch in protocol['feature_epochs']:torch.save(common,a.output/f'epoch{epoch:03d}.pth')
            torch.save({**common,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                        'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'loader_rng':generator.get_state()},a.output/'latest.pth')
        write(a.output/'history.json',history)
        write(a.output/'status.json',dict(status='training' if epoch<epochs else 'verification_pending',epoch=epoch,mode=a.mode))
        print(json.dumps(history[-1]),flush=True)
    if a.mode=='preflight':
        assert changed
        saved=torch.load(a.output/'best.pth',map_location=device)
        model.load_state_dict(saved['model'],strict=True);model.eval()
        x=next(iter(val_loader))[0].to(device)
        with torch.no_grad(),amp(device):before=model(x)
        fresh=PureMixerUNet(a.kind,a.seed).to(device).eval()
        fresh.load_state_dict(torch.load(a.output/'best.pth',map_location=device)['model'],strict=True)
        with torch.no_grad(),amp(device):after=fresh(x)
        assert torch.equal(before,after)
    else:
        torch.save(common,a.output/'final.pth')
    write(a.output/'status.json',dict(status='preflight_passed' if a.mode=='preflight' else 'training_validation_complete_features_pending',
        epochs_completed=epochs,full_training_performed=a.mode=='train',parameter_update_verified=changed if a.mode=='preflight' else None,
        strict_reload_identical=True if a.mode=='preflight' else None,test_samples_accessed=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))


if __name__=='__main__':
    main()
