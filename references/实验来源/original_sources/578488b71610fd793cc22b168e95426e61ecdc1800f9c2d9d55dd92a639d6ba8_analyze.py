"""Trained checkpoint only: native before/after features, no model modification."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from models import ROOT, PureMixerUNet
from experiment import Images, check_protocol, sha, write, amp

sys.path.insert(0,str(ROOT/'tinyvim_frequency_analysis'))
from analyze_native import Observer


class Features:
    # Reuse the independently verified streaming FFT accumulator only.
    add=Observer.add
    results=Observer.results

    def __init__(self,model):
        self.acc={};self.handles=[]
        for stage,blocks in enumerate(model.stages,1):
            for i,block in enumerate(blocks):
                name=f'stage{stage}/block{i:02d}'
                self.handles.append(block.register_forward_pre_hook(self.pre(name+'/block_input')))
                self.handles.append(block.register_forward_hook(self.post(name+'/block_output')))

    def pre(self,key):
        def hook(m,args):self.add(key,args[0])
        return hook

    def post(self,key):
        def hook(m,args,y):self.add(key,y)
        return hook

    def close(self):
        for h in self.handles:h.remove()


def draw(folder,stats,epoch,kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib import gridspec
    plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams['axes.unicode_minus']=False
    pairs=[(f'stage3/block{i:02d}/block_input',f'stage3/block{i:02d}/block_output',f'第{i+1}个模块自身') for i in range(9)]
    pairs += [('stage3/block00/block_input',f'stage3/block{i-1:02d}/block_output',f'共同起点经过{i}个模块') for i in (1,3,5,6,9)]
    lines=[f'# {kind} 第{epoch}轮输入与输出频谱','', '每幅图只比较输入与一个输出。自身前后比较、共同起点的累积比较分开列出。','']
    for index,(before,after,title) in enumerate(pairs):
        a,b=stats[before],stats[after]
        arrays=[np.asarray(x['first8_mean_log_amplitude_maps']) for x in (a,b)]
        low=min(x.min() for x in arrays);high=max(x.max() for x in arrays)
        fig=plt.figure(figsize=(14,4.5),layout='constrained')
        grid=gridspec.GridSpec(2,2,figure=fig,width_ratios=(2.6,1),height_ratios=(.3,2))
        top=fig.add_subplot(grid[0,:]);top.axis('off');top.text(.5,.5,f'{kind} · 第{epoch}轮 · {title}：输入 → 输出',ha='center')
        maps=gridspec.GridSpecFromSubplotSpec(2,8,subplot_spec=grid[1,0],wspace=.05,hspace=.15)
        for row,array in enumerate(arrays):
            for col in range(8):
                ax=fig.add_subplot(maps[row,col]);ax.imshow(array[col],vmin=low,vmax=high,cmap='viridis');ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_ylabel('输入' if row==0 else '输出')
        ax=fig.add_subplot(grid[1,1])
        for d,style,label in [(a,'-','输入'),(b,'--','输出')]:
            y=d['relative_log_amplitude'];ax.plot(np.linspace(0,1,len(y)),y,'k'+style,label=label,marker=None if style=='-' else 'o',markersize=3,markerfacecolor='white')
        ax.legend();ax.set_xticks([0,.5,1],['0','0.5π','π']);ax.set_xlabel('低频 → 高频');ax.set_ylabel('相对零频的对数幅值')
        fig.savefig(folder/f'pair{index:02d}.png',dpi=150);plt.close(fig)
        lines += [f'## {title}','',f'![{title}](pair{index:02d}.png)','']
    (folder/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    check_protocol();torch.set_num_threads(4);device=torch.device(a.device)
    ck=torch.load(a.checkpoint,map_location='cpu')
    assert ck['formal_training'] and ck['epoch']>0,'Preflight/random weights are not trained spectral evidence'
    assert ck['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    a.output.mkdir(parents=True,exist_ok=False)
    model=PureMixerUNet(ck['kind'],ck['seed']).to(device).eval();model.load_state_dict(ck['model'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    dataset=Images('val');loader=DataLoader(dataset,batch_size=1,shuffle=False)
    x=dataset[0][0][None].to(device)
    with torch.no_grad(),amp(device):reference=model(x)
    observer=Features(model)
    with torch.no_grad(),amp(device):observed=model(x)
    assert torch.equal(reference,observed)
    observer.close();observer=Features(model)
    with torch.no_grad():
        for x,_ in loader:
            with amp(device):model(x.to(device))
    observer.close();stats=observer.results()
    assert len(stats)==42 and all(v['samples']==1010 for v in stats.values())
    write(a.output/'frequency_statistics.json',stats)
    write(a.output/'sample_ids.json',[r['image_id'] for r in dataset.rows])
    draw(a.output,stats,ck['epoch'],ck['kind'])
    write(a.output/'status.json',dict(status='complete',checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),
          kind=ck['kind'],epoch=ck['epoch'],samples=1010,feature_points=42,hook_predictions_identical=True,test_samples_accessed=False))


if __name__=='__main__':main()
