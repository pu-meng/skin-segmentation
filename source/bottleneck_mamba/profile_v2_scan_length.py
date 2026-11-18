"""Controlled synthetic scan-only timing and autograd-shape evidence, no training writes."""
from pathlib import Path
import sys,json,time,statistics,collections
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P),str(P/'source/comparison')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import torch
from Models.TinyViM.mamba_bottleneck_v2 import scan_without_delta_injection
from Models.TinyViM.mamba_bottleneck_v2_triton_pilot import scan_v2_triton

ROOT=P/'log/版本2_扫描长度取证_20261030'

def graph_shapes(y):
    queue=[y.grad_fn];seen=set();counts=collections.Counter()
    while queue:
        node=queue.pop()
        if node is None or node in seen:continue
        seen.add(node)
        name=type(node).__name__
        if 'SelectBackward' in name or 'SliceBackward' in name:
            shape=tuple(getattr(node,'_saved_self_sym_sizes',()))
            counts[(name,str(shape))]+=1
        queue.extend(x for x,_ in node.next_functions if x is not None)
    return [{'node':name,'saved_full_input_shape':shape,'count':count} for (name,shape),count in counts.items()]


def inputs(length):
    torch.manual_seed(44)
    E=1024;N=16;batch=8
    vals=[torch.randn(batch,E,length,device='cuda')*0.1,
          torch.ones(batch,E,length,device='cuda')*0.01,
          -torch.ones(E,N,device='cuda'),
          torch.randn(batch,N,length,device='cuda')*0.1,
          torch.randn(batch,N,length,device='cuda')*0.1]
    return [v.requires_grad_() for v in vals]


def main():
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(0)
    records=[]
    # Interleave lengths and implementations to avoid timing a single long block.
    for length in [64,128,256,256,128,64]:
        for name,fn in [('explicit',scan_without_delta_injection),('triton',scan_v2_triton)]:
            vals=inputs(length);samples=[]
            for repeat in range(3):
                for v in vals:v.grad=None
                torch.cuda.synchronize();t0=time.perf_counter()
                y=fn(*vals);torch.cuda.synchronize();t1=time.perf_counter()
                y.square().mean().backward();torch.cuda.synchronize();t2=time.perf_counter()
                if repeat:samples.append({'forward_seconds':t1-t0,'backward_seconds':t2-t1})
            records.append({'length':length,'backend':name,'samples':samples})
            del vals,y;torch.cuda.empty_cache()
    profiles=[]
    for length in [64,256]:
        vals=inputs(length);y=scan_without_delta_injection(*vals)
        nodes=graph_shapes(y)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=True) as prof:
            y.square().mean().backward();torch.cuda.synchronize()
        rows=[]
        wanted=['aten::select_backward','aten::slice_backward','aten::zeros','aten::zero_','SelectBackward0','SliceBackward0']
        for e in prof.key_averages():
            if any(e.key==k for k in wanted):
                rows.append({'op':e.key,'calls':e.count,'self_cuda_us':e.self_cuda_time_total,
                             'cuda_us':e.cuda_time_total,'cuda_memory_bytes_cumulative':e.cuda_memory_usage})
        profiles.append({'length':length,'graph_nodes':nodes,'backward_profile':rows})
        del vals,y;torch.cuda.empty_cache()
    summary=[]
    for name in ['explicit','triton']:
        for length in [64,128,256]:
            samples=[s for r in records if r['backend']==name and r['length']==length for s in r['samples']]
            summary.append({'backend':name,'length':length,
                            'forward_ms':1000*statistics.median(s['forward_seconds'] for s in samples),
                            'backward_ms':1000*statistics.median(s['backward_seconds'] for s in samples)})
    result={'scope':'synthetic scan-only, batch8 E1024 N16 FP32; only L/backend changed; no data loading, backbone or optimizer',
            'samples':records,'summary':summary,'profiles':profiles,
            'memory_note':'cumulative allocator bytes, not peak live memory and not directly measured DRAM traffic'}
    (ROOT/'evidence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
