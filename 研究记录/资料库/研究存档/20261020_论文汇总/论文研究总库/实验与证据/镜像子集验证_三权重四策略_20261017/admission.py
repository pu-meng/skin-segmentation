import os
from pathlib import Path
from common import ROOT,WORK,read

def upstream_ready(state,queue_alive,children_alive):
    return state.get('status')=='complete' and not queue_alive and not children_alive

def resource_ready(sample,formal_pids=(),parallel=False):
    if not parallel:
        return not sample['compute_pids'] and sample['memory_used_mib']<=1024 and sample['utilization_pct']<=5
    formal=set(formal_pids)
    # Other users are untouched and accounted for by absolute free memory.
    # Includes formal workers still preprocessing or making reports, even when
    # their CUDA context has not appeared in nvidia-smi yet.
    return (len(formal)<=2
            and sample.get('memory_free_mib',0)>=8*1024 and sample.get('available_ram_gib',0)>=12)

def final_workers():
    path=WORK/'results_skip_only_method6_screen50_v3_final_eval_v1/queue_state.json'
    state=read(path);pids=[]
    for item,pid in state.get('running',{}).items():
        process=Path('/proc')/str(pid)
        try:
            if process.stat().st_uid!=os.getuid(): continue
            args=(process/'cmdline').read_bytes().split(b'\0')
            status=(process/'stat').read_text().split(') ',1)[1].split()[0]
            if status!='Z' and b'pumengyu.skip50_final.evaluate' in args and b'--item' in args and args[args.index(b'--item')+1].decode()==item:
                pids.append(pid)
        except (FileNotFoundError,ProcessLookupError): pass
    return pids

def gpu_idle(sample):
    parallel=read(ROOT/'manifest.json').get('parallel_with_final_tests',False)
    return resource_ready(sample,final_workers() if parallel else (),parallel)
