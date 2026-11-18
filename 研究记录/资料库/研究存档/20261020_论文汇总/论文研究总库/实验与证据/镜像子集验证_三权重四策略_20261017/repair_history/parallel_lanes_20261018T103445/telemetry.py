import subprocess,time
from common import *
def sample():
    fields='uuid,memory.used,utilization.gpu,power.draw,power.limit,temperature.gpu'
    selection=read(ROOT/'manifest.json'); gpu=str(selection['physical_gpu'])
    row=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip().split(', ')
    expected=selection['gpu_uuid']
    if row[0]!=expected: raise RuntimeError(f'Physical GPU{gpu} UUID changed')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
    pids=[]
    for line in apps:
        parts=[s.strip() for s in line.split(',')]
        if parts and parts[0]==expected: pids.append(int(parts[1]))
    return dict(at_unix=time.time(),uuid=row[0],memory_used_mib=float(row[1]),utilization_pct=float(row[2]),power_w=float(row[3]),power_limit_w=float(row[4]),temperature_c=float(row[5]),compute_pids=pids,compute_apps=apps)
def upstream():
    root=WORK/'results_skip_only_method6_screen50_v3'
    state=read(root/'queue_state.json')
    ps=subprocess.check_output(['ps','-u','PuMengYu','-o','pid=,args='],text=True).splitlines()
    lines=[l.strip() for l in ps if len(l.split())>1 and l.split()[1].rsplit('/',1)[-1] in ('python','python3') and ' -m pumengyu.skip50.' in l]
    queue=any(' -m pumengyu.skip50.queue' in l for l in lines)
    children=any(' -m pumengyu.skip50.queue' not in l for l in lines)
    return state,queue,children,lines
