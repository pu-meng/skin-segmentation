"""Serial preflight, then at most two independent training/test jobs on GPU1."""
from pathlib import Path
import subprocess,sys,os,json,time,fcntl,shutil,hashlib
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/'log/八模型对比/TinyViM论文Encoder扩展_20261101'
KINDS=['swiftformer_s','efficientformerv2_s0','mobileone_s1','efficientvmamba_t']
ENTRY=Path(__file__).parent/'experiment.py'
TEST=Path(__file__).parent/'final_test.py'


def free_memory():
    value=subprocess.check_output(['nvidia-smi','-i',os.environ['EXPERIMENT_PHYSICAL_GPU'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(value.strip())


def write(state):
    state['updated_at']=datetime.now().astimezone().isoformat()
    p=LOG/'queue_status.json';t=p.with_suffix('.json.tmp')
    t.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def launch(arm,mode):
    work=LOG/arm;work.mkdir(exist_ok=True)
    command=[sys.executable,'-u',str(ENTRY),'--arm',arm,'--mode',mode,'--run-root',str(work),'--gpu','0']
    if mode=='test':command=[sys.executable,'-u',str(TEST),'--arm',arm,'--gpu','0']
    with (work/(mode+'_console.log')).open('x') as stream:
        return subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)


def main():
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (LOG/'queue_status.json').exists(),'Refusing duplicate queue'
        assert shutil.disk_usage(LOG).free>12*2**30,'Need 12 GiB disk headroom'
        state=dict(status='preflight',pid=os.getpid(),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),
                   dataset='HAM10000',seed=42,epochs=300,batch_size=8,max_parallel_jobs=2,
                   automatic_final_test=True,arms={a:dict(status='pending') for a in KINDS})
        write(state)
        passed=[]
        for arm in KINDS:
            existing=LOG/arm/'preflight/preflight_status.json'
            if existing.exists():
                report=json.loads(existing.read_text())
                protocol_path=LOG/arm/'protocol.json'
                protocol=json.loads(protocol_path.read_text())
                assert report['status']=='passed'
                assert report['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                for relative,digest in protocol['source_sha256'].items():
                    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
                state['arms'][arm].update(status='preflight_passed',reused_exact_protocol_preflight=True)
                passed.append(arm);write(state);continue
            while free_memory()<6500:
                state['arms'][arm].update(status='waiting_gpu_memory');write(state);time.sleep(20)
            process=launch(arm,'preflight')
            state['arms'][arm]=dict(status='preflight',pid=process.pid);write(state)
            code=process.wait()
            if code:
                state['arms'][arm].update(status='failed',failed_stage='preflight',exitcode=code)
            else:
                state['arms'][arm].update(status='preflight_passed');passed.append(arm)
            write(state)
        active={}
        while passed or active:
            for arm,(process,mode) in list(active.items()):
                code=process.poll()
                if code is None:continue
                if code:
                    state['arms'][arm].update(status='failed',failed_stage=mode,exitcode=code);del active[arm]
                elif mode=='train':
                    child=launch(arm,'test');active[arm]=(child,'test')
                    state['arms'][arm].update(status='testing',pid=child.pid)
                else:
                    audit=json.loads((LOG/'最终测试'/arm/'artifact_audit.json').read_text())
                    assert audit['complete'];state['arms'][arm].update(status='complete');del active[arm]
            if passed and len(active)<2:
                arm=passed[0]
                report=json.loads((LOG/arm/'preflight/preflight_status.json').read_text())
                required=max(4500,int(report['peak_allocated_mib'])+2000)
                if free_memory()>=required:
                    child=launch(arm,'train');active[arm]=(child,'train');passed.pop(0)
                    state['arms'][arm].update(status='training',pid=child.pid)
                    # Do not count memory as free until this child allocates its model.
                    time.sleep(15)
                else:
                    state['arms'][arm].update(status='waiting_gpu_memory',minimum_free_mib=required)
            state['status']='running';write(state)
            if active or passed:time.sleep(10)
        state['status']='all_complete' if all(v['status']=='complete' for v in state['arms'].values()) else 'completed_with_failures'
        write(state)


if __name__=='__main__':main()
