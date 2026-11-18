"""Retry failed tests after the original queue exits, retaining failed artifacts."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'log/八模型对比/HAM独立测试补齐_20261101'
ENTRY=ROOT/'source/comparison/final_test_previous_encoders.py'
def save(state):
    state['updated_at']=datetime.now().astimezone().isoformat()
    p=OUT/'queue_status.json';t=p.with_suffix('.json.tmp')
    t.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
    while True:
        state=json.loads((OUT/'queue_status.json').read_text())
        if state['status'] in ('completed_with_failures','all_complete'):break
        time.sleep(10)
    if state['status']=='all_complete':return
    failed=[k for k,v in state['models'].items() if v['status']=='failed']
    state.update(status='recovering_failed_tests',recovery_pid=os.getpid())
    relative=ENTRY.relative_to(ROOT)
    destination=OUT/'source_snapshot_worker_import_fix'/relative
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ENTRY,destination)
    state['worker_import_fix_sha256']=hashlib.sha256(ENTRY.read_bytes()).hexdigest()
    save(state)
    for name in failed:
        old=OUT/name
        if old.exists():
            archive=OUT/'failed_attempts'/(name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
            archive.parent.mkdir(exist_ok=True);old.rename(archive)
        python=state['models'][name]['python']
        with (OUT/(name+'_retry_console.log')).open('x') as stream:
            child=subprocess.Popen([python,'-u',str(ENTRY),'--arm',name,'--gpu','0'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['models'][name].update(status='testing',pid=child.pid,retry=True);save(state)
            code=child.wait()
        state['models'][name].update(status='complete' if code==0 else 'failed',exitcode=code);save(state)
    state['status']='all_complete' if all(v['status']=='complete' for v in state['models'].values()) else 'completed_with_failures'
    save(state)
if __name__=='__main__':main()
