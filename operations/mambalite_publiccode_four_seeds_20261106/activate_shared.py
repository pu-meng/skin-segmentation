"""Latest explicit sharing authorization replaces undeployed occupancy barrier."""
import os,json,hashlib,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');NAME='mambalite_publiccode_four_seeds_20261106';OPS=ROOT/'operations'/NAME;LOG=ROOT/'log'/NAME
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
p=OPS/'protocol.json';c=json.loads(p.read_text());accept=LOG/'cpu_acceptance.json';a=json.loads(accept.read_text())
assert a['status']=='passed' and a['protocol_sha256']==sha(p)
for name,h in c['sources'].items():assert sha(name)==h,name
assert not (OPS/'controller.pid').exists(),'refuse duplicate launch'
(OPS/'protocol_waiting_superseded.json').write_bytes(p.read_bytes());(LOG/'cpu_acceptance_before_share.json').write_bytes(accept.read_bytes())
old=OPS/'controller.py';disabled=OPS/'controller_waiting_superseded.py.disabled';old.rename(disabled)
c['sources'].pop(str(old));c['sources'][str(disabled)]=sha(disabled)
c['sources'][str(OPS/'shared_controller.py')]=sha(OPS/'shared_controller.py')
c['resource_policy']='LATEST USER AUTHORIZATION: GPU1 sharing; do NOT wait BD/B3D or CandidateA completion; existing jobs untouched; >=6000MiB free before each new slot; max2 new arms, batch8 real preflight before each train; no smaller batch, no auto retry'
c['authorization']+='; latest user explicitly authorizes immediate GPU1 sharing and independent per-arm preflight->train, no peer barrier'
c['execution_order']=['public43','public44','public45','public46','fig2b42'];c['batches']='two rolling slots; each arm preflight->train300->strict CPU best->test2020->audit independently'
c['controller']=str(OPS/'shared_controller.py');c['superseded_controller']=str(disabled);c['training_is_legacy_seed42_reference']=True
c['scope_manifests']={'public':str(OPS/'public_manifest.json'),'reconstruction':str(OPS/'reconstruction_manifest.json')}
dump(p,c)
a['rebound_protocol_sha256_from']=a['protocol_sha256'];a['protocol_sha256']=sha(p);a['reuse_basis']='All science/loader/model/restore sources identical; only undeployed operations controller renamed/disabled and new explicit sharing scheduler added. CPU proof reused after exact hash checks.';dump(accept,a)
dump(OPS/'public_manifest.json',dict(identity='author public MambaLiteUNet defaults, four channel branches',seeds=[43,44,45,46],legacy_seed42='read-only reuse, verified',authorization='user expressly authorizes these four from-scratch trainings + validationbest noTTA HAM2020',protocol_sha256=sha(p),source_manifest=str(p)))
dump(OPS/'reconstruction_manifest.json',dict(identity='frozen candidate_model.py Figure2b mixed reconstruction, NOT public model or strict full paper reproduction',seeds=[42],authorization='latest explicit user reauthorization ONLY one reconstructed seed42; NEVER old five-seed controller',resume=c['reconstruction_resume'],recovery_completed_epoch=102,next_epoch=103,old_cancel_retained=True,protocol_sha256=sha(p),excluded_from_public_statistics=True))
f=(OPS/'controller_console.log').open('x');child=subprocess.Popen([sys.executable,'-B','-u',str(OPS/'shared_controller.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close()
dump(OPS/'launch_receipt.json',dict(controller_pid=child.pid,time=time.time(),independent_session=True,entry=str(OPS/'shared_controller.py'),protocol_sha256=sha(p)));print('DETACHED_CONTROLLER_PID',child.pid,flush=True)
