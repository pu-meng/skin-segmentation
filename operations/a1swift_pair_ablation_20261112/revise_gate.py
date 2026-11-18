import hashlib,json,shutil,time
from pathlib import Path
r=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1Swift_PairAblations_ISIC_FRESH300_20261112')
entry=Path('/home/PuMengYu/experiments/skin_segmentation/operations/a1swift_pair_ablation_20261112/entry.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rev=r/'protocol_revisions'/'before_immediate_start_gate_8192'
rev.mkdir(parents=True,exist_ok=False)
shutil.copy2(r/'protocol.json',rev/'protocol.json')
shutil.copy2(r/'prepare_complete.json',rev/'prepare_complete.json')
old=json.loads((r/'protocol.json').read_text())
entry_key=next(k for k in old['protected_sha256'] if k.endswith('/operations/a1swift_pair_ablation_20261112/entry.py'))
old_snapshot=r/'source_snapshot'/Path(entry_key).relative_to('/')
shutil.copy2(old_snapshot,rev/'entry.py')
if (r/'queue_status.json').exists():shutil.copy2(r/'queue_status.json',rev/'queue_status.json')
p=old
p['scheduling']['free_vram_gate_mib']=4608
p['scheduling']['gate_revision']='Lowered before any formal batch, per user immediate-start request; admission still requires <4 compute processes and 4608 MiB free. Reference pair-ablation preflight peak was about 2.97 GiB.'
p['protected_sha256'][entry_key]=sha(entry)
tmp=r/'protocol.json.tmp';tmp.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n');tmp.replace(r/'protocol.json')
old_snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(entry,old_snapshot);assert sha(old_snapshot)==sha(entry)
pc=json.loads((r/'prepare_complete.json').read_text());pc['protocol_sha256']=sha(r/'protocol.json');pc['admission_gate_revision']='8192 -> 4608 MiB before training';tmp=r/'prepare_complete.json.tmp';tmp.write_text(json.dumps(pc,indent=2,ensure_ascii=False)+'\n');tmp.replace(r/'prepare_complete.json')
write={'status':'restarting_workers_with_revised_admission_gate','previous_gate_mib':8192,'gate_mib':4608,'reason':'User requested immediate start; scientific/training protocol unchanged.','time':time.time()}
tmp=r/'queue_status.json.tmp';tmp.write_text(json.dumps(write,indent=2)+'\n');tmp.replace(r/'queue_status.json')
print(json.dumps({'protocol_sha256':sha(r/'protocol.json'),'entry_sha256':sha(entry),'gate_mib':p['scheduling']['free_vram_gate_mib'],'backup':str(rev),'source_hash_verified':p['protected_sha256'][entry_key]==sha(entry)},indent=2))
