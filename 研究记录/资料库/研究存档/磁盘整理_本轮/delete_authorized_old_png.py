from pathlib import Path
import os,json,stat,fcntl,datetime,shutil,time
base=Path('/media/8T/PuMengYu')
roots=[base/'nnUNet_workspace_archive',base/'nnUNet_result',base/'MSD_LiverTumorSeg/experiments',base/'system_disk_archive/系统盘紧急迁移_本轮/nnUNet_workspace/results_task02',base/'system_disk_archive/系统盘紧急迁移_本轮/nnUNet_workspace/results_task02_source_plans_formal_v1']
audit=Path('/home/PuMengYu/研究存档/磁盘整理_本轮/旧实验PNG删除_20261023');audit.mkdir(parents=True,exist_ok=True)
lock=(audit/'delete.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.path.ismount('/media/8T')
assert not (audit/'status.json').exists(),'Existing cleanup record; review instead of repeat'
def stamp(p):
 s=p.lstat();return [s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns]
def status(stage,**kw):
 data={'stage':stage,'time':datetime.datetime.now().astimezone().isoformat(),'scope':[str(p) for p in roots],**kw};tmp=audit/'status.writing';tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(audit/'status.json');print(json.dumps(data,ensure_ascii=False),flush=True)
def protected_inventory():
 result={}
 for root in roots:
  for b,ds,fs in os.walk(root,followlinks=False):
   for n in ds[:]:
    p=Path(b)/n
    if p.is_symlink():result[str(p)]=stamp(p)+[os.readlink(p)];ds.remove(n)
   for n in fs:
    p=Path(b)/n
    if p.suffix.lower()=='.png' and not p.is_symlink():continue
    result[str(p)]=stamp(p)+([os.readlink(p)] if p.is_symlink() else [])
 return result
try:
 for r in roots:assert r.is_dir() and not r.is_symlink() and r.resolve().is_relative_to(base)
 hits=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   for q in [p/'cwd',p/'exe']+list((p/'fd').iterdir()):
    try:t=os.readlink(q)
    except OSError:continue
    if any(t==str(r) or t.startswith(str(r)+'/') for r in roots):hits.append([p.name,t])
   maps=(p/'maps').read_text(errors='replace')
   if any(str(r)+'/' in maps for r in roots):hits.append([p.name,'mapped_file'])
  except OSError:pass
 assert not hits,('Old experiment directory active; no deletion',hits)
 protected=protected_inventory();(audit/'保留文件元数据.json').write_text(json.dumps(protected,ensure_ascii=False))
 planned=[]
 for r in roots:
  for b,ds,fs in os.walk(r,followlinks=False):
   ds[:]=[n for n in ds if not (Path(b)/n).is_symlink()]
   for n in fs:
    p=Path(b)/n
    if p.suffix.lower()!='.png' or p.is_symlink():continue
    s=stamp(p);assert stat.S_ISREG(s[2]);planned.append({'path':str(p),'stamp':s})
 total=sum(x['stamp'][3] for x in planned);assert len(planned)==182957 and total==80796836742,'Scope changed; inspect before deletion'
 with (audit/'获准删除清单.jsonl').open('w') as f:
  for x in planned:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 free_before=shutil.disk_usage(base).free;status('deleting',planned_files=len(planned),planned_bytes=total,deleted_files=0,free_before_bytes=free_before)
 deleted=0;deleted_bytes=0
 with (audit/'实际删除记录.jsonl').open('w') as log:
  for x in planned:
   p=Path(x['path']);assert stamp(p)==x['stamp'],'File changed; refuse delete: '+str(p);assert any(p.resolve().is_relative_to(r) for r in roots);p.unlink();log.write(json.dumps({'path':str(p),'bytes':x['stamp'][3]},ensure_ascii=False)+'\n');deleted+=1;deleted_bytes+=x['stamp'][3]
   if deleted%5000==0:log.flush();status('deleting',planned_files=len(planned),deleted_files=deleted,deleted_bytes=deleted_bytes)
 after=protected_inventory();assert after==protected,'Protected non-PNG file metadata changed; investigate'
 remaining=[x['path'] for x in planned if Path(x['path']).exists() or Path(x['path']).is_symlink()];assert not remaining
 status('complete',deleted_files=deleted,deleted_bytes=deleted_bytes,protected_non_png_entries=len(protected),protected_metadata_unchanged=True,all_planned_png_absent=True,free_before_bytes=free_before,free_after_bytes=shutil.disk_usage(base).free,system_free_bytes=shutil.disk_usage('/home/PuMengYu').free,backup_created=False)
 assert json.loads((audit/'status.json').read_text())['deleted_files']==len(planned)
except Exception as e:
 status('blocked',error=repr(e));raise
