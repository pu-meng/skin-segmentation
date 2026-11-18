from pathlib import Path
import os,json,shutil,fcntl,datetime,stat
base=Path('/media/8T/PuMengYu')
roots=[base/'system_disk_archive/系统盘紧急迁移_本轮/nnUNet_workspace/preprocessed',base/'system_disk_archive/inference_lab/mednext_cpp_v1/env']
audit=Path('/home/PuMengYu/研究存档/磁盘整理_本轮/旧预处理与推理环境删除_20261023');audit.mkdir(parents=True,exist_ok=True)
lock=(audit/'delete.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.path.ismount('/media/8T');assert not (audit/'status.json').exists(),'Existing operation record; inspect before repeating'
def save(stage,**kw):
 d={'stage':stage,'time':datetime.datetime.now().astimezone().isoformat(),'scope':[str(r) for r in roots],**kw};p=audit/'status.writing';p.write_text(json.dumps(d,ensure_ascii=False,indent=2));p.replace(audit/'status.json');print(json.dumps(d,ensure_ascii=False),flush=True)
def sig(p):
 s=p.lstat();return [s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns]+([os.readlink(p)] if p.is_symlink() else [])
def keep_inventory():
 out={};parent=base/'system_disk_archive/inference_lab/mednext_cpp_v1'
 for b,ds,fs in os.walk(parent,followlinks=False):
  ds[:]=[n for n in ds if Path(b)/n!=roots[1] and not (Path(b)/n).is_symlink()]
  for n in fs:out[str(Path(b)/n)]=sig(Path(b)/n)
 return out
try:
 for r in roots:assert r.is_dir() and not r.is_symlink() and r.resolve().is_relative_to(base.resolve())
 hits=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   for q in [proc/'cwd',proc/'exe']+list((proc/'fd').iterdir()):
    try:t=os.readlink(q)
    except OSError:continue
    if any(t==str(r) or t.startswith(str(r)+'/') for r in roots):hits.append([proc.name,t])
   maps=(proc/'maps').read_text(errors='replace')
   if any(str(r)+'/' in maps for r in roots):hits.append([proc.name,'mapped_file'])
  except OSError:pass
 assert not hits,('Target active; deletion refused',hits)
 preserved=keep_inventory();(audit/'保留工程代码与模型元数据.json').write_text(json.dumps(preserved,ensure_ascii=False))
 records=[]
 with (audit/'删除前文件清单.jsonl').open('w') as f:
  for r in roots:
   count=0;logical=0;allocated=0;seen=set()
   for b,ds,fs in os.walk(r,followlinks=False):
    for n in ds+fs:
     p=Path(b)/n;s=p.lstat();f.write(json.dumps({'path':str(p),'stamp':sig(p)},ensure_ascii=False)+'\n')
     if stat.S_ISREG(s.st_mode):
      count+=1;logical+=s.st_size;k=(s.st_dev,s.st_ino)
      if k not in seen:allocated+=s.st_blocks*512;seen.add(k)
   records.append({'path':str(r),'files':count,'logical_bytes':logical,'allocated_file_bytes':allocated,'status':'planned'})
 free_before=shutil.disk_usage(base).free;save('deleting',items=records,free_before_bytes=free_before)
 for r,row in zip(roots,records):
  assert not r.is_symlink() and r.resolve().is_relative_to(base.resolve());shutil.rmtree(r);assert not r.exists();row['status']='deleted';save('deleting',items=records,free_before_bytes=free_before)
 assert all(not r.exists() for r in roots);assert keep_inventory()==preserved,'Preserved code/model metadata changed'
 save('complete',items=records,free_before_bytes=free_before,free_after_bytes=shutil.disk_usage(base).free,code_and_models_metadata_unchanged=True,backup_created=False,all_targets_absent=True)
 assert json.loads((audit/'status.json').read_text())['all_targets_absent']
except Exception as e:save('blocked',error=repr(e));raise
