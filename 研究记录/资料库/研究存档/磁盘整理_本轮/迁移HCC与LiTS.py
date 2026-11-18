import pathlib,os,json,subprocess,shutil,time,datetime,fcntl
job=pathlib.Path('/home/PuMengYu/研究存档/磁盘整理_本轮');job.mkdir(parents=True,exist_ok=True)
lock=(job/'migration.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
archive=pathlib.Path('/media/8T/PuMengYu/system_disk_archive/HCC_LiTS_20261021')
items=[('/home/PuMengYu/HCC','HCC'),('/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCReferencedCT','nnUNet_workspace/raw/Dataset013_HCCReferencedCT'),('/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver','nnUNet_workspace/raw/Dataset003_Liver'),('/home/PuMengYu/nnUNet_workspace/raw/Dataset004_LiverTumor','nnUNet_workspace/raw/Dataset004_LiverTumor')]
state={'pid':os.getpid(),'scope':'HCC metadata plus HCC/LiTS raw datasets only; no code, weights, preprocessed data, or duplicate deletion','items':[]}
def write():
 state['updated']=datetime.datetime.now().astimezone().isoformat();p=job/'迁移状态.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2));p.replace(job/'迁移状态.json')
def inventory(root):
 data={}
 for b,ds,fs in os.walk(root,followlinks=False):
  for n in ds+fs:
   p=pathlib.Path(b)/n;s=p.lstat();data[str(p.relative_to(root))]=[s.st_mode,0 if p.is_dir() and not p.is_symlink() else s.st_size,s.st_mtime_ns,os.readlink(p) if p.is_symlink() else None]
 return data
def active(root):
 hits=[];uid=os.getuid()
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   if p.stat().st_uid!=uid:continue
   cmd=(p/'cmdline').read_bytes()
   if '重复扫描.py'.encode() in cmd:continue
   cwd=os.readlink(p/'cwd')
   if cwd==str(root) or cwd.startswith(str(root)+'/'):hits.append([p.name,'cwd',cwd])
   for f in (p/'fd').iterdir():
    try:
     target=os.readlink(f)
     if target==str(root) or target.startswith(str(root)+'/'):hits.append([p.name,'fd',target])
    except OSError:pass
  except OSError:pass
 return hits
try:
 assert os.path.ismount('/media/8T'),'8T mount unavailable'
 archive.mkdir(parents=True,exist_ok=True)
 for source,relative in items:
  src=pathlib.Path(source);dst=archive/relative
  record={'source':source,'target':str(dst),'stage':'checking'};state['items'].append(record);write()
  if src.is_symlink():
   assert src.resolve()==dst.resolve() and dst.exists(),'Unexpected existing symlink'
   record['stage']='already_moved';write();continue
  assert src.is_dir() and not dst.exists(),'Missing source or destination conflict'
  stage=dst.with_name(dst.name+'.copying');old=src.with_name(src.name+'.migration_original_20261021')
  assert not old.exists() and not stage.exists(),'Previous migration remnants require review'
  use=active(src);assert not use,('Source currently in use',use)
  before=inventory(src);size=sum(v[1] for v in before.values() if v[3] is None and not __import__('stat').S_ISDIR(v[0]))
  assert shutil.disk_usage('/media/8T').free>size+16*1024**3,'Insufficient 8T free space with 16GiB reserve'
  dst.parent.mkdir(parents=True,exist_ok=True);stage.mkdir()
  record.update(stage='copying',file_entries=len(before),bytes=size);write();print('COPYING',src,flush=True)
  subprocess.run(['rsync','-aH','--no-owner','--no-group',str(src)+'/',str(stage)+'/'],check=True)
  record['stage']='checksum_verifying';write();print('VERIFYING',src,flush=True)
  check=subprocess.run(['rsync','-aHnci','--no-owner','--no-group','--delete',str(src)+'/',str(stage)+'/'],capture_output=True,text=True,check=True)
  assert not check.stdout.strip(),('Copy mismatch',check.stdout[:1000])
  assert inventory(src)==before,'Source changed during copying'
  assert inventory(stage)==before,'Destination inventory differs'
  assert not active(src),'Source became active; retained both copies'
  stage.rename(dst);src.rename(old)
  try:
   src.symlink_to(dst,target_is_directory=True)
   assert src.is_symlink() and src.resolve()==dst.resolve() and inventory(src)==before
  except Exception:
   if src.is_symlink():src.unlink()
   old.rename(src);raise
  record['stage']='linked_source_retained';write()
  # Explicit user-authorized relocation: release only the exact checksum-verified old copy.
  assert old.parent==src.parent and old.name.endswith('.migration_original_20261021')
  assert inventory(old)==before
  shutil.rmtree(old)
  assert not old.exists() and src.is_symlink() and dst.is_dir()
  record['stage']='complete';write();print('COMPLETE',src,'->',dst,flush=True)
 state['stage']='complete';write()
except Exception as e:
 state['stage']='blocked';state['error']=repr(e);write();print('BLOCKED',repr(e),flush=True);raise
print(json.dumps(state,ensure_ascii=False),flush=True)
