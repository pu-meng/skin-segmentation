import os,pathlib,json,subprocess,hashlib,datetime,select,sys,fcntl
job=pathlib.Path('/home/PuMengYu/研究存档/磁盘整理_本轮')
lock=(job/'export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
statefile=job/'本地备份服务器状态.json'
def status(stage,**data):
 d={'stage':stage,'pid':os.getpid(),'time':datetime.datetime.now().astimezone().isoformat(),**data};t=statefile.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(statefile)
status('waiting_for_migration')
s=json.loads((job/'迁移状态.json').read_text())
if s.get('stage')!='complete':
 try:
  fd=os.pidfd_open(s['pid']); ready,_,_=select.select([fd],[],[],5400);os.close(fd)
  if not ready:raise TimeoutError('Migration did not finish in 90 minutes')
 except ProcessLookupError:pass
s=json.loads((job/'迁移状态.json').read_text());assert s.get('stage')=='complete',('Migration incomplete',s)
root=pathlib.Path('/media/8T/PuMengYu/system_disk_archive/HCC_LiTS_20261021')
names=['HCC','nnUNet_workspace/raw/Dataset013_HCCReferencedCT','nnUNet_workspace/raw/Dataset003_Liver','nnUNet_workspace/raw/Dataset004_LiverTumor']
def inventory():
 out={}
 for name in names:
  r=root/name;assert r.is_dir() and not r.is_symlink()
  for b,ds,fs in os.walk(r,followlinks=False):
   for n in fs:
    p=pathlib.Path(b)/n;assert p.is_file() and not p.is_symlink(),str(p)
    z=p.stat();out[str(p.relative_to(root))]={'size':z.st_size,'mtime_ns':z.st_mtime_ns}
 return out
before=inventory(); manifest={'source_root':str(root),'directories':names,'files':before,'count':len(before),'bytes':sum(p['size'] for p in before.values())}
(job/'本地备份源文件清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
status('streaming_to_local',files=len(before),bytes=manifest['bytes'])
h=hashlib.sha256();sent=0
with (job/'本地备份tar日志.txt').open('wb') as log:
 p=subprocess.Popen(['tar','-C',str(root),'-cf','-','--']+names,stdout=subprocess.PIPE,stderr=log)
 try:
  while True:
   b=p.stdout.read(4*1024*1024)
   if not b:break
   sys.stdout.buffer.write(b);h.update(b);sent+=len(b)
  sys.stdout.buffer.flush();rc=p.wait();assert rc==0,('tar exit code',rc)
 except BaseException:
  # Do not remove or alter source data on interruption.
  status('export_failed',streamed_bytes=sent);raise
assert inventory()==before,'Source changed during export; do not remove server copies'
result={'stage':'export_complete','pid':os.getpid(),'time':datetime.datetime.now().astimezone().isoformat(),'tar_sha256':h.hexdigest(),'tar_bytes':sent,'file_count':len(before),'source_file_bytes':manifest['bytes'],'source_unchanged':True,'server_data_deleted':False}
statefile.write_text(json.dumps(result,ensure_ascii=False,indent=2))
