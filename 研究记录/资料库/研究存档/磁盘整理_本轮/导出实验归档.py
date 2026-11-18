import os,pathlib,json,sys,subprocess,hashlib,datetime,fcntl
name=sys.argv[1];job=pathlib.Path('/home/PuMengYu/研究存档/磁盘整理_本轮')
assert name in ('old_8T','system_results')
lock=(job/(name+'.export.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if name=='old_8T':root=pathlib.Path('/media/8T/PuMengYu/nnUNet_workspace_archive');names=['.']
else:
 root=pathlib.Path('/home/PuMengYu/nnUNet_workspace')
 names=[p.name for p in sorted(root.iterdir()) if p.is_dir() and not p.is_symlink() and p.name.startswith('results')]
assert os.path.ismount('/media/8T')
# Explicitly exclude credential/config directories; no environment directories are selected.
skip={'.aws','.ssh','.codex','.claude','.git','credentials','auth.json','.env'}
def inventory():
 entries={}
 for selected in names:
  for b,ds,fs in os.walk(root/selected,followlinks=False):
   ds[:]=[n for n in ds if n not in skip]
   for n in ds+fs:
    if n in skip or n.startswith('.env.'):continue
    p=pathlib.Path(b)/n;k=str(p.relative_to(root));s=p.lstat()
    if p.is_symlink():entries[k]={'kind':'symlink','target':os.readlink(p),'mtime':s.st_mtime_ns}
    elif p.is_file():entries[k]={'kind':'file','size':s.st_size,'mtime':s.st_mtime_ns}
 return entries
before=inventory();manifest={'root':str(root),'directories':names,'entries':before,'files':sum(v['kind']=='file' for v in before.values()),'bytes':sum(v.get('size',0) for v in before.values())}
(job/(name+'.manifest.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
state=job/(name+'.export.json')
state.write_text(json.dumps({'stage':'streaming','pid':os.getpid(),'bytes':manifest['bytes'],'files':manifest['files']}))
cmd=['tar','-C',str(root),'-cf','-']
for n in skip:cmd+=['--exclude='+n]
cmd+=['--exclude=.env.*','--']+names
h=hashlib.sha256();sent=0
with (job/(name+'.tar.log')).open('wb') as log:
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
 while True:
  b=p.stdout.read(4*1024*1024)
  if not b:break
  sys.stdout.buffer.write(b);h.update(b);sent+=len(b)
 sys.stdout.buffer.flush();rc=p.wait();assert rc==0,('tar failed',rc)
assert inventory()==before,'Source changed during backup; original remains untouched'
result={'stage':'complete','tar_sha256':h.hexdigest(),'tar_bytes':sent,'source_unchanged':True,'server_deleted':False,'files':manifest['files'],'time':datetime.datetime.now().astimezone().isoformat()}
state.write_text(json.dumps(result,ensure_ascii=False,indent=2))
