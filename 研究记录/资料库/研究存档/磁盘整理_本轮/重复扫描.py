import os,pathlib,json,hashlib,collections,time,datetime
base=pathlib.Path('/home/PuMengYu/研究存档/磁盘整理_本轮')
base.mkdir(parents=True,exist_ok=True)
roots=[pathlib.Path('/media/8T/PuMengYu'),pathlib.Path('/home/PuMengYu/nnUNet_workspace')]
status=base/'重复扫描状态.json'
def save_status(stage,**kw):
 p={'stage':stage,'time':datetime.datetime.now().astimezone().isoformat(),'pid':os.getpid(),**kw};t=status.with_suffix('.tmp');t.write_text(json.dumps(p,ensure_ascii=False,indent=2));t.replace(status)
def stamp(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
def digest(p,partial=False):
 h=hashlib.sha256(); size=p.stat().st_size
 with p.open('rb') as f:
  if partial:
   for pos in sorted(set([0,max(0,size//2-32768),max(0,size-65536)])):
    f.seek(pos);h.update(f.read(65536))
  else:
   while True:
    b=f.read(4*1024*1024)
    if not b:break
    h.update(b)
 return h.hexdigest()
by_size=collections.defaultdict(list);errors=[];inode=set();n=0
skip={'env','venv','.venv','site-packages','.git','.aws','.ssh','.cache','__pycache__'}
suffixes=('.pth','.pt','.ckpt','.model','.npy','.npz','.nii.gz','.zip','.tar','.tar.gz','.tar.zst','.tgz','.b2nd','.png','.jpg')
for root in roots:
 for folder,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if d not in skip and not (pathlib.Path(folder)/d).is_symlink()]
  for name in files:
   p=pathlib.Path(folder)/name
   if p.is_symlink() or not name.lower().endswith(suffixes):continue
   try:
    s=p.stat();n+=1
    if s.st_size<1024*1024:continue
    key=(s.st_dev,s.st_ino)
    if key in inode:continue
    inode.add(key);by_size[s.st_size].append({'path':str(p),'stamp':stamp(p)})
   except OSError as e:errors.append({'path':str(p),'error':str(e)})
save_status('sampling',files_seen=n,size_candidates=sum(len(a) for a in by_size.values() if len(a)>1))
matched=[];sampled=0
for size,group in sorted(by_size.items(),reverse=True):
 if len(group)<2:continue
 buckets=collections.defaultdict(list)
 for x in group:
  try:buckets[digest(pathlib.Path(x['path']),True)].append(x);sampled+=1
  except OSError as e:errors.append({'path':x['path'],'error':str(e)})
 matched.extend([a for a in buckets.values() if len(a)>1])
 if sampled%100<10:save_status('sampling',sampled=sampled,potential_groups=len(matched))
save_status('full_hash_verifying',potential_groups=len(matched),bytes_to_read=sum(x['stamp'][2] for g in matched for x in g))
verified=[];processed=0
for group in matched:
 buckets=collections.defaultdict(list)
 for x in group:
  p=pathlib.Path(x['path'])
  try:
   before=stamp(p)
   if tuple(x['stamp'])!=before:errors.append({'path':str(p),'error':'changed_before_hash'});continue
   sha=digest(p)
   if stamp(p)!=before:errors.append({'path':str(p),'error':'changed_during_hash'});continue
   buckets[sha].append(x)
  except OSError as e:errors.append({'path':str(p),'error':str(e)})
 for sha,items in buckets.items():
  if len(items)>1:verified.append({'sha256':sha,'size':items[0]['stamp'][2],'files':items})
 processed+=1
 if processed%10==0:save_status('full_hash_verifying',groups_processed=processed,groups_total=len(matched),verified_duplicate_groups=len(verified))
result={'roots':[str(r) for r in roots],'screened_files':n,'minimum_file_bytes':1024*1024,'excluded_directories':sorted(skip),'duplicate_groups':verified,'errors':errors,'completed_at':datetime.datetime.now().astimezone().isoformat(),'readonly':True}
(base/'内容重复核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
reclaim8=0
for g in verified:
 paths=[x['path'] for x in g['files'] if x['path'].startswith('/media/8T/PuMengYu/')]
 reclaim8+=max(0,len(paths)-1)*g['size']
summary={'verified_duplicate_groups':len(verified),'potential_reclaim_within_8T_bytes':reclaim8,'errors':len(errors),'report':str(base/'内容重复核验结果.json'),'note':'No files deleted, moved, compressed, or modified.'}
save_status('complete',**summary)
print(json.dumps(summary,ensure_ascii=False),flush=True)
for g in sorted(verified,key=lambda x:x['size']*(len(x['files'])-1),reverse=True)[:25]:print(json.dumps({'size':g['size'],'paths':[x['path'] for x in g['files']]},ensure_ascii=False),flush=True)
