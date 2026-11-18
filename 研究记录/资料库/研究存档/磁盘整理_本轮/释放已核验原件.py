import pathlib,os,json,sys,shutil,datetime,fcntl
name=sys.argv[1];assert name in ('raw','old_8T','system_results')
proof=json.load(sys.stdin);assert proof['local_verified'] is True and proof['sha256']
job=pathlib.Path('/home/PuMengYu/研究存档/磁盘整理_本轮');lock=(job/(name+'.release.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.path.ismount('/media/8T')
if name=='raw':
 m=json.loads((job/'本地备份源文件清单.json').read_text());root=pathlib.Path(m['source_root']);names=m['directories'];expected={k:{'kind':'file','size':v['size'],'mtime':v['mtime_ns']} for k,v in m['files'].items()}
 assert str(root)=='/media/8T/PuMengYu/system_disk_archive/HCC_LiTS_20261021'
 aliases={x['target']:x['source'] for x in json.loads((job/'迁移状态.json').read_text())['items']}
else:
 m=json.loads((job/(name+'.manifest.json')).read_text());root=pathlib.Path(m['root']);names=m['directories'];expected=m['entries'];aliases={}
 assert str(root)==('/media/8T/PuMengYu/nnUNet_workspace_archive' if name=='old_8T' else '/home/PuMengYu/nnUNet_workspace')
 if name=='old_8T':assert names==['.']
 else:assert names and all('/' not in n and n.startswith('results') for n in names)
roots=[root/n for n in names]
record=job/(name+'.release.json')
if record.exists():
 prior=json.loads(record.read_text())
 if prior.get('stage')=='complete' and all(not p.exists() for p in roots):print(json.dumps(prior,ensure_ascii=False));sys.exit(0)
 raise RuntimeError('Previous release record requires review; no repeated deletion')
for r in roots:
 assert r.is_dir() and not r.is_symlink(),str(r)
 if str(r) in aliases:
  p=pathlib.Path(aliases[str(r)]);assert (not p.exists() and not p.is_symlink()) or (p.is_symlink() and p.resolve()==r.resolve()),'Original alias was replaced by a different object'
actual={}
for r in roots:
 for b,ds,fs in os.walk(r,followlinks=False):
  for n in ds+fs:
   p=pathlib.Path(b)/n;k=str(p.relative_to(root));s=p.lstat()
   if p.is_symlink():actual[k]={'kind':'symlink','target':os.readlink(p),'mtime':s.st_mtime_ns}
   elif p.is_file():actual[k]={'kind':'file','size':s.st_size,'mtime':s.st_mtime_ns}
assert actual==expected,'Source changed or contains unbacked files; release blocked'
hits=[];uid=os.getuid()
def inside(t):return any(t==str(r) or t.startswith(str(r)+'/') for r in roots)
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=uid:continue
  cwd=os.readlink(p/'cwd')
  if inside(cwd):hits.append([p.name,'cwd',cwd])
  for f in (p/'fd').iterdir():
   try:
    t=os.readlink(f)
    if inside(t):hits.append([p.name,'fd',t])
   except OSError:pass
  args=(p/'cmdline').read_bytes().split(b'\0')
  for a in args[1:]:
   t=a.decode(errors='replace')
   if '\n' not in t and len(t)<4096 and inside(t.split('=',1)[-1]):hits.append([p.name,'argv',t])
 except OSError:pass
assert not hits,('Source currently used; release blocked',hits[:12])
result={'stage':'releasing','job':name,'proof':proof,'entries':len(actual),'file_bytes':sum(v.get('size',0) for v in actual.values()),'items':[],'time':datetime.datetime.now().astimezone().isoformat()}
record.write_text(json.dumps(result,ensure_ascii=False,indent=2))
for r in roots:
 item={'path':str(r),'stage':'releasing'};result['items'].append(item);record.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 assert str(r).startswith('/media/8T/PuMengYu/') or (name=='system_results' and r.parent==pathlib.Path('/home/PuMengYu/nnUNet_workspace'))
 shutil.rmtree(r);assert not r.exists()
 if str(r) in aliases:
  alias=pathlib.Path(aliases[str(r)])
  if alias.is_symlink():alias.unlink()
  else:assert not alias.exists(),'Original alias changed during release'
  note=alias.with_name(alias.name+'.已转存本地.json');note.write_text(json.dumps({'original_server_path':str(alias),'local_backup':proof['local_path'],'restore_required_before_use':True},ensure_ascii=False,indent=2))
 item['stage']='removed';record.write_text(json.dumps(result,ensure_ascii=False,indent=2))
result['stage']='complete';result['all_selected_roots_absent']=all(not p.exists() for p in roots);result['time']=datetime.datetime.now().astimezone().isoformat();record.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
