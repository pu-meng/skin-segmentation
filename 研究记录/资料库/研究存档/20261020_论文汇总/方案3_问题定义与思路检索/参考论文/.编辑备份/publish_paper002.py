from pathlib import Path
import hashlib, json, re, datetime, os
root=Path('/home/PuMengYu/论文汇总/方案3_问题定义与思路检索/参考论文')
target=root/'论文研究问题数据库_方案3_10篇_Evidence扩展版.md'
backup_dir=root/'.编辑备份'
candidate=backup_dir/'Paper002_candidate_scientific_rewrite.md'
expected='7375049bf7fd4aa6e4201e7cb749a5ea3760a6db82d40f15492bca7ca295e05d'
old=target.read_bytes(); new=candidate.read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(old)==expected, 'CONCURRENT CHANGE: publication stopped'
a=b'<a id="paper-002"></a>'; b=b'<a id="paper-003"></a>'
def split(data):
 i=data.index(a); j=data.index(b,i); return data[:i],data[i:j],data[j:]
op,ob,osuffix=split(old); np,nb,nsuffix=split(new)
# Only the nine paper-002 table-of-contents labels may differ outside its body.
def mask_toc(data):
 return re.sub(rb'  - \[2\.[1-9] [^\r\n]*\]\(#paper-002-[1-9]\)',b'TOC_PAPER002',data)
assert mask_toc(op)==mask_toc(np), 'UNAUTHORIZED PREFIX CHANGE'
assert osuffix==nsuffix, 'UNAUTHORIZED SUFFIX CHANGE'
for n in range(1,10):
 assert nb.count(('id="paper-002-%s"'%n).encode())==1
 text=nb.decode('utf-8'); heading=re.search(r'^## 2\.%s (.+)$'%n,text,re.M).group(0)[3:]
 assert ('['+heading+'](#paper-002-'+str(n)+')').encode() in np
for image in ['figures_reading/Method_002.svg','figures_reading/Evidence_002.svg']:
 assert image.encode() in nb
 assert (root/image).is_file()
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
backup=backup_dir/(target.name+'.'+stamp+'.bak')
with backup.open('xb') as f: f.write(old)
assert sha(backup.read_bytes())==expected
# Recheck immediately before replacing the target.
assert sha(target.read_bytes())==expected, 'CONCURRENT CHANGE AFTER BACKUP'
temp=backup_dir/('publish_'+stamp+'.tmp')
with temp.open('xb') as f: f.write(new); f.flush(); os.fsync(f.fileno())
os.replace(temp,target)
readback=target.read_bytes(); assert readback==new
single=backup_dir/'Paper002_Tip-Adapter_重写单篇.md'
single.write_bytes(nb)
receipt={'target_path':str(target),'backup_path':str(backup),'before_sha256':expected,'after_sha256':sha(readback),'unchanged_regions_check':{'prefix_except_nine_toc_labels_identical':True,'suffix_identical':True,'paper001_and_paper003_to010_unchanged':True,'anchors_and_toc_match':True,'existing_image_links_preserved_and_exist':True},'remote_readback_exact_match':True,'paper002_chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',nb.decode())), 'local_database_path':'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/论文研究问题数据库_方案3_10篇_Evidence扩展版.md','local_single_path':'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/Paper002_Tip-Adapter_重写单篇.md','remote_single_staging':str(single),'scope':'Only paper002 body and its nine TOC labels; no image edits, experiments or configuration changes'}
receipt_path=backup_dir/'Paper002_重写收据.json'
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
print('READBACK_BEGIN\n'+nb.decode('utf-8')+'READBACK_END')
