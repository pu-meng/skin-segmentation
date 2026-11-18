from pathlib import Path
import hashlib,json,datetime,shutil,os
root=Path('/home/PuMengYu/论文汇总'); stage=root/'.reroute_stage'
before=json.loads((stage/'manifest_before.json').read_text())
expected=json.loads((stage/'manifest_after.json').read_text())
hashf=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Only graphics and drawing source are compare-and-swap guarded; document edits by the mentor are normal.
for rel in expected:
    p=root/rel
    if rel in before:
        assert p.exists() and hashf(p)==before[rel], 'Concurrent graphics/source change: '+rel
    elif p.exists():
        raise RuntimeError('Unexpected existing target: '+rel)
for rel,h in expected.items(): assert hashf(stage/'publish'/rel)==h, 'Staged hash mismatch '+rel
backup=root/'figure_backups'/('reroute_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True)
doc_before=hashf(root/'当前汇总.md')
for rel in expected:
    p=root/rel
    if p.exists():
        q=backup/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(root/'当前汇总.md',backup/'当前汇总_只读快照.md')
for rel in expected:
    p=root/rel;tmp=p.with_name(p.name+'.reroute-new');shutil.copy2(stage/'publish'/rel,tmp);os.replace(tmp,p)
actual={rel:hashf(root/rel) for rel in expected}
assert actual==expected
result={'backup':str(backup),'figures_svg':len([x for x in actual if x.endswith('.svg')]),'figures_png':len([x for x in actual if x.endswith('.png')]),'sources':len([x for x in actual if x.endswith('.py')]),'document_before_sync':doc_before,'document_after_sync':hashf(root/'当前汇总.md'),'document_written_by_this_job':False,'all_readback_hashes_match':True,'hashes':actual}
(stage/'remote_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
(backup/'deployment_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
