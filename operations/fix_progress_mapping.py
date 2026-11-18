from pathlib import Path
import subprocess
p=Path('/home/PuMengYu/experiments/skin_segmentation/scripts/实验进展.sh');raw=p.read_bytes();s=raw.decode();assert '# test-source registry v2' not in s
old='''    return entries


def test_record(e):'''
new='''    # test-source registry v2: link completed tests by stable run identity, not name/score.
    closeout = eight/'Closeout_Test6_20261107'
    mappings = {
        'CandidateA_E2_Paired100_20261107_retry/Full': ('AFull', '候选A完整·100轮'),
        'CandidateA_E2_Paired100_20261107_retry/NoE2': ('ANoE2', '候选A关闭E2·100轮'),
        'DualA_E2_FirstBatch100_20261107/NoE2': ('DNoE2', 'DualA瓶颈恒等·100轮'),
        'DualA_E2_FirstBatch100_20261107/Input': ('DInput', 'A2·原100轮阶段'),
        'DualA_E2_Cross_AFNO_GPU1_20261107/Cross': ('DCross', 'A3·原100轮阶段'),
        'DualA_E2_Cross_AFNO_GPU1_20261107/AFNO': ('DAFNO', 'A4·原100轮阶段'),
    }
    jobs = load(eight/'A234_RESUME100to300_20261107/protocol.json').get('jobs', {})
    queue = load(root/'operations/a234_resume300_20261107/queue_status.json')
    for e in entries:
        resolved = e['base'].resolve()
        for rel, (job, label) in mappings.items():
            if resolved == (eight/rel).resolve():
                e.update(summary=closeout/job/'none/summary.json', mirror=closeout/job/'mirror4/summary.json', name=label, total=100)
        for arm in ('R0','R1','R2','R3'):
            if resolved == (eight/'TinyViM_E1232_FourArm_RESUME100to300_20261106'/arm).resolve():
                target=eight/'TinyViM_E1232_FourArm_RESUME100to300_20261106/final_test'/arm
                e.update(summary=target/'test/summary.json', mirror=target/'mirror4/summary.json')
        if resolved == (eight/'DecoderDualA_RESUME100to300_20261106/DualA').resolve():
            target=eight/'DecoderDualA_RESUME100to300_20261106/final_test/DualA'
            e.update(summary=target/'test/summary.json', mirror=target/'mirror4/summary.json', name='A1·原DualA300轮')
        for arm, job in jobs.items():
            if e['run'].resolve() == Path(job['resume_run']).resolve():
                target=eight/'A234_RESUME100to300_20261107'/arm/'final_test'
                e.update(total=300, name=arm+'·续至300轮', summary=target/'none/summary.json', mirror=target/'mirror4/summary.json', continuation=True)
                if arm not in queue.get('completed', []) and queue.get('job') != arm and queue.get('status') in ('train','preflight','test'):
                    e['queue_note']='已排队，等待GPU0前一组训练及测试完成'
    return entries


def test_record(e):'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("        return summary, audited", """        mirror = load(e.get('mirror', b/'final_test/mirror4/summary.json'))
        if mirror.get('global') and mirror.get('split') == 'test' and mirror.get('images') == summary.get('images') and mirror.get('checkpoint_sha256') == summary.get('checkpoint_sha256'):
            summary = dict(summary, mirror_dice=mirror['global']['dice'])
        return summary, audited""")
s=s.replace("f\"{m['dice']*100:.4f}\", avg_text])", "f\"{m['dice']*100:.4f}\", f\"{summary['mirror_dice']*100:.4f}\" if 'mirror_dice' in summary else '—', avg_text])")
s=s.replace("['实验','训练','测试状态','测试Dice%','末5轮秒/轮']", "['实验','训练','测试状态','无TTA Dice%','四视图TTA Dice%','末5轮秒/轮']")
s=s.replace("            if total and completed >= total:\n                desc", "            if e.get('queue_note'):\n                desc = e['queue_note']\n            elif total and completed >= total:\n                desc")
# Syntax-check the embedded Python before publishing.
embedded=s.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0];compile(embedded,str(p),'exec')
assert p.read_bytes()==raw;p.write_text(s);assert p.read_text()==s
subprocess.run(['bash','-n',str(p)],check=True)
print('SCRIPT_UPDATED_READBACK_VERIFIED')
