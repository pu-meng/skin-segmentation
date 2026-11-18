import csv,json,hashlib,sys
from pathlib import Path
root=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_SKIP1_ISIC_SEED42_PATIENCE50_20261114')
w=root/'A1-skip1-patience50'; run=w/'run'
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
p=read(root/'protocol.json')
for path,digest in p['protected_sha256'].items(): assert sha(Path(path))==digest,path
assert sha(Path(p['config']['data']['split_file']))==p['config']['data']['split_sha256']
state=read(run/'status.json'); summary=read(run/'validation/summary.json')
with (run/'history.csv').open() as f: rows=list(csv.DictReader(f))
assert [int(r['epoch']) for r in rows]==list(range(1,84))
assert state['status']=='early_stopped_patience' and state['epoch_completed']==83 and state['best_epoch']==33 and state['epochs_without_improvement']==50 and state['optimizer_updates']==83*223
assert min(range(len(rows)),key=lambda i:float(rows[i]['val_loss']))+1==33
assert summary['selected_epoch']==33 and summary['images']==258 and summary['checkpoint_sha256']==sha(run/'checkpoints/best.pth')
with (run/'validation/per_image_metrics.csv').open() as f: metrics=list(csv.DictReader(f))
pred=list((run/'validation/predictions').glob('*.png')); visuals=list((run/'validation/visualizations').glob('*.png'))
assert len(metrics)==len(pred)==258 and len(visuals)==12
assert {r['image_id']+'_segmentation.png' for r in metrics}=={x.name for x in pred}
assert all((run/'checkpoints'/n).is_file() for n in ('best.pth','latest.pth','final.pth'))
assert not (run/'artifact_audit.json').exists() and not (w/'final_test').exists()
summary['dataset']='ISIC2018'; (run/'validation/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
a={'training_complete':True,'validation_artifacts_complete':True,'epochs':83,'max_epochs':300,'early_stopping_patience':50,'stopped_early':True,'epochs_without_improvement':50,'optimizer_updates':18509,'validation_predictions':258,'per_image_metrics':258,'visualizations':12,'selected_epoch':33,'test_evaluation_complete':False,'test_evaluation_deferred':True,'recovery':'Early stop was valid; original post-training checker incorrectly expected 300 history rows. Independently verified 83 rows, validation and checkpoint.'}
(run/'artifact_audit.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
(w/'early_stop_recovery_audit.json').write_text(json.dumps({'passed':True,'original_failure_preserved':True,'best_sha256':sha(run/'checkpoints/best.pth'),'explanation':a['recovery']},ensure_ascii=False,indent=2)+'\n')
print('RECOVERY_AUDIT_PASS',sha(run/'checkpoints/best.pth'))
