"""Freeze isolated adapters; never modify legacy sources or experiment outputs."""
import ast,hashlib,json,sys,shutil
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
NAME='mambalite_publiccode_four_seeds_20261106'
SRC=ROOT/'source'/NAME;OPS=ROOT/'operations'/NAME;LOG=ROOT/'log'/NAME;MODELS=ROOT/'Models'/NAME
for p in (SRC,OPS,LOG,MODELS):p.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
# Snapshot actual imported training loop. Changes are metadata seed only, not math.
p=ROOT/'source/comparison/ham_fast_runner.py';text=p.read_text();tree=ast.parse(text)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
train=ast.get_source_segment(text,node).replace('seed=42,','seed=SEED,').replace('seed42\\n','seed{SEED}\\n').replace('checkpoints.mkdir(exist_ok=resume)','checkpoints.mkdir(exist_ok=True)')
header='''import csv,json,os,random,time\nfrom datetime import datetime\nimport numpy as np\nimport torch\nfrom pathlib import Path\nfrom ham_fast_pipeline import evaluate_fast\nfrom ham_fast_runner import baseline_api,write_json\nfrom pretrained_unet2d import sha256_file\nSTOP_REQUESTED=False\nSEED=None\n'''
(SRC/'public_train.py').write_text(header+train+'\n')
# Seed-dependent clone of existing loader, preserving the seed42 reference stream exactly.
p=ROOT/'source/comparison/ham_fast_pipeline.py';text=p.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='make_fast_loaders')
load=ast.get_source_segment(text,node).replace('validation_batch=8):','validation_batch=8, seed=42):').replace('torch.manual_seed(42)','torch.manual_seed(seed)').replace('manual_seed(42001)','manual_seed(seed*1000+1)').replace('manual_seed(42002)','manual_seed(seed*1000+2)')
(SRC/'seed_loaders.py').write_text('from ham_fast_pipeline import *\n'+load+'\n')
# Isolated reconstruction adapter: old controller is NEVER imported/executed.
old=ROOT/'operations/mambalite_finite_resume_20261105/experiment_revision.py'
text=old.read_text().replace("LOG=ROOT/'log/mambalite_fig2b_20261105'",f"LOG=ROOT/'log/{NAME}/fig2b'").replace("MODELS=ROOT/'Models/mambalite_fig2b_20261105'",f"MODELS=ROOT/'Models/{NAME}/fig2b'")
text=text.replace('from restore_state import restore_cpu','from reconstruction_restore import restore_cpu')
(SRC/'reconstruction_engine.py').write_text(text)
# Explicit boundary: source snapshot keeps old dormant aggregate function, but permitted launcher only seed42.
oldrun=ROOT/'log/mambalite_fig2b_20261105/seed42';oldmodel=ROOT/'Models/mambalite_fig2b_20261105/seed42'
rd=LOG/'fig2b/seed42';md=MODELS/'fig2b/seed42';rd.mkdir(parents=True,exist_ok=True);md.mkdir(parents=True,exist_ok=True)
for name in ['history.csv','selection.json','provenance.json']:
 target=rd/name
 if target.exists():raise FileExistsError(target)
 shutil.copy2(oldrun/name,target)
for name in ['latest.pth','best.pth']:
 target=md/name
 if target.exists():raise FileExistsError(target)
 shutil.copy2(oldmodel/name,target)
legacy=ROOT/'log/八模型对比/runs/protocol_v5_dedup_seed42/mambalite/seed42'
legacytest=ROOT/'log/八模型对比/HAM独立测试补齐_20261101/mambalite'
protocol=json.loads((ROOT/'source/config/protocol_v5_dedup_seed42.json').read_text())
from_old=dict(training_run=str(legacy),test_root=str(legacytest),policy='reuse only after CPU strict/source/split/history/output audit; mismatch excludes reuse; never auto fifth public training')
protected=[ROOT/'source/comparison'/n for n in ['ham_fast_runner.py','ham_fast_pipeline.py','baseline_runner_seed42.py','pretrained_unet2d.py','project_paths.py']]
protected += [ROOT/'source/config/paths.json',ROOT/'source/config/protocol_v5_dedup_seed42.json',ROOT/'references/code/MambaLiteUNet/models/MambaLiteUNet.py',ROOT/'references/code/MambaLiteUNet/utils.py',ROOT/'source/bottleneck_mamba/final_test_pipeline.py',old]
protected += list((ROOT/'source/mambalite_paper_candidate_20261105').glob('*.py'))+[ROOT/'source/mambalite_paper_candidate_20261105/protocol.json']
protected += [p for p in SRC.glob('*.py')]+[p for p in OPS.glob('*.py')]
split=ROOT/'data/processed/ham10000_group42_dedup_20261023/split.csv'
import csv
rows=list(csv.DictReader(split.open()));testids=sorted(r['image_id'] for r in rows if r['split']=='test')
dump(LOG/'test_ids.json',testids)
manifest=dict(identity='public author code four channel branches, not four spatial directions',authorization='public seeds43-46 from scratch plus ONE separately authorized frozen Figure2b seed42 continuation; old five-reconstruction queue remains cancelled',public_seeds=[43,44,45,46],reconstruction_seeds=[42],gpu=1,gpu_uuid='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',batches=[['public43','public44'],['public45','public46'],['fig2b42']],training=protocol['training'],public_validation='reuse ham_fast_runner.train; batched validation8 plus original batch1 recheck for potential best/ties within.001; final validation1010 batch1',seed_policy='each initialization seed; sampler generator state after isolated public model construction at that seed; Python/NumPy enhancement seed; worker seeds seed*1000+1/2',legacy_seed42=from_old,reconstruction_resume=dict(original_run=str(oldrun),original_model=str(oldmodel),input_latest_sha256=sha(md/'latest.pth'),input_best_sha256=sha(md/'best.pth'),original_history_sha256=sha(oldrun/'history.csv'),old_cancel_record=str(ROOT/'operations/mambalite_finite_resume_20261105/user_cancelled.json'),scope='resume only seed42 through300; own copies; old roots read-only; no old controller launch'),split_file=str(split),split_sha256=sha(split),test_ids_file=str(LOG/'test_ids.json'),test_ids_sha256=sha(LOG/'test_ids.json'),counts={'train':6983,'val':1010,'test':2020},sources={str(p):sha(p) for p in protected},minimum_disk_free_GiB=8,resource_policy='wait BD and B3D controller complete, phase exits0, training/validation audit and final_test audit2020, exact test IDs/predictions/CSV, best hash; main process exit; also wait pre-existing E123 CandidateA GPU1 100epoch validation-only controller exit0, then no PuMengYu GPU1 compute PID and at least12000MiB free; preserve other-user existing job without interference; never wait GPU0 independent queues',deadline_hours=168,failure_policy='block next batch on any failure; no retries,no cleanup,no config downgrade; ongoing sibling allowed to finish',testing='each seed validation-loss best, strict CPU load then fullHAM2020 batch1 threshold.5 noTTA; independent prediction/target confusion andHD95 readback plus twelve visualizations; no test selection',model_parameters=910627)
dump(OPS/'protocol.json',manifest)
print(json.dumps(dict(protocol=str(OPS/'protocol.json'),sources=len(protected),latest_sha256=manifest['reconstruction_resume']['input_latest_sha256']),indent=2))
