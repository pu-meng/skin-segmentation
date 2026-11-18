from pathlib import Path
import json,csv,hashlib,time,shutil
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=Path(__file__).resolve().parent;OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/HAM_comparison_TTA_20261108');DOC=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
assert not OUT.exists();OUT.mkdir();shutil.copy2(DOC,OPS/'document_before.md')
LOG=ROOT/'log';split=ROOT/'data/processed/ham10000_group42_dedup_20261023/split.csv';ids=sorted(r['image_id'] for r in csv.DictReader(split.open()) if r['split']=='test');assert len(ids)==len(set(ids))==2020
entries=[]
def add(ident,name,src,module,fun,args,mode='direct'):
 s=json.loads(Path(src).read_text());assert s['images']==2020 and s['split_sha256']==sha(split) and sha(s['checkpoint'])==s['checkpoint_sha256']
 cp=Path(s['checkpoint']);run=cp.parent.parent
 hist=list(csv.DictReader((run/'history.csv').open()));assert len(hist)==300 and int(min(hist,key=lambda r:float(r['val_loss']))['epoch'])==s['selected_epoch']
 e=dict(id=ident,name=name,original_summary=str(src),original_summary_sha256=sha(src),reference=s,checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256'],epoch=s['selected_epoch'],split_file=str(split),split_sha256=s['split_sha256'],dataset='HAM10000',test_unique_ids=2020,env={},import_prefix=[],builder_module=module,builder_function=fun,builder_kwargs=args,logit_mode=mode,output=str(OUT/ident/'mirror4'))
 save(OUT/ident/'test_ids.json',ids);entries.append(e)
for seed in range(43,47):add('mambalite'+str(seed),'MambaLite seed'+str(seed),LOG/f'mambalite_publiccode_four_seeds_20261106/public{seed}/test/summary.json','final_test_previous_encoders','build',dict(name='mambalite'),'mambalite')
items=[('rmt_t','RMT-T＋UNet'),('rmt_s','RMT-S＋UNet'),('swiftformer_s','swiftformer_s＋UNet'),('mobileone_s1','mobileone_s1＋UNet'),('tinyvim_base','TinyViM-S＋UNet'),('fastervit0','FasterViT-0＋UNet'),('efficientformerv2_s0','efficientformerv2_s0＋UNet'),('mambavision_t','MambaVision-T＋UNet'),('su_rmt','SU-RMT'),('lowformer_e1','LowFormer-E1＋UNet'),('unet_base','UNet2D'),('cpubone_b0_bftraining','CPUBone-B0修订版＋UNet')]
for ident,name in items:
 if ident in ['rmt_t','rmt_s','fastervit0','mambavision_t','su_rmt']:
  src=LOG/f'八模型对比/HAM独立测试补齐_20261101/{ident}/test/summary.json';module='final_test_previous_encoders';fun='build';args=dict(name=ident)
 elif ident in ['tinyvim_base','unet_base']:
  src=LOG/f'瓶颈Mamba_最终测试_20261029/{ident}/test/summary.json';module='final_test_pipeline';fun='build_model';args=dict(arm=ident)
 elif ident in ['lowformer_e1','cpubone_b0_bftraining']:
  src=LOG/f'八模型对比/2026_encoder扩展_20261030/最终测试/{ident}/test/summary.json';module='Models.Encoder2026.model';fun='build_model';args=dict(name=ident,pretrained=False)
 else:
  src=LOG/f'八模型对比/TinyViM论文Encoder扩展_20261101/最终测试/{ident}/test/summary.json';module='Models.TinyViMPeers.model';fun='build_model';args=dict(name=ident,pretrained=False)
 add(ident,name,src,module,fun,args,'surmt' if ident=='su_rmt' else 'unet' if ident=='unet_base' else 'direct')
reuse=LOG/'mambalite_public_mirror4_test_20261105/mirror4/summary.json';s=json.loads(reuse.read_text());assert s['images']==2020 and json.loads((reuse.parent.parent/'audit.json').read_text())['complete'];old=json.loads((LOG/'八模型对比/HAM独立测试补齐_20261101/mambalite/test/summary.json').read_text());assert s['checkpoint_sha256']==old['checkpoint_sha256'] and s['split_sha256']==old['split_sha256']
save(OPS/'manifest.json',dict(document=str(DOC),created_unix=time.time(),gpu_uuid='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',rows=entries,reused_seed42=str(reuse),contract='HAM10000 dedup test2020;fixed original validationbest;no training;FP32 batch1 inverseflip mean true logits then sigmoid; no test selection'))
print('Prepared',len(entries),'missing TTA evaluations; seed42 reused')
