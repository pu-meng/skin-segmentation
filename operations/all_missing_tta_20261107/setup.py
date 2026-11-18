from pathlib import Path
import json,re,csv,hashlib,collections,time
R=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=R/'operations/all_missing_tta_20261107';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/all_missing_tta_20261107');DOC=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
assert not (OPS/'manifest.json').exists();OPS.mkdir(exist_ok=True);OUT.mkdir(exist_ok=False)
raw=DOC.read_bytes();text=raw.decode();(OPS/'original_document.md').write_bytes(raw)
same=DOC.parent/'all_experiment_inventory.json';inv=same if same.exists() else R/'operations/resume_tta_20261107/all_experiment_inventory.json';inventory=json.loads(inv.read_text());save(OPS/'inventory_snapshot.json',inventory)
app={m[1]:re.findall(r'`([^`]+summary.json)`',m[2]) for m in re.finditer(r'^### ([^\n]+)\n(.*?)(?=^### |\Z)',text,re.M|re.S)}
section=0;rows=[];allcounts=collections.Counter()
for l in text.splitlines():
 if re.match(r'## [1-4]\. ',l):section=int(l[3])
 if section not in (1,2,3) or not l.startswith('|'):continue
 v=l.strip('|').split('|')
 if len(v)!=4 or not re.fullmatch(r'\d+',v[1]):continue
 allcounts[section]+=1
 if v[-1]!='未取得':continue
 matches=[]
 for p in app[v[0]]:
  s=json.loads(Path(p).read_text())
  if abs(100*s['global']['dice']-float(v[2]))<.000051 and s.get('selected_epoch')==int(v[1]):matches.append((p,s))
 assert len(matches)==1,(v,len(matches));p,s=matches[0];arm=s.get('model');name=v[0]
 e={'id':f'S{section}-'+hashlib.sha256(name.encode()).hexdigest()[:10],'section':section,'name':name,'original_row':l,'original_summary':p,'original_summary_sha256':sha(p),'reference':s,'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256'],'epoch':int(v[1]),'split_sha256':s['split_sha256'],'status':'queued','blocker':None,'env':{},'python':'/home/PuMengYu/anaconda3/envs/medseg/bin/python','logit_mode':'four_stage' if section==1 or s['images']==518 else 'direct'}
 module=None;fun='build_candidate';args={'pretrained':False};prefix=[]
 if section==1:
  if arm in ['D4-BD','D4-B1','D4-B3D']:module='source.bd_controls_20261105.candidate';e['env']['BD_ARM']=arm;args.update(count=1 if arm=='D4-B1' else 3 if arm=='D4-B3D' else 0,med=arm!='D4-B1')
  elif arm=='D4-B3D-R2':module='source.b3d_repeat_20261106.candidate';args.update(count=3,med=True)
  elif arm.startswith('WPO-S'):module='source.wpo_skips_20261105.candidate';e['env']['WPO_ARM']=arm
  elif arm.startswith('E31-'):module='source.e31_skip_positions_20261106.candidate';e['env']['E31_ARM']=arm
  elif arm in ['E2','E3']:module='source.three_module_e23_20261104.candidate';args['arm']=arm
  elif arm=='E1.1':module='source.e1_variants.candidate'
  elif arm=='E1.2':module='source.e1_variants.gated_lambda';prefix=['source/e1_variants']
  elif arm=='E1.3':module='source.e1_additive.candidate'
  elif arm=='E2.1':module='source.e2_simplified.candidate'
  elif arm=='E2.2':module='source.e22_current_20261105.candidate'
  elif arm=='E1_E3':module='source.e1_e3_combo.candidate';args['arm']=arm
  elif arm=='B3-E1-E3.1':module='source.b3_e1_e31_20261105.candidate'
  elif arm=='B-WPO-MR':module='source.wpo_mr_20261105.candidate'
  elif arm=='DG-S0':module='source.direct_skip_gate_20261106.candidate'
  elif arm=='four_stage':module='experiment_four_stage_laplacian';fun='build_model'
 elif section==2:
  e['logit_mode']='mambalite'
  if '图2(b)' in name:module='source.mambalite_paper_candidate_20261105.candidate_model';fun='build_model';args={}
  elif '未去重' in name:
   module='experiment';fun='model';args={};prefix=['source/mambalite_未去重_seed42_20261106'];c=json.loads((R/prefix[0]/'protocol.json').read_text());e['python']=c['environment']+'/bin/python';e['api_module']='engine';e['split_file']=c['data']['split_file']
  else:module='final_test_previous_encoders';fun='build';args={'name':'mambalite'}
 else:
  if arm in ['E0','E1_E3']:module='source.e1_e3_combo.candidate';args['arm']=arm
  elif arm=='laplacian':module='experiment_laplacian_skip';fun='build_model'
  elif arm in ['conv','channel','spatial','region']:module='experiment_shallow_skip_refiners';fun='build_model';args['arm']=arm
  elif arm=='tinyvim_v2_skip_lora_r8':module='experiment_skip_lora';fun='build_model';args['arm']=arm
  elif arm=='tinyvim_v4_skip_fusion':module='experiment_v4_skip_fusion';fun='build_model';args['arm']=arm
  elif arm=='tinyvim_base_shared_adgbc_s0_fast':module='experiment_shared_adgbc_fast';fun='build_model';args['arm']=arm
 assert module,(name,arm)
 e.update(builder_module=module,builder_function=fun,builder_kwargs=args,import_prefix=prefix)
 split=Path(e.get('split_file',str(R/('data/processed/isic2018_256_dedup_seed42_20261031/split.csv' if s['images']==518 else 'data/processed/ham10000_group42_dedup_20261023/split.csv'))));e['split_file']=str(split)
 assert sha(split)==s['split_sha256'],(name,'split');assert sha(e['checkpoint'])==s['checkpoint_sha256'],(name,'checkpoint')
 members=list(csv.DictReader(split.open()));ids=sorted(r['image_id'] for r in members if r['split']=='test');assert len(ids)==len(set(ids))==s['images'];assert not set(ids)&{r['image_id'] for r in members if r['split']!='test'}
 e.update(dataset='ISIC2018' if s['images']==518 else 'HAM10000',test_unique_ids=len(ids),test_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),output=str(OUT/e['id']/'mirror4'));save(OUT/e['id']/'test_ids.json',ids)
 e['dedup_key']=hashlib.sha256(json.dumps([e['checkpoint_sha256'],e['split_sha256'],e['test_ids_sha256'],'FP32_batch1_logits_4views_threshold0.5']).encode()).hexdigest();rows.append(e)
assert len(rows)==37 and dict(allcounts)=={1:35,2:8,3:10}
manifest={'created_unix':time.time(),'document':str(DOC),'document_initial_sha256':hashlib.sha256(raw).hexdigest(),'table_rows':dict(allcounts),'missing_by_section':dict(collections.Counter(e['section'] for e in rows)),'expected_missing_rows':len(rows),'unique_inference_count':len({e['dedup_key'] for e in rows}),'inventory':str(inv),'same_directory_inventory_present':same.exists(),'inventory_sha256':sha(inv),'gpu':1,'gpu_uuid':'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b','deadline_unix':time.time()+72*3600,'rules':'read-only original best/source; CPU strict; first3 real noTTA PNG equality; identity/axis2/axis3/both inverse flip mean logits then sigmoid threshold .5; FP32 batch1; no CPU fallback; independent arm failures continue; no training','rows':rows}
save(OPS/'manifest.json',manifest);save(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},ensure_ascii=False))
