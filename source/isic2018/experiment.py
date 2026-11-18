"""ISIC2018-only adapter around the existing locked HAM model/train/eval math.
No HAM protocol, cache, model implementation or output is changed.
"""
from pathlib import Path
from datetime import datetime
import argparse,csv,fcntl,json,os,shutil,signal,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(Path(__file__).parent)]
import torch
import ham_fast_runner as parent
from pretrained_unet2d import PretrainedUNet2D,prepare_imports
from final_test_pipeline import fixed_best_record,validate_outputs,ProgressLoader,training_ready
import data
LOG=ROOT/'log/ISIC2018_瓶颈对比_20261031'
ARMS=['tinyvim_base','tinyvim_v2','tinyvim_v2_no_proj','tinyvim_v1','tinyvim_v3','tinyvim_v2_ds_proj','unet_base','unet_v2','unet_v2_no_proj','unet_v1','unet_v3','unet_v2_ds_proj']
EXPECTED={'tinyvim_base':8675341,'unet_base':7763041,'tinyvim_v1':11358253,'unet_v1':10445953,'tinyvim_v2':10371085,'unet_v2':9458785,'tinyvim_v2_no_proj':8737293,'unet_v2_no_proj':7824993,'tinyvim_v2_ds_proj':10384909,'unet_v2_ds_proj':9472609,'tinyvim_v3':18744333,'unet_v3':17832033}

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);parent.write_json(p,v)
def build(arm,pretrained=True):
 prepare_imports('mambavision_t')
 if arm=='tinyvim_base':return PretrainedUNet2D('tinyvim_s',pretrained=pretrained,new_layers_seed=42)
 if arm=='unet_base':
  from unet2d import UNet2D
  return UNet2D()
 if arm=='tinyvim_v1':
  from experiment_v1 import build_experiment_model
  return build_experiment_model(pretrained=pretrained)
 if arm in ['tinyvim_v2','unet_v1']:
  from experiment_extra import build_model
  return build_model(arm,pretrained=pretrained)
 if arm=='unet_v2':
  from experiment_v2_triton_full import build
  return build()
 module='experiment_v3' if arm.endswith('_v3') else ('experiment_v2_no_proj' if arm.endswith('_no_proj') else 'experiment_v2_ds_proj')
 import importlib
 return importlib.import_module(module).build_model(arm,pretrained=pretrained)

def sources():
 paths=set()
 for directory in [ROOT/'source/isic2018',ROOT/'Models/TinyViM',ROOT/'source/bottleneck_mamba',ROOT/'references/code/TinyViM']:
  paths.update(directory.rglob('*.py'))
 paths.update(ROOT/'source/comparison'/n for n in ['ham_fast_runner.py','ham_fast_pipeline.py','baseline_runner_seed42.py','pretrained_unet2d.py','unet2d.py','project_paths.py'])
 paths.update((ROOT/'references/code/MambaLiteUNet').glob('utils.py'))
 paths.update((ROOT/'references/code/MambaLiteUNet/models').rglob('*.py'))
 paths.add(ROOT/'source/isic2018/run_queue.sh')
 return {str(p.relative_to(ROOT)):data.sha(p) for p in sorted(paths) if p.is_file()}

def prepare_protocol():
 LOG.mkdir(exist_ok=True)
 cache=data.verify();original=json.loads((ROOT/'source/config/protocol_v5_dedup_seed42.json').read_text())
 weights=ROOT/'Models/pretrained/tinyvim_s_1000e.pth';expected=original['models']['tinyvim_s']['pretrained_source']['sha256'];assert data.sha(weights)==expected
 t=original['training'].copy();t.update(updates_per_epoch=223,last_batch_size=4,total_optimizer_updates=66900,image_exposures=534000,num_workers=2,validation_batch_size=8)
 pp=original['preprocessing'].copy();pp['mask_resize']='PIL L nearest then >0; predefined ISIC plan, differs from historical HAM bilinear-mask cache'
 config={'version':'isic2018_dedup_seed42_20261031','dataset':'ISIC2018','arms':ARMS,'data':{'dataset':'ISIC2018','split_file':str(data.CACHE/'split.csv'),'split_sha256':cache['split_sha256'],'dedup_split_file':str(data.DEDUP/'split.csv'),'dedup_split_sha256':cache['dedup_split_sha256'],'counts':cache['counts'],'cache_root':str(data.CACHE),'cache_manifest_sha256':data.sha(data.CACHE/'cache_complete.json'),'preprocessing_reads_all_splits':True,'test_never_selects_weights':True},'training':t,'preprocessing':pp,'normalization':original['normalization'],'initialization':{'seed':42,'bottleneck_seed':44,'ImageNet_weight':str(weights),'ImageNet_sha256':expected,'HAM_segmentation_checkpoint_loaded':False,'preflight_weights_reused':False},'evaluation':{'selection':'minimum validation loss','threshold':.5,'tta':False,'test_images':518,'primary_metrics':['per-image Dice','per-image IoU','per-image HD95 pixels'],'completion':'300 epochs -> full258 validation -> fixed best ->518 test predictions/CSV/summary/report/12 visuals/artifact audit'},'model_implementation':{'tinyvim_v2':'unchanged explicit FP32 recurrence as original HAM TinyViM V2','unet_v2':'existing HAM UNet V2 direct Triton formal builder; strict equivalent state loading'},'source_sha256':sources(),'authorization':'User requested start experiments and reuse existing tmux sessions; approved12-arm documented matrix','models':{a:{'run_dir':str(LOG/'runs'/a/'run'),'initialization':'ImageNet encoder and fresh alignment/decoder/module' if a.startswith('tinyvim') else 'random'} for a in ARMS}}
 p=LOG/'protocol.json'
 if p.exists():assert json.loads(p.read_text())==config,'Frozen ISIC protocol differs'
 else:write(p,config)
 for relative,expected in config['source_sha256'].items():
  target=LOG/'source_snapshot'/relative
  if target.exists():assert data.sha(target)==expected
  else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
 write(LOG/'protocol_identity.json',{'protocol_sha256':data.sha(p),'dedup_split_sha256':cache['dedup_split_sha256'],'derived_cache_split_sha256':cache['split_sha256']})
 print('PROTOCOL_READY',data.sha(p),flush=True)
 return config

def protocol():
 p=LOG/'protocol.json';c=json.loads(p.read_text());identity=json.loads((LOG/'protocol_identity.json').read_text());assert data.sha(p)==identity['protocol_sha256']
 assert data.sha(data.CACHE/'split.csv')==c['data']['split_sha256']
 assert data.sha(data.DEDUP/'split.csv')==c['data']['dedup_split_sha256']
 assert data.sha(data.CACHE/'cache_complete.json')==c['data']['cache_manifest_sha256']
 for relative,expected in c['source_sha256'].items():assert data.sha(ROOT/relative)==expected,relative
 assert data.sha(Path(c['initialization']['ImageNet_weight']))==c['initialization']['ImageNet_sha256']
 return c

def close(loaders):
 for value in loaders.values():
  if hasattr(value,'close'):value.close()
  elif hasattr(value,'_iterator') and value._iterator is not None:value._iterator._shutdown_workers()

def run(arm,mode):
 config=protocol();work=LOG/'runs'/arm;work.mkdir(parents=True,exist_ok=True)
 with (work/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stage=work/('preflight' if mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
  if mode=='train':
   passed=json.loads((work/'preflight/preflight_status.json').read_text());assert passed['status']=='passed' and passed['protocol_sha256']==data.sha(LOG/'protocol.json')
   assert not any((stage/n).exists() for n in ['status.json','history.csv','checkpoints']),'Refusing overwrite'
  api=parent.baseline_api();api.seed_all(42);torch.set_num_threads(4);device=torch.device('cuda:0')
  if mode!='test':
   torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.65,0)
  model=build(arm,pretrained=mode!='test');params=sum(p.numel() for p in model.parameters())
  if arm in EXPECTED:assert params==EXPECTED[arm],(arm,params,EXPECTED[arm])
  info={'dataset':'ISIC2018','model':arm,'seed':42,'pid':os.getpid(),'parameters':params,'protocol_file':str(LOG/'protocol.json'),'protocol_sha256':data.sha(LOG/'protocol.json'),'split_sha256':config['data']['split_sha256'],'dedup_split_sha256':config['data']['dedup_split_sha256'],'physical_gpu':int(os.environ.get('EXPERIMENT_PHYSICAL_GPU','1')),'fresh_start':True,'test_data_accessed':False,'preprocessing_reads_all_splits':True,'test_used_for_model_selection':False,'training':config['training'],'source_sha256':config['source_sha256'],'engineering':{'runner':'existing ham_fast_runner.train/preflight, ISIC-only loader and metadata adapter','workers':2,'validation_batch':8}}
  if mode=='test':
   assert training_ready(stage),'Training and full validation must complete before test'
   fixed=fixed_best_record(stage)
   model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu'),strict=True)
   assert not torch.cuda.is_initialized(),'Test loading must precede CUDA initialization'
   write(work/'cpu_test_load_audit.json',{'strict_load_passed':True,'checkpoint_sha256':fixed['checkpoint_sha256'],'cuda_initialized_during_load':torch.cuda.is_initialized()})
   torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.65,0)
   output=work/'test';assert not output.exists(),'Refusing overwrite of test artifacts'
   from torch.utils.data import DataLoader
   ds=data.ISICDataset(data.CACHE,'test');assert not ds.training and len(ds)==518
   loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
   model=model.to(device).eval();criterion=api.BceDiceLoss().to(device);state={'arm':arm,'status':'testing','total_images':518,**fixed}
   result=api.evaluate(model,ProgressLoader(loader,work/'test_status.json',state),criterion,device,output)
   result.update(dataset='ISIC2018',split='test',model=arm,**fixed)
   write(output/'summary.json',result)
   (output/'report.txt').write_text(f'ISIC2018 {arm}；seed42；固定验证损失best第{fixed["selected_epoch"]}轮\n518张独立测试图；逐图Dice {100*result["per_image_mean"]["dice"]:.4f}%；IoU {100*result["per_image_mean"]["iou"]:.4f}%；HD95 {result["per_image_mean"]["hd95_pixels"]:.4f}像素\n测试不参与权重选择。\n')
   write(work/'test_artifact_audit.json',validate_outputs(output,{r['image_id'] for r in ds.rows},result));write(work/'test_status.json',dict(state,status='test_complete',completed_images=518));print('TEST_COMPLETE',arm,flush=True);return
  model=model.to(device);loaders,generator=data.loaders(config,api);criterion=api.BceDiceLoss().to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  write(stage/'provenance.json',info)
  if hasattr(model,'encoder'):write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
  signal.signal(signal.SIGUSR1,parent.request_stop)
  try:
   if mode=='preflight':
    # Check a model parameter and every inserted-module parameter after actual optimizer steps.
    insertion=None if arm.endswith('_base') else (model.align[-1][1] if arm.startswith('tinyvim') else model.encoders[-1][1])
    watched=list(insertion.parameters()) if insertion is not None else list(model.parameters())[:1]
    before=[p.detach().cpu().clone() for p in watched]
    report=parent.preflight(model,loaders,criterion,optimizer,device,stage,[])
    changed=sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,watched));assert changed>0
    report.update(protocol_sha256=info['protocol_sha256'],split_sha256=info['split_sha256'],parameters=params,parameter_tensors_updated=changed,formal_training_started=False,test_loader_constructed=False)
    write(stage/'preflight_status.json',report);print('PREFLIGHT_PASSED',arm,json.dumps(report),flush=True)
   else:
    parent.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
    # Parent math is reused unchanged; replace its legacy HAM report metadata before handoff.
    st=json.loads((stage/'status.json').read_text());assert st['epoch_completed']==300
    summary=json.loads((stage/'validation/summary.json').read_text());summary.update(dataset='ISIC2018',model=arm,dedup_split_sha256=config['data']['dedup_split_sha256']);write(stage/'validation/summary.json',summary)
    (stage/'validation/report.txt').write_text(f'ISIC2018 {arm}；seed42；300轮，66900次更新\n验证258张，按最小验证损失选best第{summary["selected_epoch"]}轮；完整指标见summary.json与逐图CSV。\n')
    print('TRAIN_AND_VALIDATION_COMPLETE',arm,flush=True)
  except BaseException as e:
   write(stage/'failure.json',{'error':repr(e),'mode':mode,'pid':os.getpid(),'time':datetime.now().astimezone().isoformat()});raise
  finally:close(loaders)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','preflight','train','test'],required=True);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
 if a.mode=='prepare':prepare_protocol()
 else:
  assert a.arm is not None;run(a.arm,a.mode)
if __name__=='__main__':main()
