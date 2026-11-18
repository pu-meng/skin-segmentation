import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import ast, csv, hashlib, json, math, subprocess, sys
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'source/mambalite_paper_candidate_20261105'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'source/comparison'))
from ham_fast_runner import baseline_api
api=baseline_api()
import torch
from torch import nn
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
torch.set_num_threads(1)
torch.manual_seed(42)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
public=ROOT/'references/code/MambaLiteUNet/models/MambaLiteUNet.py'
vm=ROOT/'references/code/EfficientVMamba/classification/models/vmamba.py'
checks=[]
def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    assert condition,name
before={str(p):sha(p) for p in [public,vm,ROOT/'source/comparison/ham_fast_runner.py',ROOT/'source/comparison/ham_fast_pipeline.py',ROOT/'source/comparison/baseline_runner_seed42.py']}
# Tests the exact published class and only its six AMF residual scalars in memory.
model=api.MambaLiteUNet().cpu()
scales={n:float(p.detach()) for n,p in model.named_parameters() if n.endswith('.skip_scale')}
negative={'assertion':'all AMF skip_scale initial values are zero','passed':all(v==0 for v in scales.values()),'observed':scales,'expected_failure':True}
save('alpha_original_negative_check.json',negative)
check('original_public_alpha_zero_check_correctly_fails',not negative['passed'])
params_before={n:p.detach().clone() for n,p in model.named_parameters()}
with torch.no_grad():
    for n,p in model.named_parameters():
        if n in scales:p.zero_()
changed=[n for n,p in model.named_parameters() if not torch.equal(p,params_before[n])]
check('only_six_AMF_branch_residual_scalars_changed_in_memory',set(changed)==set(scales) and len(changed)==6)
check('six_AMF_scalars_now_zero',all(float(p.detach())==0 for n,p in model.named_parameters() if n in scales))
check('public_parameter_count_unchanged',sum(p.numel() for p in model.parameters())==910627)
# Extract the installed upstream SS2D class unchanged, not TinyViM's split/pool variant.
source=vm.read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SS2D')
class_text=ast.get_source_segment(source,node)
(OUT/'upstream_SS2D_class_snapshot.py').write_text(class_text+'\n')
class ReferenceScan:
    captured=None
    identity=False
    @staticmethod
    def apply(u,delta,A,B,C,D,delta_bias,delta_softplus,nrows):
        ReferenceScan.captured=u.detach().clone()
        if ReferenceScan.identity:return u
        return selective_scan_ref(u,delta,A,B,C,D=D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
namespace=dict(torch=torch,nn=nn,math=math,repeat=repeat,SelectiveScan=ReferenceScan)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(vm),'exec'),namespace)
# Ratio1 is a DIAGNOSTIC configuration, not an approved architecture choice.
op=namespace['SS2D'](d_model=8,d_state=16,ssm_ratio=1.0,ssm_rank_ratio=1.0,d_conv=3,forward_type='v0').cpu()
real_norm=op.out_norm
op.out_norm=nn.Identity()
x=torch.arange(2*8*3*5,dtype=torch.float32).reshape(2,8,3,5)
ReferenceScan.identity=True
y=op.forward_corev0(x,channel_first=True)
seq=ReferenceScan.captured.reshape(2,4,8,15)
expected=torch.stack([x.flatten(2),x.transpose(2,3).flatten(2),x.flatten(2).flip(-1),x.transpose(2,3).flatten(2).flip(-1)],1)
check('rectangular_map_exact_four_scan_permutations',torch.equal(seq,expected))
check('inverse_permute_and_SUM_not_average',torch.equal(y,4*x.permute(0,2,3,1)))
op.out_norm=real_norm
ReferenceScan.identity=False
z=torch.randn(2,3,5,8,requires_grad=True)
result=op(z)
check('upstream_complete_SS2D_CPU_reference_forward_finite',result.shape==z.shape and bool(torch.isfinite(result).all()))
result.square().mean().backward()
check('CPU_reference_input_gradient_finite',z.grad is not None and bool(torch.isfinite(z.grad).all()))
grads={n:{'present':p.grad is not None,'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),'norm':None if p.grad is None else float(p.grad.norm())} for n,p in op.named_parameters()}
check('CPU_reference_all_SS2D_parameter_gradients_present_finite',all(g['present'] and g['finite'] for g in grads.values()))
check('direction_specific_projection_storage',op.x_proj_weight.shape[0]==4 and all(op.x_proj_weight[k].data_ptr()!=op.x_proj_weight[j].data_ptr() for k in range(4) for j in range(k)))
check('direction_specific_A_D_storage',op.A_logs.shape==(32,16) and op.Ds.shape==(32,))
check('CUDA_never_initialized',not torch.cuda.is_initialized())
check('all_existing_source_hashes_unchanged',all(sha(p)==h for p,h in before.items()))
report={'scope':'server CPU diagnostic ONLY, not paper-aligned network implementation or formal preflight','device':'cpu','CUDA_initialized':torch.cuda.is_initialized(),'checks':checks,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'AMF_original_scalars':scales,'in_memory_changed_parameters':changed,'residual_actual_operand':'AMF chunks of self.norm(flatten(X)); not unnormalized raw X','upstream_SS2D_diagnostic_configuration':{'d_model':8,'d_state':16,'ssm_ratio':1,'d_conv':3,'forward_type':'v0','shape':[2,3,5,8]},'scan_backend':'installed mamba_ssm selective_scan_ref; CPU diagnostic substitution only; never training fallback','gradients':grads,'existing_source_sha256':before,'class_snapshot_sha256':sha(OUT/'upstream_SS2D_class_snapshot.py'),'formal_network_CPU_passed':False,'formal_batch8_GPU_passed':False,'formal_training_started':False}
save('cpu_diagnostic_report.json',report)
paths=json.loads((ROOT/'source/config/paths.json').read_text())
split=Path(paths['active_split'])
rows=list(csv.DictReader(split.open()))
counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']}
assert counts=={'train':6983,'val':1010,'test':2020}
protocol={'identity':'MambaLiteUNet paper-description-alignment CANDIDATE; NOT strict reproduction','state':'blocked_scientific_definition_not_frozen','seeds':[42,43,44,45,46],'physical_gpu':1,'resource_authorization':'user authorizes GPU1 sharing and sequential five runs; do not wait for idle; existing E1.3 untouched','channels':[16,32,48,64,96,128],'alpha_scope':'only six AMF skip_scale branch-residual scalars; init0; do not change SSM D or other residuals','alpha_operand_unresolved':'paper X_k vs public normalized x_chunk; requires explicit architecture interpretation','SS2D_verified':'row-major and column-major full flattened traversals plus reversals; each independent x_proj/dt_proj/A/D; inverse-permute and SUM; LN; not four whole 1D Mambas averaged','unresolved':['Mamba Fig2(b): projection -> DWConv3x3 -> SiLU -> SS2D -> LN, gated by SiLU(project K), then output projection; prose3.2 places SiLU before DWConv and omits final projection','Prose W_g,W_z are CxC; upstream VMamba default expands to2C while public 1D Mamba also expands2. Choose literal C-width versus full referenced default explicitly; cannot identify author hidden version','AMF equation residual operand raw X_k versus actual public normalized chunks; zeroing skip_scale alone does not resolve this operand'], 'data':{'split_file':str(split),'split_sha256':sha(split),'counts':counts,'read_only':True,'author_split_recovered':False},'training':{'initialization':'fresh independent from scratch per seed; no old checkpoint reuse','epochs':300,'batch_size':8,'input_size':[256,256],'loss':'official BceDiceLoss','optimizer':'AdamW','lr':0.001,'weight_decay':0.01,'betas':[0.9,0.999],'eps':1e-8,'amp':False,'scheduler':'CosineAnnealingLR','T_max':50,'eta_min':1e-5,'step_frequency':'once every epoch','selection':'minimum complete validation segmentation loss; earliest tie','loader':'MUST private seed-dependent sampler+augmentation RNG; shared ham_fast_pipeline hardcoded seed42 prohibited','preflight':'real CUDA physical GPU1 formal batch8 input256 forward/backward/optimizer; failure blocks; no smaller batch or CPU fallback; discard preflight weights'},'testing':{'after_each_seed':'strict load validation-best and full2020 test','tta':False,'threshold':0.5,'outputs':['2020 predictions','per_image_metrics.csv','summary.json','report.txt','12 visualizations','checkpoint/source/split hashes'],'metrics':'global confusion-count metrics plus per-image HD95 on resized256 pixel grid; preserve old HD95 conventions','aggregate':'five independent run metric mean and sample std(ddof=1), only after all five complete'},'execution':{'tmux':'one persistent session, actual foreground controller output piped through tee','singleton_lock':True,'failure_policy':'stop downstream, no auto retries or configuration changes','controller_deployed':False,'reason':'do not privately assemble a network while above scientific relationships unresolved'},'existing_source_sha256':before}
save('protocol_draft.json',protocol)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
save('resource_snapshot.json',{'gpu':gpu,'compute_processes':processes,'GPU1_sharing_authorized':True,'no_GPU_work_started':True})
print(json.dumps({'output':str(OUT),'passed':report['passed'],'failed':report['failed'],'CUDA_initialized':report['CUDA_initialized'],'counts':counts,'blocked':protocol['unresolved']},ensure_ascii=False,indent=2))
