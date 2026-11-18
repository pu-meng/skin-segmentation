import os,sys,json,hashlib,subprocess,ast
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');HERE=ROOT/'source/three_module_e23_20261104';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
old=ROOT/'source/three_module_impl_20261104'
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in old.glob('*.py')}
(HERE/'E1_protected.json').write_text(json.dumps(protected,indent=2))
s=(old/'experiment.py').read_text()
s=s.replace('RUN=ROOT/\'log/TinyViM_三模块_E1_DWPO_HAM_20261104\'\nNAME=\'E1_DWPO\'',"ARM=os.environ['E23_ARM']\nRUN=ROOT/('log/TinyViM_三模块_'+ARM+('_HSA' if ARM=='E2' else '_FDCG')+'_HAM_20261104')\nNAME=ARM")
s=s.replace("p['model']['deepest']='replace entire D3 with official directional WaveOperator + gamma residual'","p['model']['new_module']=ARM+' only; E2 align bottleneck512x8x8 or E3 align highest skip64x64x64; pre conv -> mixer residual -> FFN residual -> post conv'")
a=s.index(" p['model']['implementation_choices']=");b=s.index(" p['source_sha256']",a)
s=s[:a]+" p['model']['implementation_choices']=json.loads((HERE/'implementation_choices.json').read_text())\n"+s[b:]
s=s.replace("HERE/'cpu_acceptance.json'","HERE/('cpu_'+ARM+'_acceptance.json')").replace("HERE/'gpu_acceptance.json'","HERE/('gpu_'+ARM+'_acceptance.json')")
s=s.replace('build_candidate(pretrained=True,d=True)','build_candidate(pretrained=True,arm=ARM)').replace('physical_gpu=1','physical_gpu=int(os.environ[\'EXPERIMENT_PHYSICAL_GPU\'])')
(HERE/'experiment.py').write_text(s)
choices={'status':'implementation choices, not all specified by user','pre_post':'one ordinary C->C 3x3 biasFalse + BN + GELU each; existing align separate; no whole-module outer residual','block':'one complete fused mixer residual, FFN C->4C->C (official BN/GELU) residual','gate':'A*(1+tanh(raw)*sigmoid(Conv1x1(B))); channelwise target width; raw0; three-cycle B->A C->B A->C from original features; parallel cross frequency','HSA_groups':[172,170,170],'HSA_fft':'ideal circular cutoff .25 cycles/pixel inclusive, ortho fp32, complementary masks','SS2D':'full channels, state16, rankceil(C/16), input1x1BN -> DW3SiLU -> official four directional cross_selective_scan -> LN -> output1x1BN; no split/pool','attention':'qkv512->256, 4heads64; low query + K/V MaxPool 2/3/5/6 vanilla attention; high official FourierAttention and RoPE unchanged AST, MultiDW dilation combinations1/3/5, norm/context/out preserved; dual low/high branch fusion replaces original weighted pool/frequency mixture explicitly','skip':'5x5 binomial blur replicate pad -> AvgPool2 -> bilinear up align_cornersFalse; high=x-low; low1x1+FullScan; highRepDW3 and DW1x7->7x1 concat1x1; parallel gates concat1x1','E2_only':True,'E3_only':True,'E4_7':False}
(HERE/'implementation_choices.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2))
def ident(pid):
 p=Path(f'/proc/{pid}/stat');return p.read_text().split()[21] if p.exists() else None
e1=ROOT/'log/TinyViM_三模块_E1_DWPO_HAM_20261104'
assert (e1/'controller.exitcode').read_text().strip()=='0'
audit=json.loads((e1/'final_test/E1/artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==2020 and audit['visualizations']==12
assert json.loads((e1/'run/status.json').read_text())['epoch_completed']==300
assert ident(2006888) is None,'E1 controller still alive'
# Current owned controllers have been inspected: only laplacian GPU0 test is active;
# all existing GPU1 tmux panes are dead. Both new arms reserve GPU1 in E2->E3 order.
reservations={'1':[]}
(HERE/'reservation_evidence.json').write_text(json.dumps({'E1_exitcode':0,'E1_audit':audit,'E1_epochs':300,'new_order':['E2','E3'],'physical_gpu':1,'GPU0_excluded':True},indent=2))
(HERE/'reservations.json').write_text(json.dumps(reservations,indent=2))
for p in HERE.glob('*.py'):ast.parse(p.read_text())
for arm in ('E2','E3'):
 r=ROOT/('log/TinyViM_三模块_'+arm+('_HSA' if arm=='E2' else '_FDCG')+'_HAM_20261104');r.mkdir(exist_ok=True)
 assert not (r/'controller_status.json').exists()
 with (r/'controller_console.log').open('x') as f:
  p=subprocess.Popen([PY,'-u',str(HERE/'wait_queue.py'),arm],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 print('DETACHED',arm,p.pid)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
