from pathlib import Path
import subprocess,concurrent.futures,json,urllib.request,hashlib,zipfile,sys,shutil
root=Path('/home/PuMengYu/研究存档/20261020_论文汇总/方案1_问题定义与思路检索/候选编码器_ImageNet1K');root.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(root).free>3*1024**3
items=[{'name':'SHViT','repo':'ysj9909/SHViT','variant':'SHViT-S1','params_M':6.3,'url':'https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth'}, {'name':'FasterViT','repo':'NVlabs/FasterViT','variant':'FasterViT-0','params_M':31.4,'url':'https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_0_224_1k.pth.tar'},{'name':'TinyViM','repo':'xwmaxwma/tinyvim','variant':'TinyViM-S','params_M':5.6,'url':'https://drive.google.com/drive/folders/1cLgX_Mlk9W5_nMtgwmLC71ap-RlvT7eW?usp=sharing','training_epochs':1000},{'name':'MambaVision','repo':'NVlabs/MambaVision','variant':'MambaVision-T','params_M':31.8,'url':'https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar'}]
(root/'下载清单.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
def run(x):
 d=root/x['name'];d.mkdir(exist_ok=True);code=d/'code';w=d/'weights';w.mkdir(exist_ok=True)
 try:
  if not (code/'.git').exists():subprocess.run(['git','clone','--depth','1','https://github.com/'+x['repo']+'.git',str(code)],check=True,timeout=180,capture_output=True)
  x['commit']=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip();x['code_path']=str(code)
  if x['name']=='TinyViM':
   deps=root/'_download_dependencies'
   subprocess.run([sys.executable,'-m','pip','install','--target',str(deps),'gdown'],check=True,timeout=120,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   sys.path.insert(0,str(deps));import gdown
   files=gdown.download_folder(url=x['url'],output=str(w)+'/',quiet=True,remaining_ok=True)
   assert files,'Google Drive folder returned no files'
  else:
   target=w/x['url'].split('/')[-1]
   if not target.exists():
    part=target.with_suffix(target.suffix+'.part')
    with urllib.request.urlopen(x['url'],timeout=60) as response,part.open('wb') as out:
     expected=response.headers.get('Content-Length');shutil.copyfileobj(response,out,1024*1024)
    if expected:assert part.stat().st_size==int(expected)
    assert part.stat().st_size>1024**2
    part.rename(target)
  records=[]
  for p in w.rglob('*'):
   if p.is_file() and p.name.endswith(('.pth','.pt','.pth.tar','.ckpt','.safetensors')):
    h=hashlib.sha256()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    rec={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'validation':'file size and SHA256 recorded; model loading not tested'}
    if zipfile.is_zipfile(p):
     with zipfile.ZipFile(p) as z:assert z.testzip() is None;rec['archive_crc']='pass';rec['archive_members']=len(z.namelist())
    records.append(rec)
  assert records,'No checkpoint found';x['weights']=records;x['status']='downloaded'
 except Exception as e:x['status']='blocked';x['error_type']=type(e).__name__;x['error']=str(e)[:700]
 (d/'来源与校验.json').write_text(json.dumps(x,ensure_ascii=False,indent=2));print(json.dumps(x,ensure_ascii=False),flush=True);return x
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,items))
(root/'下载结果.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
lines=['# 候选编码器：官方代码与ImageNet-1K权重','','范围：按此前最小型号、50M以下要求，每个系列一个最小型号；未下载整个ImageNet数据集，未安装训练依赖，未改旧实验，未启动训练。','参数量是官方分类模型表格值，不是接入UNet后整网参数量。','TinyViM-S官方当前公开链接为1000 epoch权重，不与其他模型假定同一预训练预算。','','|系列|型号|官方参数量M|状态|','|---|---|---:|---|']
for x in results:lines.append('|'+x['name']+'|'+x['variant']+'|'+str(x['params_M'])+'|'+x['status']+'|')
lines+=['','各系列 code/ 为GitHub代码，weights/ 为权重；来源与校验.json记录URL、代码commit、权重路径、大小及SHA256。','','验收范围：下载文件与归档检查，不等于模型严格加载、UNet多尺度接口接入、显存或分割效果验收。Mamba相关CUDA算子和版本兼容性尚未测试。','','用户引用的第二轮B组文档是旧3D CT MoE诊断报告，本批是新的2D编码器候选，未修改该报告、未把其结果套到新候选。']
(root/'README_候选清单.md').write_text('\n'.join(lines))
print('RESULT',sum(x['status']=='downloaded' for x in results),len(results),str(root),flush=True)
