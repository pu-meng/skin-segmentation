from pathlib import Path
import json,hashlib,subprocess,zipfile,datetime,time,sys,fcntl
root=Path('/home/PuMengYu/研究存档/20261020_论文汇总/方案1_问题定义与思路检索/候选编码器_ImageNet1K')
lock=(root/'材料核验.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
expected=json.loads((root/'材料核验依据.json').read_text());verified={}
variants={'SHViT':('SHViT-S1',6.3),'FasterViT':('FasterViT-0',31.4),'TinyViM':('TinyViM-S',5.6),'MambaVision':('MambaVision-T',31.8)}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,d):
 q=p.with_suffix('.writing');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
# Preserve obsolete download records before correcting current status.
for name in ['README_候选清单.md','下载结果.json']:
 p=root/name;b=root/(name+'.更新前备份')
 if p.exists() and not b.exists():b.write_bytes(p.read_bytes())
def snapshot():
 rows=[]
 for e in expected['expected_weights']:
  name=e['name'];d=root/name;p=d/'weights'/e['file'];code=d/'code';paper=next(x for x in expected['papers'] if x['name']==name);pp=Path(paper['path'])
  assert pp.is_file() and digest(pp)==paper['sha256']
  commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
  row={'name':name,'variant':variants[name][0],'official_classification_params_M':variants[name][1],'code_path':str(code),'commit':commit,'paper_path':str(pp),'paper_sha256':paper['sha256'],'weights_path':str(p),'expected_bytes':e['bytes'],'download_status':'pending','model_loading_tested':False,'unet_integration_completed':False,'training_started':False}
  if p.is_file() and p.stat().st_size==e['bytes']:
   key=(str(p),p.stat().st_mtime_ns,p.stat().st_size)
   if key not in verified:
    assert digest(p)==e['sha256'],'Weight hash mismatch: '+name
    assert zipfile.is_zipfile(p),'Expected a PyTorch zip checkpoint: '+name
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
    verified[key]=True
   row.update(download_status='verified',sha256=e['sha256'],bytes=e['bytes'],archive_crc='passed')
  else:
   parts=[q for q in (d/'weights').glob('*part') if q.is_file()];row['partial_bytes']=sum(q.stat().st_size for q in parts)
  rows.append(row)
 done=sum(x['download_status']=='verified' for x in rows);data={'updated_at':datetime.datetime.now().astimezone().isoformat(),'verified':done,'expected':4,'candidates':rows,'RMT':'excluded from current preparation; previous evidence retained; no further download attempts','strategy_changed':False,'training_started':False}
 atomic(root/'材料准备状态.json',data)
 lines=['# 候选编码器：论文、官方代码与ImageNet-1K权重','','**本次只准备材料，不修改训练策略，不接入UNet，不启动实验。**','',f'当前文件核验完成：{done}/4。RMT官方权重未取得，按用户允许减少候选的决定，暂不纳入本批；既有代码、论文及拒绝镜像证据保留。','','|候选|官方分类参数量|论文与代码|权重文件|','|---|---:|---|---|']
 for x in rows:
  state='已完成，SHA256与本机源文件一致、归档CRC通过' if x['download_status']=='verified' else '续传中：临时文件不算完成'
  lines.append(f"|{x['variant']}|{x['official_classification_params_M']}M|[论文]({x['name']}/paper.pdf) · [官方代码]({x['name']}/code/)|{state}|")
 lines+=['','## 直接使用的材料位置','','每个系列的 `code/` 保存固定Git提交的官方仓库，`paper.pdf` 为用户提供的论文，`weights/` 为官方公开链接取得的ImageNet-1K预训练权重。`材料准备状态.json` 记录真实路径、代码提交、文件大小、SHA256和完成范围。`下载清单.json` 保存原始官方来源；`材料核验依据.json` 保存本机源文件哈希与论文来源。','','## 限制','','- 参数量是官方分类模型表格值；接入UNet后的整网参数量尚未统计，不能保证整网已小于50M。','- TinyViM-S当前下载为作者公开的1000-epoch预训练版本；其余模型的预训练预算不能假定与之相同。','- SHA256用于核对本机文件与服务器副本，不称为与未发布的官方校验和一致。','- 已核验的是下载文件和归档完整性，尚未严格加载模型、核验多尺度接口或安装各模型训练依赖。','- 不修改RMT接入约定、已有统一解码器方案、数据划分、优化器、训练预算或旧实验结果。','- 用户引用的旧B组3D CT MoE报告保持原样，本批新的2D编码器材料不代表该旧结果可直接用于新模型。']
 (root/'README_候选清单.md').write_text('\n'.join(lines)+'\n')
 if done==4:
  atomic(root/'下载结果.json',rows)
  for x in rows:atomic(root/x['name']/'来源与校验.json',x)
 # Read back exact externally written status and index before reporting.
 assert json.loads((root/'材料准备状态.json').read_text())==data
 assert f'当前文件核验完成：{done}/4' in (root/'README_候选清单.md').read_text()
 return data
end=time.monotonic()+3*3600
while True:
 result=snapshot();print(json.dumps({'verified':result['verified'],'expected':4,'updated_at':result['updated_at']},ensure_ascii=False),flush=True)
 if result['verified']==4 or '--wait' not in sys.argv:break
 if time.monotonic()>end:raise TimeoutError('Material audit deadline reached; incomplete items remain explicitly marked')
 time.sleep(30)
