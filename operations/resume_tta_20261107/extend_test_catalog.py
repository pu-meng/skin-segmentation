from pathlib import Path
import re,json,html,xml.etree.ElementTree as ET
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');summary=B/'实验结果汇总.md';s=summary.read_text();doc=B/'靠前实验方案_图文目录.md';old=doc.read_text();assert '## P13' not in old
rows=[]
for section in re.split(r'\n## ',s):
 if not section.startswith(('1. 当前','2. MambaLite','3. 其他')):continue
 for l in section.splitlines():
  if not l.startswith('|'):continue
  r=l.strip('|').rsplit('|',3)
  if len(r)==4 and all(re.fullmatch(r'\d+\.\d+|未取得',x) for x in r[-2:]):
   values=[float(x) for x in r[-2:] if x!='未取得']
   if max(values)>=94.9:rows.append(r)
assert len(rows)==18
# Read exact raw test summaries; require all displayed scores to match original Global Dice.
raws=[]
for root in [Path('/home/PuMengYu/experiments/skin_segmentation/log'),Path('/media/8T/PuMengYu/experiments/skin_segmentation')]:
 for p in root.rglob('summary.json'):
  if any(x in p.parts for x in ['validation','val']):continue
  try:d=json.loads(p.read_text());score=d['global']['dice']*100
  except (ValueError,KeyError,TypeError):continue
  if d.get('images')==2020 and ('test' in str(p) or 'mirror4' in str(p)):raws.append((score,str(p)))
evidence=[]
for r in rows:
 found={}
 for mode,value in zip(['none','tta'],r[-2:]):
  if value=='未取得':continue
  matches=[p for score,p in raws if f'{score:.4f}'==value]
  assert matches,(r[0],mode,value)
  # Distinct values identify these selected summaries; retain all matching provenance paths.
  found[mode]={'dice_percent':float(value),'summary_paths':matches}
 evidence.append({'name':r[0],'split':'test','images':2020,'metrics':found})
G=B/'靠前方案图/结构差异图'
extra=[
('三层Mamba＋MedNeXt Decoder原配置重跑','瓶颈增加三个完整Mamba残差块，同时将Decoder处理换成MedNeXt式块；是原配置重跑，不是原运行best3，也不是仅改Decoder的D4-BD。',['最深对齐特征','三层完整Mamba','MedNeXt式Decoder','原上采样／Skip边界'],None),
('最深WPO＋输入门控（E1.2）','仍在最深D3采用WPO；相对原E1，增加当前输入生成的控制以调节传播强度，不是R1的Mamba／WPO双分支融合门控。',['最深对齐特征 X','X生成传播控制','最深D3 WPO处理','其余Decoder保持'],None),
('Skip平滑／细节门控（E3）','只改最浅Skip，分开处理平滑结构与细节并门控融合；不是候选A双源两级多支E3，也不是E3.1扫描／动态卷积。',['最浅 S','平滑／细节分解','两路处理＋门控融合','回到原D0融合'], '方案与结构图/figures/三模块/E3_主图_整网位置与完整计算.svg'),
('四层底座＋拉普拉斯Skip','四层基础网络只在最浅Skip使用拉普拉斯5／2处理；不额外加入候选A瓶颈。与后面的历史laplacian分开记录。',['四层基础网络','最浅 S0','拉普拉斯5／2处理','原最浅融合＋D0'],'方案与结构图/figures/21_四层对应Decoder_拉普拉斯总图.svg'),
('瓶颈WPO并行split-Mamba／RepDW','瓶颈使用WPO与split-Mamba／RepDW处理并行；这是单独的B-WPO-MR候选，不等于完整512通道三层Mamba＋WPO的R1系列。简图仅表达已确认的并行分支身份，不补画未核实的内部融合公式。',['最深对齐特征','WPO ／ split-Mamba＋RepDW','瓶颈分支汇总','Decoder'],'方案与结构图/figures/20_四层对应Decoder_基础版.svg'),
('扫描＋动态卷积仅接S1（E31-S1）','把E3.1扫描／动态卷积融合放在S1，其他Skip保持基础版本；不是在S0和S1同时执行。下方E3.1原模块详图只看内部计算，其S0位置标注不适用于本变体。',['基础Encoder／Decoder','仅 S1 输入模块','扫描／动态卷积＋D选择','回到对应Decoder融合'],'方案与结构图/figures/三模块/E3_1_最浅Skip跨层选择融合.svg'),
('laplacian','历史TinyViM最浅Skip拉普拉斯5／2方案，测试2020张。与当前四层底座的拉普拉斯运行不是同一实验，不能因同名直接合并；旧图为该历史方案的阅读参照。',['历史TinyViM底座','最浅Skip','拉普拉斯5／2处理','历史Decoder输出'],'方案与结构图/figures/25_方案1_整网与模块一体图.svg'),
('tinyvim_v2_skip_lora_r8','历史TinyViM最浅Skip低秩更新方案，LoRA的秩为8；不把它称为当前四层网络的同协议改造。这里仅展示已核对的插入范围，不臆画具体低秩矩阵接法。',['历史TinyViM底座','最浅Skip','LoRA低秩更新 r=8','原融合与输出'],None)
]
existing={'旧候选A':'p01','DualA':'p02','E3.1/S0':'p03','（R1）':'p04','（BS）':'p05','（AU）':'p06','（E1）':'p07','（D4-B3）':'p08','（D4-BD）':'p11','WPO-S0':'p12'}
ids={}
for r in rows:
 matches=[v for k,v in existing.items() if k in r[0]]
 if matches:assert len(matches)==1;ids[r[0]]=matches[0]
for i,(name,desc,steps,img) in enumerate(extra,13):ids[name]=f'p{i:02d}'
assert set(ids)=={r[0] for r in rows}
rank=sorted(rows,key=lambda r:max(float(v) for v in r[-2:] if v!='未取得'),reverse=True)
table='## 测试Dice达到94.9%的方案\n\n**仅收录HAM测试集2020张的Global Dice；验证集、验证旁路及类别子集分数全部排除。** 无TTA／四视图TTA任一达到94.9%即收录，按两者最高值降序。原12项保留作参照，其中旧E123.1和R3未达本阈值，不混入下表。\n\n|方案与图|无TTA Dice%|四视图TTA Dice%|达标方式|\n|---|---:|---:|---|\n'
for r in rank:
 modes=[label for label,v in zip(['无TTA','TTA'],r[-2:]) if v!='未取得' and float(v)>=94.9]
 table+=f'|[{r[0]}](#{ids[r[0]]})|{r[-2]}|{r[-1]}|'+('、'.join(modes))+'|\n'
table+='\n18个达标运行：当前四层系列16个，历史不同底座2个（laplacian、Skip LoRA r8）；历史两项不作为严格同底座排名。当前收集范围内没有仅凭验证高分补入的方案。每个分数均核对原始测试summary的global.dice，路径见同目录`94.9测试筛选证据.json`。\n\n'
new=old.replace('覆盖截图中的12个方案，顺序对应截图，**不是本次重新排序**。','保留截图中的12个方案，并补入其他测试Dice达到94.9%的方案。原12项目录顺序对应截图；新增阈值表单独按最高测试Dice排序。',1)
new=new.replace('## 目录\n',table+'## 目录\n',1)
# Add entries into existing directory table, before P01 section.
pos=new.index('\n## P01');before=new[:pos]
for i,(name,desc,steps,img) in enumerate(extra,13):
 r=next(r for r in rows if r[0]==name);before+=f'|[P{i:02d} {name}](#p{i:02d})|{desc.split("；")[0].split("。")[0]}|{r[-2]}|{r[-1]}|\n'
new=before+new[pos:]
sections=''
for i,(name,desc,steps,img) in enumerate(extra,13):
 parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#526b87"/></marker></defs><rect width="1000" height="560" fill="white"/><g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif" fill="#203650">',f'<text x="500" y="40" text-anchor="middle" font-size="23">{html.escape(name)}</text>']
 for j,label in enumerate(steps):
  y=80+110*j
  if j:parts.append(f'<path d="M500 {y-40} L500 {y-8}" stroke="#526b87" stroke-width="2" marker-end="url(#a)"/>')
  parts.append(f'<rect x="100" y="{y}" width="800" height="70" rx="12" fill="#edf4fc" stroke="#8fa4ba"/><text x="500" y="{y+43}" text-anchor="middle" font-size="25">{html.escape(label)}</text>')
 parts.append('<text x="500" y="535" text-anchor="middle" font-size="18">结构范围简图；不补充未核实的内部公式</text></g></svg>');svg=''.join(parts);ET.fromstring(svg);p=G/f'P{i:02d}.svg';assert not p.exists();p.write_text(svg)
 sections+=f'\n## P{i:02d}\n\n**{name}**\n\n{desc}\n\n![{name}结构范围](靠前方案图/结构差异图/P{i:02d}.svg)\n'
 if img and (B/img).exists():sections+=f'\n![已有模块图或明确标注的阅读参照]({img})\n'
 sections+='\n[返回目录](#目录)\n'
new=new.replace('\n## 原始入口',sections+'\n## 原始入口',1)
assert len(re.findall(r'^## P\d+',new,re.M))==20
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',new):
 if not link.startswith('#'):assert (B/link).exists(),link
assert doc.read_text()==old;doc.write_text(new)
(B/'94.9测试筛选证据.json').write_text(json.dumps({'threshold':94.9,'metric':'global.dice * 100','split':'test','images':2020,'selected_count':len(rows),'new_sections':len(extra),'records':evidence},ensure_ascii=False,indent=2))
# Update obsolete screenshot-only entry label without changing scores.
s2=summary.read_text();s2=s2.replace('靠前实验方案图文目录（截图12项）','靠前实验方案图文目录（含测试≥94.9%补充）');summary.write_text(s2)
check=doc.read_text();e=json.loads((B/'94.9测试筛选证据.json').read_text());assert check==new and e['selected_count']==18
print(json.dumps({'updated':str(doc),'qualifying_test_runs':len(rows),'new_sections':len(extra),'total_sections':20,'selected_by_none':sum(float(r[-2])>=94.9 for r in rows),'raw_test_summaries_verified':True},ensure_ascii=False))
