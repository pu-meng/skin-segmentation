from pathlib import Path
import json,re,statistics
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');P=B/'实验结果汇总.md';ROOT=Path('/home/PuMengYu/experiments/skin_segmentation/log/八模型对比')
# Read completed testing indices, then exact summaries, never their old displayed mean-Dice.
reports=[ROOT/'HAM完整测试汇总_20261101/results.md',ROOT/'TinyViM论文Encoder扩展_20261101/最终测试/results.md']
extra=ROOT/'2026候选骨干_HAM_20261104/最终测试'
if (extra/'results.md').exists():reports.append(extra/'results.md')
rows=[];seen=set()
for doc in reports:
 for line in doc.read_text().splitlines():
  links=re.findall(r'\[summary\]\(([^)]+)\)',line)
  if not links:continue
  path=(doc.parent/links[0]).resolve()
  if path in seen:continue
  d=json.loads(path.read_text());assert d.get('images')==2020 and d.get('global')
  if d.get('split') not in ['test',None]:raise AssertionError((path,d.get('split')))
  seen.add(path);name=line.split('|')[1].strip();rows.append(dict(name=name,dice=d['global']['dice']*100,source=str(path)))
# Five independent public-code results: precise source collection from known run tree and original seed42 test.
base=Path('/home/PuMengYu/experiments/skin_segmentation/log/mambalite_publiccode_four_seeds_20261106');seeds=[]
for seed in [42,43,44,45,46]:
 if seed==42:source=ROOT/'HAM独立测试补齐_20261101/mambalite/test/summary.json'
 else:
  candidates=list((base/f'public{seed}').glob('**/summary.json'))
  valid=[]
  for c in candidates:
   d=json.loads(c.read_text())
   if d.get('images')==2020 and d.get('global') and ('test' in str(c)) and not d.get('tta',False) and 'mirror' not in str(c):valid.append(c)
  assert len(valid)==1,(seed,[str(v) for v in valid]);source=valid[0]
 d=json.loads(source.read_text());seeds.append(dict(seed=seed,dice=d['global']['dice']*100,source=str(source)))
assert len(seeds)==5
block='''<!-- EXTERNAL_COMPARISONS_START -->
## 其他方法与Encoder对比

这些是A系列之外的既有实验，不应因精简结构图而丢掉。以下只展示HAM测试2020张的Global Dice，读取原summary的global.dice；不使用旧报告中的逐图均值，不重新训练或推理。

### 1. MambaLite公开代码版｜5次独立运行

固定去重划分，seed42、43、44、45、46各自训练300轮、验证选best后测试。不是把同一权重测五遍。未去重版和图2(b)重建版是另外两个身份，不计入这五次。

|Seed|无TTA测试Dice%|
|---|---:|
'''
for r in seeds:block+=f'|{r["seed"]}|{r["dice"]:.4f}|\n'
block+=f'\n五次无TTA运行均值 **{statistics.mean(r["dice"] for r in seeds):.4f}%**，样本标准差 **{statistics.stdev(r["dice"] for r in seeds):.4f}个百分点**。这是运行间统计，不是病例间标准差。\n'
block+='''
### 2. 其他网络与换Encoder对比

Encoder替换组使用原比较中的共同UNet解码器，不自动等同A系列当前四层Decoder。MambaLite、SU-RMT为完整网络对照，UNet2D为基础网络，不称为仅换Encoder的对照。以下为各方法原单次seed42结果；MambaLite五次统计见上节。

|方法|无TTA测试Global Dice%|
|---|---:|
'''
for r in sorted(rows,key=lambda x:x['dice'],reverse=True):block+=f'|{r["name"]}|{r["dice"]:.4f}|\n'
block+='\nEfficientVMamba-T的旧测试队列记录为失败，本次未发现上述索引中的完整summary，不将其写成已有最终成绩。此处只回收现有结果，不授权补跑或新方法训练。\n'
block+='\n### 3. 原始来源\n\n'
for r in seeds:block+=f'- MambaLite seed{r["seed"]}：`{r["source"]}`。\n'
for r in rows:block+=f'- {r["name"]}：`{r["source"]}`。\n'
block+='<!-- EXTERNAL_COMPARISONS_END -->\n\n'
raw=P.read_bytes();s=raw.decode();assert '<!-- EXTERNAL_COMPARISONS_START -->' not in s
anchor='## 100轮实验续训范围';assert s.count(anchor)==1
new=s.replace(anchor,block+anchor,1);assert P.read_bytes()==raw;P.write_text(new);assert P.read_text()==new
idx=B/'其他对比来源索引.json';idx.write_text(json.dumps({'mambalite_five_runs':seeds,'single_seed_comparisons':rows},ensure_ascii=False,indent=2));assert len(json.loads(idx.read_text())['mambalite_five_runs'])==5
print(json.dumps({'seeds':seeds,'comparisons':len(rows),'models':[r['name'] for r in rows],'mean':statistics.mean(r['dice'] for r in seeds),'sd':statistics.stdev(r['dice'] for r in seeds),'published':str(P)},ensure_ascii=False))
