from pathlib import Path
import re,json
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');p=B/'实验结果汇总.md';raw=p.read_bytes();s=raw.decode();archive=B/'历史实验与诊断汇总.md';assert not archive.exists();archive.write_bytes(raw)
part=s.split('## 1. 当前四层HAM结构结果')[1].split('## 2.')[0];rows={}
for l in part.splitlines():
 if l.startswith('|'):
  a=l.strip('|').rsplit('|',3)
  if len(a)==4 and re.fullmatch(r'\d+',a[1]):rows[a[0]]=a
specs=[('A0','旧候选A完整模型','旧候选A','E2三层128通道Mamba＋频谱统计；最深D3为WPO；最浅双源交叉Skip'),('A1','D2双修正完整模型（原DualA）','DualA','在A0上仅将D2换成Mamba／RepDW7双修正，E2和Skip保留'),('A2','局部全局输入选路','DInput','在A1上替换E2：局部DW7＋单层Mamba，输入生成两路通道权重'),('A3','局部查询全局','DCross','在A1上替换E2：局部生成Q、全局Mamba生成K/V'),('A4','局部＋AFNO输入选路','DAFNO','在A1上替换E2：局部DW7＋AFNO，输入生成两路通道权重'),('A5','Mamba＋WPO输入门控／交叉Skip','（R1）','完整三层512通道Mamba与WPO并行；X生成权重；基础D3；双向交叉Skip'),('A6','Mamba＋WPO内容门控／D侧控制','（BS）','两路处理后内容生成权重；基础D3；D控S、D控自身'),('A7','Mamba＋WPO输入门控／仅修正S','（AU）','X生成权重；基础D3；D提供控制但原样输出，仅修正S'),('A8','Mamba＋WPO输入门控／D侧控制','（R3）','X生成权重；基础D3；D控S、D控自身'),('A9','频谱E2／D侧控制','（R2）','保留A0 E2与WPO D3；仅改为D控S、D控自身'),('A10','Mamba＋WPO内容门控／交叉Skip','（BT）','两路处理后内容生成权重；基础D3；双向交叉Skip'),('A11','旧全局局部E2联合','旧E123.1','旧全局／局部多分支E2＋WPO D3＋双源交叉Skip，不是A0频谱E2')]
d=json.loads((Path('/media/8T/PuMengYu/experiments/skin_segmentation/Closeout_Test6_20261107/results.json')).read_text());new={r['job']:r for r in d['results']};inventory=[]
text='''# 完整模型实验汇总｜A系列

仅保留完整联合模型及其结构改版，统一使用A编号。单模块、去E2消融、基线对照、频谱与测速不在主表展开；旧材料完整保留在[历史备查](历史实验与诊断汇总.md)，不删除训练或数据。A编号是阅读编号，不重命名原run，也不代表新实验授权。

## 目录

- [完整模型测试成绩](#完整模型测试成绩)
- [各方案改了哪里](#各方案改了哪里)
- [目前怎么选](#目前怎么选)
- [旧图与数据来源](#旧图与数据来源)

## 完整模型测试成绩

HAM独立测试2020张，Global Dice百分比，无TTA和固定四视图TTA分列；全部已有测试。100轮与300轮分开标明，不把不同预算混成严格配对。A编号固定，不随分数变化重排身份。

|编号|完整模型|训练预算|best轮|无TTA Dice%|四视图TTA Dice%|
|---|---|---:|---:|---:|---:|
'''
for id,name,term,desc in specs:
 if term in new:
  r=new[term];epoch=r['selected_epoch'];none=r['none_dice'];tta=r['tta_dice'];budget=100;oldname=r['name'];source=r['result_root']
 else:
  matches=[v for k,v in rows.items() if term in k];assert len(matches)==1,(term,matches);r=matches[0];oldname=r[0];epoch=int(r[1]);none=float(r[2]);tta=float(r[3]);budget=300;source='历史实验与诊断汇总.md：'+oldname
 inventory.append(dict(id=id,name=name,original=oldname,budget=budget,best_epoch=epoch,none=none,tta=tta,description=desc,source=source));text+='|{}|{}|{}|{}|{:.4f}|{:.4f}|\n'.format(id,name,budget,epoch,none,tta)
# Same architecture, different runs remain visibly distinct, not new A IDs.
r=new['AFull'];r0=next(v for k,v in rows.items() if 'R0独立运行' in k)
text+='''
同结构不同运行不另编成新方案：

|归属|运行身份|训练预算|best轮|无TTA Dice%|四视图TTA Dice%|
|---|---|---:|---:|---:|---:|
'''
text+='|A0|本次配对Full独立训练|100|{}|{:.4f}|{:.4f}|\n'.format(r['selected_epoch'],r['none_dice'],r['tta_dice'])
text+='|A0同结构|原R0独立运行，不与A0合并成一次结果|300|{}|{}|{}|\n'.format(r0[1],r0[2],r0[3])
text+='\n## 各方案改了哪里\n\n'
for r in inventory:text+='- **{}｜{}**：{}。原名：{}。\n'.format(r['id'],r['name'],r['description'],r['original'])
text+='''
A1相对A0只换D2；A2—A4相对A1只换E2，所以相对A0一共改了D2和E2两处。A5—A10不是把瓶颈和D3都原样保留的同一种改法，不能仅归因于门控。

## 目前怎么选

- **原完整模型A0仍是已收集完整模型中的TTA最高：95.0706%。**
- **A1（原DualA）无TTA最高：94.9322%，TTA为95.0459%。** 当前先看A0与A1两条，不因微小单run差距声称稳定胜出。
- 新E2的A2／A3／A4都只训100轮：A2的TTA最高94.9629%，A4的无TTA最高94.7972%；没有一项在两列上都超过A1。不同预算，不能直接判最终上限。
- A3没有表现出值得立即优先推进的成绩优势；A5—A10也没有明确超过A0／A1，不继续自动展开更多门控组合。
- A0同结构的不同run差异明显，不能只挑最高一次当稳定水平；这里保留原R0读数以免掩盖这一点。

**下一步先决定：以A0原完整模型，还是A1的D2双修正版作为唯一继续修改的底座。** 这是供选择的读数，不启动训练、不补新矩阵、不自动更改模型。测试成绩已反复查看，当前选择属于探索，不能充作独立确认。

## 旧图与数据来源

- [A0固定旧图（基础位置、E1、E2、E3）](方案与结构图/主线方案_E123.1_三模块联合网络.md)
- [A1的D2双修正图](靠前方案图/六组补测/DualA的D2.svg)
- [A2—A4新E2区别图](靠前方案图/六组补测/新E2三个候选.svg)
- [A0到A1及新E2的修改范围图](靠前方案图/六组补测/候选A到DualA.svg)
- [历史原始summary路径、其余实验及诊断](历史实验与诊断汇总.md)
- 本次100轮原始结果：`/media/8T/PuMengYu/experiments/skin_segmentation/Closeout_Test6_20261107/results.json`。
- A系列旧名映射及成绩：同目录`A系列模型索引.json`；原始run不改名。原图内旧标识保留，由本表映射，不花额度重画。
'''
assert p.read_bytes()==raw;p.write_text(text);(B/'A系列模型索引.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2));assert p.read_text()==text and archive.read_bytes()==raw
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
 if not link.startswith('#'):assert (B/link).exists(),link
print(json.dumps({'main':str(p),'archive':str(archive),'A_models':len(inventory),'old_contents_preserved':True,'no_training_changes':True},ensure_ascii=False))
