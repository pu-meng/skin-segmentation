from pathlib import Path
import shutil,re,json,hashlib
BASE=Path('/home/PuMengYu/论文汇总/四层分割实验总览');SRC=Path('/home/PuMengYu/experiments/skin_segmentation/研究记录/模块设计');SPEC=Path('/media/8T/PuMengYu/diagnostics/fixed_best_spectral_gates_20261106');copied=[]
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.read_bytes()==src.read_bytes(),'target collision '+str(dst)
 else:shutil.copy2(src,dst)
 assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest();copied.append(str(dst.relative_to(BASE)))
dest=BASE/'方案与结构图';dest.mkdir(exist_ok=True)
for src in SRC.glob('*.md'):
 text=src.read_text()
 def link(m):
  target=m.group(2)
  if target.startswith(('http:','https:','#','/')):return m.group(0)
  rel=target.split('#')[0];anchor=('#'+target.split('#',1)[1]) if '#' in target else ''
  p=(SRC/rel).resolve()
  if p.is_relative_to(SRC) and (p.suffix.lower() in ['.md','.svg','.png','.jpg','.jpeg','.pdf','.json']):return m.group(0)
  return m.group(1)+'('+str(p)+anchor+')'
 text=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',link,text)
 out=dest/src.name;assert not out.exists();out.write_text('> 集中阅读副本：保留原设计正文和图；文内历史状态不作为当前运行状态。来源：`'+str(src)+'`。旧编号含义见目录中的《阅读导航》。\n\n'+text)
for src in SRC.rglob('*'):
 if src.is_file() and src.suffix.lower() in ['.svg','.png','.jpg','.jpeg','.pdf','.json']:
  cp(src,dest/src.relative_to(SRC))
spectral=BASE/'频谱与门控分析';spectral.mkdir(exist_ok=True)
for n in ['报告.md','protocol.json','audit.json','bypass_correlations.json','validation_identity.json']:
 cp(SPEC/n,spectral/n)
for arm in ['CandidateA','B3-E1-E3.1','D4-B3D-R2']:
 for n in ['bands.png','hf_ratio.png','radial_curves.png','gates.png','branch_amplitudes.png','summary.json','node_manifest.json','per_image_spectrum.csv','per_image_gates.csv','per_image_pairs.csv','implementation_rectangular_hf.csv']:
  src=SPEC/arm/n
  if src.exists():cp(src,spectral/arm/n)
read='''# 从这里看方案和分析图

## 先看哪几套方案

|对象|打开的方案|重点图|身份说明|
|---|---|---|---|
|旧候选A（原完整模块）|[原主线文档](方案与结构图/主线方案_E123.1_三模块联合网络.md)|[E2瓶颈](方案与结构图/figures/新分工/E2_A_单向三层_频谱通道辅助.png) · [双源Skip](方案与结构图/figures/新分工/E3_双源两级多支完整图.png)|原文同时含旧联合和候选A；候选A为三层128通道Mamba＋频谱统计辅助|
|Mamba＋WPO双分支，R1/R3等|[瓶颈与Skip方案](方案与结构图/主线方案_E123.2_瓶颈双分支与Skip选择.md)|[整网](方案与结构图/figures/E123.2/E123.2_整网与瓶颈Skip.png) · [瓶颈展开](方案与结构图/figures/E123.2/E123.2_瓶颈内部展开.png) · [Skip展开](方案与结构图/figures/E123.2/E123.2_Skip内部展开.png)|图主要对应输入门控＋D侧控制；BT/BS的内容门控变化要结合结果汇总组成说明，不冒称全部变体共用完全相同接线|
|E1最深方向WPO|[E1位置说明](方案与结构图/E1_总图位置与放置说明.md)|[整网与模块图](方案与结构图/figures/28_E1_DWPO_整网与模块一体图.svg)|是替换D3，不是最浅Skip追加WPO|
|E3.1扫描＋动态卷积Skip|[E3.1原文](方案与结构图/E3.1_最浅Skip跨层选择融合.md)|[完整结构图](方案与结构图/figures/三模块/E3_1_最浅Skip跨层选择融合.png)|不是候选A的双源E3|
|旧全局／局部E2及历史候选|[E2设计稿](方案与结构图/E2_新设计独立草稿_全局局部多支互控_v1.md) · [模块方案](方案与结构图/模块方案.md)|[旧E2完整图](方案与结构图/figures/新分工/E2_新方案完整图.png)|旧稿状态与结构版本不可覆盖实际运行身份|
|当前DualA及第一批E2候选|[下一轮实验方案](下一轮实验方案.md)|当前没有从原目录核到这几个新候选的专用结构图，不能拿旧图冒充|已有实现：Input/Cross/AFNO；内部可跳过版另行准备，尚未开训|

复制的是原方案和已有配图，不删除来源、不搬权重、不改训练。相对图片链接保持可用；指向运行代码或外部资料的链接改为服务器原绝对路径。

## 频谱结果不是没有：图和原始读数已集中

[频谱报告](频谱与门控分析/报告.md) · [实验成绩](实验结果汇总.md)

对象均为固定best、HAM验证1010图；不是测试集排名。raw保留空间均值，demean为逐通道去空间均值；两种口径不要混看。

|对象|频带能量|高频占比|径向功率曲线|分支幅度|门控|
|---|---|---|---|---|---|
|旧候选A e173|[图](频谱与门控分析/CandidateA/bands.png)|[图](频谱与门控分析/CandidateA/hf_ratio.png)|[图](频谱与门控分析/CandidateA/radial_curves.png)|[图](频谱与门控分析/CandidateA/branch_amplitudes.png)|[图](频谱与门控分析/CandidateA/gates.png)|
|B3＋E1＋E3.1 e67|[图](频谱与门控分析/B3-E1-E3.1/bands.png)|[图](频谱与门控分析/B3-E1-E3.1/hf_ratio.png)|[图](频谱与门控分析/B3-E1-E3.1/radial_curves.png)|[图](频谱与门控分析/B3-E1-E3.1/branch_amplitudes.png)|[图](频谱与门控分析/B3-E1-E3.1/gates.png)|
|B3D原配置重跑 e129|[图](频谱与门控分析/D4-B3D-R2/bands.png)|[图](频谱与门控分析/D4-B3D-R2/hf_ratio.png)|[图](频谱与门控分析/D4-B3D-R2/radial_curves.png)|[图](频谱与门控分析/D4-B3D-R2/branch_amplitudes.png)|本轮没有动态门控记录|

每对象summary.json是汇总，per_image_spectrum.csv为逐图频谱读数；node_manifest.json说明接口。候选A implementation_rectangular_hf.csv是模块实际使用的矩形分频，不是诊断径向分频。

### 最先值得对照图看的三项

- 候选A E2输入／输出raw高频比0.105285／0.105265，基本不变；实际修正相对输入范数0.00195645，固定best关闭E2几乎不改变验证Dice。这些一起提示接口影响小，不等于直接证明可删。
- 候选A E2修正raw高频比约6.53e-7，去均值后为0.222339；不要只看后者就认为高频增强很强。
- 分支内部幅度、输出修正幅度与指标贡献不是同一件事。频谱高频不等于病灶边缘，不同特征分辨率也不是同一原图物理频率。
'''
(BASE/'阅读导航.md').write_text(read)
# Validate all local image links in copied design documents and navigation.
missing=[];checked=0
for doc in list(dest.glob('*.md'))+[BASE/'阅读导航.md']:
 for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
  path=link.split('#')[0]
  if not path or path.startswith(('http:','https:')):continue
  if Path(path).suffix.lower() in ['.svg','.png','.jpg','.jpeg']:
   checked+=1
   if not (doc.parent/path).exists():missing.append(str(doc.parent/path))
write={'copied_files':len(copied),'verified_by_sha256':True,'image_links_checked':checked,'missing_image_links':missing,'destination':str(BASE)}
(BASE/'集中整理清单.json').write_text(json.dumps(write,ensure_ascii=False,indent=2));print(json.dumps(write,ensure_ascii=False))
