from pathlib import Path
import re,html,xml.etree.ElementTree as ET,json,hashlib
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');G=B/'靠前方案图/结构差异图';G.mkdir(exist_ok=True)
# Compact connection cards, not a replacement for detailed inherited diagrams.
data=[
('旧候选A','候选A E2 + 最深WPO + 双向交叉Skip','Align3 → 候选A E2 → WPO替换D3 → 其余Decoder；最浅融合用双源Skip。E2是三层128通道Mamba＋频谱统计通道辅助，不是旧全局／局部E2。',['Align3','Mamba＋频谱统计 E2','WPO替换 D3','原D2／D1','双向交叉Skip＋D0'],['方案与结构图/figures/新分工/E2_A_单向三层_频谱通道辅助.svg','方案与结构图/figures/新分工/E3_双源两级多支完整图.svg']),
('DualA','旧候选A基础上，只替换D2处理','保留候选A的E2、最深WPO及双源Skip。上采样与Encoder同层特征拼接后，D2公共投影产生Z；输出为Z＋缩放的Mamba纯修正＋缩放的RepDW7修正。不是把瓶颈E2换成双分支。',['保留候选A E2／D3','D2拼接输入','公共投影 Z','Z＋Mamba修正＋RepDW7修正','保留候选A Skip／D0'],[]),
('E3.1/S0','只改最浅Skip：扫描＋动态卷积，由D选择','以基础网络为起点，S经全局扫描与局部卷积两路处理，D产生融合选择；输出再进入最浅Decoder原融合边界。不继承候选A E2或最深WPO。',['基础Encoder／Decoder','最浅 S','扫描路／动态卷积路','D产生融合选择','原最浅融合＋D0'],['方案与结构图/figures/三模块/E3_1_最浅Skip跨层选择融合.svg']),
('R1','完整Mamba＋WPO瓶颈，输入门控；双向交叉Skip','瓶颈两路读取同一输入X：完整三层512通道Mamba与WPO并行，由X生成融合权重；后接基础D3。Skip保留D控S、S控D两侧交叉控制。',['瓶颈输入 X','完整Mamba ／ WPO','X生成融合权重','基础D3／D2／D1','双向交叉Skip＋D0'],['方案与结构图/figures/E123.2/E123.2_瓶颈内部展开.svg','方案与结构图/figures/新分工/E3_双源两级多支完整图.svg']),
('BS','完整Mamba＋WPO瓶颈，内容门控；D侧控制Skip','与R1同类双路内容，但融合权重从处理后的两路内容联合生成，而非原始X。Skip仍处理S和D两侧内容；D控S，D也控自身。不是删去D内容路。',['瓶颈输入 X','完整Mamba ／ WPO','处理后内容生成权重','基础D3／D2／D1','D控S＋D控D → D0'],['方案与结构图/figures/E123.2/E123.2_Skip内部展开.svg']),
('AU','完整Mamba＋WPO瓶颈，输入门控；仅修正S','瓶颈与输入门控方案相同。Skip仍计算D侧内容供控制S，但D直接原样输出；只把修正后的S送回原融合。不是完全不计算D侧。',['瓶颈输入 X','完整Mamba ／ WPO','X生成融合权重','基础D3／D2／D1','D控S；D原样 → D0'],[]),
('E1','只用方向WPO替换最深D3','只替换最深Decoder处理块，不额外插入候选A E2，也没有候选A双源Skip；其他Decoder与Skip沿用基础网络。',['基础Encoder＋对齐','WPO替换最深 D3','原 D2','原 D1','原Skip＋D0'],['方案与结构图/figures/28_E1_DWPO_整网与模块一体图.svg']),
('D4-B3','只在瓶颈插入三层完整512通道Mamba','在最深对齐特征后串联三个完整Mamba残差块，然后进入原D3。它不是候选A E2内的128通道Mamba，也不附带候选A频谱统计辅助。',['最深对齐特征','完整Mamba残差块 1','完整Mamba残差块 2','完整Mamba残差块 3','原D3→D2→D1→D0'],[]),
('旧E123.1','旧全局／局部E2＋最深WPO＋双源Skip','原全局／局部多分支E2和候选A频谱统计E2是不同结构；E1与双源Skip共用同类设计。这个旧联合结果不能标成候选A。',['Align3','旧全局／局部多支 E2','WPO替换 D3','原D2／D1','双向交叉Skip＋D0'],['方案与结构图/figures/新分工/E2_新方案完整图.svg','方案与结构图/figures/新分工/E3_双源两级多支完整图.svg']),
('R3','完整Mamba＋WPO瓶颈，输入门控；D侧控制Skip','瓶颈为X生成权重的完整Mamba／WPO双分支；Skip改为D控S、D控自身。与BS的区别在瓶颈门控读原输入还是处理后内容；与R1区别在Skip控制方向。',['瓶颈输入 X','完整Mamba ／ WPO','X生成融合权重','基础D3／D2／D1','D控S＋D控D → D0'],['方案与结构图/figures/E123.2/E123.2_整网与瓶颈Skip.svg','方案与结构图/figures/E123.2/E123.2_瓶颈内部展开.svg','方案与结构图/figures/E123.2/E123.2_Skip内部展开.svg']),
('D4-BD','只将Decoder处理换成MedNeXt式块','保留基础Encoder及上采样、Skip连接边界，将Decoder处理换为MedNeXt式处理。不含D4-B3的瓶颈三层Mamba，也不等于DualA。这里只画已核实替换范围，不臆画MedNeXt内部扩张率等细节。',['基础Encoder＋对齐','MedNeXt式 D3','上采样＋Skip → D2','上采样＋Skip → D1','上采样＋Skip → D0'],[]),
('WPO-S0','只在最浅Skip追加WPO修正','保留原D3，不将它换成WPO；只对最浅Skip加入WPO修正，送回原最浅融合。与E1的核心区别是插入位置及原D3是否保留。',['基础Encoder／原D3','原D2／D1','最浅 S → WPO修正','回到原最浅融合','原 D0＋输出头'],[])
]
s=(B/'实验结果汇总.md').read_text();rows=[]
for line in s.splitlines():
 if line.startswith('|'):
  parts=line.strip('|').rsplit('|',3)
  if len(parts)==4 and all(re.fullmatch(r'\d+\.\d+|未取得',v) for v in parts[-2:]):rows.append(parts)
patterns=['旧候选A','DualA','E3.1/S0','（R1）','（BS）','（AU）','（E1）','（D4-B3）','旧E123.1','（R3）','（D4-BD）','WPO-S0']
metrics=[]
for term in patterns:
 matches=[x for x in rows if term in x[0]];assert len(matches)==1,(term,matches);metrics.append(matches[0])
# Order is screenshot order for easy correspondence, not a fresh ranking claim.
text='# 靠前实验方案：目录、组成与结构图\n\n覆盖截图中的12个方案，顺序对应截图，**不是本次重新排序**。Dice从当前《实验结果汇总》读取；后续新增测试可能改变名次。S指Encoder Skip，D指Decoder特征；WPO为方向波传播算子，TTA为四视图镜像融合。\n\n每节先给本方案的简短接线图，再放对应已有模块详图。接线图只表达位置与改动，不是训练代码，也不替代原详图。共用图注明适用范围，不把其他变体的图当本方案完整图。\n\n## 目录\n\n|方案|具体改动|无TTA Dice%|四视图TTA Dice%|\n|---|---|---:|---:|\n'
for i,((name,title,desc,steps,imgs),row) in enumerate(zip(data,metrics),1):text+=f'|[P{i:02d} {name}](#p{i:02d})|{title}|{row[-2]}|{row[-1]}|\n'
for i,(name,title,desc,steps,imgs) in enumerate(data,1):
 # Fixed reusable vertical SVG layout, no external renderer or AI image call.
 parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="690" viewBox="0 0 900 690"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#526b87"/></marker></defs><rect width="900" height="690" fill="white"/><g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif" fill="#203650">',f'<text x="450" y="44" text-anchor="middle" font-size="28">{html.escape(name)} · 位置与改动</text>']
 for j,label in enumerate(steps):
  y=85+j*112
  if j:parts.append(f'<path d="M450 {y-36} L450 {y-7}" stroke="#526b87" stroke-width="2" marker-end="url(#a)"/>')
  parts.append(f'<rect x="105" y="{y}" width="690" height="76" rx="12" fill="'+('#edf4fc' if j%2==0 else '#f0f7ef')+f'" stroke="#8fa4ba"/><text x="450" y="{y+46}" text-anchor="middle" font-size="24">{html.escape(label)}</text>')
 parts.append('<text x="450" y="668" text-anchor="middle" font-size="19">结构范围简图；分支内部及真实残差见本节说明与原模块图</text></g></svg>');svg=''.join(parts);ET.fromstring(svg);path=G/f'P{i:02d}.svg';assert not path.exists();path.write_text(svg)
 text+=f'\n## P{i:02d}\n\n**{name}｜{title}**\n\n{desc}\n\n![{name}结构范围图](靠前方案图/结构差异图/P{i:02d}.svg)\n'
 for image in imgs:
  assert (B/image).exists();text+=f'\n![{name}对应已有模块详图]({image})\n'
 if name=='BS':text+='\n上方原详图仅对应D侧控制Skip；瓶颈内容门控以本节说明和接线图为准，不套用输入门控图。\n'
 if name=='R1':text+='\n上方两张分别对应输入门控瓶颈与原双向交叉Skip，不使用R3的D侧控制Skip冒充R1。\n'
 text+='\n[返回目录](#目录)\n'
text+='\n## 原始入口\n\n- [全部实验成绩及来源](实验结果汇总.md)\n- [旧候选A固定图稿](方案与结构图/主线方案_E123.1_三模块联合网络.md)\n- [频谱结论与结果图](当前候选_频谱结论与方案图.md)\n- [当前第一批方案](下一轮实验方案.md)\n- 绘图复用说明：`/home/PuMengYu/4090图片生成.md`。本页简图生成源码保存在`靠前方案图/结构差异图/build_scheme_catalog.py`；可沿用固定模板局部改图，勿再次从零设计。\n'
p=B/'靠前实验方案_图文目录.md';assert not p.exists();p.write_text(text)
links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text());missing=[x for x in links if not x.startswith('#') and not (p.parent/x).exists()];assert not missing
assert len(re.findall(r'^## P\d+',p.read_text(),re.M))==12
# Existing summary is the user's active entrance: add one local link, no metric changes.
p2=B/'实验结果汇总.md';raw=p2.read_bytes();original=raw.decode();needle='\n';at=original.find(needle);line='\n**各方案组成与对应结构图：** [靠前实验方案图文目录（截图12项）](靠前实验方案_图文目录.md)\n';updated=original[:at+1]+line+original[at+1:];assert p2.read_bytes()==raw;p2.write_text(updated)
assert (B/'靠前实验方案_图文目录.md').read_text()==text;assert line in p2.read_text()
print(json.dumps({'document':str(p),'schemes':len(data),'compact_svg':len(list(G.glob('P*.svg'))),'inline_images':text.count('!['),'missing_links':missing,'summary_entry_verified':True},ensure_ascii=False))
