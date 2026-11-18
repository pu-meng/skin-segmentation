from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');P=B/'实验结果汇总.md';raw=P.read_bytes();s=raw.decode()
records=[]
for i in range(2,9):
 folder='A2-A4' if i<=4 else ('A5_A8' if i in (5,8) else 'A6_A7')
 detail=f'A{i}_E2_算子图' if i<=4 else (f'A{i}_完整模块图' if i in (5,8) else f'A{i}_完整改动模块图')
 records.append({'model':f'A{i}','general':f'A系列结构图/{folder}/A{i}_总图.svg','detail':f'A系列结构图/{folder}/{detail}.svg'})
assert len(records)==7 and len({r['model'] for r in records})==7
manifest=[]
for r in records:
 for field in ['general','detail']:
  p=B/r[field];ET.parse(p);png=p.with_suffix('.png');assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n');manifest.append({'model':r['model'],'kind':field,'svg':r[field],'svg_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
idx=B/'A系列结构图/发布清单_A2至A8.json'
# Exact published files; no model or training outputs touched.
idx.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));assert len(json.loads(idx.read_text()))==14
intro={
'A2':'保留A1的D2；E2改为局部卷积与单层Mamba，处理前的共同输入生成两路Softmax权重。E1、E3保留。',
'A3':'保留A1的D2；E2由局部特征产生Q、Mamba特征产生K/V，以四头空间注意力读取全局内容。E1、E3保留。',
'A4':'保留A1的D2；E2用局部卷积与AFNO并行，输入产生两路融合权重。AFNO取处理结果，不重复加输入。E1、E3保留。',
'A5':'E2用完整三层512通道Mamba与完整E1块并行，原输入经context生成两个独立Sigmoid通道权重；D3恢复基础双卷积，E3保持双向交叉门控。不是A1的D2。',
'A6':'E2两路处理后的Mamba与波传播内容共同产生两个Sigmoid权重；D3恢复基础双卷积。E3由D侧内容分别控制S与D，两侧输出处理都保留。',
'A7':'E2沿用A5的原输入门控；D3恢复基础双卷积。E3仍处理D内容来指导S，但D原样输出，不执行D侧finish；不是删除D的全部计算。',
'A8':'E2沿用A5的原输入门控；D3恢复基础双卷积。E3由D侧内容分别生成控制S和D的两组权重，两个门控头独立，不共享同一权重。'}
block='<!-- A2_A8_FIGURES_START -->\n## A2—A8｜总图与改动模块图\n\n'
block+='图内沿用E1／E2／E3及原解码阶段编号。以下不是逐项累加：A2—A4从A1替换E2；A5—A8不继承A1的D2，并恢复基础D3。\n\n'
block+=' |方案|总图|改动模块图|\n|---|---|---|\n'.lstrip()
for i in range(2,9):block+=f'|A{i}|[位置](#a{i}-network)|[内部计算](#a{i}-modules)|\n'
for r in records:
 a=r['model'];block+=f'\n<a id="{a.lower()}-figures"></a>\n\n## {a}｜结构图\n\n{intro[a]}\n\n<a id="{a.lower()}-network"></a>\n\n### {a}.1 总图\n\n![{a}整网位置]({r["general"]})\n\n<a id="{a.lower()}-modules"></a>\n\n### {a}.2 改动模块内部计算\n\n![{a}完整改动模块]({r["detail"]})\n'
block+='''
### 图注与代码来源

Conv／PW为卷积／1×1卷积，DW为深度卷积，LN为通道归一化，GAP为空间平均；＋为相加，×为相乘，Concat为通道拼接。A3注意力中的乘法为矩阵乘，按Q/K/V连接区分，不是普通逐元素门控。

A5—A8的波传播支路是带自身输入旁路的完整E1块，E2外层另加原输入；图中两处相加均保留实际代码，不能自行简化成裸D-WPO。A5、A7、A8使用两个独立Sigmoid通道门控，非Softmax竞争；A6读取两路处理后内容再生成权重。A5—A8图中上下两个面板分别是同一模型的E2与E3，不是两个候选方案。

代码相对`/home/PuMengYu/experiments/skin_segmentation/`：A2—A4见`operations/duala_E2_batch1_20261107/candidates.py`及`official_afno2d.py`；A5、A8见`source/e1232_fourarm_screen_20261106/candidate.py`；A6、A7见`source/e1232_matrix_20261106/candidate.py`。继承模块来自`source/e123_joint_20261105/candidate.py`、`source/bd_controls_20261105/candidate.py`与`source/three_module_impl_20261104/candidate.py`。本次只新增图与引用，不改变实验结构、预算或指标。
<!-- A2_A8_FIGURES_END -->

'''
assert '<!-- A2_A8_FIGURES_START -->' not in s
anchor='## 100轮实验续训范围';assert s.count(anchor)==1
out=s.replace(anchor,block+anchor,1)
assert P.read_bytes()==raw;P.write_text(out);assert P.read_text()==out
assert out.replace(block,'',1)==s
print(json.dumps({'document':str(P),'models':len(records),'svg':len(manifest),'png':len(manifest),'exact_file_readback':True,'existing_content_unchanged':True},ensure_ascii=False))
