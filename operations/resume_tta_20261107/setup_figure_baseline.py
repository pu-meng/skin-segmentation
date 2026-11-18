from pathlib import Path
import shutil,json,hashlib,subprocess
H=Path('/home/PuMengYu');B=H/'论文汇总/四层分割实验总览';S=H/'experiments/skin_segmentation/研究记录/模块设计';R=B/'旧候选A绘图底稿';R.mkdir(exist_ok=True)
entries=[('base','基础网络',S/'figures/draw_four_stage_decoder.py','figures/20_四层对应Decoder_基础版.svg'),('e1','最深D3方向WPO',S/'figures/draw_e1_integrated.py','figures/28_E1_DWPO_整网与模块一体图.svg'),('e2','瓶颈三层Mamba与频谱统计',S/'绘图源/候选A/draw_candidateA.py','figures/新分工/E2_A_单向三层_频谱通道辅助.svg'),('e3','最浅双源Skip',S/'figures/新分工/draw_e3_final.py','figures/新分工/E3_双源两级多支完整图.svg')]
manifest=[]
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert src.read_bytes()==dst.read_bytes(),str(dst)
 else:shutil.copy2(src,dst)
for key,title,src,fig in entries:
 target=B/'方案与结构图'/fig;assert target.exists() and src.exists()
 copy(src,R/'sources'/key/src.name);copy(target,R/'approved'/target.name)
 manifest.append(dict(id=key,title=title,source_original=str(src),source_snapshot=str((R/'sources'/key/src.name).relative_to(R)),approved=str((R/'approved'/target.name).relative_to(R)),published=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
copy(S/'figures/draw_module_figures.py',R/'sources/shared/draw_module_figures.py')
style=H/'.codex/skills/research-figure-style/scripts'
if style.exists():
 for src in style.rglob('*'):
  if src.is_file() and src.suffix in ['.py','.json','.yaml','.yml','.toml']:copy(src,R/'sources/shared/figure_style'/src.relative_to(style))
(R/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
runner='''#!/usr/bin/env python3
import argparse,json,hashlib,shutil,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'manifest.json').read_text())
p=argparse.ArgumentParser(description='旧候选A固定底图工具：不联网、不调用模型、不改正式图')
p.add_argument('action',choices=['list','check','export','render-e2'],nargs='?',default='list')
p.add_argument('--out',help='导出或试绘输出目录；必须是新目录')
a=p.parse_args()
if a.action=='list':
 for row in DATA:print(row['id'],row['title'],'\\n  图:',row['published'],'\\n  原源码:',row['source_original'])
elif a.action=='check':
 for row in DATA:
  b=(ROOT/row['approved']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
  assert Path(row['published']).read_bytes()==b,'正式图与固定底图不同: '+row['id']
  assert (ROOT/row['source_snapshot']).exists()
 print('CHECK_OK: 四张正式图与固定底图完全相同，源码快照齐全')
else:
 if not a.out:p.error('需要 --out，且必须是尚不存在的新目录')
 out=Path(a.out).expanduser().resolve()
 if out.exists():p.error('输出目录已存在，拒绝覆盖：'+str(out))
 out.mkdir(parents=True)
 if a.action=='export':
  for row in DATA:shutil.copy2(ROOT/row['approved'],out/Path(row['approved']).name)
  print('EXPORTED:',out)
 else:
  row=next(x for x in DATA if x['id']=='e2');script=out/'draw_candidateA.py';shutil.copy2(ROOT/row['source_snapshot'],script)
  subprocess.run([sys.executable,str(script)],check=True,cwd=out)
  svg=next(out.glob('*.svg'))
  import xml.etree.ElementTree as ET
  ET.parse(svg)
  print('E2_RENDER_OK:',svg)
  print('与已确认旧图字节一致:',svg.read_bytes()==(ROOT/row['approved']).read_bytes())
'''
(R/'tool.py').write_text(runner)
sh=H/'4090图片生成.sh';assert not sh.exists();sh.write_text('''#!/usr/bin/env bash
set -euo pipefail
exec python /home/PuMengYu/论文汇总/四层分割实验总览/旧候选A绘图底稿/tool.py "$@"
''');sh.chmod(0o755)
md='''# 4090图片生成：复用旧候选A底图

## 固定入口

正式基线图稿：`/home/PuMengYu/论文汇总/四层分割实验总览/方案与结构图/主线方案_E123.1_三模块联合网络.md`。

以这里的四张现有图为准，不重画、不重新设计风格；基础整网图仅作位置参照，不把它说成已画全三个新增模块的候选A总图。这是**绘图底稿基线**，不代表改变科学对照模型、实验配置或训练路线。

快捷脚本：`/home/PuMengYu/4090图片生成.sh`。不联网、不调用LLM／图片模型、不占GPU，不启动训练。默认只列出图和源码，不生成也不覆盖任何图。

```bash
# 查看四张图与原始生成代码的位置
bash /home/PuMengYu/4090图片生成.sh list

# 核对当前正式图与固定底图逐字节一致
bash /home/PuMengYu/4090图片生成.sh check

# 无需重绘，直接导出四张已确认的SVG（输出目录必须不存在）
bash /home/PuMengYu/4090图片生成.sh export --out /home/PuMengYu/候选A图导出

# 仅用原E2源码试绘到新目录，不触碰正式图（只需Python标准库）
bash /home/PuMengYu/4090图片生成.sh render-e2 --out /home/PuMengYu/候选A_E2试绘
```

重复使用export／render-e2时换一个尚不存在的输出目录；脚本拒绝覆盖已有目录，不会擅自清理。SVG可以直接在浏览器或Markdown预览中打开，讨论时无需再转PNG。

## 四张图的原始生成代码

|图|原始代码|
|---|---|
|基础网络|`/home/PuMengYu/experiments/skin_segmentation/研究记录/模块设计/figures/draw_four_stage_decoder.py`|
|E1最深方向WPO|`/home/PuMengYu/experiments/skin_segmentation/研究记录/模块设计/figures/draw_e1_integrated.py`|
|E2三层Mamba＋频谱统计|`/home/PuMengYu/experiments/skin_segmentation/研究记录/模块设计/绘图源/候选A/draw_candidateA.py`|
|E3双源Skip|`/home/PuMengYu/experiments/skin_segmentation/研究记录/模块设计/figures/新分工/draw_e3_final.py`|

固定副本、代码快照和映射：`/home/PuMengYu/论文汇总/四层分割实验总览/旧候选A绘图底稿/`。其中`approved/`是四张已确认图，`sources/`是找到的绘图源码及共享绘图器快照，`manifest.json`保存出处和哈希；**不声称这些源码已在Linux完整重现全部旧图**。

## 已知依赖，不再重复踩坑

- E2：纯Python标准库生成SVG，可单独在新目录运行。历史代码与已确认SVG可能有后续局部修订差异；以固定图为准，不自动覆盖。
- 基础网络／E1：依赖`draw_module_figures.py`、`FigureStyle`与matplotlib。FigureStyle原路径是`/home/PuMengYu/.codex/skills/research-figure-style/scripts/figure_style.py`。当前默认Python未发现matplotlib，不能直接保证原命令可运行；本次不安装、不更改环境。默认直接复用固定SVG。
- E3：原脚本导出PNG依赖resvg_py，且字体写死`C:/Windows/Fonts/msyh.ttc`，不能直接在Linux原样运行。原SVG已固定，默认复用；将来确需改E3时，只在工作副本里修改，先输出SVG，再按Linux现有字体和渲染环境导出PNG，不改原脚本。

## 后续省额度的工作方式

1. 未改变结构：直接引用／export已确认图，不读全部历史、不重新绘制。
2. 只改E2：只读E2原图、对应源码及本次确认的接线；不重画E1、E3、整网。
3. 先明确本次增删的节点、连线和残差起终点，再从源码快照复制到工作目录进行最小修改；不从零设计排版。
4. 新候选输出到自己的目录，不覆盖旧候选A。确认后再更新对应Markdown图链接；旧实验继续引用旧图。
5. 只进行一次针对改动位置的检查；只有检查失败才扩大排查。训练配置与科学结论不由绘图脚本更改。

普通复用不需要模型绘图调用；理解新结构、修改绘图代码和核对接线仍会用少量对话额度，不能保证完全零额度。
'''
p=H/'4090图片生成.md';assert not p.exists();p.write_text(md)
for action in [['check'],['export','--out',str(R/'首次导出验证')],['render-e2','--out',str(R/'E2源码试绘验证')]]:
 subprocess.run(['bash',str(sh),*action],check=True)
assert p.read_text()==md;assert sh.read_text().startswith('#!/usr/bin/env bash')
print('DELIVERED',p,sh)
