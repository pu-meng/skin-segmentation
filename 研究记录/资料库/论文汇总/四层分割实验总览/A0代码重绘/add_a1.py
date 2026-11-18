from pathlib import Path
import ast,html,xml.etree.ElementTree as ET,resvg_py
O=Path(__file__).parent
code=ast.parse((O/'operator_figures.py').read_text(encoding='utf-8'));node=next(n for n in code.body if isinstance(n,ast.ClassDef) and n.name=='G');exec(compile(ast.Module(body=[node],type_ignores=[]),'figure_class','exec'))
s=(O/'A0_总图.svg').read_text(encoding='utf-8');s=s.replace('A0｜整网总图','A1｜整网总图').replace('A0新增或替换的位置','保留A0，另替换D2').replace('D2｜原双卷积','D2｜双分支融合').replace('拼接512 → 输出256','Conv 1×1 → 两路融合').replace('A0不含A1的D2双分支','A1只替换D2，E1／E2／E3保留')
ET.fromstring(s);(O/'A1_总图.svg').write_text(s,encoding='utf-8');(O/'A1_总图.png').write_bytes(resvg_py.svg_to_bytes(svg_string=s,width=1100,dpi=96,font_family='Microsoft YaHei'))
f=G(1000,1000,'A1｜D2全局与局部双分支融合')
f.box('in',300,80,400,'Concat(Decoder↑, Encoder Skip)')
f.box('proj',300,160,400,'Conv 1×1 · 512→256');f.edge('in','proj');f.text(500,245,'U · 256×16×16',20)
f.box('ln1',110,290,280,'LayerNorm');f.box('ln2',610,290,280,'LayerNorm');f.edge('proj','ln1');f.edge('proj','ln2')
f.box('m',110,380,280,'Mamba');f.edge('ln1','m');f.box('r',610,380,280,'RepDW7');f.edge('ln2','r')
f.box('act',610,470,280,'GELU');f.edge('r','act');f.box('pw',610,560,280,'Conv 1×1 · 256→256');f.edge('act','pw')
f.op('mm',250,680,'×');f.edge('m','mm');f.box('alpha',40,590,120,'α');f.edge('alpha','mm','b','l',via=[(100,680)])
f.op('mc',750,680,'×');f.edge('pw','mc');f.box('beta',840,620,120,'β');f.edge('beta','mc','b','r',via=[(900,680)])
f.op('sum',500,810,'+');f.edge('mm','sum',via=[(250,765),(500,765)]);f.edge('mc','sum',via=[(750,765),(500,765)])
f.edge('proj','sum','l','l',via=[(20,182),(20,810)])
f.box('out',300,870,400,'Y · 256×16×16 → 原后续解码');f.edge('sum','out');f.text(500,970,'两条分支均读取完整U；不是各分128通道',21);f.save('A1_D2_算子图')
print('A1 total and D2 rendered')
