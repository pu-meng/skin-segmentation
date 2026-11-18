from pathlib import Path
import html,xml.etree.ElementTree as ET
import resvg_py
O=Path(__file__).parent
class Fig:
 def __init__(self,w,h,title):
  self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L7 3L0 6" fill="#50677b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif" fill="#23384c">'];self.text(w/2,42,title,28)
 def text(self,x,y,t,size=22):self.a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{html.escape(t)}</text>')
 def box(self,x,y,w,h,lines,new=False):
  self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{"#fff0dc" if new else "#edf3f9"}" stroke="{"#cc8127" if new else "#68859f"}" stroke-width="2"/>')
  for i,l in enumerate(lines):self.text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*29+7,l,21)
 def path(self,pts,dash=False):
  d='M'+' L'.join(f'{x} {y}' for x,y in pts);self.a.append(f'<path d="{d}" fill="none" stroke="#50677b" stroke-width="2.3" marker-end="url(#a)"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 def save(self,n):
  s=''.join(self.a)+'</g></svg>';ET.fromstring(s);(O/(n+'.svg')).write_text(s,encoding='utf-8');(O/(n+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,width=1100,dpi=96,font_family='Microsoft YaHei'))
f=Fig(1100,890,'A0｜整网总图')
f.text(550,80,'E0：四尺度基础网络；橙色：A0新增或替换的位置',21)
f.box(35,110,235,60,['输入图像 3×256×256']);f.path([(152,170),(152,205)])
ys=[205,365,525,685];cs=[64,128,256,512];sz=[64,32,16,8]
for i,y in enumerate(ys):
 f.box(35,y,235,72,[f'Encoder 第{i+1}尺度',f'对齐后 {cs[i]}×{sz[i]}×{sz[i]}'])
 if i<3:f.path([(152,y+72),(152,ys[i+1])])
f.box(410,685,235,72,['E2｜最深瓶颈','输入／输出 512×8×8'],True);f.path([(270,721),(410,721)])
f.box(795,685,270,72,['E1｜替换最深处理块','输出 512×8×8'],True);f.path([(645,721),(795,721)])
f.box(795,525,270,72,['D2｜原双卷积','拼接512 → 输出256']);f.path([(930,685),(930,615),(930,597)]);f.text(985,642,'上采样×2',18);f.path([(270,561),(795,561)]);f.text(500,547,'同层Skip；与上采样结果拼接',19)
f.box(795,365,270,72,['D1｜原双卷积','拼接256 → 输出128']);f.path([(930,525),(930,437)]);f.text(985,485,'上采样×2',18);f.path([(270,401),(795,401)]);f.text(500,386,'同层Skip；与上采样结果拼接',19)
f.box(410,205,235,82,['E3｜最浅Skip处理','读取两侧S、D','输出S′、D′'],True);f.path([(270,241),(410,241)]);f.path([(930,365),(930,315),(527,315),(527,287)]);f.text(725,337,'上采样×2得到D',18)
f.box(795,205,270,82,['D0｜原双卷积','Concat(D′, S′)','128 → 64通道']);f.path([(645,246),(795,246)])
f.box(795,110,270,60,['1×1输出头 → 插值×4','Sigmoid → 256×256分割']);f.path([(930,205),(930,170)])
f.text(550,815,'传播顺序：最深Encoder → E2 → E1 → D2 → D1 → E3与D0 → 输出',21)
f.text(550,852,'E1/E2/E3是模块编号，D0/D1/D2是解码阶段；A0不含A1的D2双分支',20);f.save('A0_总图')
f=Fig(1100,1390,'E1｜最深解码处理块内部计算')
f.text(550,81,'按 DStage.forward → DirectionalWaveOperator → 原WaveOperator实际调用绘制',19)
f.box(230,110,640,65,['输入 X：512通道、8×8']);f.path([(550,175),(550,215)])
f.box(230,215,640,75,['3×3深度卷积 → 逐位置Linear：512→1536','沿通道分成 u₀、v₀、z，各512通道']);f.path([(550,290),(550,330)])
f.box(230,330,640,62,['u₀、v₀分别做实数二维FFT，z留在空间域']);f.path([(550,392),(550,450)])
f.box(100,450,900,105,['方向频率项 λ = (1+a) kx² + (1−a) ky²；a = tanh(r)','r为逐隐藏通道可学习参数；kx、ky为水平、垂直角频率','得到 λ 的可学习幂次 Λ = λᵖ；传播时间 t 固定为1'],True);f.path([(550,555),(550,605)])
f.box(100,605,900,165,['U、V分别是u₀、v₀的频谱；d为正阻尼系数','阻尼结果 A = exp(−dΛt) · U','振荡结果 B = exp(−dΛt/2) · [cos(ωt)U + sin(ωt)/ω · (V+dΛU/2)]','ω = sqrt(Λ)，实际计算有极小值保护','按可学习混合系数 w 融合：T = A + w(B−A)']);f.path([(550,770),(550,820)])
f.box(230,820,640,105,['T → 逆FFT回到8×8 → 通道LayerNorm','与保留的 SiLU(z) 逐元素相乘','逐位置Linear：512→512，得到特征 Q']);f.path([(550,925),(550,980)])
f.box(230,980,640,80,['Q乘逐通道可学习γ，初值0.001','与本模块原输入相加：Y = X + γ⊙Q'],True)
f.path([(230,142),(48,142),(48,1020),(230,1020)]);f.text(112,950,'原X直通',18)
f.text(550,1120,'Y保持512通道、8×8，随后进入第一次上采样和D2',23)
f.text(550,1170,'d = Softplus(stage频谱参数＋通道参数)，带正下界保护',21)
f.text(550,1210,'p、w、γ均由训练学习；z门控是原算子内部已有路径',21)
f.text(550,1250,'E1替换整个最深双卷积块；没有额外FFN，也不重复叠加前后卷积',21)
f.text(550,1305,'Linear沿通道计算；图中⊙与乘号表示逐元素乘法',20);f.save('E1_内部图')
print('TWO_SVG_AND_PNG_RENDERED',O)
