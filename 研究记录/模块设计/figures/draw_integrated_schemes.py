"""One complete figure per scheme: network and its expanded Skip in one canvas."""
from draw_module_figures import Diagram

def network(d,number,decoder_level=None):
 def n(k,x,y,t,c='projection'):return d.node(k,x,y,t,c,role=None)
 n('input',125,100,'输入 RGB\n3 ch · 256×256','input')
 n('stem',125,220,'Stem\n3→48；256→64')
 d.edge('input','stem')
 for i,(raw,c,size,y) in enumerate([(48,64,64,345),(64,128,32,455),(168,256,16,565),(224,512,8,675)]):
  n(f'e{i}',125,y,f'E{i} · {raw} ch\n{size}×{size}','input')
  special=(i==0 and decoder_level is None)
  n(f'a{i}',390,y,(f'对齐48→64 + 方案{number}' if special else f'1×1 对齐 {raw}→{c}'),'state' if special else 'projection')
  text=f'D{i} → 方案{number}（候选）\n{c} ch · {size}×{size}' if i==decoder_level else f'D{i} · {c} ch\n{size}×{size}'
  n(f'd{i}',665,y,text,'state' if i==decoder_level else 'output')
  d.edge(f'e{i}',f'a{i}',source='right',target='left');d.edge(f'a{i}',f'd{i}',source='right',target='left')
  if i:d.edge(f'e{i-1}',f'e{i}');d.edge(f'd{i}',f'd{i-1}',source='top',target='bottom')
 d.edge('stem','e0')
 n('head',665,170,'输出头\n64→1；插值×4\n256×256；Sigmoid','output');d.edge('d0','head',source='top',target='bottom')
 if number==1:
  a,t=d.nodes['d1'],d.nodes['a0']
  d.edge('d1','a0',source='top',target='bottom',route=[(a['x'],a['top']),(a['x'],405),(t['x'],405),(t['x'],t['bottom'])])
  d.label(515,395,'上采样后的 D')
 # Dashed line denotes module expansion; it is not a feature input.
 a=d.nodes['a0'] if decoder_level is None else d.nodes[f'd{decoder_level}']
 d.ax.plot([a['right'] if decoder_level is None else a['left'],545,545,400],[a['y'],a['y'],750,750],ls='--',lw=1,color=d.style.cfg['colors']['ink'],zorder=0)
 d.label(400,785,f'方案{number} · '+('S0 → D0 模块内部' if decoder_level is None else 'Decoder Block 内部'))


def scheme1():
 d=Diagram('方案 1 · 双向 Mamba + 三路门控',2010);network(d,1)
 def n(k,x,y,t,c='projection'):return d.node(k,x,y,t,c,role=None)
 n('s',220,875,'Skip S\n64 ch · 64×64','input');n('d',620,875,'上采样后的 D\n64 ch · 64×64','input')
 n('split',220,995,'① 拉普拉斯分解\n$L=R(S),\\ H=S-E(L)$')
 n('dlow',620,995,'② 低频提取 R\n64 ch · 32×32')
 n('l',100,1120,'低频 L\n展开 E 后保留')
 n('h',345,1120,'细节残差 H','state')
 n('m',620,1120,'双向 Mamba\n正／反向融合 + 残差')
 n('hp',345,1240,'③ 修正 H′\nRepDW3 → GELU → PW1\n缩放 + H 残差','state')
 n('c',620,1240,'展开 E → 提示 C\n64 ch · 64×64')
 n('gh',480,1370,'④ 细节联合门控\n读取 C 与 H′ → 增益 g_H','state')
 n('selected',345,1490,'选后细节\n$\\widehat H=g_H\\odot H\\prime$','output')
 n('gl',100,1610,'⑤ 低频门控\n读取 E(L)、H′\n$\\widehat L=g_L\\odot E(L)$','state')
 n('gd',650,1610,'⑥ D 门控\n读取 D、H′\n$\\widehat D=g_D\\odot D$','state')
 n('rec',345,1750,'⑦ 重建 Skip\n$\\widehat S=\\widehat L+\\widehat H$','output')
 n('cat',345,1870,'拼接门控 D 与新 Skip\n128 ch → 原 D0 卷积','output')
 for a,b in [('s','split'),('split','l'),('split','h'),('d','dlow'),('dlow','m'),('m','c'),('h','hp'),('hp','gh'),('c','gh'),('gh','selected'),('selected','rec'),('l','gl'),('gl','rec'),('rec','cat')]:d.edge(a,b)
 a,t=d.nodes['hp'],d.nodes['selected']
 d.edge('hp','selected',source='left',target='left',route=[(a['left'],a['y']),(235,a['y']),(235,t['y']),(t['left'],t['y'])])
 a,t=d.nodes['hp'],d.nodes['gl']
 d.edge('hp','gl',source='left',target='right',route=[(a['left'],a['y']),(200,a['y']),(200,t['y']),(t['right'],t['y'])])
 a,t=d.nodes['hp'],d.nodes['gd']
 d.edge('hp','gd',source='bottom',target='top',route=[(a['x'],a['bottom']),(a['x'],1310),(745,1310),(745,1510),(t['x'],1510),(t['x'],t['top'])])
 a,t=d.nodes['d'],d.nodes['gd']
 d.edge('d','gd',source='right',target='right',route=[(a['right'],a['y']),(780,a['y']),(780,t['y']),(t['right'],t['y'])])
 a,t=d.nodes['gd'],d.nodes['cat']
 d.edge('gd','cat',source='bottom',target='right',route=[(a['x'],a['bottom']),(a['x'],t['y']),(t['right'],t['y'])])
 d.label(400,1970,'三套门控：PW128→32 → GELU → Conv3：32→1 → 2σ；空间权重共享通道')
 d.save('25_方案1_整网与模块一体图')


def scheme2():
 d=Diagram('方案 2 · LP-WPO 双分支 Skip（设计稿）',1760);network(d,2)
 def n(k,x,y,t,c='projection'):return d.node(k,x,y,t,c,role=None)
 n('x',400,885,'对齐后的 Skip X\n64 ch · 64×64','input')
 n('lp',220,1020,'① 拉普拉斯分解\n低频 L + 残差 H')
 n('fft',610,1020,'② FFT(X)\n完整 Skip 的频率系数')
 n('rep',220,1160,'H → RepDW3\n低频 L 保留','state')
 n('wpo',610,1160,'阻尼波响应 T_c\n按频率调制','state')
 n('ylp',220,1300,'低频 + 处理后细节\n得到 Y_LP','output')
 n('yw',610,1300,'逆 FFT\n得到 Y_WPO','output')
 n('mix',400,1440,'③ 按通道融合\n(1−β) × Y_LP + β × Y_WPO','state')
 n('res',400,1570,'④ 输入残差\nZ = X + γ × 融合结果','output')
 for a,b in [('x','lp'),('x','fft'),('lp','rep'),('rep','ylp'),('fft','wpo'),('wpo','yw'),('ylp','mix'),('yw','mix'),('mix','res')]:d.edge(a,b)
 a,t=d.nodes['x'],d.nodes['res']
 d.edge('x','res',source='left',target='left',route=[(a['left'],a['y']),(25,a['y']),(25,t['y']),(t['left'],t['y'])])
 d.label(85,1490,'原始 X')
 d.label(400,1680,'Z 与上采样 D 拼接后送入原 D0；β 为通道参数，不是逐像素门控')
 d.save('26_方案2_整网与模块一体图')

if __name__=='__main__':scheme1();scheme2()
