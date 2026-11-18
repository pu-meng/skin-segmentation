"""Mamba diagram content/layout; typography is owned by the shared figure tool."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'研究记录/figures'

def draw(version='original',numeric=False,stem=None):
    style=FigureStyle();cfg=style.cfg;colors=cfg['colors'];width=cfg['output']['width_px']
    fig,ax=style.canvas()
    B,L,D,N,R,C0=('8','64','1024','16','32','512') if numeric else ('B','L','D','N','R','C_0')
    C0plain='512' if numeric else 'C0'
    code_order=stem.startswith(('14_','15_','16_'))
    shape=f'[{B},{D},{L}]' if code_order else f'[{B},{L},{D}]'
    short=f'[{B},{D}]';state=f'[{B},{D},{N}]';coef=f'[{B},{L},{N}]'
    if code_order:coef=f'[{B},{N},{L}]'
    input_shape='[8,512,8,8]' if numeric else '[B,C0,H,W]'
    projection='512 → 2048' if numeric else 'C0 → 2D'
    proj=(r'1. 归一化、输入投影与拆分'+'\n'+r'$(F,Z)=\operatorname{Split}(\operatorname{Linear}(\operatorname{LN}(X)))$'+'\n'+projection+f'　 $F,Z$：{shape}')
    conv=('2. 因果 DWConv1d（核4）+ SiLU\n'+r'$T=\operatorname{SiLU}(\operatorname{DWConv1d}(F))$'+'\n'+f'Conv1d：[{B},{D},{L}]\n'+f'$T$：{shape}')
    if version=='v1':
        coeff=('3. 额外的因果卷积生成参数\n'+r'$V=\operatorname{SiLU}(\operatorname{DWConv1d}_{k=3}(T))$'+'\n'+r'$(\widetilde\Delta,B,C)=\operatorname{Split}(\operatorname{Conv1d}_{k=1}(V))$'+'\n'+f'$\widetilde\Delta$：{shape}　 $B,C$：{coef}')
        delta=('4. 步长生成\n'+r'$\Delta=\operatorname{softplus}(\widetilde\Delta)$'+'\n'+f'$\Delta$：{shape}')
    elif version=='v3':
        coeff=('3. 步长中间特征\n'+r'$\rho=\operatorname{Linear}(T)$'+'\n'+fr'$\rho$：[{B},{L},{R}]')
        delta=('4. 步长生成\n'+r'$\Delta=\operatorname{softplus}(\operatorname{Linear}(\rho))$'+'\n'+f'$\Delta$：{shape}')
    else:
        coeff=('3. 线性生成系数与拆分\n'+r'$Q=\operatorname{Linear}(T)$'+'\n'+f'$Q$：[{B},{L},'+('64' if numeric else 'R+2N')+']\n'+r'$(\rho,B,C)=\operatorname{Split}(Q)$'+'\n'+fr'$\rho$：[{B},{L},{R}]　 $B,C$：{coef}')
        delta=('4. 步长生成\n'+r'$\Delta=\operatorname{softplus}(\operatorname{Linear}(\rho))$'+'\n'+('32 → 1024　 ' if numeric else 'R → D　 ')+f'$\Delta$：{shape}')
    if code_order:
        proj=proj.replace('归一化、输入投影与拆分','输入投影、拆分与转置')
        coeff=coeff.replace(r'\operatorname{Linear}(T)',r'\operatorname{Linear}(T^\top)')
    if version=='v2light':
        coeff=coeff.replace('R+2N','R+N').replace(r'(\rho,B,C)',r'(\rho,B)').replace('$B,C$', '$B$')
    state_text=('5. 状态更新\n'+r'$A=-\exp(A_{\log})\quad h_{-1}=0$'+'\n'+f'$A_{{\log}},A$：[{D},{N}]\n'+
        (r'$h_t=\exp(\Delta_t A)\odot h_{t-1}+B_tT_t$' if version in ('v2','v2light') else r'$h_t=\exp(\Delta_t A)\odot h_{t-1}+\Delta_t B_tT_t$')+'\n'+f'$\Delta_t,T_t$：{short}　 $B_t$：[{B},{N}]\n'+f'$h_{{t-1}},h_t$：{state}')
    y0_element=r'$Y_0[b,d,t]=\sum_{n=0}^{N-1}C[b,n,t]h_t[b,d,n]$' if code_order else r'$Y_0[b,t,d]=\sum_{n=0}^{N-1}C[b,t,n]h_t[b,d,n]$'
    read=('6. 状态读出：沿 N 求和\n'+y0_element+'\n'
          +f'$C$：{coef}　 $h_t$：{state}'+'\n'+f'$Y_0$：{shape}')
    if numeric:read=read.replace('N 求和','16个状态分量求和').replace('N-1','15')
    if version=='v2light':
        state_text=state_text.replace(f'$A_{{\\log}},A$：[{D},{N}]',f'$A_{{\\log}},A$：[1,{N}]（通道共享）')
        read=('6. 固定平均读出：无 C 分支\n'+r'$Y_0[b,d,t]=\frac{1}{N}\sum_{n=0}^{N-1}h_t[b,d,n]$'+'\n'+f'每个位置：{state} → {short}\n'+f'依次收集 L 个位置：$Y_0$：{shape}')
    y1_element=r'$Y_1[b,d,t]=Y_0[b,d,t]+D[d]T[b,d,t]$' if code_order else r'$Y_1[b,t,d]=Y_0[b,t,d]+D[d]T[b,t,d]$'
    skip=('7. 逐通道直连\n'+y1_element+'\n'+f'$D$：[{D}]　 $Y_1$：{shape}')
    specs=[('input','输入特征图 → 展平并转置\n'+input_shape+f' → $X$：[{B},{L},{C0plain}]','input'),('proj',proj,'projection'),('conv',conv,'projection'),('coeff',coeff,'projection'),('delta',delta,'projection')]
    if version=='v3':
        specs.extend([
          ('state','5. 内层门控：无状态递推\n'+r'$A=-\exp(A_{\log})$'+'\n'+r'$G_U=\sum_{n=0}^{N-1}\exp(\Delta A_n)$'+'\n'+r'$G_T=\Delta$'+'\n'+f'$G_U,G_T$：{shape}','state'),
          ('read','6. 融合（D 为可学习参数）\n'+r'$Y_1=G_U\odot U+D\odot G_T\odot T$'+'\n'+f'$D$：[{D}]　 $Y_1$：{shape}','state'),
        ])
    else:
        specs.extend([('state',state_text,'state'),('read',read,'state'),('skip',skip,'state')])
    specs.extend([
      ('gate',('7' if version=='v3' else '8')+'. Z 门控\n'+r'$Y_2=Y_1\odot S$'+'\n'+f'$Y_2$：{shape}','projection'),
      ('out',('8' if version=='v3' else '9')+'. 输出投影与残差\n'+r'$Y=X+\operatorname{Linear}(Y_2)$'+'\n'+('1024 → 512　 ' if numeric else 'D → C0　 ')+f'$Y$：[{B},{L},{C0plain}]','output'),
      ('restore','还原特征图：'+input_shape,'output'),
    ])
    if code_order:
        specs=[(key,text.replace(r'\operatorname{Linear}(Y_2)',r'\operatorname{Linear}(Y_2^\top)') if key=='out' else text,color) for key,text,color in specs]
    if stem=='14_原始Mamba_逐块维度与下标':
        explicit={
            'input':('输入特征图：维度均用字母表示\n'+r'$[B,C_0,H,W]\quad L=H\times W$'+'\n展平空间轴：Flatten(2) → [B,C0,L]\n交换轴1、2：Transpose(1,2)\n'+r'$X$：[B,L,C0]'),
            'proj':('1. 归一化、输入投影、拆分与转置\n'+r'$\operatorname{Linear}(\operatorname{LN}(X))=\operatorname{LN}(X)W_{\mathrm{in}}^\top$'+'\n'+r'$W_{\mathrm{in}}$：[2D,C0]　 输出：[B,L,2D]'+'\n沿末轴拆分 → 两个 [B,L,D]\n各自转置 → $F,Z$：[B,D,L]'),
            'out':('9. 输出投影与残差\n转置 $Y_2$：[B,D,L] → [B,L,D]\n'+r'$Y=X+Y_2^\top W_{\mathrm{out}}^\top$'+'\n'+r'$W_{\mathrm{out}}$：[C0,D]　 $Y$：[B,L,C0]'),
            'restore':('输出特征图：转置再恢复空间轴\nTranspose(1,2)：[B,L,C0] → [B,C0,L]\nReshape(B,C0,H,W) → [B,C0,H,W]'),
        }
        specs=[(key,explicit.get(key,text),color) for key,text,color in specs]
    title={'original':'标准Mamba（公式参照）','v1':'V1：卷积生成参数','v2':'V2（基础版）','v3':'V3：二维卷积与门控融合','v2light':'V2＋共享A＋平均读出'}[version]
    if numeric:title+=' · TinyViM 瓶颈 · 批量 8'
    nodes=[]
    for key,text,color in specs:
        w,h=style.measure(ax,text);nodes.append(dict(key=key,text=text,color=colors[color],w=w,h=h))
    ztext='Z 门控\n'+r'$S=\operatorname{SiLU}(Z)$'+'\n'+shape
    zw,zh=style.measure(ax,ztext);z=dict(key='z',text=ztext,color=colors['input'],w=zw,h=zh)
    side=None
    if version=='v3':
        utext=('二维卷积分支\n恢复空间排列\nDWConv3×3 → GroupNorm\nPWConv扩展4倍 → GELU\nPWConv压缩 → 展平\n'+f'$U$：{shape}')
        uw,uh=style.measure(ax,utext);side=dict(key='u',text=utext,color=colors['state'],w=uw,h=uh)
    top=58;gap=cfg['box']['node_gap'];lookup={n['key']:n for n in nodes}
    for n in nodes:
        n['x']=width/2
        if n['key']=='conv':
            total=n['w']+z['w']+24
            if total>width-80:raise ValueError('字号使双分支超宽；缩短标签或提高画布宽度，不能挤小字体')
            n['x']=(width-total)/2+n['w']/2
            z.update(x=n['x']+n['w']/2+24+z['w']/2,y=top+n['h']/2)
        if side and n['key'] in ('coeff','delta','state'):
            main_w=max(lookup[k]['w'] for k in ('coeff','delta','state'))
            total=main_w+side['w']+36
            if total>width-100:raise ValueError('V3双分支超宽')
            n['x']=(width-total)/2+main_w/2
        n['y']=top+n['h']/2;top+=n['h']+gap
    if side:
        start=lookup['coeff']['y']-lookup['coeff']['h']/2
        bottom=lookup['state']['y']+lookup['state']['h']/2
        side.update(x=lookup['coeff']['x']+max(lookup[k]['w'] for k in ('coeff','delta','state'))/2+36+side['w']/2,y=(start+bottom)/2)
    height=top+35;dpi=cfg['output']['dpi'];fig.set_size_inches(width/dpi,height/dpi);ax.set_ylim(height,0)
    style.text(ax,width/2,24,title,role='heading')
    extras=[z]+([side] if side else [])
    for n in nodes+extras:
        node=style.box(ax,n['x'],n['y'],n['text'],n['color']);n.update(node)
    for prev,n in zip(nodes,nodes[1:]):
        if prev['x']==n['x']:points=[(prev['x'],prev['bottom']),(n['x'],n['top'])]
        else:
            mid=(prev['bottom']+n['top'])/2
            points=[(prev['x'],prev['bottom']),(prev['x'],mid),(n['x'],mid),(n['x'],n['top'])]
        style.arrow(ax,points)
    p=lookup['proj'];cv=lookup['conv'];co=lookup['coeff'];st=lookup['state'];rd=lookup['read'];ga=lookup['gate'];ot=lookup['out'];inp=lookup['input']
    # Align branch departure with the projection box; both output arrows read it.
    z_depart=min(z['x'],p['right']-5)
    style.arrow(ax,[(z_depart,p['bottom']),(z_depart,z['top']-14),(z['x'],z['top']-14),(z['x'],z['top'])])
    xz=max(n['right'] for n in nodes+extras)+20;xx=xz+20
    style.arrow(ax,[(z['right'],z['y']),(xz,z['y']),(xz,ga['y']),(ga['right'],ga['y'])])
    style.arrow(ax,[(inp['right'],inp['y']),(xx,inp['y']),(xx,ot['y']),(ot['right'],ot['y'])])
    left=min(n['left'] for n in nodes);xb=left-18;xc=left-36;xt=left-54
    if xt<0 or xx>width:raise ValueError('连线越出画布；增加画布宽度或缩短标签')
    if not side:
        style.arrow(ax,[(co['left'],co['y']-20),(xb,co['y']-20),(xb,st['y']),(st['left'],st['y'])])
        style.text(ax,xb+10,st['y']-24,'$B$',role='annotation')
        if version!='v2light':
            style.arrow(ax,[(co['left'],co['y']+20),(xc,co['y']+20),(xc,rd['y']),(rd['left'],rd['y'])])
            style.text(ax,xc+10,rd['y']-24,'$C$',role='annotation')
    target=rd if side else lookup['skip']
    style.arrow(ax,[(cv['left'],cv['y']),(xt,cv['y']),(xt,target['y']),(target['left'],target['y'])])
    style.text(ax,xt+10,target['y']-24,'$T$',role='annotation')
    if side:
        # U is computed from T, in parallel with rho/delta/GU.
        mid=(cv['bottom']+co['top'])/2
        style.arrow(ax,[(cv['right'],cv['y']),(cv['right']+8,cv['y']),(cv['right']+8,mid),(side['x'],mid),(side['x'],side['top'])])
        style.arrow(ax,[(side['x'],side['bottom']),(side['x'],rd['top']-12),(rd['x']+rd['w']/4,rd['top']-12),(rd['x']+rd['w']/4,rd['top'])])
    if numeric:style.text(ax,width/2,height-17,'UNet 瓶颈为16×16：序列长度256，通道数和状态数不变。',role='annotation')
    audit=style.save(fig,OUT/stem)
    print(f'{stem}：{len(audit["boxes"])}个色块，样式检查通过')
