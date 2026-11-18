"""Independent full diagrams for the two implemented V2 projection variants.
Sources: Models/TinyViM/mamba_bottleneck_v2_{no_proj,ds_proj}.py.
Preserves V2's block IDs 1..9, symbols, B/C readout and no-delta injection.
"""
from draw_block_details import Diagram


def draw_variant(spatial):
    width = 1024 if spatial else 512
    title = 'V2＋深度可分离投影' if spatial else 'V2＋无投影'
    d = Diagram(title+' · TinyViM瓶颈', 2400)
    specs = [('input','输入特征图 → 按行展平空间\n[B,512,8,8] → X：[B,64,512]','input')]
    if spatial:
        proj = ('1. 通道归一化 → 二维输入投影\n'
                'LN：每个位置沿512通道归一化\n'
                'DWConv2d 3×3：512→512\nPWConv2d 1×1：512→2048\n'
                '拆成F、Z，再展平；各 [B,1024,64]')
    else:
        proj = ('1. 每个位置沿512通道归一化（LN）\n'
                '交换通道轴与位置轴，同一结果送入F、Z\n'
                +r'$F=Z=\mathrm{LN}(X)^\top$'
                +'\n各 [B,512,64]；无输入投影')
    specs += [
        ('proj',proj,'projection'),
        ('conv','2. 每通道独立做一维卷积 → SiLU\n'
         '核4：当前位置＋前3个序列位置\n'
         '不取后续位置；序列开头补零\n'
         +r'$T=\mathrm{SiLU}(\mathrm{DWConv1d}(F))$'
         +f'\n[B,{width},64]','projection'),
        ('coeff',f'3. 逐位置生成ρ、B、C\nLinear：{width}→64，拆为32＋16＋16\nρ：[B,64,32]\nB/C：[B,64,16] → 转置为 [B,16,64]','projection'),
        ('delta','4. 正步长Δ\n'+r'$\Delta=\mathrm{softplus}(\mathrm{Linear}(\rho))^\top$'+f'\n32→{width}；Δ：[B,{width},64]','projection'),
        ('state','5. 状态更新：写入不乘Δ\n'+r'$A=-\exp(A_{\log}),\quad h_{-1}=0$'+'\n'+r'$h_t=\exp(\Delta_tA)\odot h_{t-1}+B_tT_t$'+f'\nA：[{width},16]；当前状态：[B,{width},16]','state'),
        ('read','6. C加权读出：沿16个状态求和\n'+r'$Y_0[b,d,t]=\sum_{n=0}^{15}C[b,n,t]h_t[b,d,n]$'+f'\n[B,{width},64]','state'),
        ('skip','7. 逐通道直连\n'+r'$Y_1=Y_0+D\odot T$'+f'\nD：[{width}]，可学习；初始为1','state'),
        ('gate','8. Z门控\n'+r'$Y_2=Y_1\odot S$'+f'\n[B,{width},64]','projection'),
    ]
    if spatial:
        out = ('9. 恢复二维 → 输出投影 → 残差\n'
               '[B,1024,8,8]\nDWConv2d 3×3：1024→1024\nPWConv2d 1×1：1024→512\n'
               '展平转置为 [B,64,512]，再加回X')
    else:
        out = ('9. 删除输出投影；转置后加回X\n'+r'$Y=X+Y_2^\top$'+'\n[B,64,512]')
    specs.extend([('out',out,'output'),('restore','恢复二维输出\n[B,512,8,8]','output')])
    s,ax = d.s,d.ax
    sizes = {key:s.measure(ax,text) for key,text,color in specs}
    gate_text='Z分支\n'+r'$S=\mathrm{SiLU}(Z)$'+f'\n[B,{width},64]'
    gate_size=s.measure(ax,gate_text)
    conv_width=sizes['conv'][0]
    total=conv_width+gate_size[0]+s.cfg['box']['node_gap']
    if total>740:
        raise ValueError('Content too wide for parallel T/Z branches')
    conv_x=(800-total)/2+conv_width/2
    z_x=conv_x+conv_width/2+s.cfg['box']['node_gap']+gate_size[0]/2
    y=65
    for key,text,color in specs:
        h=sizes[key][1]
        x=conv_x if key=='conv' else 400
        d.box(key,x,y+h/2,text,color)
        if key=='conv':
            d.box('z',z_x,y+gate_size[1]/2,gate_text,'input')
        y+=h+s.cfg['box']['node_gap']
    height=y+30
    d.fig.set_size_inches(800/s.cfg['output']['dpi'],height/s.cfg['output']['dpi'])
    ax.set_ylim(height,0)
    keys=[row[0] for row in specs]
    for a,b in zip(keys,keys[1:]):
        an,bn=d.nodes[a],d.nodes[b]
        if an['x']==bn['x']:
            d.link(a,b)
        else:
            elbow=(an['bottom']+bn['top'])/2
            d.link(a,b,[(an['x'],elbow),(bn['x'],elbow)])
    p,z=d.nodes['proj'],d.nodes['z']
    fork_x=min(z['x'],p['right']-5)
    elbow=(p['bottom']+z['top'])/2
    s.arrow(ax,[(fork_x,p['bottom']),(fork_x,elbow),(z['x'],elbow),(z['x'],z['top'])])
    max_right=max(n['right'] for n in d.nodes.values())
    z_route=max_right+18
    residual_route=z_route+20
    if residual_route>792:
        raise ValueError('Residual route outside canvas')
    d.bypass('z','gate',z_route)
    d.bypass('input','out',residual_route)
    min_left=min(n['left'] for n in d.nodes.values())
    b_route,c_route,t_route=min_left-18,min_left-36,min_left-54
    if t_route<8:
        raise ValueError('Coefficient route outside canvas')
    for symbol,start,end,route,offset in [('B','coeff','state',b_route,-18),('C','coeff','read',c_route,18),('T','conv','skip',t_route,0)]:
        a,b=d.nodes[start],d.nodes[end]
        y0=a['y']+offset
        s.arrow(ax,[(a['left'],y0),(route,y0),(route,b['y']),(b['left'],b['y'])])
        s.text(ax,route+10,b['y']-22,'$'+symbol+'$',role='annotation')
    d.save('37_V2深度可分离投影完整流程' if spatial else '36_V2无投影完整流程')


def draw_all():
    draw_variant(False)
    draw_variant(True)


if __name__ == '__main__':
    draw_all()
