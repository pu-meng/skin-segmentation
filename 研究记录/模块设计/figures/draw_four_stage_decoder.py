"""Four real decoder stages, aligned to the four encoder output scales."""
from draw_module_figures import Diagram


def main(modified=False):
    d = Diagram('四层对应修改版 · S0 → D0 拉普拉斯引导' if modified else '四层对应基础版 · 4 Encoder / 4 Decoder', 1040)
    def node(key,x,y,text,color):
        return d.node(key,x,y,text,color,role=None)
    node('input',125,90,'RGB 输入\n3 ch · 256×256','input')
    node('stem',125,215,'输入投影 Stem\n两次步长 2 卷积\n3→48；256→64','projection')
    d.edge('input','stem')
    for i,(raw,ch,size,y) in enumerate([(48,64,64,385),(64,128,32,565),(168,256,16,745),(224,512,8,925)]):
        node(f'e{i}',125,y,f'E{i} · Encoder\n{raw} ch · {size}×{size}','input')
        if modified and i==0:
            node(f'a{i}',380,y,f'1×1 对齐 → 拉普拉斯5.2\n{raw}→{ch} ch；S0 → D0','state')
        else:
            node(f'a{i}',380,y,f'1×1 对齐\n{raw}→{ch} ch','projection')
        title=f'D{i} · '+('最深层处理' if i==3 else '融合')
        node(f'd{i}',665,y,f'{title}\n{ch} ch · {size}×{size}','output')
        d.edge(f'e{i}',f'a{i}',source='right',target='left')
        d.edge(f'a{i}',f'd{i}',source='right',target='left')
        if not (modified and i==0):d.label(495,y-25,'最深特征' if i==3 else f'S{i}')
    d.edge('stem','e0')
    for i in range(3):
        d.edge(f'e{i}',f'e{i+1}')
        d.edge(f'd{i+1}',f'd{i}',source='top',target='bottom')
        d.label(745,(d.nodes[f'd{i+1}']['top']+d.nodes[f'd{i}']['bottom'])/2,'×2')
    if modified:
        a,t=d.nodes['d1'],d.nodes['a0']
        d.edge('d1','a0',source='top',target='bottom',route=[(a['x'],a['top']),(a['x'],480),(t['x'],480),(t['x'],t['bottom'])])
        d.label(530,460,'D1 上采样后的 D')
    node('head',665,200,'输出头\n64→1；插值 ×4\n256×256；Sigmoid','output')
    d.edge('d0','head',source='top',target='bottom')
    d.label(400,1010,'D3 → D2 → D1 → D0；输出头没有额外 Decoder 卷积块')
    d.save('21_四层对应Decoder_拉普拉斯总图' if modified else '20_四层对应Decoder_基础版')


if __name__ == '__main__':
    main()
