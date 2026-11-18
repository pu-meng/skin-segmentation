"""仅重画方案3：阶段分开、操作名称与正文一致，不生成其他方案的图。"""
from pathlib import Path
from html import escape
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / '方案3图文对齐版'
INK = '#28384A'
COL = {'base':'#FFF0BD', 'aux':'#E4EFFB', 'new':'#ECE0F8', 'gate':'#FFF0D8', 'out':'#DCF1E4', 'gray':'#F1F4F7'}
FIGURES = []
LABELS = {7:'建库展开图',8:'评分展开图',9:'强度选择展开图',10:'检索展开图',11:'修正展开图'}
def relabel(value):
    return re.sub(r'图(11|10|9|8|7)(?!\d)', lambda m: LABELS[int(m[1])], value)

class SVG:
    def __init__(self, width, title, phase):
        self.width, self.parts = width, []
        self.text(24, 38, title, 26, True)
        self.text(24, 75, phase, 19)
    def text(self, x, y, value, size=20, bold=False, center=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{"middle" if center else "start"}">{escape(relabel(value))}</text>')
    def box(self, x, y, w, title, sub='', color='base', h=74):
        self.parts.append(f'<rect x="{x-w/2}" y="{y-h/2}" width="{w}" height="{h}" rx="8" fill="{COL[color]}" stroke="{INK}" stroke-width="1.5"/>')
        self.text(x, y-7 if sub else y+8, title, getattr(self, 'box_title_size', 21), True, True)
        if sub: self.text(x, y+24, sub, getattr(self, 'box_sub_size', 18), center=True)
    def arrow(self, points, dashed=False):
        self.parts.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{INK}" stroke-width="1.7" '+('stroke-dasharray="6 4" ' if dashed else '')+'marker-end="url(#arr)"/>')
    def save(self, name, height, footer=None):
        self.text(24, height-20, footer if footer is not None else '待验证的方案流程；尚无真实 CT 完整实现或效果证据。', 18)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{height}" viewBox="0 0 {self.width} {height}"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="{INK}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g fill="{INK}" font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif">'+''.join(self.parts)+'</g></svg>'
        name = relabel(name)
        path = OUT / (name+'.svg')
        tree=ET.fromstring(svg)
        for index, node in enumerate(tree.iter()):
            if node.tag.endswith(('text','rect')):
                node.set('id',f'element_{index}')
        ET.indent(tree)
        path.write_text(ET.tostring(tree,encoding='unicode'), encoding='utf-8')
        ET.parse(path)
        FIGURES.append({'name':name,'width':self.width,'height':height})

def panel(number, title, phase, cards):
    s=SVG(850, f'图{number}　{title}', phase)
    y=102
    bounds=[]
    for i,(head,lines,color) in enumerate(cards):
        h=50+30*len(lines)
        left=55 if number==17 else 24
        w=740 if number==17 else 802
        bounds.append((y,h))
        s.parts.append(f'<rect x="{left}" y="{y}" width="{w}" height="{h}" rx="8" fill="{COL[color]}" stroke="{INK}" stroke-width="1.5"/>')
        s.text(left+18,y+31,head,22,True)
        for j,line in enumerate(lines): s.text(left+18,y+62+30*j,line,20)
        y+=h
        if i<len(cards)-1: s.arrow([(425,y+3),(425,y+29)])
        if number==17 and i==2:s.text(448,y+24,'无合格者',17)
        y+=36
    if number==17:
        trial=bounds[1][0]+bounds[1][1]/2
        criterion=bounds[2][0]+bounds[2][1]/2
        s.arrow([(55,criterion),(25,criterion),(25,trial),(55,trial)])
    s.save(f'图{number}_{title}', y+30)




def network(number,n):
    if n != 3:
        raise ValueError('This generator is restricted to scheme3')
    s=SVG(1000,f'图{number}　方案3新病例预测主网络',
          '阶段3：全部网络参数冻结，M、S 固定；只预测，不训练。')
    s.box_title_size, s.box_sub_size = 24, 22
    s.text(24,110,'主路径向下编码、向上解码；中央 + 将原 skip 与同级上采样特征相加。',18)
    e,d,merge=145,840,500
    ys=[365,565,765,965]
    s.box(e,160,230,'新 CT','输入块 128³','aux',h=56)
    s.box(e,250,230,'Stem',h=50)
    s.arrow([(e,188),(e,225)])
    s.arrow([(e,275),(e,328)])
    for i,y in enumerate(ys):
        channels=32*2**i;size=128//2**i
        sub='局部特征 A，256通道' if i==3 else f'{channels} 通道，{size}³'
        s.box(e,y,250,f'Encoder Stage {i}',sub,h=74)
        s.box(d,y,260,f'Decoder Stage {i}',f'{channels} 通道，{size}³',h=74)
        s.parts.append(f'<circle cx="{merge}" cy="{y}" r="18" fill="white" stroke="{INK}" stroke-width="1.5"/>')
        s.text(merge,y+8,'+',26,center=True)
        s.arrow([(270,y),(merge-18,y)])
        s.text(375,y-16,f'原 skip {i}',18,center=True)
        s.arrow([(merge+18,y),(710,y)])
        if i<3:
            mid=y+100
            s.box(e,mid,214,f'DownBlock {i}','stride 2',h=60)
            s.arrow([(e,y+37),(e,mid-30)])
            s.arrow([(e,mid+30),(e,ys[i+1]-37)])
            s.box(d,mid,220,f'UpBlock {i}','上采样 ×2',h=60)
            s.arrow([(d,ys[i+1]-37),(d,mid+30)])
            s.arrow([(d,mid-30),(d,y+60),(merge,y+60),(merge,y+18)])
    s.box(e,1100,225,'DownBlock 3','256 → 512 通道',h=60)
    s.arrow([(e,1002),(e,1070)])
    s.box(e,1220,250,'Conv bottleneck','512 通道，8³')
    s.arrow([(e,1130),(e,1183)])
    s.box(e,1340,250,'已有 MHA＋MoE','原上下文模块')
    s.arrow([(e,1257),(e,1303)])
    s.box(d,1220,260,'UpBlock 3 → U','256 通道，16³')
    s.arrow([(e,1377),(e,1415),(d,1415),(d,1257)])
    s.arrow([(d,1183),(d,1040),(merge,1040),(merge,983)])
    s.box(d,160,260,'输出头 → 原分数 z','三个类别的原始分数','out')
    s.arrow([(d,328),(d,197)])
    s.box_title_size, s.box_sub_size = 28, 24
    s.box(190,1525,320,'已评分记忆表 M','特征k、纠错v、评分u','aux',h=92)
    s.box(650,1525,520,'逐位置检索相似记忆','按特征相似度取前三条记录','new',h=92)
    s.arrow([(350,1525),(390,1525)])
    s.arrow([(270,965),(300,965),(300,1445),(650,1445),(650,1479)])
    s.box(650,1675,520,'用评分 u 过滤并计算占比','剔除 u≤0；w=归一化(相似度×u)','new',h=92)
    s.arrow([(650,1571),(650,1629)])
    s.box(650,1825,520,'组合纠错并上采样','按 w 组合 v → 修正图 C_base','new',h=92)
    s.arrow([(650,1721),(650,1779)])
    s.box(190,1975,320,'固定整体强度 S','支持标注学习，验证后保存','aux',h=92)
    s.box(650,1975,520,'用 S 缩放，再加到原分数','z′ = z + S × C_base','gate',h=92)
    s.arrow([(650,1871),(650,1929)])
    s.arrow([(350,1975),(390,1975)])
    s.arrow([(970,160),(970,1975),(910,1975)])
    s.text(943,1900,'z',24,center=True)
    s.box(650,2125,520,'分类 → 最终分割','z′ → softmax → 最大概率类别','out',h=92)
    s.arrow([(650,2021),(650,2079)])
    s.text(24,2200,'预测只读取新 CT 图像；记忆表 M、评分 u、整体强度 S 均已固定。',22)
    s.text(24,2232,'预测完成后，将分割结果与人工标注比较，计算评价指标。',22)
    s.save(f'图{number}_方案3新病例预测主网络',2280)


def note(s,x,y,lines,size=19):
    for i,line in enumerate(lines):s.text(x,y+29*i,line,size)

def building_cards(title, phase, cards):
    """按数据去向排卡片；长句分行，给相邻步骤保留箭身。"""
    s=SVG(900, title, phase)
    y=110
    for i,(heading,lines,color) in enumerate(cards):
        h=60+29*len(lines)
        s.parts.append(f'<rect x="42" y="{y}" width="816" height="{h}" rx="8" fill="{COL[color]}" stroke="{INK}" stroke-width="1.5"/>')
        s.text(62,y+33,heading,23,True)
        note(s,62,y+66,lines,20)
        y+=h
        if i<len(cards)-1:
            s.arrow([(450,y+3),(450,y+35)])
            y+=40
    return s,y+60


def record_building():
    s,height=building_cards('建库展开图　A 和输出怎样共同生成一条记忆',
        '先看完整操作①至⑤，再看局部算例⑥；空间配对规则仍待选定。',[
        ('① 逐病例运行冻结网络',[
            '甲的 CT → 中间特征 A；输出分数 z → softmax → 三类概率 P。',
            '人工标注 y 与同一输入块对齐；y 不输入网络。'],'base'),
        ('② 同一位置取 f、P、y → 一行样本',[
            '先统一空间网格：配对展开图演示输出网格候选，尚未选定。',
            '取同一病例、同一位置的256维 f、3类概率 P、真实类别 y。',
            '遍历有效位置 → 样本表；保留每行来源病例和空间位置。'],'aux'),
        ('③ 逐行比较预测类别与 y → 写入表现组',[
            '预测类别 = P 中最大概率对应的类别；不使用 f 判组。',
            '真实肿瘤、预测非肿瘤 → 漏检；真实非肿瘤、预测肿瘤 → 误报。',
            '另保留肝脏预测正确、肿瘤预测正确；其他关系不进当前四组。'],'gate'),
        ('④ 甲乙丙各自的四组 → 最多12个组分别聚类',[
            '甲：肝脏正确组｜肿瘤正确组｜漏检组｜误报组。',
            '乙：肝脏正确组｜肿瘤正确组｜漏检组｜误报组。',
            '丙：肝脏正确组｜肿瘤正确组｜漏检组｜误报组。',
            '上述每个非空组都单独聚类；不同病例、不同组不混合。',
            '每行 f 单位化 → 与各中心比较 → 分到最相似中心。',
            '每簇更新中心并归一化 → 重复分配，直到规定停止条件。',
            '候选每组最多4簇；实际簇数、初始化、停止规则仍待固定。'],'new'),
        ('⑤ 每簇收集其中各位置的数据 → 汇总成一条记忆',[
            '例如簇1包含位置 a1、a2，就收集这两个位置的特征、概率和标注。',
            'f：配对时为每个位置保存的256个特征数，尚未做长度归一化。',
            '将簇内这些 f 逐维求平均，再把平均向量长度缩放为1 → k。',
            'y：人工标注在该位置写的真实类别；P：模型在该位置的3类概率。',
            '把 y 写成3个数：背景=[1,0,0]；肝脏=[0,1,0]；肿瘤=[0,0,1]。',
            '每个位置用这3个数减去自己的 P，再在簇内逐类别求平均 → v。',
            '来源病例、组别、k、v 存成一条记忆；评分 u 在下一阶段计算。'],'out'),
        ('⑥ 代入⑤的规则手算：假定某簇最终只有 a1、a2',[
            'a1：f₁=[1,0,0,…]；P₁=[0.10,0.70,0.20]；y₁=肿瘤。',
            'a2：f₂=[1,1,0,…]；P₂=[0.20,0.60,0.20]；y₂=肿瘤。',
            '本算例中，第3至256维都设为0；真实特征不要求这些维度为0。',
            '算 k 第一步：两个位置的特征，对应维度分别取平均。',
            '第1维：(1+1)/2=1；第2维：(0+1)/2=0.5；其余维度为0。',
            '得到平均特征 [1,0.5,0,…]；仍有256维，没有压缩维度。',
            '算 k 第二步：平均特征的长度 = √(1²+0.5²) ≈ 1.118。',
            '算 k 第三步：平均特征的每个数都除以这个长度。',
            'k=[1/1.118,0.5/1.118,0,…]≈[0.8944,0.4472,0,…]。',
            '这叫长度归一化：方向不变，长度变为1；k作为这一簇的代表。',
            '新病例取一个位置的256维特征，与k比方向相似度来检索记忆。',
            '算 v：两处真实都是肿瘤，人工答案各写成 [0,0,1]。',
            'a1：[0,0,1]−P₁=[−0.10,−0.70,+0.80]。',
            'a2：[0,0,1]−P₂=[−0.20,−0.60,+0.80]。',
            '两个位置的差值逐类别平均；除以位置数2，不是三个类别均分。',
            'v=[−0.15,−0.65,+0.80]，依次对应背景、肝脏、肿瘤。'],'gray'),
        ('⑦ 合并所有簇记录 → 未评分记忆表 M',[
            '算例记录 m01：来源=甲；组别=漏检肿瘤。',
            '已算出 k≈[0.8944,0.4472,0,…]；v=[−0.15,−0.65,+0.80]。',
            '只有评分 u 尚未计算；下一阶段跨病例试用后写入。',
            '甲其他簇、乙各簇、丙各簇分别生成记录，再合并。',
            '3病例 × 4组 × 每组最多4簇 = 最多48条；空组不生成记录。'],'out')])
    s.save('建库展开图_方案3建库模块',height,
           footer='建库步骤产物：来源、组别、k、v已得到；下一阶段计算评分u。')


def pairing_module():
    s,height=building_cards('配对模块　从同一位置取数，再逐行判组',
        '完整演示输出网格候选；尚未选定，不能直接混用粗网格预测。',[
        ('① 输入同一病例、同一输入块的 A、P、y',[
            'A：[1,256,16,16,16]；P：[1,3,128,128,128]。',
            'y：同一输出网格，每体素一个类别；图像与标签须空间对齐。'],'aux'),
        ('② 仅对 A 做三线性空间插值 → 对齐特征 F',[
            '256个通道分别由16³插值到128³ → F：[1,256,128,128,128]。',
            'P、y 保持原网格；不把256通道压成3通道，不连续插值标签。'],'new'),
        ('③ 固定 p=(40,50,60)，从三份数据取同一位置',[
            'f=F[0,:,40,50,60]：256个数。',
            'P_p=P[0,:,40,50,60]：3个概率；y_p：该位置1个真实类别。',
            '空间下标一致，通道数不同；p 是输出坐标。'],'aux'),
        ('④ 用 P_p 与 y_p 判组 → 写成一行',[
            '例如 P_p=[0.10,0.70,0.20]，预测类别为1（非肿瘤肝脏）。',
            '若 y_p=2（肿瘤），则这行属于漏检肿瘤组。',
            '行 a1：来源甲、p、f、P_p、y_p、预测类别1、漏检肿瘤。'],'gate'),
        ('⑤ 遍历其他位置 → 样本表 → 按病例＋组别筛行',[
            '每行重复③④；不符合四组则不入库，有效位置规则仍待定。',
            '甲、乙、丙各分四组：肝脏正确、肿瘤正确、漏检、误报。',
            '最多12个组，每个非空组都分别按 f 聚类，不混合病例或组别。',
            '组别由 P 与 y 判定；同组内的簇才由 f 相似性决定。'],'out'),
        ('空间选择的边界：本图演示怎样做，不表示已经实现',[
            '插值坐标约定、边界设置、有效位置与跨块去重规则仍待固定。',
            '若选本候选，评分和预测也按输出位置查询、直接构造修正图。',
            '若保留粗网格 q，则关联输出位置、概率汇总与混合标签规则待定。'],'gray')])
    s.save('配对模块_两个网格怎样对应',height)
    target=ROOT/'方案3_问题定义与思路检索'/'figures'
    target.mkdir(parents=True,exist_ok=True)
    (target/'配对模块_两个网格怎样对应.svg').write_bytes((OUT/'配对模块_两个网格怎样对应.svg').read_bytes())

def scoring():
    """单条记忆的跨病例试用：先展开乙，再并列呈现两例结果。"""
    s,height=building_cards('评分展开图　方案3跨病例评分模块',
        '阶段2：甲的一条记忆分别帮助乙、丙预测，再用标注检查有没有改善。',[
        ('① 取出甲的一条记忆 m01',[
            '代表特征 k：256个数，用来找相似位置。',
            '修正量 v：背景、肝脏、肿瘤各一个调整数。',
            '此例 v=[−0.15, −0.65, +0.80]；下面数字均为构造示例。'],'aux'),
        ('② 乙的 CT 输入冻结源模型，乙的标注另存用于评分',[
            '模型读取乙的 CT，得到乙各粗位置的256维特征和输出三类分数。',
            '乙的人工标注不作为模型输入，也不参与检索或修正位置选择。',
            '本条记忆试用完成后，才用乙的标注比较前后损失、计算评分 u。'],'base'),
        ('③ 比较特征，给乙每个有效粗位置算三个修正数',[
            '比较甲的代表特征 k 与乙当前位置的256维特征，得到余弦相似度。',
            '若相似度为0.8：0.8 × v = [−0.12, −0.52, +0.64]。',
            '这三个数依次用于调整背景、肝脏、肿瘤分数；下一步才加到输出。',
            '若这个余弦相似度 ≤ 0，该粗位置修正量为 [0,0,0]。',
            '这里判断的是特征相似度，不是类别分数，也不是尚未得到的 u。',
            '甲、乙的同一坐标不保证对应同一组织；不按甲的位置复制修正。',
            '遍历乙自身网格，用乙各位置的特征决定该处修正量。'],'new'),
        ('④ 将乙的粗修正图上采样，再加到乙的输出分数',[
            '乙各粗位置的三个修正数 → 三通道粗修正图 → 上采样到输出网格。',
            '在输出位置，分别给背景、肝脏、肿瘤的原始分数加上修正数。',
            '例如分数 [0.20,0.70,0.30] + 修正 [−0.12,−0.52,+0.64]',
            '得到 [0.08,0.18,0.94]；最大分数类别从肝脏变为肿瘤。',
            '此例是假设某输出位置上采样后得到该修正量，不是概率直接相加。',
            '只试 m01，强度为1；不组合其他记忆，不改网络参数或中间特征。'],'new'),
        ('⑤ 用乙的标注比较前后损失；对丙独立重复②至⑤',[
            '损失衡量预测与标注的差距，越小越好；前后采用同一计算规则。',
            '比较整例预测，修好的位置与受损的位置都计入。',
            '丙从自己的 CT 和原始分数开始；不接着乙的结果修正。'],'gate')])
    # 两例的对照与平均分别占一层，避免堆叠带病例下标的公式。
    top=height-20
    for cx,case,before,after,change,meaning in [
        (245,'乙','0.50','0.46','+0.04','正值：损失下降'),
        (655,'丙','0.40','0.42','−0.02','负值：损失上升')]:
        s.parts.append(f'<rect x="{cx-195}" y="{top}" width="390" height="158" rx="8" fill="{COL["gate"]}" stroke="{INK}" stroke-width="1.5"/>')
        s.text(cx,top+31,f'{case}：损失减少了多少？',23,True,True)
        s.text(cx,top+65,f'修正前 {before} → 修正后 {after}',20,center=True)
        s.text(cx,top+99,f'{before} − {after} = {change}',23,True,True)
        s.text(cx,top+133,meaning,20,center=True)
    s.arrow([(450,top-40),(450,top-22),(245,top-22),(245,top-3)])
    s.arrow([(450,top-22),(655,top-22),(655,top-3)])
    end=top+238
    s.box(450,end+48,816,'⑥ 两例的损失减少量取平均 → 这条记忆的评分 u',
          '（+0.04 +（−0.02））÷ 2 = +0.01','out',h=96)
    s.parts.append(f'<polyline points="245,{top+161} 245,{end-26} 450,{end-26}" fill="none" stroke="{INK}" stroke-width="1.7"/>')
    s.arrow([(450,end-26),(450,end-3)])
    s.parts.append(f'<polyline points="655,{top+161} 655,{end-26} 450,{end-26}" fill="none" stroke="{INK}" stroke-width="1.7"/>')
    note(s,62,end+124,[
        '写回 m01：评分 +0.01，并保存乙、丙各自的前后损失；k、v 保持不变。',
        '平均为正不代表两例都改善；此例乙改善、丙变差，不能只看平均值。',
        '下一条记忆仍从乙、丙的原始分数开始；所有记录逐条独立试用。'],20)
    s.save('评分展开图_方案3跨病例评分模块',end+250,
           '候选流程：损失配置、空间配对与整病例合并规则待固定；尚无真实 CT 结果。')


def strength_selection():
    """先展开组合修正图，再只学习全局S；源网络和已评分记忆库保持固定。"""
    s,height=building_cards('强度学习展开图　S 怎样学出来',
        '接评分 u：先生成组合修正图，用支持标注学习 S，再独立验证训练结果。',[
        ('① 固定源网络与 M，为支持病例和独立验证病例分别生成图',[
            'M 汇集甲、乙、丙的记录，每条已有代表特征 k、修正量 v 和评分 u。',
            '甲乙丙生成训练图时先排除自身记录；丁戊是独立验证病例。',
            '丁的标注另存，只在最后评价；不建库、不重评 u、不参与检索。'],'aux'),
        ('② 从丁的一个粗位置 q 取特征，与 M 各条 k 比较',[
            'A 示例形状为 [1,256,16,16,16]；q 是这张16³网格上的一个位置。',
            '取 A 在 q 的256个数，与甲、乙、丙全部记录的 k 算余弦相似度。',
            '这是从已评分的全部记忆中检索；不是再次逐条单独评分。'],'new'),
        ('③ 先取正相似度最高的最多3条，再删评分 u ≤ 0 的记录',[
            '例如前三条相似度为0.8、0.7、0.6，评分为 +0.01、−0.005、+0.02。',
            '删去中间负评分记录，留下两条；不补排名第四条。',
            '没有留下记录：q 的三个修正数为 [0,0,0]。'],'gate'),
        ('④ 相似度乘评分，再换算成留下记录各占多少',[
            '读取已存 u：m01为0.01、m03为0.02；不重新评分。',
            '相似度乘 u：m01 得0.008，m03 得0.012；总和0.020。',
            'm01 占比 = 0.008 ÷ 0.020 = 40%；m03 = 0.012 ÷ 0.020 = 60%。',
            '这个比例在丁的每个位置分别计算，不是所有位置共用一组比例。'],'new'),
        ('⑤ 按比例组合两条 v，得到 q 的三个修正数',[
            '从 M 读取：m01 的 v=[−0.15,−0.65,+0.80]，m03 的 v=[−0.05,+0.20,−0.15]。',
            'q 的修正 = 40% × m01的v + 60% × m03的v = [−0.09,−0.14,+0.23]。',
            '三个数依次对应背景、肝脏、肿瘤；此时还没乘整体强度 S。'],'new'),
        ('⑥ 遍历丁的所有有效粗位置，把三个数填回各自位置',[
            '每个位置重复②至⑤：各自检索、算比例、组合，得到三通道粗修正图 T。',
            'T 示例形状为 [1,3,16,16,16]；未参与查询的位置按规则填零。',
            '将 T 的三个通道分别上采样到输出网格 → 完整组合修正图 C_base。',
            'C_base 示例形状为 [1,3,128,128,128]，与原始三类分数 z 一致。',
            '图的坐标属于丁；不把甲、乙、丙的坐标直接复制到丁。'],'out'),
        ('⑦ 缓存各病例 z 与 C_base，初始化唯一可训练标量 S=1',[
            '甲乙丙使用排除自身记录后的修正图；丁戊使用完整 M 的修正图。',
            '只有 S 交给优化器；源网络参数、k、v、u及检索规则都保持固定。',
            '每次更新后将负 S 截为0；S 是所有病例、位置和类别共用的一个数。'],'gate'),
        ('⑧ 用甲乙丙的标注计算分割损失，反向传播只更新 S',[
            '读取支持病例缓存：修正后分数 = z + S × C_base。',
            '与该支持病例标注计算soft Dice和交叉熵损失，再求对 S 的梯度。',
            '优化器只修改 S；不重跑记忆评分、不修改库，也不更新源网络。',
            '重复训练，记录 S 与损失；学习率、步数和早停规则实施前固定。'],'gate'),
        ('⑨ 用独立验证丁戊评价训练中保存的 S，选择训练快照',[
            '丁戊各用自己的 z 与 C_base，读取当次 S 生成分割并评价。',
            '验证标注只用于评价，不反向传播；验证病例不建库、不重评 u。',
            '不再枚举0、0.25、0.5、1；按预定指标与并列规则保存训练结果。'],'gate'),
        ('⑩ 保存学得的一个 S，用于所有新病例',[
            '预测时固定源权重、M和 S；新病例仍从自身 CT 生成自身 C_base。',
            'S=0不加修正，S=0.5加一半，S=1加完整图；不预设最终学到哪个值。',
            '支持病例复用于建库、评分和拟合 S，训练损失不能当独立验证成绩。'],'out')])
    note(s,42,height,[
        '这是待实现的当前设计，尚无真实 CT 的 S 训练或效果验证。',
        '原验证网格选择 S 保留为历史对照；图中组合数字仅用于说明计算。'])
    s.save('强度选择展开图_方案3验证选择强度',height+120)


def retrieval_calculation():
    """明确区分现算相似度、读取已有u、乘积换算比例及用比例组合v。"""
    s,height=building_cards('检索展开图　一个位置怎样选记忆并算混合比例',
        '验证与新病例预测使用相同规则；u 已经算好，以下数值为构造示例。',[
        ('① 读取当前位置特征，以及记忆库每行已有的 k、v、u',[
            '当前 CT 经冻结网络得到特征图 A；取一个粗位置 q 的256维特征 f。',
            'M 每行对应一个簇：k 是代表特征，v 是三个修正数，u 是已存评分。',
            '这里不重新试用乙丙，不重算 u，也不读取当前病例的人工标注。'],'aux'),
        ('② 只按当前相似度排序，取正相似度最高的最多3条',[
            '用当前 f 与 M 的每条 k 比较，重新计算这个位置的余弦相似度。',
            '例：m01为0.8、m02为0.7、m03为0.6、m04为0.5。',
            '本例前三名是 m01、m02、m03；m04 不进入此次候选。',
            '相似度随位置变化；不是按已有评分 u 排序，也不是按相似度乘 u 排序。'],'new'),
        ('③ 读取前三条的已存评分，去掉 u ≤ 0 的记忆',[
            'm01 已存 u=+0.01，m02 已存 u=−0.005，m03 已存 u=+0.02。',
            '删除 m02，本例只用 m01 与 m03；不再从第4名补一条。',
            '以下算比例时只包括这两条；其他记录不进入分母，也不参与组合。',
            '如果一条都没留下，直接令当前位置的三个修正数为 [0,0,0]。'],'gate'),
        ('④ 给两条记忆分别计算“当前位置相似度 × 已存评分”',[
            'm01 的乘积：0.8 × 0.01 = 0.008。',
            'm03 的乘积：0.6 × 0.02 = 0.012。',
            '这两个乘积用于决定两条修正量各占多少；不是重新给记忆评分。'],'new'),
        ('⑤ 每个乘积除以两者总和，得到40%与60%',[
            '只把本例两条参与组合的乘积相加：0.008 + 0.012 = 0.020。',
            'm01 的比例：0.008 ÷ 0.020 = 0.4，也就是40%。',
            'm03 的比例：0.012 ÷ 0.020 = 0.6，也就是60%；两者相加为100%。',
            '这一步才叫归一化：把两个乘积换算成占比，不改整条记录或已有 u。'],'new'),
        ('⑥ 用40%和60%分别乘对应的 v，再逐类别相加',[
            '从 M 读取：m01 的 v=[−0.15,−0.65,+0.80]。',
            '从 M 读取：m03 的 v=[−0.05,+0.20,−0.15]。',
            '背景：0.4×(−0.15) + 0.6×(−0.05) = −0.09。',
            '肝脏：0.4×(−0.65) + 0.6×(+0.20) = −0.14。',
            '肿瘤：0.4×(+0.80) + 0.6×(−0.15) = +0.23。'],'new'),
        ('⑦ 得到当前粗位置 q 的三个修正数 [−0.09,−0.14,+0.23]',[
            '这只是 q 的修正量，不是整张修正图，也还没有加到原始分数 z。',
            '下一个位置重新算相似度、选记忆和算比例；M 中的 k、v、u 不变。',
            '遍历有效位置后填成三通道图，接下图上采样，再乘整体强度 S。'],'out')])
    note(s,42,height,[
        '只剩一条记忆时，其正乘积除以自身得到比例1；没有记录时不做除法。',
        '上述百分比只表示修正量的混合占比，不是背景、肝脏、肿瘤的预测概率。',
        '复核甲、乙、丙自身时，排名前先排除该病例贡献的全部记录。'])
    s.save('检索展开图_方案3检索模块',height+130)

def correction_calculation():
    """承接检索图的同一组记忆与占比，逐步展开填图、上采样和分类。"""
    s=SVG(900,'修正展开图　从两条记忆到整张修正图，再到分割',
          '接上一张检索图：m01占40%、m03占60%；以下数值均为构造示例。')
    def card(y,heading,lines,color,arrow=True):
        h=60+29*len(lines)
        s.parts.append(f'<rect x="42" y="{y}" width="816" height="{h}" rx="8" fill="{COL[color]}" stroke="{INK}" stroke-width="1.5"/>')
        s.text(62,y+33,heading,23,True)
        note(s,62,y+66,lines,20)
        if arrow:s.arrow([(450,y+h+3),(450,y+h+35)])
        return y+h+40
    y=card(110,'① 接收上一张图已算好的比例，读取对应两条 v',[
        'm01：0.008÷0.020=0.4；m03：0.012÷0.020=0.6，分别占40%和60%。',
        '读取 m01 的 v=[−0.15,−0.65,+0.80]，m03 的 v=[−0.05,+0.20,−0.15]。',
        '比例已包含相似度与评分 u 的作用；本步不再乘一次相似度或 u。'],'aux')
    y=card(y,'② 在同一个粗位置 q，分别计算三个类别的组合修正',[
        '背景：40%×(−0.15) + 60%×(−0.05) = −0.09。',
        '肝脏：40%×(−0.65) + 60%×(+0.20) = −0.14。',
        '肿瘤：40%×(+0.80) + 60%×(−0.15) = +0.23。',
        '得到 q 的三个修正数 [−0.09,−0.14,+0.23]；还没有乘整体强度 S。'],'new')
    y=card(y,'③ 把这三个数填到三通道粗修正图 T 的同一个位置',[
        '例如 q=(5,6,7)，把−0.09写入背景通道在 q 的格子。',
        '把−0.14写入肝脏通道在 q 的格子，把+0.23写入肿瘤通道在 q 的格子。',
        '不是把同一组三个数复制到整张图；它们只属于本次查询的这个 q。'],'new',arrow=False)
    # Three channel slices make the destination of each correction explicit.
    grid_top=y-8
    for cx,title,value,color in [(180,'背景通道','−0.09',COL['aux']),
                                  (450,'肝脏通道','−0.14',COL['gate']),
                                  (720,'肿瘤通道','+0.23',COL['out'])]:
        s.text(cx,grid_top+16,title,21,True,True)
        for row in range(3):
            for col in range(3):
                x=cx-75+col*50;gy=grid_top+35+row*50
                cell_fill = color if row == 1 and col == 1 else COL['gray']
                s.parts.append(f'<rect x="{x}" y="{gy}" width="50" height="50" fill="{cell_fill}" stroke="{INK}"/>')
        s.text(cx,grid_top+100,value,15,True,True)
    note(s,42,grid_top+220,[
        '三个二维切片的中心格表示同一粗位置 q；其他格子有各自的计算结果。',
        '图中二维格子仅用于说明三个通道，实际 T 是三维空间网格。'],19)
    s.arrow([(450,grid_top+262),(450,grid_top+296)])
    y=grid_top+300
    y=card(y,'④ 遍历其余有效粗位置，各自查询后填回 T',[
        '下一个位置重新与各条 k 比较、选top3、读取 u、算占比并组合 v。',
        '因此，不同位置可选中不同记忆，也可得到不同的比例和三个修正数。',
        '没有可用记忆的位置填 [0,0,0]；未参与查询的位置按规则初始化为零。',
        '所有位置填完，T 的示例形状为 [1,3,16,16,16]：1个块、3类、16³网格。'],'out')
    y=card(y,'⑤ 把 T 的三个通道分别上采样，得到 C_base',[
        '每个通道从16³插值到128³；得到 [1,3,128,128,128] 的组合修正图。',
        'C_base 与输出头原始三类分数 z 在空间网格和类别通道上一一对应。',
        '上采样的是三个修正通道；一个细位置可能混合邻近粗位置的修正。',
        '以下假设某个细位置插值后得到 [−0.09,−0.14,+0.23]，不是必然保持原值。'],'aux')
    y=card(y,'⑥ 在该输出细位置，修正图乘 S 后逐类别加到原分数',[
        '例：原分数 z=[0,1.0,0.9]，假设已保存的学得强度 S=0.5。',
        '背景新分数：0 + 0.5×(−0.09) = −0.045。',
        '肝脏新分数：1.0 + 0.5×(−0.14) = 0.93。',
        '肿瘤新分数：0.9 + 0.5×(+0.23) = 1.015。',
        '合并为 [−0.045,0.93,1.015]；这是原始类别分数，不是概率。'],'gate')
    y=card(y,'⑦ 每个输出位置取最大分数的类别，得到最终分割',[
        '原来 [0,1.0,0.9] 中肝脏最高；本例修正后肿瘤分数1.015最高。',
        '若需要概率，再对三个修正后分数做softmax；取最大类别的结果相同。',
        '对所有输出位置执行同样操作，形成分割图；不修改网络参数或人工标注。',
        '这个例子只说明类别如何改变，是否改对需要在评价时与标注比较。'],'out',arrow=False)
    note(s,42,y+8,[
        '支持标注只训练共同 S；独立验证选训练快照；新病例直接使用保存的 S。',
        '当前仍是粗网格逐位置修正候选；有效位置、插值和边界规则需在实现前固定。'])
    s.save('修正展开图_方案3组合修正与分类',y+110)

def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    FIGURES.clear()
    panel(5,'方案3阶段与准备产物','阶段1训练源网络；阶段2准备 M、S；阶段3预测新病例。',[
        ('阶段1　LiTS 源训练 → 源权重',['LiTS CT＋标注训练原网络；此时更新网络参数。','随后冻结源网络；阶段2只学习外部标量 S。'],'base'),
        ('阶段2　建库 → 无评分的 M',['甲、乙、丙 CT＋标注 → 冻结网络读取 A、z。','按四组建记录：来源、组别、中心 k、修正方向 v。'],'aux'),
        ('阶段2　跨病例评分 → 给 M 写入 u',['甲记录只试乙、丙；每条记录独立试用。','u＝另外两例原损失减修正后损失的平均。'],'new'),
        ('阶段2　支持病例学习一个 S → 独立验证保存',['固定源网络和 M，支持病例查询时排除自身记录，只更新 S。','额外验证病例只评价训练快照，不反向传播、不建库或重评 u。'],'gate'),
        ('阶段2保存产物 → 用于阶段3预测',['固定源权重、已评分 M、强度 S、检索与空间规则。','预测时只读取新 CT 图像，不使用这例的人工标注。','预测完成后，将分割结果与人工标注比较，计算评价指标。'],'out')])
    network(6,3)
    record_building()
    pairing_module()
    scoring()
    strength_selection()
    retrieval_calculation()
    correction_calculation()
    (OUT/'manifest.json').write_text(json.dumps(FIGURES,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Generated {len(FIGURES)} scheme3 figures in {OUT}')

def generate_document_main_figures():
    """独立方案3文档使用本地图号，保留总览图5、图6。"""
    target = ROOT / '方案3_问题定义与思路检索' / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    for source_name, target_name, old_title, new_title in [
        ('图5_方案3阶段与准备产物.svg', '图2_阶段与准备产物.svg',
         '图5　方案3阶段与准备产物', '图2　阶段与准备产物'),
        ('图6_方案3新病例预测主网络.svg', '图1_新病例预测主网络.svg',
         '图6　方案3新病例预测主网络', '图1　新病例预测主网络'),
        ('建库展开图_方案3建库模块.svg', '图3_建库与单条记忆.svg',
         '建库展开图　A 和输出怎样共同生成一条记忆', '图3　完整建库流程与局部算例'),
        ('评分展开图_方案3跨病例评分模块.svg', '图4_跨病例评分.svg',
         '评分展开图　方案3跨病例评分模块', '图4　甲的一条记忆怎样在乙、丙上试用并评分'),
        ('强度选择展开图_方案3验证选择强度.svg', '图5_验证选择整体强度.svg',
         '强度学习展开图　S 怎样学出来', '图5　先生成组合修正图，再学习一个整体强度 S'),
    ]:
        tree = ET.parse(OUT / source_name)
        matches = 0
        for node in tree.iter():
            if node.text and old_title in node.text:
                node.text = node.text.replace(old_title, new_title)
                matches += 1
        if matches != 1:
            raise ValueError(f'Expected one figure title: {source_name}, got {matches}')
        tree.write(target / target_name, encoding='unicode')


if __name__=='__main__':
    import sys
    if '--score-only' in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        scoring()
        target = ROOT / '方案3_问题定义与思路检索' / 'figures' / '图4_跨病例评分.svg'
        tree = ET.parse(OUT / '评分展开图_方案3跨病例评分模块.svg')
        for node in tree.iter():
            if node.text == '评分展开图　方案3跨病例评分模块':
                node.text = '图4　甲的一条记忆怎样在乙、丙上试用并评分'
        tree.write(target, encoding='unicode')
    else:
        generate()
        generate_document_main_figures()
