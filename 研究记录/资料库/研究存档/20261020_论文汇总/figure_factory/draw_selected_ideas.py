"""Reader diagrams for four candidate ideas, not implemented architectures.

Run with the medseg environment. SVG assets are linked in 论文选择的idea.md;
PNG previews under /tmp are for visual review only.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'selected_ideas'
PREVIEW = Path('/tmp/selected_ideas_preview')
REG = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK = '#334155'
COLORS = {
    'base': ('#edf2f7', '#64748b'),
    'feature': ('#e0effc', '#28679d'),
    'learn': ('#efe5fa', '#8054a1'),
    'memory': ('#fff0d5', '#b67517'),
    'out': ('#e5f2e7', '#42764b'),
    'inhibit': ('#fce3e9', '#b54e70'),
}


def start(height, title, subtitle):
    global fig, ax, texts, boxes
    fig, ax = plt.subplots(figsize=(9, height * .5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 18), ylim=(0, height))
    ax.axis('off')
    texts, boxes = [], []
    txt(9, height-.65, title, 18, bold=True)
    txt(9, height-1.4, subtitle, 12, color='#64748b')


def txt(x, y, value, size=13, bold=False, color=INK):
    t = ax.text(x, y, value, ha='center', va='center', fontsize=size,
                fontproperties=BOLD if bold else REG, color=color,
                linespacing=1.5, zorder=5)
    texts.append(t)
    return t


def box(x, y, title, sub='', kind='base', width=6, height=1.4):
    fill, edge = COLORS[kind]
    p = FancyBboxPatch((x-width/2, y-height/2), width, height,
                       boxstyle='round,pad=0,rounding_size=.15',
                       facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ts = [txt(x, y+.25 if sub else y, title, 14, True)]
    if sub:
        ts.append(txt(x, y-.32, sub, 12))
    boxes.append((p, ts))


def arrow(a, b, dashed=False, color=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                 linewidth=1.5, color=color, linestyle='--' if dashed else '-',
                 shrinkA=0, shrinkB=0, zorder=2))


def path(points, dashed=False, color=INK):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.5,
                linestyle='--' if dashed else '-', zorder=1)
    arrow(points[-2], points[-1], dashed, color)


def plus(x, y):
    ax.add_patch(Circle((x, y), .27, edgecolor=INK, facecolor='white', lw=1.5, zorder=3))
    txt(x, y, '+', 17)


def save(name):
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in texts:
        bb = t.get_window_extent(renderer)
        assert fig.bbox.contains(bb.x0, bb.y0) and fig.bbox.contains(bb.x1, bb.y1), t.get_text()
    for p, ts in boxes:
        bb = p.get_window_extent(renderer)
        for t in ts:
            tb = t.get_window_extent(renderer)
            assert tb.x0 >= bb.x0+2 and tb.x1 <= bb.x1-2, t.get_text()
            assert tb.y0 >= bb.y0 and tb.y1 <= bb.y1, t.get_text()
    filename = OUT / (name+'.svg')
    fig.savefig(filename, facecolor='white', metadata={'Date': None})
    ET.parse(filename)
    fig.savefig(PREVIEW / (name+'.png'), dpi=120, facecolor='white')
    plt.close(fig)
    print(filename)


def overview():
    start(12, '四个方案：分别在哪一步判断“新信息是否有用”？',
          '四条独立候选路线，不是要串起来的四个模块')
    box(9, 9.5, '共同现象：适配有收益，也会改坏原本正确的预测',
        '研究目标：保留收益，并减少有害改变', width=16, height=1.6)
    cards = [
        (4.65, 6.8, '原方案5·方法1｜选特征', '源特征／适配特征', '传给解码器之前选', 'feature'),
        (13.35, 6.8, '原方案3·方法1｜选记忆', '支持患者的纠错经验', '检索后按跨患者收益筛选', 'memory'),
        (4.65, 3.8, '原方案1·方法8｜选路径', '多层特征的修正支路', '训练中按路径收益调偏好', 'learn'),
        (13.35, 3.8, '原方案7｜更新状态', '当前状态＋其他层／邻域', '预测中多轮增强、抑制、再核查', 'inhibit'),
    ]
    for x, y, title, sub, footer, kind in cards:
        box(x, y, title, sub, kind, width=7.6, height=1.6)
        txt(x, y-1.25, footer, 12)
    txt(9, .65, '当前：候选设计；是否能解决问题，需要各自的实验与对照。', 12)
    save('00_four_ideas_overview')


def features():
    start(15, '原方案5·方法1｜源特征与适配特征混合',
          '图示一个skip；不同层分别混合，不把四层压成一份')
    box(9, 12.65, '同一张CT', '两路读取同一空间位置', width=6)
    box(4.5, 10.2, '源编码器原件', '参数冻结，但仍计算特征', 'base', width=7)
    box(13.5, 10.2, '适配编码器副本', '从源权重复制，用支持标签训练', 'learn', width=7)
    path([(7, 11.95), (7, 11.5), (4.5, 11.5), (4.5, 10.9)])
    path([(11, 11.95), (11, 11.5), (13.5, 11.5), (13.5, 10.9)])
    box(4.5, 7.9, '这一层的源特征', '形状、位置与另一路对应', 'feature', width=7)
    box(13.5, 7.9, '这一层的适配特征', '每个通道对应一份空间特征', 'feature', width=7)
    arrow((4.5, 9.5), (4.5, 8.6)); arrow((13.5, 9.5), (13.5, 8.6))
    box(9, 5.5, '按通道学习两路比例，再相加',
        '混合特征 = (1−比例) × 源特征 ＋ 比例 × 适配特征', 'learn', width=15, height=1.6)
    path([(4.5, 7.2), (4.5, 6.7), (6.5, 6.7), (6.5, 6.3)])
    path([(13.5, 7.2), (13.5, 6.7), (11.5, 6.7), (11.5, 6.3)])
    plus(9, 3.7); arrow((9, 4.7), (9, 3.97))
    box(3.7, 3.7, '同级上采样特征', kind='feature', width=5.4, height=1.1)
    arrow((6.4, 3.7), (8.73, 3.7))
    box(9, 1.8, '解码块 → 后续解码 → 最终分割', '混合比例在训练时学习，预测时固定', 'out', width=14)
    arrow((9, 3.43), (9, 2.5))
    txt(9, .5, '对照：全部适配、固定混合、每层一个比例、输出集成。', 12)
    save('01_feature_selection')


def memory():
    start(18, '原方案3·方法1｜纠错记忆先试用，再检索',
          '上半部分提前准备记忆；下半部分用于未见患者')
    box(9, 15.55, '① 支持病例建库', '固定源模型，保存局部特征、纠错方向和来源', 'memory', width=15)
    box(9, 13.15, '② 每条记忆单独在其他支持患者上试用',
        '例如：甲的记录分别用于乙、丙，排除甲自身', 'memory', width=15)
    arrow((9, 14.85), (9, 13.85))
    box(9, 10.75, '③ 比较纠错前后损失，写入有效性评分',
        '从接收患者原预测开始，不把其他记录的修正累加', 'learn', width=15)
    arrow((9, 12.45), (9, 11.45))
    box(9, 8.35, '④ 固定记忆与评分，学习共同修正强度',
        '支持病例训练强度；独立验证选择快照', 'learn', width=15)
    arrow((9, 10.05), (9, 9.05))
    ax.plot([1, 17], [6.95, 6.95], color='#94a3b8', ls='--', lw=1)
    txt(9, 6.5, '新病例预测：不读取新病例标签', 13, True)
    box(4.5, 4.95, '新CT → 固定源模型', '得到查询特征与原始三类分数', 'base', width=7.2)
    box(13.5, 4.95, '按特征检索 → 按评分过滤', '组合纠错量；无合格记录则不修正', 'memory', width=7.2)
    arrow((8.1, 4.95), (9.9, 4.95))
    box(9, 2.3, '原始分数 ＋ 强度 × 纠错量 → 最终分割',
        '相似度回答“像不像”；评分回答“试用后有没有帮助”', 'out', width=16, height=1.6)
    path([(4.5, 4.25), (4.5, 3.6), (6.5, 3.6), (6.5, 3.1)])
    path([(13.5, 4.25), (13.5, 3.6), (11.5, 3.6), (11.5, 3.1)])
    txt(9, .65, '对照：只按相似度、加入置信度、同数量随机过滤。', 12)
    save('02_memory_filtering')


def routes():
    start(18, '原方案1·方法8｜这条路径真的帮忙了吗？',
          '一个接收层的示意；历史收益在训练轮末更新，推理时固定')
    box(9, 15.55, '当前CT的多层编码特征', '分别对齐并经过专家，得到三路修正特征', 'feature', width=15)
    box(5.2, 13.05, '当前特征 → 路由分数', '随病例和位置变化', 'learn', width=7.4)
    box(13.3, 13.05, '已保存的历史收益', '每条路径一个偏好', 'memory', width=7.4)
    path([(7, 14.85), (7, 14.35), (5.2, 14.35), (5.2, 13.75)])
    box(9, 10.6, '结合分数 → 三路权重 → 加权修正',
        '加回原skip，与上采样特征相加后继续解码', 'learn', width=15)
    path([(5.2, 12.35), (5.2, 11.8), (6.5, 11.8), (6.5, 11.3)])
    path([(13.3, 12.35), (13.3, 11.8), (11.5, 11.8), (11.5, 11.3)])
    txt(9, 9.25, '训练时：同一输入、同一参数，额外做一次指定路径关闭比较', 12)
    box(5, 7.65, '全部新增路径开启', '最终分割与标签 → 损失A', 'out', width=7)
    box(13, 7.65, '只关闭指定新增路径', '最终分割与同一标签 → 损失B', 'inhibit', width=7)
    path([(1.5, 10.6), (.65, 10.6), (.65, 8.95), (5, 8.95), (5, 8.35)])
    path([(16.5, 10.6), (17.35, 10.6), (17.35, 8.95), (13, 8.95), (13, 8.35)])
    box(9, 5.15, '路径条件收益 = 损失B − 损失A',
        '正值：关闭更差；负值：关闭反而更好', 'memory', width=15)
    path([(5, 6.95), (5, 6.4), (6.5, 6.4), (6.5, 5.85)])
    path([(13, 6.95), (13, 6.4), (11.5, 6.4), (11.5, 5.85)])
    box(9, 2.75, '训练轮末汇总并平滑 → 下一轮历史偏好',
        '额外比较不反传；正常分割训练仍更新网络参数', 'memory', width=15)
    arrow((9, 4.45), (9, 3.45))
    path([(16.5, 2.75), (17.55, 2.75), (17.55, 13.05), (17, 13.05)], True, '#b67517')
    txt(9, .8, '虚线：下一训练轮读取；不表示同一次预测内部循环。', 12)
    save('03_path_feedback')


def brain():
    start(20, '原方案7｜仿脑循环：增强、抑制，再返回更新',
          '候选机制示意：共享参数、多轮改变状态；不是已有实验结果')
    box(9, 17.55, 'CT → 编码器 → 初始特征状态',
        '示意跨stage版本：读取Stage 2与Stage 3输出', 'base', width=15)
    box(4.8, 15.1, '本轮较浅层状态', '来自Stage 2的初始特征', 'feature', width=7.2)
    box(13.2, 15.1, '本轮较深层状态', '来自Stage 3的初始特征', 'feature', width=7.2)
    path([(6.5, 16.85), (6.5, 16.3), (4.8, 16.3), (4.8, 15.8)])
    path([(11.5, 16.85), (11.5, 16.3), (13.2, 16.3), (13.2, 15.8)])
    box(9, 12.65, '对齐空间与通道 → 两层交换信息',
        '每层读取自己的状态与另一层发来的消息', 'learn', width=15)
    path([(4.8, 14.4), (4.8, 13.85), (6.5, 13.85), (6.5, 13.35)])
    path([(13.2, 14.4), (13.2, 13.85), (11.5, 13.85), (11.5, 13.35)])
    box(4.8, 10.2, '兴奋分支：提出增强量', '帮助有用信息继续传递', 'learn', width=7.2)
    box(13.2, 10.2, '抑制分支：提出削弱量', '尝试限制有害信息传递', 'inhibit', width=7.2)
    path([(6.5, 11.95), (6.5, 11.45), (4.8, 11.45), (4.8, 10.9)])
    path([(11.5, 11.95), (11.5, 11.45), (13.2, 11.45), (13.2, 10.9)])
    box(9, 7.75, '保留当前状态 ＋ 增强量 − 削弱量',
        '得到两层的新状态，作为下一轮输入', 'out', width=15)
    path([(4.8, 9.5), (4.8, 8.95), (6.5, 8.95), (6.5, 8.45)])
    path([(13.2, 9.5), (13.2, 8.95), (11.5, 8.95), (11.5, 8.45)])
    path([(1.5, 7.75), (.55, 7.75), (.55, 15.1), (1.2, 15.1)], True, '#8054a1')
    path([(16.5, 7.75), (17.45, 7.75), (17.45, 15.1), (16.8, 15.1)], True, '#8054a1')
    box(9, 5.05, '有限轮后的特征 → skip连接 → 解码与分割',
        '示意先试两轮；轮数由验证确定，不承诺稳定', 'out', width=15)
    arrow((9, 7.05), (9, 5.75))
    txt(9, 3.85, '虚线：返回下一轮，复用参数；向下实线：结束循环后解码。', 12)
    box(9, 2.55, '局部版本：同一block内的通道与邻近位置循环',
        '也可让当前候选状态反复读取内部与周围特征', 'base', width=15, height=1.6)
    txt(9, 1.15, '局部循环与跨stage循环分开试验，先不叠加。', 12)
    txt(9, .5, '验证：单轮 / 普通循环 / 加显式抑制 / 计算量匹配前馈。', 12)
    save('04_brain_recurrent')


if __name__ == '__main__':
    overview()
    features()
    memory()
    routes()
    brain()
