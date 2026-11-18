"""把方案3的关键公式渲染为SVG，供Markdown直接显示，不依赖预览器数学插件。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '方案3_问题定义与思路检索/figures'
FONT = FontProperties(family='Noto Sans CJK SC')
plt.rcParams['svg.fonttype'] = 'path'

def card(filename, title, rows):
    fig = plt.figure(figsize=(11, 1.0 + 1.1 * len(rows)), facecolor='#f4f7fb')
    fig.text(.05, .93, title, fontproperties=FONT, fontsize=19, color='#28384A', va='top', weight='bold')
    for i, (formula, explanation) in enumerate(rows):
        y = .77 - i * (.73 / len(rows))
        fig.text(.06, y, formula, fontsize=23, color='#223b55', va='center')
        fig.text(.06, y-.065, explanation, fontproperties=FONT, fontsize=12, color='#53677b', va='top')
    fig.savefig(OUT / filename, format='svg', bbox_inches='tight', pad_inches=.25)
    plt.close(fig)

def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    card('公式_检索到最终输出.svg', '从一条记忆到整张修正图：每一步的输出接到哪里', [
        (r'$a_i(q)=s_i(q)u_i,\qquad w_i(q)=\frac{a_i(q)}{\sum_{j\in I(q)}a_j(q)}$', 'I(q)：筛选后保留的记录；s：正相似度；w：这些记录的占比'),
        (r'$r(q)=\sum_{i\in I(q)}w_i(q)v_i\ \in\ \mathbb{R}^{3}$', '一个粗位置q得到三个数，分别调整背景、非肿瘤肝脏、肿瘤'),
        (r'$T[0,:,q]=r(q),\qquad C_{\mathrm{base}}=\mathrm{Upsample}(T)$', '遍历q形成三通道粗修正图T；上采样后才是与输出头同网格的C_base'),
        (r'$z_{\mathrm{corrected}}=z+S\,C_{\mathrm{base}}$', 'z来自最后一级解码器的输出头，尚未softmax；S控制整张图的修正幅度'),
        (r'$P_{\mathrm{corrected}}=\mathrm{softmax}(z_{\mathrm{corrected}}),\quad \hat y=\mathrm{argmax}_{c}\ P_{\mathrm{corrected},c}$', '沿三个类别通道转为概率、取最大类别；没有合格记录的位置r(q)=0'),
    ])
    card('公式_中心k.svg', 'k：同一簇的特征平均后，保存其单位方向', [
        (r'$\bar f=\frac{1}{n}\sum_{j=1}^{n}f_j,\qquad k=\frac{\bar f}{\Vert\bar f\Vert_2}$', 'n：这簇的位置数；f_j和k均为256维；平均向量为零时需另定处理规则'),
    ])
    card('公式_修正量v.svg', 'v：人工答案指示数与预测概率的差，按同一簇平均', [
        (r'$Y_j=\mathrm{onehot}(y_j)\ \in\ \{0,1\}^{3},\qquad d_j=Y_j-P_j$', 'y_j：人工标注类别；Y_j：三类指示数；P_j：这位置的三个预测概率'),
        (r'$v=\frac{1}{n}\sum_{j=1}^{n}d_j\ \in\ \mathbb{R}^{3}$', 'v三个分量依次对应背景、非肿瘤肝脏、肿瘤；当前候选将它用于修正输出分数'),
    ])
    print('Rendered three formula SVGs.')

if __name__ == '__main__': generate()
