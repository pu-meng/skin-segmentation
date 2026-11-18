"""Scheme 2 proposal only: parallel LP/RepDW and FFT-wave branches."""
from draw_module_figures import Diagram

def main():
 d=Diagram('方案 2 · LP-WPO 双分支 Skip（设计稿）',1150)
 def n(k,x,y,t,c='projection'):return d.node(k,x,y,t,c,role=None)
 n('x',400,95,'对齐后的 Skip X\n64 ch · 64×64','input')
 n('split',215,245,'A. 拉普拉斯分解\n低频 L + 残差 H')
 n('fft',605,245,'B. 二维 FFT\n读取完整 Skip X')
 n('rep',215,415,'H → RepDW3\n低频 L 保留','state')
 n('wave',605,415,'阻尼波频率响应\n按频率调制；参数待定','state')
 n('lp',215,575,'低频 + 处理后细节\n局部结果 Y_LP','output')
 n('iw',605,575,'逆 FFT\n传播结果 Y_WPO','output')
 n('mix',400,745,'按通道融合\n每通道一个 β，空间共享','state')
 n('res',400,895,'输入残差 + 缩放修正\nZ = X + γ × 融合结果','output')
 n('out',400,1035,'与上采样后的 D 拼接 → D0\n保留四层版 Decoder 接口','output')
 for a,b in [('x','split'),('x','fft'),('split','rep'),('rep','lp'),('fft','wave'),('wave','iw'),('lp','mix'),('iw','mix'),('mix','res'),('res','out')]:d.edge(a,b)
 a,t=d.nodes['x'],d.nodes['res']
 d.edge('x','res',source='left',target='left',route=[(a['left'],a['y']),(25,a['y']),(25,t['y']),(t['left'],t['y'])])
 d.label(100,810,'原始 X')
 d.label(400,1120,'空间响应与频率响应是不同坐标；本图未加入 Decoder 条件门控')
 d.save('24_方案2_LP_WPO双分支')
if __name__=='__main__':main()
