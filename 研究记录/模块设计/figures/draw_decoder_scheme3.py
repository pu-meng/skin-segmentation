"""Scheme 3: full-sized frequency groups and bidirectional cross gates, proposal only."""
from draw_module_figures import Diagram
from draw_integrated_schemes import network

def main():
 d=Diagram('方案 3 · 频率解耦多分支 Decoder Block V1',2040)
 network(d,3,decoder_level=1)
 def n(k,x,y,t,c='projection'):return d.node(k,x,y,t,c,role=None)
 n('x',400,875,'输入 X\n128 ch · 32×32','input')
 n('norm',400,985,'① PW1 + 通道 LayerNorm\n得到 U；原 X 走残差')
 n('split',400,1100,'② 单层拉普拉斯分解\nU_L = E(R(U))；U_H = U − U_L')
 n('low',210,1230,'③ 低频投影并二分\n两路各 64 ch')
 n('high',620,1230,'④ 高频投影\n两路均 128 ch','state')
 n('m',110,1360,'SS2D\n选择性扫描')
 n('a',325,1360,'Q/K/V Attention\n全空间 token 关系')
 n('r',540,1360,'RepDW3\n局部响应','state')
 n('k',725,1360,'DW1×7\n→ DW7×1','state')
 n('yl',210,1490,'Concat → PW1\n低频结果 Y_L：128 ch','output')
 n('yh',620,1490,'Concat → PW1\n高频结果 Y_H：128 ch','output')
 n('gate',400,1640,'⑤ 双向交叉门控\nY_L → G_H；Y_H → G_L\n每路乘以 1 + λ × G','state')
 n('out',400,1810,'⑥ Concat → PW1 → 缩放残差\nZ = X + γ × 融合结果','output')
 n('next',400,1940,'128 ch · 32×32\n回到 Decoder 主路','output')
 for a,b in [('x','norm'),('norm','split'),('split','low'),('split','high'),('low','m'),('low','a'),('high','r'),('high','k'),('m','yl'),('a','yl'),('r','yh'),('k','yh'),('yl','gate'),('yh','gate'),('gate','out'),('out','next')]:d.edge(a,b)
 a,t=d.nodes['x'],d.nodes['out']
 d.edge('x','out',source='left',target='left',route=[(a['left'],a['y']),(25,a['y']),(25,t['y']),(t['left'],t['y'])])
 d.label(90,1740,'原始 X')
 d.label(400,2010,'设计稿；D1 融合后为候选示例位置，SS2D／门控参数约束待固定')
 d.save('27_方案3_Decoder多分支一体图')
if __name__=='__main__':main()
