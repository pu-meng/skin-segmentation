"""Code/paper-traced RWKV figures. Typography and boxes use the shared FigureStyle."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'研究记录/figures/rwkv'
NAMES=[]

def stack(stem,title,rows,bypasses=()):
 st=FigureStyle();fig,ax=st.canvas(2000);width=st.cfg['output']['width_px'];gap=st.cfg['box']['node_gap']
 sizes=[st.measure(ax,text)for text,color in rows]
 assert max(w for w,h in sizes)<width-80,(stem,max(w for w,h in sizes))
 title_height=st.measure(ax,title,role='heading')[1]
 top=title_height+12;total=top+sum(h for w,h in sizes)+gap*(len(rows)-1)+15
 fig.set_size_inches(width/st.cfg['output']['dpi'],total/st.cfg['output']['dpi']);ax.set_ylim(total,0)
 st.text(ax,width/2,title_height/2,title,role='heading');nodes=[]
 for (text,color),(w,h) in zip(rows,sizes):
  node=st.box(ax,width/2,top+h/2,text,st.cfg['colors'][color]);nodes.append(node);top+=h+gap
 for a,b in zip(nodes[:-1],nodes[1:]):st.arrow(ax,[(a['x'],a['bottom']),(b['x'],b['top'])])
 for a,b,side in bypasses:
  n1,n2=nodes[a],nodes[b];x=width-18 if side=='right' else 18
  e1='right' if side=='right'else'left'
  st.arrow(ax,[(n1[e1],n1['y']),(x,n1['y']),(x,n2['y']),(n2[e1],n2['y'])])
 st.save(fig,OUT/stem);NAMES.append(stem)

def fourway(stem,title,input_text,branches,output_text):
 st=FigureStyle();fig,ax=st.canvas(700);width=st.cfg['output']['width_px']
 mh=max(st.measure(ax,t)[1]for t in branches);ih=st.measure(ax,input_text)[1];oh=st.measure(ax,output_text)[1]
 input_y=60+ih/2;row1=input_y+ih/2+65+mh/2;row2=row1+mh+65;out_y=row2+mh/2+65+oh/2
 total=out_y+oh/2+15;fig.set_size_inches(width/st.cfg['output']['dpi'],total/st.cfg['output']['dpi']);ax.set_ylim(total,0)
 st.text(ax,400,22,title,role='heading');entry=st.box(ax,400,input_y,input_text,st.cfg['colors']['input'])
 nodes=[]
 for text,x,y in zip(branches,[195,605,195,605],[row1,row1,row2,row2]):
  node=st.box(ax,x,y,text,st.cfg['colors']['projection']);nodes.append(node)
  assert node['w']<330,(stem,node['w'])
  outside=18 if x<400 else 782;edge='left'if x<400 else'right'
  split=entry['bottom']+20
  st.arrow(ax,[(400,entry['bottom']),(400,split),(outside,split),(outside,y),(node[edge],y)])
  inner='right'if x<400 else'left'
  st.arrow(ax,[(node[inner],y),(400,y)])
 out=st.box(ax,400,out_y,output_text,st.cfg['colors']['output'])
 st.arrow(ax,[(400,row1),(400,out['top'])])
 st.save(fig,OUT/stem);NAMES.append(stem)

def mixing(stem,title,visual=False):
 st=FigureStyle();fig,ax=st.canvas(1000);width=st.cfg['output']['width_px']
 rows=[('输入X\n[B,N,C]'if visual else'输入X\n[B,L,C]','input'),
       ('1. Q-Shift：四组通道移动'if visual else'1. Token Shift：向后移位，首位补0','projection'),
       ('2. 分别与原输入混合\n生成Xr、Xk、Xv；三组参数独立','projection')]
 top=60;nodes=[]
 for text,color in rows:
  w,hh=st.measure(ax,text);node=st.box(ax,400,top+hh/2,text,st.cfg['colors'][color]);nodes.append(node);top+=hh+28
 for a,b in zip(nodes[:-1],nodes[1:]):st.arrow(ax,[(400,a['bottom']),(400,b['top'])])
 branch_y=top+48;branches=[]
 for x,text in [(130,'3. R分支\nLinear → Sigmoid'),(400,'3. K分支\nLinear'),(650,'3. V分支\nLinear')]:
  node=st.box(ax,x,branch_y,text,st.cfg['colors']['projection']);branches.append(node)
  st.arrow(ax,[(400,nodes[-1]['bottom']),(400,top-6),(x,top-6),(x,node['top'])])
 top=max(n['bottom']for n in branches)+35
 wtext='4. Bi-WKV\nRWKV-UNet：接LN\nURWKV默认：不接LN'if visual else'4. 因果WKV\nK决定权重，V提供内容'
 w,hh=st.measure(ax,wtext);wkv=st.box(ax,520,top+hh/2,wtext,st.cfg['colors']['state'])
 for node in branches[1:]:st.arrow(ax,[(node['x'],node['bottom']),(node['x'],top-12),(520,top-12),(520,wkv['top'])])
 top=wkv['bottom']+35;out_text='5. R门控乘WKV输出 → 输出Linear\n形状回到与输入相同'
 w,hh=st.measure(ax,out_text);out=st.box(ax,400,top+hh/2,out_text,st.cfg['colors']['output'])
 st.arrow(ax,[(520,wkv['bottom']),(520,out['top'])])
 st.arrow(ax,[(130,branches[0]['bottom']),(18,branches[0]['bottom']),(18,out['y']),(out['left'],out['y'])])
 total=out['bottom']+15;fig.set_size_inches(width/st.cfg['output']['dpi'],total/st.cfg['output']['dpi']);ax.set_ylim(total,0)
 st.text(ax,400,22,title,role='heading');st.save(fig,OUT/stem);NAMES.append(stem)

def network(stem,kind):
 st=FigureStyle();width=st.cfg['output']['width_px'];fig,ax=st.canvas(1400)
 if kind=='unet':
  title='RWKV-UNet-T：编码器、CCM与解码器'
  left=[('RGB输入','[B,3,256,256]'),('1. Stem','24通道，256×256'),('2. Stage I：LP×2','32通道，128×128'),('3. Stage II：LP×2','48通道，64×64'),('4. Stage III','下采样＋GLSP×3\n96通道，32×32'),('5. Stage IV','下采样＋GLSP×1\n160通道，16×16')]
  right=[('10. 输出logits','[B,K,256,256]'),('9. Decoder 4','拼接F1′ → UpBlock\n24通道，256×256'),('8. Decoder 3','拼接F2′ → UpBlock\n32通道，128×128'),('7. Decoder 2','拼接F3′ → UpBlock\n48通道，64×64'),('6. Decoder 1','UpBlock\n96通道，32×32'),('最深层输入','160通道，16×16')]
  middle='CCM\n三尺度融合';incoming=[2,3,4];outgoing=[1,2,3]
 else:
  title='URWKV：三阶段编码器与三阶段解码器'
  left=[('输入图像I','[B,3,H,W]'),('1. 输入Conv3×3','32通道，H×W'),('2. 编码阶段1','URWKV×3 → 投影\n32通道，H×W'),('3. 编码阶段2','下采样 → URWKV×3\n投影：64通道，H/2'),('4. 编码阶段3','下采样 → URWKV×3\n投影：128通道，H/4'),('5. 最深特征','双线性下采样\n128通道，H/8')]
  right=[('9. 图像恢复输出','Conv3×3输出3通道\n再与原图I相加'),('8. 解码阶段3','上采样 → SSF → 块×2\n投影：32通道，H'),('7. 解码阶段2','上采样 → SSF → 块×2\n投影：32通道，H/2'),('6. 解码阶段1','上采样 → SSF → 块×2\n投影：64通道，H/4'),('解码输入','128通道，H/8'),('阶段状态历史','头部和各阶段块输出\n送后续LAN')]
  middle='SSF×3\n三尺度预测门控';incoming=[2,3,4];outgoing=[1,2,3]
 # Fixed columns are layout positions, with each box fitted to measured text.
 measurements=[]
 for l,r in zip(left,right):measurements.append(max(st.measure(ax,'\n'.join(l))[1],st.measure(ax,'\n'.join(r))[1]))
 top=65;ys=[]
 for hh in measurements:ys.append(top+hh/2);top+=hh+45
 total=top+10;fig.set_size_inches(width/st.cfg['output']['dpi'],total/st.cfg['output']['dpi']);ax.set_ylim(total,0)
 st.text(ax,400,23,title,role='heading');ln=[];rn=[]
 for i,((a,b),(c,d),y)in enumerate(zip(left,right,ys)):
  ln.append(st.box(ax,154,y,a+'\n'+b,st.cfg['colors']['input' if i==0 else 'projection']))
  rn.append(st.box(ax,646,y,c+'\n'+d,st.cfg['colors']['output' if i==0 else 'state']))
  assert ln[-1]['right']<300 and rn[-1]['left']>500,(stem,i,ln[-1]['right'],rn[-1]['left'])
 for a,b in zip(ln[:-1],ln[1:]):st.arrow(ax,[(154,a['bottom']),(154,b['top'])])
 if kind=='unet':
  for i in range(5,0,-1):st.arrow(ax,[(646,rn[i]['top']),(646,rn[i-1]['bottom'])])
  st.arrow(ax,[(ln[-1]['right'],ln[-1]['y']),(rn[-1]['left'],rn[-1]['y'])])
 else:
  st.arrow(ax,[(ln[-1]['right'],ln[-1]['y']),(410,ln[-1]['y']),(410,rn[4]['y']),(rn[4]['left'],rn[4]['y'])])
  for i in range(4,0,-1):st.arrow(ax,[(646,rn[i]['top']),(646,rn[i-1]['bottom'])])
 if kind=='urwkv':
  st.arrow(ax,[(ln[0]['right'],ln[0]['y']),(rn[0]['left'],rn[0]['y'])])
 center=st.box(ax,400,ys[3]-5,middle,st.cfg['colors']['state'])
 for offset,i in zip([-18,0,18],incoming):
  st.arrow(ax,[(ln[i]['right'],ln[i]['y']),(center['left']-12,ln[i]['y']),(center['left']-12,center['y']+offset),(center['left'],center['y']+offset)])
 for offset,i in zip([-18,0,18],outgoing):
  st.arrow(ax,[(center['right'],center['y']+offset),(center['right']+12,center['y']+offset),(center['right']+12,rn[i]['y']),(rn[i]['left'],rn[i]['y'])])
 st.save(fig,OUT/stem);NAMES.append(stem)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 stack('01_RWKV总图','RWKV-4：2023论文的语言模型总图',[
 ('词元ID\n[B,L]','input'),('1. Embedding与入口LayerNorm\n[B,L,C]','projection'),
 ('2. RWKV残差块重复D层\n每层：TimeMix残差 → ChannelMix残差','state'),
 ('3. 最后LayerNorm → Linear词表投影\n[B,L,Vocab]','projection'),('4. 输出词元logits\nSoftmax用于得到预测概率','output')])
 stack('02_RWKV残差块','RWKV-4：一个残差块的两次相加',[
 ('输入X\n[B,L,C]','input'),('1. LN1 → TimeMix\n跨序列位置汇总信息','projection'),
 ('2. 与输入相加\n'+r'$X^{(1)}=X+\operatorname{TimeMix}(\operatorname{LN}_1(X))$','state'),
 ('3. LN2 → ChannelMix\n每个位置做门控通道变换','projection'),
 ('4. 与上一步结果相加\n'+r'$Y=X^{(1)}+\operatorname{ChannelMix}(\operatorname{LN}_2(X^{(1)}))$','output')],[(0,2,'right'),(2,4,'left')])
 mixing('03_TimeMix','RWKV-4 TimeMix：WKV支路与R门控支路')
 stack('04_WKV因果状态','RWKV-4 WKV：先读取，再写入历史',[
 ('当前位置t：输入k、v\n向量均为[B,C]；历史a、b初始为0','input'),
 ('1. 读历史与当前词元\n'+r'$\mathrm{wkv}_t=\frac{a_{t-1}+e^{u+k_t}\odot v_t}{b_{t-1}+e^{u+k_t}}$','state'),
 ('2. 更新加权值的历史\n'+r'$a_t=e^{-w}\odot a_{t-1}+e^{k_t}\odot v_t$','state'),
 ('3. 更新权重和的历史\n'+r'$b_t=e^{-w}\odot b_{t-1}+e^{k_t}$','state'),
 ('4. 保存a、b供下一位置使用\nw为非负衰减；u单独控制当前词元','output')])
 stack('05_ChannelMix','RWKV-4 ChannelMix：门控前馈网络',[
 ('输入X\n[B,L,C]','input'),('1. Token Shift后分别混合\n'+r'$X_k=\mu_k\odot X+(1-\mu_k)\odot X_{\mathrm{prev}}$'+'\nR支路使用自己的混合参数','projection'),
 ('2. K支路扩展通道并平方ReLU\n'+r'$K=X_kW_k^\top,\quad K^+=\max(K,0)^2$'+'\nC → 4C','projection'),
 ('3. Value投影压回原通道\n'+r'$V=K^+W_v^\top$'+'\n4C → C','state'),
 ('4. R支路生成门控并逐元素相乘\n'+r'$O=\sigma(X_rW_r^\top)\odot V$','output')])
 fourway('08_QShift','Q-Shift：四组通道分别移动，不是串行移位',
 '输入[B,N,C] → 还原[B,C,H,W]',
 ['1. 第1组向右移动\n读左邻居；左边界补0','2. 第2组向左移动\n读右邻居；右边界补0',
 '3. 第3组向下移动\n读上邻居；上边界补0','4. 第4组向上移动\n读下邻居；下边界补0'],
 '5. 四组通道放回各自通道位置\n展平回[B,N,C]；默认各占C/4')
 network('06_RWKV_UNet总图','unet')
 stack('07_GLSP与LP','GLSP：按公开forward展开；LP跳过SpatialMix',[
 ('输入X\n[B,Cin,H,W]；普通块Cin=Cout','input'),
 ('1. 输入Norm → Conv1×1＋BN＋ReLU\n通道扩展为Cmid','projection'),
 ('2. 仅GLSP开启：展平 → SpatialMix\n'+r'$Z\leftarrow Z+\operatorname{DP}(\operatorname{LN}(\operatorname{SM}(Z)))$'+'\n随后还原二维；LP跳过此步','state'),
 ('3. DWConv5×5＋BN＋SiLU\n普通块加回该卷积的输入','projection'),
 ('4. Dropout → Conv1×1线性压缩\n输出Cout通道','projection'),
 ('5. 仅同形状普通块：加回原输入X\n各阶段首块步长2、无这两条残差','output')],[(0,5,'right')])
 mixing('09_视觉SpatialMix','视觉SpatialMix：Bi-WKV支路与R门控支路',visual=True)
 stack('10_CCM总流程','CCM：三尺度一起做通道混合，再分别送回',[
 ('输入F3、F2、F1\nTiny：96×32²、48×64²、32×128²','input'),
 ('1. 三路双线性上采样到128×128\n各自Conv1×1压到32通道','projection'),
 ('2. 按通道拼接并展平\n[B,96,128,128] → [B,16384,96]','projection'),
 ('3. ChannelMix → LN → DropPath\n再与未处理的拼接序列相加','state'),
 ('4. 还原二维，按通道拆成三份\n每份32通道','projection'),
 ('5. 各自Conv1×1恢复96、48、32通道\n再缩回32²、64²、128²送给对应解码层','output')],[(2,3,'right')])
 stack('11_CCM通道混合','CCM内部ChannelMix：没有WKV扫描',[
 ('输入X\n[B,N,3C1]；Tiny的3C1=96','input'),
 ('1. Q-Shift与原X分别混合\n生成Xk与Xr','projection'),
 ('2. K支路：Linear扩展2倍\n平方ReLU → LayerNorm → Value Linear\n96 → 192 → 96','state'),
 ('3. R支路：Linear → Sigmoid\n与K支路压回后的结果逐元素相乘','projection'),
 ('4. 返回CCM外层\n外层还有一次LN、DropPath与残差','output')])
 stack('12_解码UpBlock','RWKV-UNet解码器：公开UpBlock顺序',[
 ('输入：最深特征或与Skip拼接后的特征\n[B,Cin,H,W]','input'),
 ('1. Conv1×1＋BN＋ReLU\nCmid=Cin（当前解码器exp_ratio=1）','projection'),
 ('2. DWConv9×9＋BN＋SiLU\n当前has_skip=False，不加局部残差','projection'),
 ('3. Conv1×1＋BN＋ReLU\n通道压到Cout','projection'),
 ('4. 双线性上采样2倍\n[B,Cout,2H,2W]','output')])
 network('13_URWKV总图','urwkv')
 stack('14_URWKV阶段','URWKV阶段：三个状态容器的生存范围',[
 ('阶段输入二维特征\n[B,C,H,W]','input'),
 ('1. PatchEmbed：Conv3×3，步长1\ncorner补边 → 展平 → 加位置编码\n[B,N,C]','projection'),
 ('2. 阶段入口创建两个空列表\nspatial_shift_list与channel_shift_list','state'),
 ('3. 连续调用Block\n编码阶段3个；解码阶段2个\n两个列表只在本阶段内累计','state'),
 ('4. 最后LayerNorm → 还原二维输出\n[B,C,H,W]','projection'),
 ('5. 阶段输出加入inter_feat\n跨阶段传递给后面的LAN','output')])
 stack('15_URWKV块','URWKV Block：默认pre-norm分支',[
 ('输入X与跨阶段inter_feat\n本阶段另有空间、通道两个历史列表','input'),
 ('1. 若为本阶段第一个Block\n先额外执行一次LAN0','projection'),
 ('2. LAN1 → 空间历史混合 → Q-Shift\nLinear生成Rs、Ks、Vs → Bi-WKV\nSigmoid门控 → Linear','state'),
 ('3. 空间分支输出加回输入\n'+r'$X\leftarrow X+O_s$','projection'),
 ('4. LAN2 → 通道历史混合 → Q-Shift\nK支路扩展4倍、平方ReLU并压回\nR支路Sigmoid门控','state'),
 ('5. 通道分支输出再相加\n'+r'$Y=X+O_c$','output')])
 stack('16_LAN','LAN：跨阶段摘要决定当前归一化尺度',[
 ('输入：当前X与已有阶段特征Mi\nX为[B,N,C]，Mi可有不同空间尺寸','input'),
 ('1. 当前图与各Mi分别做空间GAP\n各得到一个通道向量','projection'),
 ('2. 短向量沿通道补0到Cmax\n按状态堆叠：[B,1,T,Cmax]','projection'),
 ('3. 三种卷积核：(T,1)、(T,3)、(T,5)\n各自Linear到C → 拼接为3C','state'),
 ('4. MLP：3C→C→C → tanh\n'+r'$\widehat\gamma=\gamma+\Delta\gamma$','projection'),
 ('5. 当前X沿通道标准化并做color矩阵变换\n乘动态尺度，再加偏置β\n尺度[B,1,C]沿N个位置广播','output')])
 stack('17_阶段内历史EMA','SQ-Shift之前：公开代码的历史遍历顺序',[
 ('当前LAN输出X\n同一阶段内历史按旧→新保存','input'),
 ('1. 起点为当前X\n'+r'$Z^{(0)}=X$','projection'),
 ('2. 逐个读历史Hj\n'+r'$Z^{(j)}=0.5Z^{(j-1)}+0.5H_j$'+'\n历史是前面各块已混合后的结果','state'),
 ('3. 将最终混合结果加入本分支历史\n再对它做Q-Shift','projection'),
 ('4. 空间、通道历史分别保存\n新阶段重新置空；并非WKV的像素递推状态','output')])
 stack('18_SSF','SSF：三个尺度预测门控，只门控匹配层',[
 ('输入：当前解码特征D\n和编码器三个尺度E1、E2、E3','input'),
 ('1. 三路分别沿通道求均值\n双线性对齐到D的空间尺寸\n拼接[B,3,Hd,Wd]','projection'),
 ('2. Conv3×3：3→3\nInception四分支 → Conv1×1：12→1\nSigmoid得到[B,1,Hd,Wd]','state'),
 ('3. 找到空间尺寸匹配D的原编码特征\n保留全通道，乘同一张空间权重图','projection'),
 ('4. 门控编码特征与D按通道拼接\nConv3×3：2Cd→Cd\n输出交给本级URWKV阶段','output')])
 fourway('19_SSF的Inception','SSF Inception：同一输入并行进入四路',
 '输入[B,3,Hd,Wd]',
 ['1. Conv1×1\n3→3通道','2. Conv3×3\n3→3通道','3. Conv5×5\n3→3通道','4. MaxPool3×3\n再Conv1×1，3→3'],
 '5. 四路按通道拼接：3×4=12通道\n6. Conv1×1压到1通道 → SSF外层Sigmoid')
 print('\n'.join(NAMES))

if __name__=='__main__':main()
