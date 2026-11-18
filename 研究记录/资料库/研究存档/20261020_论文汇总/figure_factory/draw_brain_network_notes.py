"""个人理解稿的四幅教学示意图；不是脑解剖图或实测连接图。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
PREVIEW=Path('/tmp/brain_network_notes_preview')
OUT.mkdir(exist_ok=True); PREVIEW.mkdir(exist_ok=True)
font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(font_path))
font_name=font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({'font.family':font_name,'svg.fonttype':'none'})
C={'blue':'#e7f1fc','yellow':'#f6e8a6','purple':'#e6d5f7','green':'#cdefd8','gray':'#eef0f3'}
INK='#293e55'

def canvas(title, subtitle, height=650):
    fig,ax=plt.subplots(figsize=(12,height/85))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax.set(xlim=(0,1000),ylim=(height,0)); ax.axis('off')
    label(ax,35,37,title,20,True,ha='left')
    label(ax,35,72,subtitle,13,ha='left')
    return fig,ax

def label(ax,x,y,s,size=15,bold=False,ha='center',color=INK):
    return ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va='center',color=color)

def box(ax,x,y,w,h,title,sub='',color='blue',size=15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',facecolor=C[color],edgecolor=INK,linewidth=1.3))
    label(ax,x+w/2,y+h*(.36 if sub else .5),title,size,True)
    if sub: label(ax,x+w/2,y+h*.73,sub,size-2)

def arrow(ax,pts,color=INK,dashed=False):
    if len(pts)>2:
        ax.plot([x for x,y in pts[:-1]],[y for x,y in pts[:-1]],color=color,lw=1.4,linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=13,linewidth=1.4,color=color,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0))

def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(PREVIEW/(name+'.png'),dpi=95,facecolor='white')
    plt.close(fig)
    print(OUT/(name+'.svg'))

fig,ax=canvas('图1｜脑由哪些部分组成，脑区之间怎样联系','上半图看结构组成；下半图看连接关系。两种划分角度不同。',690)
box(ax,365,110,270,64,'整个脑','包含多个大体解剖结构')
for x,t,sub in [(35,'大脑半球','皮层与深部组织'),(275,'间脑','丘脑、下丘脑等'),(515,'小脑','自身也有皮层与核团'),(755,'脑干','中脑、脑桥、延髓')]:
    box(ax,x,225,210,74,t,sub,'yellow')
    arrow(ax,[(500,174),(500,198),(x+105,198),(x+105,225)])
label(ax,35,347,'同一个脑区内部有局部回路，也能参与跨脑区的网络',16,True,ha='left')
# 大框是区域范围，不是运算节点。
box(ax,45,389,500,193,'','','gray')
label(ax,295,415,'脑区A：内部的局部回路',16,True)
box(ax,75,459,190,72,'细胞群1','内部有突触连接')
box(ax,325,459,190,72,'细胞群2','内部有突触连接')
arrow(ax,[(265,481),(325,481)])
arrow(ax,[(325,511),(265,511)])
box(ax,695,459,250,72,'脑区B','也有自己的局部回路','purple')
arrow(ax,[(545,481),(695,481)])
arrow(ax,[(695,511),(545,511)])
label(ax,745,561,'跨脑区联系',14)
label(ax,500,636,'下半图只是关系示意；功能活动相关不等于已经证实直接突触连接。',13)
save(fig,'大脑网络_图1_整体结构与网络尺度')

fig,ax=canvas('图2(a)｜前一个神经元怎样影响后一个','左：常见化学突触的传递过程。右：存在反馈连接，不等于信号永久循环。',720)
label(ax,270,116,'(a) 信号怎样传递',16,True)
for y,t,sub,col in [(153,'A产生动作电位','电脉冲沿A的轴突传播','blue'),(281,'突触传递','释放递质，作用于B的受体','purple'),(409,'B整合多个输入','结合自身状态，改变放电可能性','yellow'),(537,'B向后续细胞发送信号','满足条件时产生自己的脉冲','green')]:
    box(ax,65,y,410,85,t,sub,col)
for y in [238,366,494]: arrow(ax,[(270,y),(270,y+43)])
label(ax,745,116,'(b) 三个细胞的简化回路',16,True)
for y,t,sub in [(153,'神经元A','接收输入并影响B'),(345,'神经元B','整合输入并影响C'),(537,'神经元C','也可反馈影响A')]:
    box(ax,630,y,230,85,t,sub)
arrow(ax,[(745,238),(745,345)])
arrow(ax,[(745,430),(745,537)])
arrow(ax,[(860,580),(945,580),(945,196),(860,196)],color='#7e57a6')
label(ax,906,390,'反馈',13,color='#7e57a6')
label(ax,500,674,'每次传递都受细胞状态影响；闭环连接不保证活动一直维持或不断增强。',13)
save(fig,'大脑网络_图2_神经元传递与反馈')

fig,ax=canvas('图3｜layer、block、stage分别有多大','这张图解释人工网络的组织方式；不把它当作脑的固定解剖层级。',700)
# Stage容器内明确画出Block，而不以箭头指向空容器边框。
box(ax,45,128,430,204,'','','gray')
label(ax,260,156,'Stage：一个处理阶段',17,True)
label(ax,260,187,'通常在某个尺度上组织多个block',13)
box(ax,70,225,170,70,'Block 1','一组组合运算','yellow')
box(ax,280,225,170,70,'Block 2','一组组合运算','yellow')
arrow(ax,[(240,260),(280,260)])
box(ax,150,394,220,70,'尺度或通道变换','例如下采样')
arrow(ax,[(365,295),(365,359),(260,359),(260,394)])
box(ax,45,526,430,105,'下一Stage','在下一尺度继续处理','gray')
arrow(ax,[(260,464),(260,526)])
label(ax,735,151,'展开一个示例Block',17,True)
for y,t in [(210,'卷积 layer'),(315,'归一化 layer'),(420,'激活 layer'),(525,'卷积 layer')]:
    box(ax,600,y,275,64,t,color='blue')
for y in [274,379,484]: arrow(ax,[(737,y),(737,y+41)])
label(ax,737,624,'实际block的操作与次序由模型决定',13)
label(ax,500,674,'多个stage及编码、解码、skip和输出头，共同组成完整的U-Net式网络。',13)
save(fig,'大脑网络_图3_layer_block_stage')

fig,ax=canvas('图4｜skip把特征送到后面，反馈让前面再次处理','左：一次无环的前向计算。右：增加回传和更新后的循环示意。',720)
for cx,title in [(265,'(a) 前向计算＋skip'),(735,'(b) 带反馈的处理')]:
    label(ax,cx,115,title,17,True)
    for y,t,sub in [(165,'前部处理','得到早期信息'),(340,'后续处理','继续变换信息'),(515,'融合或输出','使用处理结果')]:
        box(ax,cx-135,y,270,85,t,sub,'yellow' if y!=515 else 'green')
    arrow(ax,[(cx,250),(cx,340)])
    arrow(ax,[(cx,425),(cx,515)])
arrow(ax,[(130,208),(55,208),(55,558),(130,558)],color='#2f6fa3')
label(ax,74,380,'skip',13,color='#2f6fa3')
arrow(ax,[(870,558),(950,558),(950,208),(870,208)],color='#7e57a6')
label(ax,913,368,'回传',13,color='#7e57a6')
label(ax,913,399,'再处理',13,color='#7e57a6')
label(ax,265,643,'前部不会因这条skip重新计算',14,True)
label(ax,735,643,'需定义回传内容、更新和循环次数',14,True)
label(ax,500,687,'当前方法6属于左侧：U2参与修正E2，但不让Encoder Stage 2重新运行。',13)
save(fig,'大脑网络_图4_skip与反馈区别')
