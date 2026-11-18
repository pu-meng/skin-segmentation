from pathlib import Path
from html import escape
import subprocess
import xml.etree.ElementTree as ET
ROOT=Path(__file__).parent
FONT='Microsoft YaHei,SimHei,Arial,sans-serif'

def begin(h,title,sub):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{h}" viewBox="0 0 1040 {h}"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs><rect width="1040" height="{h}" fill="white"/><g font-family="{FONT}" fill="#18324b">',f'<text x="520" y="45" text-anchor="middle" font-size="29" font-weight="700">{escape(title)}</text>',f'<text x="520" y="85" text-anchor="middle" font-size="21" fill="#52677e">{escape(sub)}</text>']
def tx(a,x,y,s,size=22,color='#18324b'):
 a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" fill="{color}">{escape(s)}</text>')
def box(a,x,y,w,h,title,sub='',color='#297eb3',fill='#eff7fd'):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" stroke="{color}" fill="{fill}" stroke-width="2.3"/>');tx(a,x+w/2,y+h/2+(7 if not sub else -5),title)
 if sub:tx(a,x+w/2,y+h/2+24,sub,19,'#52677e')
def arrow(a,d,color='#297eb3'):
 a.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#a)"/>')
def end(a,name):
 svg=''.join(a)+'</g></svg>';ET.fromstring(svg);(ROOT/(name+'.svg')).write_text(svg,encoding='utf-8');return svg

a=begin(1170,'图像也能逐token：区别是输入是否已经给定','相同状态机制，两种任务组织；“图像不能逐token”是不准确的')
tx(a,520,135,'A. 自回归文字生成：下一次输入来自上一次生成',25)
arrow(a,'M300 235H355');arrow(a,'M625 235H680')
arrow(a,'M825 285V335H490V285','#9863b6')
box(a,60,185,240,100,'已知提示词','例如：“今天天气”')
box(a,355,185,270,100,'模型处理＋保留状态','状态接着前面更新')
box(a,680,185,290,100,'生成下一个token','示意结果：“很”','#9863b6','#f7f0fc')
tx(a,520,380,'把刚生成的“很”作为新输入，再生成后续token',22,'#80539a')
tx(a,520,417,'未来token原本未知，生成与下一次输入之间有依赖。',21)
a.append('<path d="M45 455H995" stroke="#ccd8e4" stroke-width="2"/>')
tx(a,520,505,'B. 图像分割：图像特征token一开始就已知',25)
arrow(a,'M360 610H460')
box(a,65,555,295,110,'图像已提取出的特征','T₁、T₂、T₃、T₄均已知')
box(a,460,555,500,110,'方式1：整串T一起传入模块','内部扫描更新状态 → 输出整串结果')
tx(a,520,725,'方式2：也能保留状态，按顺序逐个送入图像token',23)
# Left-to-right state chain; token labels above each state update.
for x,t,h in [(65,'T₁','更新 H₁'),(310,'T₂','更新 H₂'),(555,'T₃','更新 H₃'),(800,'T₄','更新 H₄')]:
 tx(a,x+85,780,t,25)
 arrow(a,f'M{x+85} 795V825')
 box(a,x,825,170,85,h,'读取本步输出')
for x1,x2 in [(235,310),(480,555),(725,800)]:arrow(a,f'M{x1} 867H{x2}')
tx(a,520,960,'每一步传过去的是状态；下一图像token不靠上一步预测来产生。',21)
box(a,95,1000,850,80,'同一输入顺序、同一初始状态：两种方式应得到相同输出','已用CPU小程序验证；不是训练模型成绩或GPU速度测试','#3b946c','#eff9f3')
tx(a,520,1135,'图像可以逐token，但不必为了读取下一块图像而等待上一块预测。',22)
s1=end(a,'01_文字生成与图像分割')

a=begin(930,'同一个递推公式，可以有不同执行方式','Python不是不能快；关键是交给GPU的计算怎样组织')
tx(a,520,140,'当前版本2：Python逐token循环',25)
arrow(a,'M305 240H390');arrow(a,'M650 240H735')
box(a,55,190,250,100,'Python循环','处理第t个token')
box(a,390,190,260,100,'多个PyTorch小运算','乘、加、状态读出等')
box(a,735,190,250,100,'GPU执行运算','随后处理下一步')
arrow(a,'M860 290V340H180V290','#9863b6')
tx(a,520,390,'循环调度、小算子启动、中间读写与反向图，都会影响速度。',21)
a.append('<path d="M45 435H995" stroke="#ccd8e4" stroke-width="2"/>')
tx(a,520,490,'原始Mamba：Python接口调用编译后的CUDA扩展',25)
arrow(a,'M305 600H390');arrow(a,'M650 600H735')
box(a,55,550,250,100,'Python调用扫描接口','selective_scan_fn')
box(a,390,550,260,100,'C++ / CUDA扩展','专用扫描与融合计算')
box(a,735,550,250,100,'GPU高效调度','优化读写与扫描组织')
tx(a,520,705,'已核实当前环境：selective_scan_cuda 的 .so 扩展',21)
tx(a,520,745,'Python包装器调用 fwd；训练反向调用 bwd。',21)
box(a,95,795,850,85,'可以写自定义CUDA或Triton实现，但需要验证前向、梯度和真实速度','这张图不表示我们已经实现或验证了版本2的融合加速核','#3b946c','#eff9f3')
s2=end(a,'02_Python循环与CUDA扫描')
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Mamba扫描与图像逐token</title><style>body{margin:0;background:#edf2f7;font-family:Microsoft YaHei,sans-serif}main{max-width:1100px;margin:auto}svg{display:block;width:100%;height:auto;margin:18px 0;background:white}</style><main>'+s1+s2+'</main></html>'
(ROOT/'图像也能逐token_图解.html').write_text(html,encoding='utf-8')
browser=next(p for p in [Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')] if p.exists())
for n,h in [('01_文字生成与图像分割',1170),('02_Python循环与CUDA扫描',930)]:
 subprocess.run([str(browser),'--headless','--disable-gpu','--hide-scrollbars','--no-first-run','--force-device-scale-factor=1','--user-data-dir='+str(ROOT/'render_profile'),'--window-size=1040,'+str(h),'--screenshot='+str(ROOT/(n+'.png')),(ROOT/(n+'.svg')).as_uri()],capture_output=True,timeout=40,check=True)
 assert (ROOT/(n+'.png')).stat().st_size>10000
print(ROOT)
