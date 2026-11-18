from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET
import subprocess
import hashlib
import json

OUT=Path(__file__).parent/'原始Mamba_清晰模块图'
OUT.mkdir(exist_ok=True)
FONT='Microsoft YaHei,SimHei,Arial,sans-serif'

def canvas(h,title,subtitle):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{h}" viewBox="0 0 1040 {h}"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>',f'<rect width="1040" height="{h}" fill="#ffffff"/>',f'<g font-family="{FONT}" fill="#17283b">',f'<text x="520" y="44" text-anchor="middle" font-size="30" font-weight="700">{escape(title)}</text>',f'<text x="520" y="80" text-anchor="middle" font-size="20" fill="#52677e">{escape(subtitle)}</text>']
def text(a,x,y,s,size=22,color='#17283b',anchor='middle'):
 a.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}">{escape(s)}</text>')
def box(a,x,y,w,h,title,sub='',color='#2878b5',fill='#edf6fd',size=23):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="2.3"/>')
 text(a,x+w/2,y+(h/2+8 if not sub else h/2-4),title,size)
 if sub:text(a,x+w/2,y+h/2+24,sub,19,'#52677e')
def line(a,d,color='#2878b5',dash=False):
 a.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arr)"'+(' stroke-dasharray="9 6"' if dash else '')+'/>')
def dot(a,x,y,color='#2878b5'):
 a.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
def op(a,x,y,s):
 a.append(f'<circle cx="{x}" cy="{y}" r="23" fill="#fff" stroke="#29435e" stroke-width="2.5"/>');text(a,x,y+9,s,31)
def finish(a,name):
 s=''.join(a)+'</g></svg>';ET.fromstring(s);p=OUT/(name+'.svg');p.write_text(s,encoding='utf-8');return s

# Figure 1: module wiring, with all bypasses attached to exact operands.
a=canvas(1480,'原始 Mamba：先看清三条路线','X＝模块输入　F＝内容分支　T＝卷积后内容　Z＝门控分支')
# Edges before opaque nodes.
line(a,'M420 170V215');line(a,'M420 285V325');line(a,'M420 395V435');line(a,'M420 500V550')
line(a,'M420 500H810V550','#9463b8');dot(a,420,500)
line(a,'M420 620V730');dot(a,420,675)
line(a,'M420 675H680V770','#d68a18')
line(a,'M680 840V970H443','#d68a18')
line(a,'M420 900V947');line(a,'M420 993V1047')
line(a,'M810 620V995','#9463b8');line(a,'M810 1065V1070H443','#9463b8')
line(a,'M420 1093V1150');line(a,'M420 1220V1272');line(a,'M420 1318V1360')
line(a,'M280 135H75V1295H397','#6d7f91',True)
text(a,96,730,'X保留到最后',21,'#6d7f91','start')
box(a,280,105,280,65,'输入 X','[b,L,512]')
box(a,265,215,310,70,'① LayerNorm','得到 V：[b,L,512]')
box(a,265,325,310,70,'② 线性映射 512 → 2048','得到 P：[b,L,2048]')
box(a,265,435,310,65,'③ 拆分 P → F、Z','各1024维；这里不相乘')
box(a,265,550,310,70,'④ F → 深度卷积 → SiLU','输出 T：[b,L,1024]')
box(a,710,550,200,70,'门控分支 Z','[b,L,1024]','#9463b8','#f6effb')
text(a,438,665,'T',25,anchor='start')
box(a,245,730,350,170,'⑤—⑫ 状态计算','生成 Δ、B、C → 更新 H',size=25)
text(a,420,860,'用 C 读出状态 → Y₀',23)
box(a,620,770,120,70,'× D','逐通道缩放','#d68a18','#fff7e7')
text(a,680,715,'T 的直通路线',21,'#a5650a')
text(a,438,934,'Y₀',23,anchor='start');op(a,420,970,'＋');text(a,446,1025,'⑬ 得到 Y₁',21,anchor='start')
box(a,710,995,200,70,'SiLU(Z)','','#9463b8','#f6effb')
op(a,420,1070,'×');text(a,446,1123,'⑭ 得到 Y₂',21,anchor='start')
box(a,265,1150,310,70,'⑮ 输出映射 1024 → 512','得到 O：[b,L,512]')
op(a,420,1295,'＋');text(a,460,1303,'⑯ X ＋ O',23,anchor='start')
box(a,265,1360,310,65,'最终输出 Y','[b,L,512] → 恢复二维特征')
text(a,520,1460,'蓝：内容与状态路线　橙：T直通　紫：Z门控　灰虚线：模块外残差',20)
s1=finish(a,'01_原始Mamba_模块接线')

# Figure 2: parameter generation and one-token state update, with repeated inputs explicit.
a=canvas(1340,'展开⑤—⑫：T 怎样变成状态输出 Y₀','只看原始版本；下半图使用上半图生成的 Δ、B、C')
line(a,'M520 175V225');line(a,'M520 295V340')
line(a,'M520 405H225V450');line(a,'M520 405V450');line(a,'M520 405H815V450')
box(a,370,110,300,65,'④得到的 T','[b,L,1024]')
box(a,350,225,340,70,'⑤ 线性映射 1024 → 64','得到参数包 Q')
box(a,350,340,340,65,'⑥ 拆分 Q → ρ、B、C','32维、16维、16维')
box(a,120,450,210,65,'ρ：32维')
box(a,425,450,190,65,'B：16维')
box(a,720,450,190,65,'C：16维')
line(a,'M225 515V555');box(a,60,555,330,85,'⑦ 32 → 1024，再softplus','得到 Δ：1024维')
text(a,655,580,'B、C不经过这次维度映射',22)
a.append('<path d="M55 685H985" stroke="#ccd7e3" stroke-width="2"/>')
text(a,520,728,'下面展开：当前一个token如何更新状态、产生输出',25)
# State update: left old state -> multiply; R downward; J downward into plus.
line(a,'M300 920H407');line(a,'M430 865V897');line(a,'M453 920H697');line(a,'M720 865V897')
line(a,'M720 943V985');line(a,'M720 1055V1115');line(a,'M720 1190V1235')
line(a,'M860 1020H985','#6d7f91')
line(a,'M960 1153H895','#9463b8')
box(a,75,885,225,70,'旧状态 H旧','每通道16个状态')
box(a,280,780,300,85,'⑧ A = −exp(A_log)','⑨ R = exp(Δ A)')
box(a,610,780,250,85,'⑩ J = Δ × B × T','当前内容的写入项')
op(a,430,920,'×');op(a,720,920,'＋')
text(a,530,968,'⑪',22)
box(a,580,985,280,70,'新状态 H新','H新 = R × H旧 ＋ J')
text(a,985,1078,'传给下一个token',20,'#6d7f91','end')
box(a,545,1115,350,75,'⑫ 与 C 相乘，再沿状态求和','每通道：Σ Cₙ × H新,ₙ')
text(a,965,1140,'C',24,'#9463b8')
box(a,580,1235,280,65,'状态输出 Y₀','每个token1024维')
text(a,520,1325,'Y₀回到主图⑬；T也同时走主图的D直通路线。',21)
s2=finish(a,'02_参数生成与状态更新')
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原始Mamba模块图</title><style>body{margin:0;background:#eef2f7;font-family:Microsoft YaHei,sans-serif}main{max-width:1100px;margin:auto}svg{display:block;width:100%;height:auto;background:white;margin:16px 0}p{padding:12px 24px;font-size:20px}</style><main>'+s1+s2+'<p>这是当前512维瓶颈接口下的原始Mamba-1结构说明，不是版本1或版本2，也不代表已经新增了原版对照实验。</p></main></html>'
(OUT/'原始Mamba_模块图.html').write_text(html,encoding='utf-8')
# Installed browser, isolated scratch profile; no signed-in browser modification.
candidates=[Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')]
browser=next(p for p in candidates if p.exists())
for name,h in [('01_原始Mamba_模块接线',1480),('02_参数生成与状态更新',1340)]:
 svg=OUT/(name+'.svg');png=OUT/(name+'.png')
 args=[str(browser),'--headless','--disable-gpu','--hide-scrollbars','--no-first-run','--force-device-scale-factor=1','--user-data-dir='+str(OUT/'render_profile'),'--screenshot='+str(png),'--window-size=1040,'+str(h),svg.as_uri()]
 subprocess.run(args,capture_output=True,timeout=40,check=True)
 assert png.exists() and png.stat().st_size>10000
print(str(OUT))
