"""Editable SVG source for user-supplied E2 revised to candidate A."""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parent/'E2_A_单向三层_频谱通道辅助.svg'
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1130" viewBox="0 0 1100 1130"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#243958"/></marker></defs><rect width="1100" height="1130" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172d50">']
def text(x,y,t,size=23):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{escape(t)}</text>')
def box(x,y,w,h,lines,color='#eaf2ff'):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#6383ac" stroke-width="1.5"/>')
 start=y+h/2-(len(lines)-1)*15+8
 for i,t in enumerate(lines):text(x+w/2,start+i*30,t)
def arrow(d,dash=False):a.append(f'<path d="{d}" fill="none" stroke="#243958" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
text(550,40,'候选A｜E2：三层单向 Mamba ＋ 频谱统计通道辅助',29)
box(45,65,1010,85,['整网位置：Align3 → 本E2 → 原E1(D3) → D2 → D1', '双源E3修正最浅 S / D → Concat(D′, S′) → 原D0 → Head'],'#f1f5fa')
text(550,185,'X  [B,512,8,8]')
arrow('M550 195 L550 215')
box(220,215,660,65,['Pre：DW3(512) → PW512→256 → LN → GELU'])
arrow('M550 280 L550 315 L295 315 L295 350');arrow('M550 315 L805 315 L805 350')
text(550,305,'U：256通道，沿通道分成128＋128',20)
box(85,350,420,65,['空间输入 Us  [B,128,8,8]'])
box(595,350,420,65,['通道输入 Uc  [B,128,8,8]'],'#fff1df')
for y,num in [(445,1),(540,2),(635,3)]:
 arrow(f'M295 {y-30} L295 {y}');box(85,y,420,65,[f'单向残差 Mamba Block {num}','tokens ＋ Mamba(LN(tokens))'],'#f1edff')
arrow('M805 415 L805 445');box(595,445,420,65,['FP32 FFT2，norm = ortho','逐通道功率 |F|² 的低／高频求和'],'#fff1df')
arrow('M805 510 L805 540');box(595,540,420,65,['统计 log(1+EL), log(1+EH), rH','共享 MLP 3→16→1，再 2×Sigmoid'],'#fff1df')
arrow('M805 605 L805 635');box(595,635,420,65,['a [B,128]，空间广播','Fc = a × Uc；初始 a = 1'],'#fff1df')
arrow('M1015 382 L1040 382 L1040 667 L1015 667')
text(295,732,'Fs  [B,128,8,8]');text(805,732,'Fc  [B,128,8,8]')
arrow('M295 700 L295 713');arrow('M805 700 L805 713')
arrow('M295 740 L295 765 L550 765 L550 785');arrow('M805 740 L805 765 L550 765')
box(110,785,880,90,['z = Concat(GAP(Fs), GAP(Fc)) → MLP256→32→256', '两路 Softmax 得 as / ac，初始各0.5', 'F = PW256→256 (Concat(2as×Fs, 2ac×Fc))'],'#e7f5ed')
arrow('M550 875 L550 900');box(220,900,660,65,['Post：LN → FFN(256→512→256) → GELU','FFN内部含GELU；不另加FFN残差'])
arrow('M550 965 L550 990');box(220,990,660,55,['补齐接口：PW256→512 → 逐通道 γ，γ₀ = 0.01'],'#fff0dc')
arrow('M550 1045 L550 1070');arrow('M550 195 L45 195 L45 1085 L530 1085')
a.append('<circle cx="550" cy="1085" r="20" fill="#eaf2ff" stroke="#6383ac"/>');text(550,1093,'＋',28);arrow('M570 1085 L645 1085');text(815,1094,'Y = X + γ × output → 原E1',22)
text(295,765,'',18)
a.append('</g></svg>');OUT.write_text(''.join(a),encoding='utf-8');print(OUT)
