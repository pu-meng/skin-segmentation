"""Illustrate three defined frequency separations; no model or patient data.

Run with the medseg Python. FigureStyle owns fonts, boxes and preview exports.
The synthetic example also checks reconstruction for all three definitions.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

OUT = Path(__file__).resolve().parents[2] / '研究记录/figures'


class Diagram:
    def __init__(self, title):
        self.s = FigureStyle()
        self.fig, self.ax = self.s.canvas()
        self.s.text(self.ax, 400, 30, title, role='heading')

    def box(self, x, top, text, color):
        width, height = self.s.measure(self.ax, text)
        if x-width/2 < 12 or x+width/2 > 788:
            raise ValueError('请缩短或拆行：'+text)
        return self.s.box(self.ax, x, top+height/2, text, self.s.cfg['colors'][color])

    def link(self, a, b):
        if a['x'] == b['x']:
            points = [(a['x'], a['bottom']), (b['x'], b['top'])]
        else:
            mid = (a['bottom']+b['top'])/2
            points = [(a['x'], a['bottom']), (a['x'], mid), (b['x'], mid), (b['x'], b['top'])]
        self.s.arrow(self.ax, points)

    def save(self, name, bottom):
        height = bottom+32
        cfg = self.s.cfg['output']
        self.fig.set_size_inches(cfg['width_px']/cfg['dpi'], height/cfg['dpi'])
        self.ax.set_ylim(height, 0)
        audit = self.s.save(self.fig, OUT/name)
        if not audit['passed']:
            raise ValueError(audit['issues'])
        print(name, flush=True)


def fft_diagram():
    d = Diagram('图1 · FFT：按频率位置分成两份')
    inp = d.box(400, 78, '1. 输入实数特征X\n[B,C,H,W]；每通道独立处理', 'input')
    freq = d.box(400, inp['bottom']+42, '2. 二维FFT得到复数频谱F\n保留幅度和相位；[B,C,H,W]', 'projection')
    d.link(inp, freq)
    top = freq['bottom']+55
    low = d.box(205, top, '3a. 保留低频范围\n'+r'$F_L=M_L\odot F$'+'\n零频附近权重为1', 'state')
    high = d.box(595, top, '3b. 保留其余频率\n'+r'$F_H=(1-M_L)\odot F$'+'\n与低频掩码互补', 'state')
    d.link(freq, low); d.link(freq, high)
    top = max(low['bottom'], high['bottom'])+42
    low_out = d.box(205, top, '4a. 逆FFT → 低频L\n[B,C,H,W]\n缓慢变化与整体形状', 'output')
    high_out = d.box(595, top, '4b. 逆FFT → 高频H\n[B,C,H,W]\n快速变化、纹理与噪声', 'output')
    d.link(low, low_out); d.link(high, high_out)
    bottom = max(low_out['bottom'], high_out['bottom'])
    d.s.text(d.ax, 400, bottom+42, '两份相加恢复X；掩码需保持正负频率的共轭对称。', role='annotation')
    d.save('55_高低频分离_FFT掩码', bottom+65)


def gaussian_diagram():
    d = Diagram('图2 · 高斯平滑：低频取平滑，高频取差值')
    inp = d.box(400, 78, '1. 输入特征X\n[B,C,H,W]', 'input')
    blur = d.box(400, inp['bottom']+42, '2. 每通道做高斯平滑\n高斯核只混合空间邻域\n步长1，保持H与W', 'projection')
    d.link(inp, blur)
    low = d.box(205, blur['bottom']+55, '3. 得到低频L\n'+r'$L=G_\sigma*X$'+'\n[B,C,H,W]', 'output')
    d.link(blur, low)
    high = d.box(595, low['bottom']+80, '4. 原输入减去低频\n'+r'$H=X-L$'+'\n[B,C,H,W]', 'state')
    # The low feature and untouched original input are separate operands.
    y = (low['bottom']+high['top'])/2
    d.s.arrow(d.ax, [(low['x'], low['bottom']), (low['x'], y),
                    (high['x'], y), (high['x'], high['top'])])
    d.s.arrow(d.ax, [(inp['right'], inp['y']), (765, inp['y']),
                    (765, high['y']), (high['right'], high['y'])])
    d.s.text(d.ax, 735, high['y']-26, '原X', role='annotation')
    out = d.box(595, high['bottom']+42, '5. 得到高频残差H\n边缘、纹理、噪声均可能保留\n可正可负；与L相加恢复X', 'output')
    d.link(high, out)
    d.s.text(d.ax, 245, out['y'], 'σ越大，平滑越强\n分离是渐变的\n没有硬频率分界', role='annotation')
    d.save('56_高低频分离_高斯平滑相减', out['bottom'])


def wavelet_diagram():
    d = Diagram('图3 · 一层Haar小波：低频＋三个细节子带')
    inp = d.box(400, 78, '1. 输入特征X\n[B,C,H,W]；H、W为偶数', 'input')
    top = inp['bottom']+55
    low = d.box(205, top, '2a. 沿高度低通，再↓2\n相邻两行相加／√2\n[B,C,H/2,W]', 'projection')
    high = d.box(595, top, '2b. 沿高度高通，再↓2\n相邻两行相减／√2\n[B,C,H/2,W]', 'projection')
    d.link(inp, low); d.link(inp, high)
    top = max(low['bottom'], high['bottom'])+55
    ll = d.box(205, top, '3a. 再沿宽度低通／高通↓2\nLL：高度低通、宽度低通\nLH：高度低通、宽度高通\n各为 [B,C,H/2,W/2]', 'state')
    hh = d.box(595, top, '3b. 再沿宽度低通／高通↓2\nHL：高度高通、宽度低通\nHH：高度高通、宽度高通\n各为 [B,C,H/2,W/2]', 'state')
    d.link(low, ll); d.link(high, hh)
    out = d.box(400, max(ll['bottom'], hh['bottom'])+65,
                '4. 保留四个子带，供分别处理或逆变换\nLL表示粗尺度；LH、HL、HH表示不同方向细节\n逆小波将四份系数重建为原尺寸X', 'output')
    # Use separate entry points to keep both input arrowheads visible.
    for a in (ll, hh):
        d.s.arrow(d.ax, [(a['x'], a['bottom']), (a['x'], out['top'])])
    d.save('57_高低频分离_Haar小波', out['bottom'])


def examples():
    n = 128
    yy, xx = np.mgrid[:n, :n]/n
    x = .12+.20*xx+.42*np.exp(-((xx-.5)**2+(yy-.5)**2)/.055)
    x += .12*((xx>.25)&(xx<.75)&(yy>.3)&(yy<.7))
    x += .055*np.sin(2*np.pi*18*xx)*np.sin(2*np.pi*14*yy)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    mask = (fx**2+fy**2 <= .10**2)
    f = np.fft.fft2(x)
    low_fft = np.fft.ifft2(f*mask).real
    high_fft = np.fft.ifft2(f*(1-mask)).real
    low_gauss = gaussian_filter(x, sigma=2., mode='wrap')
    # One orthonormal Haar level, with a fixed row-then-column convention.
    a, b, c, e = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    ll, lh, hl, hh = (a+b+c+e)/2, (a-b+c-e)/2, (a+b-c-e)/2, (a-b-c+e)/2
    def inverse(ll, lh, hl, hh):
        out = np.empty_like(x)
        out[0::2, 0::2] = (ll+lh+hl+hh)/2
        out[0::2, 1::2] = (ll-lh+hl-hh)/2
        out[1::2, 0::2] = (ll+lh-hl-hh)/2
        out[1::2, 1::2] = (ll-lh-hl+hh)/2
        return out
    zero = np.zeros_like(ll)
    low_haar = inverse(ll, zero, zero, zero)
    high_haar = inverse(zero, lh, hl, hh)
    assert np.max(np.abs(inverse(ll, lh, hl, hh)-x)) < 1e-12
    rows = [('FFT：圆形截断，半径0.10周期／像素', low_fft, high_fft),
            ('高斯：σ＝2像素，循环边界', low_gauss, x-low_gauss),
            ('Haar：低频由LL重建，高频由三个细节重建', low_haar, high_haar)]
    d = Diagram('图4 · 同一份合成输入，三种分离结果')
    for center, label in [(145, '原输入X'), (400, '低频L'), (655, '高频H')]:
        d.s.text(d.ax, center, 85, label, role='heading')
    height = 1080
    dpi = d.s.cfg['output']['dpi']
    d.fig.set_size_inches(800/dpi, height/dpi); d.ax.set_ylim(height, 0)
    for row, (name, low, high) in enumerate(rows):
        err = float(np.max(np.abs(low+high-x)))
        assert err < 1e-12
        assert np.max(np.abs(high)) < .15
        top = 160+row*290
        d.s.text(d.ax, 400, top-28, name, role='annotation')
        for col, data in enumerate((x, low, high)):
            left = 40+col*255
            image_ax = d.fig.add_axes([left/800, 1-(top+210)/height, 210/800, 210/height])
            image_ax.imshow(data, cmap='RdBu_r' if col==2 else 'gray',
                            vmin=-.15 if col==2 else 0., vmax=.15 if col==2 else 1., interpolation='nearest')
            image_ax.axis('off')
        print(name, 'max reconstruction error', err, flush=True)
    d.s.text(d.ax, 400, 1030, '原图与低频：统一灰度0～1；高频：统一色标−0.15～0.15\n红＝正值，蓝＝负值，白＝接近0；未逐图拉伸对比度', role='annotation')
    d.save('58_高低频分离_同输入三种方法示例', height-32)


if __name__ == '__main__':
    fft_diagram()
    gaussian_diagram()
    wavelet_diagram()
    examples()
