"""Shared typography, content-fitted boxes and measurable figure checks.

No model-specific formulas or architecture are defined here.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

DEFAULT_CONFIG=Path('/home/PuMengYu/个人文件/工具脚本/科研绘图/style.json')

def load_config(path=None):
    path=Path(path or os.environ.get('RESEARCH_FIGURE_STYLE',DEFAULT_CONFIG)).expanduser().resolve()
    cfg=json.loads(path.read_text())
    for key in ('zh_regular','zh_heading'):
        if not Path(cfg['fonts'][key]).is_file():raise ValueError(f'字体文件不存在：{cfg["fonts"][key]}')
    if cfg['fonts']['math_family'] not in ('stix','stixsans','cm','dejavusans','dejavuserif'):
        raise ValueError('不支持的 math_family')
    for key in ('heading','formula','dimension','annotation'):
        value=cfg['sizes'][key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not 8<=value<=48:raise ValueError(f'字号无效：{key}')
    for key in ('padding_x','padding_y','row_gap','corner_radius','stroke_width','node_gap'):
        value=cfg['box'][key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or value<0:raise ValueError(f'间距无效：{key}')
    for key in ('width_px','dpi','export_dpi'):
        if not 50<=cfg['output'][key]<=4000:raise ValueError(f'输出参数无效：{key}')
    for value in cfg['output']['preview_widths']:
        if not 200<=value<=2000:raise ValueError('预览宽度须在200到2000之间')
    for key in ('min_inner_padding_px','max_extra_box_width_px','max_extra_box_height_px','max_node_overlap_px'):
        if cfg['checks'][key]<0:raise ValueError(f'检查参数无效：{key}')
    for value in cfg['colors'].values():
        if not re.fullmatch(r'#[0-9a-fA-F]{6}',value):raise ValueError(f'颜色无效：{value}')
    return path,cfg

class FigureStyle:
    def __init__(self,config=None):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.font_manager import FontProperties
        self.config_path,self.cfg=load_config(config)
        self.fingerprint=hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.regular=FontProperties(fname=self.cfg['fonts']['zh_regular'])
        self.heading=FontProperties(fname=self.cfg['fonts']['zh_heading'])
        # Explicitly set this on FontProperties: rcParams alone does not update
        # an already-constructed FontProperties object.
        self.regular.set_math_fontfamily(self.cfg['fonts']['math_family'])
        self.heading.set_math_fontfamily(self.cfg['fonts']['math_family'])
        matplotlib.rcParams.update({'svg.fonttype':'path','mathtext.fontset':self.cfg['fonts']['math_family']})
        from matplotlib.ft2font import FT2Font
        self.glyphs={kind:set(FT2Font(self.cfg['fonts'][kind]).get_charmap()) for kind in ('zh_regular','zh_heading')}
        self.texts=[];self.boxes=[]

    def canvas(self,height=2200):
        import matplotlib.pyplot as plt
        width=self.cfg['output']['width_px'];dpi=self.cfg['output']['dpi']
        fig,ax=plt.subplots(figsize=(width/dpi,height/dpi),dpi=dpi)
        fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,width),ylim=(height,0));ax.axis('off')
        return fig,ax

    def _rows(self,ax,text,role=None):
        rows=[]
        for i,line in enumerate(text.split('\n')):
            kind=role or ('dimension' if '[' in line else 'formula' if '$' in line else 'heading' if i==0 else 'annotation')
            font=self.heading if kind=='heading' else self.regular
            plain=re.sub(r'\$[^$]*\$','',line)
            glyphs=self.glyphs['zh_heading' if kind=='heading' else 'zh_regular']
            missing={ch for ch in plain if not ch.isspace() and ord(ch) not in glyphs}
            if missing:raise ValueError('字体缺字：'+''.join(sorted(missing))+'；请用数学下标或更换字体')
            size=self.cfg['sizes'][kind]
            color=self.cfg['colors']['dimension' if kind=='dimension' else 'ink']
            probe=ax.text(0,0,line,fontsize=size,fontproperties=font,color=color)
            ax.figure.canvas.draw()
            bb=probe.get_window_extent(ax.figure.canvas.get_renderer());probe.remove()
            inv=ax.transData.inverted()
            xy=inv.transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
            rows.append((line,size,font,color,abs(xy[1,0]-xy[0,0]),abs(xy[1,1]-xy[0,1]),kind))
        return rows

    def measure(self,ax,text,role=None):
        rows=self._rows(ax,text,role);b=self.cfg['box']
        return max(r[4] for r in rows)+2*b['padding_x'],sum(r[5] for r in rows)+b['row_gap']*(len(rows)-1)+2*b['padding_y']

    def text(self,ax,x,y,text,role=None):
        rows=self._rows(ax,text,role);gap=self.cfg['box']['row_gap']
        top=y-(sum(r[5] for r in rows)+gap*(len(rows)-1))/2
        artists=[]
        for line,size,font,color,w,h,kind in rows:
            t=ax.text(x,top+h/2,line,ha='center',va='center',fontsize=size,fontproperties=font,color=color,zorder=4)
            self.texts.append((t,kind));artists.append(t);top+=h+gap
        return artists

    def box(self,ax,x,y,text,color,role=None):
        from matplotlib.patches import FancyBboxPatch
        w,h=self.measure(ax,text,role);b=self.cfg['box']
        patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle=f'round,pad=0,rounding_size={b["corner_radius"]}',facecolor=color,edgecolor=self.cfg['colors']['ink'],linewidth=b['stroke_width'],zorder=3)
        ax.add_patch(patch);artists=self.text(ax,x,y,text,role)
        node={'x':x,'y':y,'w':w,'h':h,'left':x-w/2,'right':x+w/2,'top':y-h/2,'bottom':y+h/2,'text':text,'artists':artists}
        self.boxes.append(node);return node

    def arrow(self,ax,points):
        from matplotlib.patches import FancyArrowPatch
        ink=self.cfg['colors']['ink']
        for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=ink,lw=1.25,zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,lw=1.25,color=ink,shrinkA=0,shrinkB=0,zorder=2))

    def compact_vertical(self, fig):
        """Remove empty horizontal bands without scaling any text or box.

        Opt-in for dense reading diagrams. Branch routes and node order are
        preserved by applying one monotonic y mapping to all artists.
        """
        import bisect
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if len(fig.axes) != 1:
            raise ValueError('纵向紧凑排版要求单坐标系')
        ax = fig.axes[0]
        old_height = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
        bands = [(n['top'], n['bottom']) for n in self.boxes]
        for t, _ in self.texts:
            bb = t.get_window_extent(renderer)
            points = ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
            bands.append((min(points[:,1]), max(points[:,1])))
        for patch in ax.patches:
            if isinstance(patch, Circle):
                bands.append((patch.center[1]-patch.radius, patch.center[1]+patch.radius))
            elif not isinstance(patch, (FancyBboxPatch, FancyArrowPatch)):
                raise ValueError('纵向紧凑排版不支持此色块类型')
        if not bands:
            return
        merged = []
        for lo, hi in sorted(bands):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        gap_limit = self.cfg['box']['node_gap']
        margin = self.cfg['box']['padding_y']
        old_points, new_points = [], []
        cursor = margin
        previous = None
        for lo, hi in merged:
            if previous is not None:
                cursor += min(lo-previous, gap_limit)
            old_points.extend([lo, hi])
            new_points.extend([cursor, cursor+hi-lo])
            cursor += hi-lo
            previous = hi

        def remap(y):
            i = bisect.bisect_right(old_points, float(y))-1
            if i < 0:
                return new_points[0] + float(y)-old_points[0]
            if i >= len(old_points)-1:
                return new_points[-1] + float(y)-old_points[-1]
            span = old_points[i+1]-old_points[i]
            if span == 0:
                return new_points[i]
            return new_points[i] + (float(y)-old_points[i])*(new_points[i+1]-new_points[i])/span

        for patch in ax.patches:
            if isinstance(patch, FancyBboxPatch):
                y, h = patch.get_y(), patch.get_height()
                patch.set_y(remap(y))
                patch.set_height(remap(y+h)-remap(y))
            elif isinstance(patch, Circle):
                patch.center = (patch.center[0], remap(patch.center[1]))
            elif isinstance(patch, FancyArrowPatch):
                a, b = patch._posA_posB
                patch.set_positions((a[0], remap(a[1])), (b[0], remap(b[1])))
        for line in ax.lines:
            line.set_ydata([remap(y) for y in line.get_ydata()])
        for t, _ in self.texts:
            x, y = t.get_position()
            t.set_position((x, remap(y)))
        for node in self.boxes:
            node['top'], node['bottom'], node['y'] = remap(node['top']), remap(node['bottom']), remap(node['y'])
        new_height = cursor + margin
        fig.set_size_inches(fig.get_figwidth(), new_height/self.cfg['output']['dpi'])
        ax.set_ylim(new_height, 0)
        self.compaction = {'old_height': old_height, 'new_height': new_height,
                           'max_empty_band': gap_limit, 'font_scaling': 1.0}

    def audit(self,fig):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[];records=[]
        for t,kind in self.texts:
            bb=t.get_window_extent(renderer)
            if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height:issues.append('文字越出画布：'+t.get_text())
        for node in self.boxes:
            ax=node['artists'][0].axes;corners=[]
            for t in node['artists']:
                bb=t.get_window_extent(renderer)
                xy=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
                corners.extend(xy.tolist())
            xs=[xy[0] for xy in corners];ys=[xy[1] for xy in corners]
            padding=min(min(xs)-node['left'],node['right']-max(xs),min(ys)-node['top'],node['bottom']-max(ys))
            extra_x=node['w']-(max(xs)-min(xs));extra_y=node['h']-(max(ys)-min(ys))
            checks=self.cfg['checks']
            if padding<checks['min_inner_padding_px']:issues.append('文字靠边或越界：'+node['text'].split('\n')[0])
            if extra_x>checks['max_extra_box_width_px'] or extra_y>checks['max_extra_box_height_px']:issues.append('色块留白超过配置：'+node['text'].split('\n')[0])
            records.append({'title':node['text'].split('\n')[0],'width':node['w'],'height':node['h'],'extra_width':extra_x,'extra_height':extra_y,'min_padding':padding})
        for i,a in enumerate(self.boxes):
            for b in self.boxes[i+1:]:
                overlap_x=min(a['right'],b['right'])-max(a['left'],b['left'])
                overlap_y=min(a['bottom'],b['bottom'])-max(a['top'],b['top'])
                if min(overlap_x,overlap_y)>self.cfg['checks']['max_node_overlap_px']:issues.append('色块相互重叠：'+a['text'].split('\n')[0]+' / '+b['text'].split('\n')[0])
        return {'schema':1,'config':str(self.config_path),'config_sha256':self.fingerprint,'math_family':self.cfg['fonts']['math_family'],'sizes':self.cfg['sizes'],'boxes':records,'issues':issues,'passed':not issues,'visual_acceptance':'Requires human/agent inspection at normal preview width; checks alone do not establish aesthetics.'}

    def save(self,fig,stem):
        import matplotlib.pyplot as plt
        stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
        audit=self.audit(fig)
        if hasattr(self, 'compaction'):
            audit['vertical_compaction'] = self.compaction
        if not audit['passed']:raise ValueError('\n'.join(audit['issues']))
        fig.savefig(stem.with_suffix('.svg'),facecolor='white',metadata={'Description':json.dumps(audit,ensure_ascii=False)})
        fig.savefig(stem.with_suffix('.png'),dpi=self.cfg['output']['export_dpi'],facecolor='white')
        for width in self.cfg['output']['preview_widths']:
            fig.savefig(stem.parent/(stem.name+f'_preview{width}.png'),dpi=width/fig.get_figwidth(),facecolor='white')
        # Hash ties the sidecar to the exact SVG, detecting later manual changes.
        audit['svg_sha256']=hashlib.sha256(stem.with_suffix('.svg').read_bytes()).hexdigest()
        stem.with_suffix('.style.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        plt.close(fig)
        return audit
