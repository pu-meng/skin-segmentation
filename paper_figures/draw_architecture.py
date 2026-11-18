#!/usr/bin/env python3
"""Generate an editable diagrams.net overall architecture using only Python stdlib."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from figure_config import FIGURE

HERE = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Node:
    id: str
    x: float
    y: float
    w: float
    h: float
    label: str
    kind: str
    size: int

@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    exit: tuple[float, float]
    entry: tuple[float, float]
    points: tuple[tuple[float, float], ...] = ()

class Diagram:
    def __init__(self, cfg):
        self.cfg=cfg; self.nodes={}; self.edges=[]
    def node(self, id, x, y, w, h, label, kind='operation', size=None):
        if id in self.nodes: raise ValueError(f'duplicate node: {id}')
        if w<=0 or h<=0: raise ValueError(f'bad dimensions: {id}')
        n=Node(id,x,y,w,h,label,kind,size or self.cfg['font']['detail'])
        self.nodes[id]=n;return n
    def edge(self,id,source,target,exit,entry,points=()):
        if any(e.id==id for e in self.edges): raise ValueError(f'duplicate edge: {id}')
        if source not in self.nodes or target not in self.nodes: raise ValueError(f'unknown endpoint: {id}')
        if any(v<0 or v>1 for v in (*exit,*entry)): raise ValueError(f'invalid port: {id}')
        self.edges.append(Edge(id,source,target,exit,entry,tuple(points)))
    def validate(self):
        pg=self.cfg['page'];boxes=[n for n in self.nodes.values() if n.kind not in ('stage','text')]
        for n in self.nodes.values():
            if n.x<0 or n.y<0 or n.x+n.w>pg['width'] or n.y+n.h>pg['height']:
                raise ValueError(f'node outside page: {n.id}')
            if n.label and n.kind!='text':
                lines=n.label.split('\n')
                if len(lines)*n.size*1.22 > n.h-4 or max(map(len,lines))*n.size*.50 > n.w-10:
                    raise ValueError(f'possible text overflow: {n.id}; enlarge the node or shorten text')
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                if min(a.x+a.w,b.x+b.w)-max(a.x,b.x)>0.01 and min(a.y+a.h,b.y+b.h)-max(a.y,b.y)>0.01:
                    raise ValueError(f'overlapping nodes: {a.id}, {b.id}')
        for e in self.edges:
            if e.source not in self.nodes or e.target not in self.nodes: raise ValueError(f'missing endpoint: {e.id}')
    def xml(self):
        self.validate();cfg=self.cfg;col=cfg['colors'];font=cfg['font'];line=cfg['line']
        mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
        d=ET.SubElement(mx,'diagram',id='a1-swift-overall',name='Overall Architecture')
        model=ET.SubElement(d,'mxGraphModel',dx=str(cfg['page']['width']),dy=str(cfg['page']['height']),grid='1',gridSize='10',guides='1',page='1',pageScale='1',pageWidth=str(cfg['page']['width']),pageHeight=str(cfg['page']['height']),background=col['background'],math='0',shadow='0')
        root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
        # Container boundaries first, so all other cells remain visible and editable.
        order=sorted(self.nodes.values(),key=lambda n: 0 if n.kind=='stage' else 1)
        for n in order:
            base=f'html=0;whiteSpace=wrap;align=center;verticalAlign=middle;fontFamily={font["family"]};fontSize={n.size};fontColor={col["text"]};'
            if n.kind=='text': style=base+'text;strokeColor=none;fillColor=none;'
            elif n.kind=='stage':style=base+f'rounded=1;arcSize=12;fillColor={col["stage"]};strokeColor={col["stage_border"]};strokeWidth={line["stage"]};'
            else:
                shape='ellipse;' if n.kind=='concat' else 'rounded=1;arcSize=10;'
                fill=col['background'] if n.kind=='concat' else col[n.kind]
                style=base+shape+f'fillColor={fill};strokeColor={col["edge"]};strokeWidth={line["node"]};'
            cell=ET.SubElement(root,'mxCell',id=n.id,value=n.label,style=style,vertex='1',parent='1')
            ET.SubElement(cell,'mxGeometry',x=str(n.x),y=str(n.y),width=str(n.w),height=str(n.h),**{'as':'geometry'})
        for e in self.edges:
            ex,ey=e.exit;ix,iy=e.entry
            style=(f'edgeStyle=none;rounded=0;html=0;strokeColor={col["edge"]};strokeWidth={line["edge"]};'
                   f'endArrow=block;endFill=1;endSize={line["arrow_size"]};exitX={ex};exitY={ey};exitPerimeter=0;'
                   f'entryX={ix};entryY={iy};entryPerimeter=0;')
            cell=ET.SubElement(root,'mxCell',id=e.id,source=e.source,target=e.target,edge='1',parent='1',style=style)
            g=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
            if e.points:
                arr=ET.SubElement(g,'Array',**{'as':'points'})
                for x,y in e.points:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        ET.indent(mx,space='  ');return ET.tostring(mx,encoding='utf-8',xml_declaration=True)

def build(cfg=FIGURE):
    d=Diagram(cfg);L=cfg['layout'];f=cfg['font'];sw=L['stage_width'];sh=L['stage_height'];bw=L['block_width'];bh=L['block_height'];ex=L['encoder_x'];dx=L['decoder_x'];ax=L['align_x'];aw=L['align_width'];oh=L['operation_height'];gap=L['stage_gap'];y0=L['stage_top'];cs=L['concat_size'];cx=L['concat_x'];sx=L['skip_x'];skw=L['skip_width']; y=[y0+i*gap for i in range(4)]; by=[v+L['block_top'] for v in y]
    def block(id,x,yy,w,h,label,kind='operation',size=None):return d.node(id,x,yy,w,h,label,kind,size)
    def text(id,x,yy,w,h,label,size=None):return block(id,x,yy,w,h,label,'text',size or f['detail'])
    # Foreground blocks and frame labels are separate cells (not rasterized or grouped).
    block('input',ex,L['input_y'],sw,L['io_height'],'Input: 3 × 256 × 256')
    block('stem',ex,L['stem_y'],sw,L['io_height'],'Stem: 3×3 Conv × 2\n48 × 64 × 64')
    block('head',dx,L['stem_y'],sw,L['io_height'],'1×1 Conv → resize → sigmoid')
    block('output',dx,L['input_y'],sw,L['io_height'],'Output: 1 × 256 × 256')
    source_dims=['48 × 64 × 64','64 × 32 × 32','168 × 16 × 16','224 × 8 × 8']
    aligned=['48 → 64','64 → 128','168 → 256','224 → 512']
    decoder_labels=['DoubleConv 128 → 64','DoubleConv 256 → 128','DualStage 512 → 256','Directional Wave']
    decoder_dims=['64 × 64 × 64','128 × 32 × 32','256 × 16 × 16','512 × 8 × 8']
    repeats=['Conv × 2 + Swift × 1','Conv × 2 + Swift × 1','Conv × 8 + Swift × 1','Conv × 5 + Swift × 1']
    for i in range(4):
        for prefix,x in [('encoder',ex),('decoder',dx)]:
            block(f'{prefix}_frame_{i}',x,y[i],sw,sh,'','stage')
            text(f'{prefix}_title_{i}',x+4,y[i]+L['stage_label_top'],L['stage_label_width'],L['stage_label_height'],f'{"Encoder" if prefix=="encoder" else "Decoder"} {i}',f['stage'])
        block(f'enc_{i}',ex+(sw-bw)/2,by[i],bw,bh,f'{repeats[i]}\n{source_dims[i]}','block',f['detail'])
        block(f'dec_{i}',dx+(sw-bw)/2,by[i],bw,bh,f'{decoder_labels[i]}\n{decoder_dims[i]}','block',f['block'])
        block(f'align_{i}',ax,by[i]+7,aw,oh,f'1×1: {aligned[i]}')
        if i<3:
            block(f'down_{i}',ex+(sw-L['down_width'])/2,y[i]+L['down_offset'],L['down_width'],oh,'3×3 Conv, stride 2')
    for i in range(3):
        # Stage 0 is highest resolution; Up[i] connects decoder i+1 to decoder i.
        block(f'up_{i}',dx+(sw-L['up_width'])/2,y[i]+L['up_offset'],L['up_width'],oh,f'Up{i}: Transposed Conv',size=f['detail'])
    block('skip1',sx,by[0]+7,skw,oh,'Skip1 (E3)','bottleneck')
    for i in range(3):block(f'concat_{i}',cx,by[i]+(bh-cs)/2,cs,cs,'C','concat',f['detail'])
    block('bottleneck',L['bottleneck_x'],L['bottleneck_y'],L['bottleneck_width'],L['bottleneck_height'],'Bottleneck: E2A','bottleneck')
    text('title',400,22,560,38,'A1 + SwiftFormer-S: Overall Architecture',f['title'])
    text('labels_0',565,by[0]-24,290,22,'f0: 64 × 64 × 64',f['detail'])
    text('labels_1',576,by[1]-24,280,22,'f1: 128 × 32 × 32',f['detail'])
    text('labels_2',576,by[2]-24,280,22,'f2: 256 × 16 × 16',f['detail'])
    text('labels_3',555,by[3]-24,280,22,'f3: 512 × 8 × 8',f['detail'])
    text('skip_order',770,by[0]-10,118,22,"[d′, s′]",f['edge'])
    text('note',395,950,600,25,'C: concat    |    Conv: ConvEncoder    |    Swift: SwiftFormerEncoder',f['detail'])
    # Fixed port assignment makes every connector attached to its actual node.
    d.edge('input_to_stem','input','stem',(0.5,1),(0.5,0));d.edge('stem_to_enc0','stem','enc_0',(0.5,1),(0.5,0))
    for i in range(3):
        d.edge(f'enc{i}_down{i}',f'enc_{i}',f'down_{i}',(0.5,1),(0.5,0))
        d.edge(f'down{i}_enc{i+1}',f'down_{i}',f'enc_{i+1}',(0.5,1),(0.5,0))
    for i in range(4):d.edge(f'enc{i}_align{i}',f'enc_{i}',f'align_{i}',(1,0.5),(0,0.5))
    for i in (1,2):d.edge(f'align{i}_concat{i}',f'align_{i}',f'concat_{i}',(1,0.5),(0,0.5))
    d.edge('align0_skip1','align_0','skip1',(1,0.5),(0,0.5))
    # Skip1 returns (s', d'); two distinct arrows enter concat in [d', s'] order.
    sn=d.nodes['skip1']; cn=d.nodes['concat_0']; mid1=sx+skw+38
    d.edge('skip1_sprime_concat','skip1','concat_0',(1,.3),(0,.25),((mid1,sn.y+sn.h*.3),(mid1,cn.y+cn.h*.25)))
    d.edge('skip1_dprime_concat','skip1','concat_0',(1,.7),(0,.75),((mid1+18,sn.y+sn.h*.7),(mid1+18,cn.y+cn.h*.75)))
    for i in range(3):
        d.edge(f'concat{i}_dec{i}',f'concat_{i}',f'dec_{i}',(1,.5),(0,.5))
        d.edge(f'dec{i+1}_up{i}',f'dec_{i+1}',f'up_{i}',(.5,0),(.5,1))
    for i in (1,2):
        u=d.nodes[f'up_{i}'];c=d.nodes[f'concat_{i}'];cy=c.y+c.h+L['edge_clearance']
        d.edge(f'up{i}_concat{i}',f'up_{i}',f'concat_{i}',(.5,0),(.5,1),((u.x+u.w/2,cy),(c.x+c.w/2,cy)))
    # Up0 goes to the decoder input of Skip1, not directly to D0.
    u=d.nodes['up_0']; k=d.nodes['skip1']; targetx=k.x+k.w*.5; yroute=k.y+k.h+L['edge_clearance']
    d.edge('up0_skip1','up_0','skip1',(.5,0),(.5,1),((u.x+u.w/2,yroute),(targetx,yroute)))
    a=d.nodes['align_3'];b=d.nodes['bottleneck'];wpoint=b.x-40
    d.edge('align3_bottleneck','align_3','bottleneck',(1,.5),(0,.5),((wpoint,a.y+a.h/2),(wpoint,b.y+b.h/2)))
    z=d.nodes['dec_3'];d.edge('bottleneck_dec3','bottleneck','dec_3',(1,.5),(.5,1),((z.x+z.w/2,b.y+b.h/2),))
    d.edge('dec0_head','dec_0','head',(.5,0),(.5,1));d.edge('head_output','head','output',(.5,0),(.5,1))
    return d

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'其他图片'/'A1_SwiftFormer_Overall.drawio')
    parser.add_argument('--check',action='store_true',help='validate generated XML without writing a file')
    args=parser.parse_args();diagram=build();data=diagram.xml();tree=ET.fromstring(data)
    cells=tree.findall('.//mxCell');ids=[c.get('id') for c in cells];assert len(ids)==len(set(ids))
    assert all(c.get('source') in ids and c.get('target') in ids for c in cells if c.get('edge')=='1')
    if not args.check:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(data)
        ET.parse(args.output)
        print(f'Written: {args.output}')
    print(f'Validated XML: {len(diagram.nodes)} independent nodes; {len(diagram.edges)} attached connectors; no images')
if __name__=='__main__':main()
