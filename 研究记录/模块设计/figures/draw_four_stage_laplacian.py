"""Reuse section-5.2 content and shared style, with the real four-stage D0 interface."""
import draw_laplacian as lap
from draw_module_figures import Diagram
from draw_four_stage_decoder import main as overview

class FourStageDiagram(Diagram):
    def __init__(self,title,height):
        super().__init__(title.replace('5. 拉普拉斯引导','四层版 · 拉普拉斯引导'),height)
    def node(self,key,x,y,text,color='projection',role='heading'):
        return super().node(key,x,y,text.replace('原 D2','原 D0'),color,role)
    def save(self,name):
        names={'18_拉普拉斯引导_模块展开':'22_四层对应Decoder_拉普拉斯展开',
               '19_拉普拉斯引导_融合门控':'23_四层对应Decoder_融合门控'}
        return super().save(names[name])

if __name__=='__main__':
    overview(modified=True)
    lap.Diagram=FourStageDiagram
    lap.draw_detail();lap.draw_fusion()
