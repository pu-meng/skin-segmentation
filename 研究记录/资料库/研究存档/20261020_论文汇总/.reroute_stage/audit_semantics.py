import runpy,sys,json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch,Circle
import xml.etree.ElementTree as ET
root=Path(__file__).parent
Figure.savefig=lambda *a,**k:None
ET.parse=lambda *a,**k:None
counts={}; real_add=Axes.add_patch; real_close=plt.close
stats={'arrows':0,'plus_nodes':0}
def add(self,p):
    if isinstance(p,FancyArrowPatch): stats['arrows']+=1
    if isinstance(p,Circle): stats['plus_nodes']+=1
    return real_add(self,p)
def close(fig=None):
    if fig is not None and hasattr(fig,'axes') and fig.axes:
        texts=[t.get_text() for ax in fig.axes for t in ax.texts]
        title=texts[0]; counts[title]={'texts':sorted(texts),'arrows':stats['arrows'],'plus_nodes':stats['plus_nodes']}
        stats.update(arrows=0,plus_nodes=0)
    return real_close(fig)
Axes.add_patch=add;plt.close=close
all_results={}
for version in ['original','edited']:
    counts={};stats.update(arrows=0,plus_nodes=0)
    for f in ['draw_idea01_full.py','draw_idea02.py','draw_ideas02_06_full.py','draw_other_ideas.py']:
        sys.argv=[f]
        runpy.run_path(str(root/version/'figure_factory'/f),run_name='__main__')
        if f=='draw_idea02.py': plt.close(plt.gcf())
    all_results[version]=counts
assert all_results['original']==all_results['edited'], 'Node text / arrow multiplicity / addition nodes changed'
(root/'semantic_audit.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2))
print('PASS: all generated diagrams have identical visible node text, directed arrow count, and addition/multiplication node count. Endpoint routing reviewed separately.')
