import os
os.environ.setdefault('BD_ARM','D4-B1') # old module import requires a label; explicit counts below, no old entry called
from source.bd_controls_20261105.candidate import build_candidate as old_build, BasicMambaResidual
from source.recipe_r1_20261105.selection import ROOT,COUNT
def build_candidate(pretrained=True,count=None,med=False):
 assert med is False
 return old_build(pretrained=pretrained,count=COUNT if count is None else count,med=False)
def added_parameters(model):
 return ((n,p) for n,p in model.named_parameters() if n.startswith('decoder.enhance.'))
