"""Single R1 recipe; one update schedule and one global clipping operation."""
import math
import torch
S=300*873
W=5*873
SPEC=dict(epochs=300,batch_size=8,input_size=256,precision='FP32',optimizer='AdamW',encoder_target_lr=1e-4,other_target_lr=1e-3,weight_decay=.01,betas=[.9,.999],eps=1e-8,warmup_updates=W,total_updates=S,minimum_factor=.01,max_norm=12,clip_norm_type=2,error_if_nonfinite=True,schedule='warm .01+.99*t/(W-1); tail .01+.99*.5*(1+cos(pi*(t-W)/(S-W-1)))',step_order='backward -> one global clip -> schedule.apply(global update) -> optimizer.step -> schedule.advance',extra_schedulers=False,AMP=False,EMA=False,freezing=False)
def factor(step):
 assert 0<=step<S
 return .01+.99*step/(W-1) if step<W else .01+.99*.5*(1+math.cos(math.pi*(step-W)/(S-W-1)))
def make_optimizer(model):
 named=list(model.named_parameters());assert all(p.requires_grad for _,p in named)
 enc=[(n,p) for n,p in named if n.startswith('encoder.')]
 other=[(n,p) for n,p in named if not n.startswith('encoder.')]
 assert enc and other
 assert {id(p) for _,p in enc}=={id(p) for p in model.encoder.parameters()}
 ids=[id(p) for n,p in enc+other];assert len(ids)==len(set(ids))==len(list(model.parameters()))
 groups=[dict(params=[p for _,p in group],param_names=[n for n,_ in group],lr=target,target_lr=target,group_name=name) for group,target,name in [(enc,SPEC['encoder_target_lr'],'encoder'),(other,SPEC['other_target_lr'],'align_decoder_head_new')]]
 return torch.optim.AdamW(groups,weight_decay=SPEC['weight_decay'],betas=tuple(SPEC['betas']),eps=SPEC['eps'])
class UpdateSchedule:
 def __init__(self,optimizer):self.optimizer=optimizer;self.update=0
 def apply(self):
  f=factor(self.update)
  for g in self.optimizer.param_groups:g['lr']=g['target_lr']*f
 def advance(self):self.update+=1
 def state_dict(self):return dict(global_update=self.update,S=S,W=W)
 def load_state_dict(self,state):
  assert state['S']==S and state['W']==W
  self.update=state['global_update'];assert 0<=self.update<=S
 def step(self):pass # no epoch scheduler; all LR controlled by apply immediately before optimizer step

def clip(parameters):return torch.nn.utils.clip_grad_norm_(parameters,max_norm=SPEC['max_norm'],norm_type=SPEC['clip_norm_type'],error_if_nonfinite=SPEC['error_if_nonfinite'])
