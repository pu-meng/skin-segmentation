import torch

def mirror_logits(network,x,enabled):
 prediction=network(x)
 if enabled:
  for axes in [(2,),(3,),(2,3)]:
   prediction=prediction+torch.flip(network(torch.flip(x,axes)),axes)
  prediction=prediction/4
 return prediction
