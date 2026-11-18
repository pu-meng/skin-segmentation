from pathlib import Path
import json,torch
from torch.utils.data import DataLoader
from analyze_native import ROOT,RUN,Validation,PretrainedUNet2D,validate_protocol
r=Path(__file__).resolve().parent;torch.set_num_threads(4);torch.cuda.set_device(1);device='cuda:1'
c=validate_protocol(ROOT/'protocol_v5_dedup_seed42.json','tinyvim_s',check_cache=False);ds=Validation(Path(c['data']['split_file']).parent)
m=PretrainedUNet2D('tinyvim_s',pretrained=False).to(device);m.load_state_dict(torch.load(RUN/'checkpoints/best.pth',map_location=device),strict=True);m.eval()
result={};formal=json.loads((RUN/'validation/summary.json').read_text())
for batch in [4,1]:
 d=j=0.;n=0
 with torch.no_grad():
  for x,y in DataLoader(ds,batch_size=batch,shuffle=False):
   pred=m(x.to(device))>=.5;y=y.to(device);inter=(pred&y).sum((1,2,3));p=pred.sum((1,2,3));t=y.sum((1,2,3));u=p+t-inter
   d+=torch.where(p+t>0,2*inter/(p+t),1.).sum().item();j+=torch.where(u>0,inter/u,1.).sum().item();n+=len(x)
 result[str(batch)]=dict(dice=d/n,iou=j/n,dice_delta=d/n-formal['per_image_mean']['dice'],iou_delta=j/n-formal['per_image_mean']['iou'])
 print(batch,result[str(batch)],flush=True)
(r/'best_metric_verification.json').write_text(json.dumps(dict(formal=formal['per_image_mean'],measured=result),indent=2)+'\n')
