import sys,json,torch
from pathlib import Path
p=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(p),str(p/'source/comparison')]
from pretrained_unet2d import OfficialEncoder
m=OfficialEncoder('fastervit0',pretrained=False).eval();torch.set_num_threads(2)
r={}
keys=['patch_embed','levels.0.downsample','levels.1.downsample','levels.2.downsample','levels.2.global_tokenizer','levels.2.blocks.0.attn','levels.2.blocks.0.hat_attn','levels.3.blocks.0.attn','norm']
for key in keys:
    module=dict(m.backbone.named_modules())[key]
    def hook(mod,inputs,output,key=key):r[key]={'input':list(inputs[0].shape),'output':list(output.shape)}
    module.register_forward_hook(hook)
with torch.no_grad():out=m(torch.rand(1,3,256,256))
r['features']=[list(v.shape) for v in out]
r['blocks']=[{'stage':i+1,'drop_path':[float(getattr(b.drop_path,'drop_prob',0)) for b in l.blocks],
              'head_count':[getattr(getattr(b,'attn',None),'num_heads',None) for b in l.blocks],
              'layer_scale':[isinstance(getattr(b,'gamma3',None),torch.nn.Parameter) for b in l.blocks]} for i,l in enumerate(m.backbone.levels)]
(p/'研究记录/FasterViT_图解维度核对.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(r,ensure_ascii=False))
