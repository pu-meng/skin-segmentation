"""Record architecture and real feature shapes; synthetic inference only."""
from pathlib import Path
import hashlib
import json
import sys
import torch

P=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(P),str(P/'source/comparison')]
from Models.Encoder2026.model import build_model
from pretrained_unet2d import PretrainedUNet2D
from rmt_unet2d import RMTUNet2D


def trace(names):
    torch.set_num_threads(2)
    path=P/'研究记录/Encoder结构核对.json'
    records=json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        if name=='tinyvim_s':
            from Models.TinyViM import TinyViMUNet
            model=TinyViMUNet()
        elif name in ('fastervit0','mambavision_t'):model=PretrainedUNet2D(name,pretrained=False)
        elif name in ('rmt_t','rmt_s'):model=RMTUNet2D(name,pretrained=False)
        else:model=build_model(name,pretrained=False)
        record={'input':[1,3,256,256],'weights':'random initialization; structure check only',
                'parameters':sum(v.numel() for v in model.parameters()),'modules':[]}
        handles=[]
        for key,module in model.named_modules():
            if isinstance(module,torch.nn.Conv2d):
                record['modules'].append({'name':key,'type':'Conv2d',
                    'in':module.in_channels,'out':module.out_channels,'kernel':list(module.kernel_size),
                    'stride':list(module.stride),'groups':module.groups})
        if name=='tinyvim_s':
            record['stages']=[]
            for index in (0,2,4,6):
                stage=model.encoder.backbone.network[index]
                stage_record={'network_index':index,'blocks':[type(v).__name__ for v in stage],'scans':[]}
                for i,block in enumerate(stage):
                    if hasattr(block,'op'):
                        op=block.op
                        stage_record['scans'].append({'index':i,'split':list(op.split),'N':op.d_state,
                            'dt_rank':op.dt_rank,'A_logs':list(op.A_logs.shape),'Ds':list(op.Ds.shape),
                            'x_proj_weight':list(op.x_proj_weight.shape),'dt_projs_weight':list(op.dt_projs_weight.shape),
                            'pool':op.pool.kernel_size})
                record['stages'].append(stage_record)
            if not torch.cuda.is_available():raise RuntimeError('TinyViM official scan requires CUDA')
            device=torch.device('cuda:0')
        elif name in ('lowformer_e1','cpubone_b0_bftraining'):
            record['stage_lengths']=[len(s.op_list) for s in model.encoder.backbone.stages]
            device=torch.device('cpu')
        elif name in ('fastervit0','mambavision_t'):
            record['stages']=[{'blocks':[type(b).__name__ for b in s.blocks],
                'mixers':[type(b.mixer).__name__ if hasattr(b,'mixer') else None for b in s.blocks],
                'window':s.window_size,'do_gt':s.do_gt} for s in model.encoder.backbone.levels]
            device=torch.device('cuda:0')
        else:
            record['stages']=[{'blocks':len(s.blocks),
                'retention':[type(b.retention).__name__ for b in s.blocks]} for s in model.encoder.backbone.layers]
            device=torch.device('cpu')
        model=model.to(device).eval()
        captured={}
        def shapes(m,inputs,output):captured['features']=[list(v.shape) for v in output]
        handles.append(model.encoder.register_forward_hook(shapes))
        with torch.no_grad():result=model(torch.rand(1,3,256,256,device=device))
        for handle in handles:handle.remove()
        record.update(captured,output=list(result.shape),device=str(device),forward_finite=bool(torch.isfinite(result).all()))
        assert record['output']==[1,1,256,256] and record['forward_finite']
        records[name]=record
        del model,result
        if device.type=='cuda':torch.cuda.empty_cache()
    files=['Models/TinyViM/model.py','Models/TinyViM/backbone.py','Models/TinyViM/blocks.py',
           'Models/Encoder2026/model.py','source/comparison/pretrained_unet2d.py',
           'source/comparison/rmt_unet2d.py','references/code/RMT/classfication_release/RMT.py',
           'references/code/MambaVision/mambavision/models/mamba_vision.py',
           'references/code/FasterViT/fastervit/models/faster_vit_any_res.py',
           'references/code/LowFormer/lowformer_model.py','references/code/CPUBone/cpubone_model.py']
    records['source_sha256']={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files}
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    for name in names:
        print(name,records[name]['features'],records[name]['parameters'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',required=True)
    trace(parser.parse_args().models)
