"""Short integration check of all stage shapes and observational hooks."""
from pathlib import Path
import torch
from models import KINDS, PureMixerUNet
from experiment import Images, amp, write, check_protocol
from analyze import Features


def main():
    check_protocol();torch.set_num_threads(4);device=torch.device('cuda:1')
    torch.cuda.set_device(device)
    here=Path(__file__).resolve().parent
    x=Images('val',1)[0][0][None].to(device)
    report={}
    for kind in KINDS:
        saved=torch.load(here/'preflight'/kind/'best.pth',map_location='cpu')
        model=PureMixerUNet(kind).to(device).eval();model.load_state_dict(saved['model'],strict=True)
        state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        with torch.no_grad(),amp(device):
            reference=model(x)
        observer=Features(model)
        with torch.no_grad(),amp(device):
            observed=model(x)
        observer.close()
        assert torch.equal(reference,observed)
        assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in state.items())
        stats=observer.results()
        assert len(stats)==42
        shapes=[[48,64,64],[64,32,32],[168,16,16],[224,8,8]]
        for key,v in stats.items():
            assert v['shape']==shapes[int(key[5])-1] and v['samples']==1
        report[kind]=dict(feature_points=42,shapes=shapes,hook_predictions_identical=True,
                          state_unchanged=True,trained_result=False)
    write(here/'feature_integration_audit.json',report)
    print(report)


if __name__=='__main__':main()
