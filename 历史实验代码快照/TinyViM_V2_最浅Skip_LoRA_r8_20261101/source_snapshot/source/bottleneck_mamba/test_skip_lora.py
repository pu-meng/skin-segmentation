"""Identity start, staged gradients, channel-only action, and V2 state compatibility."""
from pathlib import Path
import io,json,sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
from Models.SkipLoRA.model import LowRankSkipResidual,TinyViMV2SkipLoRA
from experiment_extra import build_model


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    block=LowRankSkipResidual()
    assert sum(p.numel() for p in block.parameters())==1024
    x=torch.randn(2,64,64,64,requires_grad=True)
    torch.testing.assert_close(block(x),x,rtol=0,atol=0)
    optimizer=torch.optim.SGD(block.parameters(),lr=.1)
    block(x).square().mean().backward()
    assert block.up.weight.grad.abs().sum()>0
    assert block.down.weight.grad is not None and torch.count_nonzero(block.down.weight.grad)==0
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    block(x).square().mean().backward()
    assert block.down.weight.grad.abs().sum()>0
    optimizer.step()
    with torch.no_grad():
        weight=torch.eye(64)+block.scale*(block.up.weight[:,:,0,0]@block.down.weight[:,:,0,0])
        torch.testing.assert_close(block(x),F.conv2d(x,weight[:,:,None,None]),rtol=1e-5,atol=1e-6)
        impulse=torch.zeros(1,64,5,5);impulse[:,:,2,2]=1
        value=block(impulse);value[:,:,2,2]=0
        assert torch.count_nonzero(value)==0
    saved=io.BytesIO();torch.save(block.state_dict(),saved);saved.seek(0)
    restored=LowRankSkipResidual();restored.load_state_dict(torch.load(saved,map_location='cpu',weights_only=True),strict=True)
    torch.testing.assert_close(block(x),restored(x),rtol=0,atol=0)
    torch.manual_seed(42);base=build_model('tinyvim_v2',pretrained=False)
    torch.manual_seed(42);model=TinyViMV2SkipLoRA(pretrained=False)
    a,b=base.state_dict(),model.state_dict()
    extra=set(b)-set(a)
    assert extra=={'skip_lora.down.weight','skip_lora.up.weight'}
    assert all(torch.equal(v,b[k]) for k,v in a.items())
    assert sum(p.numel() for p in model.parameters())==10372109
    missing=model.load_state_dict(a,strict=False)
    assert set(missing.missing_keys)==extra and not missing.unexpected_keys
    report=dict(status='passed',rank=8,alpha=8,scale=1,added_parameters=1024,parameters=10372109,
        initial_identity_exact=True,first_step_up_gradient_nonzero=True,
        first_step_down_gradient_zero_expected=True,second_step_down_gradient_nonzero=True,
        equivalent_pointwise_channel_matrix=True,no_spatial_mixing=True,
        strict_adapter_reload_identical=True,all_V2_state_tensors_initially_identical=True,
        new_state_keys=sorted(extra),test_data_accessed=False,formal_training_started=False)
    out=ROOT/'log/TinyViM_V2_最浅Skip_LoRA_r8_20261101'
    out.mkdir(exist_ok=True);(out/'cpu_contract.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
