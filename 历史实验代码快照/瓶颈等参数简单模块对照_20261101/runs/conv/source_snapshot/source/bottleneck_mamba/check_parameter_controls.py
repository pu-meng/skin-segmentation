"""CPU-only block contract; does not constitute full-network GPU preflight."""
from pathlib import Path
import importlib.util
import io
import json
import torch

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / 'Models/TinyViM/bottleneck_parameter_controls.py'
spec = importlib.util.spec_from_file_location('parameter_controls', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Block = module.ParameterMatchedBottleneck


def main():
    torch.set_num_threads(4)
    records = []
    for kind in ('mlp', 'conv', 'transformer'):
        for spatial in (8, 16):
            torch.manual_seed(42)
            block = Block(kind)
            x = torch.randn(2, 512, spatial, spatial, requires_grad=True)
            before = {n: p.detach().clone() for n, p in block.named_parameters()}
            output = block(x)
            assert output.shape == x.shape and torch.isfinite(output).all()
            output.square().mean().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            for name, p in block.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                assert p.grad.abs().sum() > 0, name
            torch.optim.AdamW(block.parameters(), lr=.001).step()
            assert all(not torch.equal(before[n], p) for n, p in block.named_parameters())
            saved = io.BytesIO()
            torch.save(block.state_dict(), saved)
            saved.seek(0)
            restored = Block(kind)
            restored.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(block(x), restored(x), rtol=0, atol=0)
            records.append(dict(kind=kind, shape=list(x.shape), width=block.width,
                                parameters=block.parameter_count,
                                target=block.target_parameters,
                                difference=block.parameter_count-block.target_parameters,
                                relative_difference=block.relative_error,
                                finite_forward_backward=True, all_parameters_updated=True,
                                strict_reload_identical=True))
    out = ROOT / 'log/瓶颈等参数简单模块对照_20261101'
    out.mkdir(exist_ok=True)
    (out/'block_contract.json').write_text(json.dumps(dict(status='cpu_block_contract_passed',
        full_network_gpu_preflight=False, formal_training_started=False,
        test_data_accessed=False, records=records), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
