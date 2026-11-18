"""Check real full models, independent variants, and E1.2 spectral behavior."""
import os
import sys
import json
import io
import importlib
from pathlib import Path

MODE = sys.argv[1]
if MODE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from selection import ARM, build_candidate
from candidate import original
from pretrained_unet2d import prepare_imports
prepare_imports('tinyvim_s')
prepare_imports('mambavision_t')
torch.set_num_threads(2)
checks = {}


def check(name, value):
    assert value, name
    checks[name] = True
    print('PASS', name, flush=True)


if MODE == 'cpu':
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    class Reference:
        @staticmethod
        def apply(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert u.device.type == 'cpu'
            return selective_scan_ref(u, delta, A, B, C, D, z=None,
                                      delta_bias=delta_bias, delta_softplus=delta_softplus)

    for name in ('Models.TinyViM.blocks', 'model.tvimblock'):
        importlib.import_module(name).SelectiveScan = Reference

torch.manual_seed(42)
model = build_candidate(pretrained=True)
baseline = original.build_candidate(pretrained=True, d=True)
check('official_encoder_loaded', model.encoder.loading_audit['all_loaded_tensors_equal'])
baseline_state = baseline.state_dict()
unchanged = {k: v for k, v in baseline_state.items()
             if ARM == 'E1.2' or not k.startswith('decoder.stages.')}
check('unchanged_model_tensors_equal', all(torch.equal(v, model.state_dict()[k]) for k, v in unchanged.items()))
device = torch.device('cpu' if MODE == 'cpu' else 'cuda:0')
model = model.to(device).eval()
x = torch.rand(1, 3, 256, 256, device=device)
if ARM == 'E1.2':
    from gated_lambda import GatedDirectionalWave
    op = model.decoder.stages[3].wave
    features = torch.randn(2, 512, 8, 8, device=device)
    check('initial_channel_gate_exact_one', torch.equal(op.channel_gate(features), torch.ones(2, 512, 1, 1, device=device)))
    baseline = baseline.to(device).eval()
    with torch.no_grad():
        check('gate_one_full_model_matches_E1', torch.allclose(model(x), baseline(x), rtol=1e-5, atol=1e-6))
    with torch.no_grad():
        op.lambda_gate[-1].weight.normal_(0, .1)
        check('gate_depends_on_image', not torch.equal(op.channel_gate(features[:1]), op.channel_gate(features[1:])))
        alpha = model.decoder.stages[3].alpha_embed
        whole = op(features, alpha)
        split = torch.cat([op(f, alpha) for f in features.split(1)])
        check('batch_does_not_mix_images', torch.allclose(whole, split, atol=2e-6, rtol=2e-5))
        specs = []
        saved = op._gate_and_project_out
        def capture(u, z, shape):
            specs.append(u)
            return saved(u, z, shape)
        op._gate_and_project_out = capture
        op(features, alpha)
        spectrum = specs[-1]
        reconstructed = torch.fft.rfft2(torch.fft.irfft2(spectrum, s=(8, 8), norm='ortho'), norm='ortho')
        check('gated_spectrum_conjugacy', torch.allclose(spectrum, reconstructed, atol=2e-5, rtol=2e-5))
        op._gate_and_project_out = saved
        op.lambda_gate[-1].weight.zero_()
del baseline
model.train()
batch = 1 if MODE == 'cpu' else 8
x = x.repeat(batch, 1, 1, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
if MODE != 'cpu':
    torch.cuda.reset_peak_memory_stats()
losses = []
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    y = model(x)
    check('output_shape_'+str(step), y.shape == (batch, 1, 256, 256))
    target = torch.rand_like(y)
    loss = torch.nn.functional.binary_cross_entropy(y, target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1)
    loss.backward()
    check('finite_loss_and_gradients_'+str(step), bool(torch.isfinite(loss)) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
    optimizer.step()
    losses.append(float(loss))
if ARM == 'E1.2':
    check('gate_parameters_receive_gradients', all(p.grad.norm()>0 for p in model.decoder.stages[3].wave.lambda_gate.parameters()))
model.eval()
with torch.no_grad():
    expected = model(x)
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)
other = build_candidate(pretrained=False).to(device).eval()
other.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
with torch.no_grad():
    check('strict_checkpoint_roundtrip', torch.equal(expected, other(x)))
record = dict(passed=True, arm=ARM, device=MODE, checks=checks, batch=batch, losses=losses,
              parameters=sum(p.numel() for p in model.parameters()))
if MODE != 'cpu':
    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    record['physical_gpu'] = os.environ['EXPERIMENT_PHYSICAL_GPU']
Path(__file__).with_name(ARM+'_'+MODE+'_acceptance.json').write_text(json.dumps(record, indent=2)+'\n')
print('ACCEPTANCE_PASS', ARM, MODE, flush=True)
