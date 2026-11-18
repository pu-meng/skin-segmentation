"""CPU forward traces for full V5 RMT and SU-RMT segmentation models.

Use the existing FMA=1 fvcore convention. No training, dataset access or GPU use.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import profile_ham_compute as base
from rmt_unet2d import RMTUNet2D
from surmt_adapter import SURMTV5

ROOT = Path(__file__).resolve().parent
OMITTED = {
    'aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::neg',
    'aten::exp', 'aten::sigmoid', 'aten::silu', 'aten::softplus',
    'aten::softmax', 'aten::gelu', 'aten::tanh', 'aten::mean', 'aten::sum',
    'aten::max_pool2d', 'aten::adaptive_max_pool2d', 'aten::add_', 'aten::mul_',
    'aten::div_', 'aten::clamp', 'aten::flip', 'aten::pad', 'aten::abs',
    'aten::pow', 'aten::sqrt', 'aten::rsqrt', 'aten::fill_', 'aten::masked_fill',
    'aten::rsub', 'aten::clone', 'aten::contiguous', 'aten::copy_',
    'aten::upsample_nearest2d', 'aten::max', 'aten::min', 'aten::meshgrid',
    'aten::sub_', 'aten::sin', 'aten::cos', 'aten::log', 'aten::log2',
    'aten::_conj', 'aten::view_as_complex',
}


def real_fft_estimate(inputs, outputs):
    # Approximate real FFT/iFFT cost: 2.5*N*log2(N) real operations.
    # Divide by 2 to express an FMA-equivalent scale alongside FMA=1 MACs.
    # This is an algorithmic estimate, not the exact backend instruction count.
    inverse = inputs[0].type().scalarType() in ('ComplexFloat', 'ComplexDouble')
    shape = base.get_shape(outputs[0] if inverse else inputs[0])
    h, w = shape[-2:]
    batches = math.prod(shape[:-2])
    return Counter(fft_estimate_fma_equivalent=1.25*batches*h*w*(math.log2(h)+math.log2(w)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['rmt_t', 'rmt_s', 'su_rmt'], required=True)
    args = parser.parse_args()
    base.torch.set_num_threads(4)
    group = 'protocol_v5_surmt_extension_seed42' if args.model == 'su_rmt' else 'protocol_v5_rmt_extension_seed42'
    protocol = ROOT / (group + '.json')
    config = json.loads(protocol.read_text())
    start = time.monotonic()
    # Weight values do not alter the graph. Preserve the full trained architecture.
    model = (SURMTV5(pretrained=False) if args.model == 'su_rmt'
             else RMTUNet2D(args.model, pretrained=False)).cpu().eval()
    count = sum(p.numel() for p in model.parameters())
    assert count == config['models'][args.model]['parameters']
    x = base.torch.zeros(1, 3, 256, 256)
    a = base.FlopCountAnalysis(model, x).tracer_warnings('none')
    a.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    a.set_op_handle('aten::scaled_dot_product_attention', base.attention)
    a.set_op_handle('aten::_native_multi_head_attention', base.native_attention)
    a.set_op_handle('aten::_convolution_mode', base.convolution_mode)
    a.set_op_handle('aten::avg_pool2d', base.average_pool)
    a.set_op_handle('aten::fft_rfft2', real_fft_estimate)
    a.set_op_handle('aten::fft_irfft2', real_fft_estimate)
    with base.torch.no_grad():
        total = a.total()
        unsupported = dict(a.unsupported_ops())
    missing = {k: v for k, v in unsupported.items() if k not in OMITTED}
    entry = dict(parameters=count, gflops_fma1=total/1e9,
                 by_operator=dict(a.by_operator()), unsupported_operators=unsupported,
                 uncovered_core_operators=missing,
                 flops_status='partial_uncovered_core_ops' if missing else 'core_ops_counted_with_declared_elementwise_omissions',
                 trace_device='cpu', seconds=time.monotonic()-start)
    report = dict(input=[1,3,256,256], dtype='float32', mode='eval',
                  scope='complete segmentation model including adapters and full decoder/head',
                  convention='fvcore one fused multiply-add counts as one; same as original five-model report',
                  omissions=sorted(OMITTED), torch=base.torch.__version__,
                  fft_convention='real FFT/iFFT estimated as 2.5*N*log2(N) real operations per transform, divided by 2 into FMA-equivalent units; approximate, backend independent',
                  protocol_sha256=base.sha256_file(protocol),
                  counter_source_sha256=base.sha256_file(Path(__file__)),
                  shared_handler_source_sha256=base.sha256_file(Path(base.__file__)),
                  synthetic_input_no_dataset_access=True, pretrained_weights_loaded=False,
                  models={args.model:entry})
    out = ROOT/'runs'/group/'efficiency/compute'/(args.model+'.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(entry), flush=True)
    if missing:
        raise SystemExit('Uncovered core operators require review: '+str(missing))


if __name__ == '__main__':
    main()
