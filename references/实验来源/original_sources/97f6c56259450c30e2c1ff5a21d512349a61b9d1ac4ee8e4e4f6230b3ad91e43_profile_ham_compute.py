"""Count full-model parameters and forward compute on independent model copies."""
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'diagnostics/efficiency_20261023/vendor'))
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

from ham_dedup_runner import PROTOCOL, make_model, validate_protocol, write_json
from pretrained_unet2d import sha256_file

OUT = ROOT / 'runs/protocol_v5_dedup_seed42/efficiency'


def selective_scan(inputs, outputs, with_z=True):
    b, d, length = get_shape(inputs[0])
    n = get_shape(inputs[2])[-1]
    count = 9 * b * length * d * n
    if len(inputs) > 5 and get_shape(inputs[5]) is not None:
        count += b * d * length
    if with_z and len(inputs) > 6 and get_shape(inputs[6]) is not None:
        count += b * d * length
    return Counter(selective_scan=count)


def mamba_inner(inputs, outputs):
    b, twice_d, length = get_shape(inputs[0])
    d = twice_d // 2
    kernel = get_shape(inputs[1])[-1]
    projection = get_shape(inputs[3])[0]
    rank = get_shape(inputs[4])[1]
    output_d = get_shape(inputs[5])[0]
    n = get_shape(inputs[7])[-1]
    # in_proj matmul is outside the fused op and counted by fvcore already.
    conv = b * length * d * kernel
    projections = b * length * (d * projection + rank * d + d * output_d)
    scan = 9 * b * length * d * n + 2 * b * d * length
    return Counter(mamba_inner=conv + projections + scan)


def attention(inputs, outputs):
    q, k, v = [get_shape(x) for x in inputs[:3]]
    batch = math.prod(q[:-2])
    return Counter(attention=batch * q[-2] * k[-2] * (q[-1] + v[-1]))


def native_attention(inputs, outputs):
    b, length, embed = get_shape(inputs[0])
    key_length = get_shape(inputs[1])[-2]
    projections = b * (length + 2 * key_length) * embed * embed
    projections += b * length * embed * embed
    return Counter(native_attention=projections + 2 * b * length * key_length * embed)


def convolution_mode(inputs, outputs):
    weight = get_shape(inputs[1])
    return Counter(conv=math.prod(get_shape(outputs[0])) * math.prod(weight[1:]))


def average_pool(inputs, outputs):
    kernel = inputs[1].toIValue()
    if kernel is None:
        kernel = [v.toIValue() for v in inputs[1].node().inputs()]
    return Counter(avg_pool=math.prod(get_shape(outputs[0])) * math.prod(kernel))


def layout_only(inputs, outputs):
    return Counter()


def main():
    torch.set_num_threads(4)
    config = validate_protocol(PROTOCOL, 'unet2d')
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'input': [1, 3, 256, 256], 'dtype': 'float32', 'mode': 'eval',
              'scope': 'complete segmentation model, encoder plus adapters plus decoder',
              'convention': 'fvcore: one fused multiply-add counts as one; not FMA=2 hardware FLOPs',
              'selective_scan_estimate': '9*B*L*D*N, plus D skip and z gate when present; VMamba convention',
              'omissions': 'standard fvcore non-MAC elementwise, activation, softmax and data movement operations',
              'cross_scan_merge': 'layout/copy and elementwise merge additions omitted, consistent with pointwise omission policy',
              'custom_handlers': {'MambaInnerFn': 'depthwise conv + x/dt/out projections + 9BLDN scan; in_proj counted outside',
                                  'SelectiveScan': '9BLDN plus optional D skip and z gate',
                                  '_native_multi_head_attention': 'packed QKV + output projections + QK and AV matmuls',
                                  '_convolution_mode': 'output numel times kernel weights per output channel',
                                  'avg_pool2d': 'output numel times pooling kernel area'},
              'references': ['https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py',
                             'https://github.com/MzeroMiko/VMamba/blob/main/vmamba.py'],
              'torch': torch.__version__, 'fvcore': '0.1.5.post20221221',
              'protocol_sha256': sha256_file(PROTOCOL),
              'counter_source_sha256': sha256_file(Path(__file__)), 'models': {}}
    windows_path = OUT / 'profiling_windows.json'
    windows = json.loads(windows_path.read_text()) if windows_path.exists() else []
    for name in config['models']:
        start = time.time()
        window = {'model_copy': name, 'start_unix': start, 'end_unix': None,
                  'purpose': 'one forward trace for static compute count; not latency measurement'}
        windows.append(window)
        write_json(OUT / 'profiling_windows.json', windows)
        model = None
        try:
            device = torch.device('cpu' if name in ('unet2d', 'fastervit0') else 'cuda:0')
            window['device'] = str(device)
            model = make_model(name, config).eval().to(device)
            count = sum(p.numel() for p in model.parameters())
            assert count == config['models'][name]['parameters']
            x = torch.zeros(1, 3, 256, 256, device=device)
            analysis = FlopCountAnalysis(model, x).tracer_warnings('none')
            analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            analysis.set_op_handle('aten::scaled_dot_product_attention', attention)
            analysis.set_op_handle('aten::_native_multi_head_attention', native_attention)
            analysis.set_op_handle('aten::_convolution_mode', convolution_mode)
            analysis.set_op_handle('aten::avg_pool2d', average_pool)
            analysis.set_op_handle('prim::PythonOp.CrossScan', layout_only)
            analysis.set_op_handle('prim::PythonOp.CrossMerge', layout_only)
            with torch.no_grad():
                analysis.total()
                unsupported = analysis.unsupported_ops()
                for op in list(unsupported):
                    if 'MambaInnerFn' in op:
                        analysis.set_op_handle(op, mamba_inner)
                    elif 'SelectiveScan' in op:
                        # TinyViM's argument 6 is delta_bias, not the z gate.
                        if op.endswith('SelectiveScanFn'):
                            analysis.set_op_handle(op, selective_scan)
                        else:
                            analysis.set_op_handle(op, lambda ins, outs: selective_scan(ins, outs, with_z=False))
                total = analysis.total()
                unsupported = dict(analysis.unsupported_ops())
            omitted = {'aten::mul', 'aten::add', 'aten::sub', 'aten::div', 'aten::neg',
                       'aten::exp', 'aten::sigmoid', 'aten::silu', 'aten::softplus',
                       'aten::softmax', 'aten::gelu', 'aten::tanh', 'aten::mean',
                       'aten::sum', 'aten::max_pool2d', 'aten::adaptive_max_pool2d',
                       'aten::add_', 'aten::mul_', 'aten::div_', 'aten::clamp',
                       'aten::flip', 'aten::pad', 'aten::abs', 'aten::pow',
                       'aten::sqrt', 'aten::rsqrt', 'aten::fill_', 'aten::masked_fill',
                       'aten::rsub', 'aten::clone', 'aten::contiguous', 'aten::copy_',
                       'aten::upsample_nearest2d', 'aten::max', 'aten::min'}
            omitted.update({'aten::meshgrid', 'aten::sub_'})
            core_missing = {op: n for op, n in unsupported.items() if op not in omitted}
            entry = {'parameters': count,
                     'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'gflops_fma1': total / 1e9, 'by_operator': dict(analysis.by_operator()),
                     'unsupported_operators': unsupported, 'uncovered_core_operators': core_missing,
                     'flops_status': 'partial_uncovered_core_ops' if core_missing else 'core_ops_counted_with_declared_elementwise_omissions',
                     'trace_device': str(device), 'seconds': time.time() - start}
            report['models'][name] = entry
            print(name, json.dumps(entry), flush=True)
            del analysis, x
        except Exception as error:
            report['models'][name] = {'parameters': config['models'][name]['parameters'],
                                      'flops_status': 'failed', 'error': repr(error)}
            print(name, 'FAILED', repr(error), flush=True)
        finally:
            window['end_unix'] = time.time()
            write_json(OUT / 'profiling_windows.json', windows)
            write_json(OUT / 'model_compute.json', report)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
