
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/fy/cfyjwfcjb5c42i6woqsz3ueqvkqgdsgxzg6mjwpwb6lpdzr326co.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp0, ymask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/aw/cawn6uobjivzv3wshr2iwhv5egtdbb7xsrml6qcl2fqvxpyrtouq.py
# Source Nodes: [l__self___side_n2_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___side_n2_norm => add_1, clone, rsqrt, var_mean
triton_per_fused_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    x3 = xindex
    x4 = xindex % 64
    x5 = (xindex // 64) % 64
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (262144*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (64*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 64.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp25 / tmp21
    tl.store(out_ptr2 + (x5 + (64*x4) + (4096*x1)), tmp26, None)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/7z/c7z434alw4qzbvjns7uqhsrtfja775phhpmbl43zowvs6lrk6w33.py
# Source Nodes: [l__self___side_n2_norm], Original ATen: [aten.native_layer_norm]
# l__self___side_n2_norm => add_1, clone, mul, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 64
    y1 = (yindex // 64)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + (64*y0) + (4096*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (4096*y4)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (64*y4)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3 + (64*y4)), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 64.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (64*x5) + (262144*y1)), tmp13, ymask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/kf/ckfairq5smezj4inbnmtuemfox2zhcdgutxyjctqbxirah5vqxna.py
# Source Nodes: [contiguous], Original ATen: [aten.clone]
# contiguous => clone_1
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 64
    y1 = (yindex // 64)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (64*y0) + (4096*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y0 + (64*x4) + (262144*y1)), tmp4, ymask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/ue/cueor4qpdyka35wjs4mcnjo7fmfyvymf43u536blzz4wix3flypa.py
# Source Nodes: [l__self___side_ffn_0, l__self___side_ffn_1], Original ATen: [aten.convolution, aten.gelu]
# l__self___side_ffn_0 => convolution_1
# l__self___side_ffn_1 => add_3, erf, mul_2, mul_3, mul_4
triton_poi_fused_convolution_gelu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/vv/cvvfilrp3htgu36rgtfh2hn6p2tt3gejhpninazexpqtlecwpl3q.py
# Source Nodes: [l__self___side_ffn_2], Original ATen: [aten.reflection_pad2d]
# l__self___side_ffn_2 => _unsafe_index, _unsafe_index_1
triton_poi_fused_reflection_pad2d_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4460544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 66
    x2 = (xindex // 8448) % 66
    x0 = xindex % 128
    x3 = (xindex // 557568)
    x7 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.full([1], 63, tl.int64)
    tmp3 = tmp2 - tmp1
    tmp4 = tl_math.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-1) + x2
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (128*tmp5) + (8192*tmp10) + (524288*x3)), None)
    tl.store(out_ptr0 + (x7), tmp11, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/3z/c3zaw5e2sniix3o3n77xvswhh3xaykgg53benhsdxghbmyrusyei.py
# Source Nodes: [l__self___side_ffn_3], Original ATen: [aten.gelu]
# l__self___side_ffn_3 => add_4, erf_1, mul_5, mul_6, mul_7
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/k3/ck3s5utcxlpf7b5b26rcqb5o7unzritck357m4wpestqdxtz43gi.py
# Source Nodes: [h, l__self___side_ffn_4, l__self___side_mix, t], Original ATen: [aten.add, aten.convolution]
# h => add
# l__self___side_ffn_4 => convolution_3
# l__self___side_mix => convolution
# t => add_5
triton_poi_fused_add_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/3h/c3hoxnmq6mwix3yhwoiwwl5h3ae6lcz7geoqc5s25xvv32zhepqw.py
# Source Nodes: [l__self___side_post_0], Original ATen: [aten.reflection_pad2d]
# l__self___side_post_0 => _unsafe_index_2, _unsafe_index_3
triton_poi_fused_reflection_pad2d_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2230272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 66
    x2 = (xindex // 4224) % 66
    x0 = xindex % 64
    x3 = (xindex // 278784)
    x7 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.full([1], 63, tl.int64)
    tmp3 = tmp2 - tmp1
    tmp4 = tl_math.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-1) + x2
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (64*tmp5) + (4096*tmp10) + (262144*x3)), None)
    tl.store(out_ptr0 + (x7), tmp11, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/f2/cf2gcwt3evvaipywrk5c7z7fbfqnfvcvmyie3nly3ibzzqc3rvvs.py
# Source Nodes: [getattr_l__self___side_post___2___norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__self___side_post___2___norm => add_6, clone_2, rsqrt_1, var_mean_1
triton_per_fused_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 64
    x3 = (xindex // 64) % 64
    x4 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp23 / tmp19
    tl.store(out_ptr2 + (x3 + (64*x2) + (4096*x4)), tmp24, None)
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/bv/cbv4kdthlwqpkiccz7wz67jb5jilu2ppl3lsriculejvjfdi4q5b.py
# Source Nodes: [getattr_l__self___side_post___2___norm], Original ATen: [aten.native_layer_norm]
# getattr_l__self___side_post___2___norm => add_6, clone_2, mul_8, rsqrt_1, sub_9, var_mean_1
triton_poi_fused_native_layer_norm_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_10', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x5 + (4096*y4)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (64*y4)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (64*y4)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 64.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (y0 + (64*x5) + (262144*y1)), tmp11, ymask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/hx/chx2jci6qso7clguvdo544a77z54h32pt6cywf6ogsatzxgs4j5a.py
# Source Nodes: [add_2, contiguous_1, l__self___side_post_3, mul], Original ATen: [aten.add, aten.clone, aten.gelu, aten.mul]
# add_2 => add_9
# contiguous_1 => clone_3
# l__self___side_post_3 => add_8, erf_2, mul_10, mul_11, mul_12
# mul => mul_13
triton_poi_fused_add_clone_gelu_mul_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_gelu_mul_11', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 64
    y1 = (yindex // 64)
    x4 = xindex
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + (64*y0) + (4096*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 + (4096*y5)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = 0.5
    tmp8 = tmp4 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp4 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp6 * tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + (y0 + (64*x4) + (262144*y1)), tmp4, ymask)
    tl.store(out_ptr1 + (x4 + (4096*y5)), tmp16, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
    args.clear()
    assert_size_stride(primals_1, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_2, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_9, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_12, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_17, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_18, (8, 64, 64, 64), (262144, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 64, 64, 64), (262144, 1, 4096, 64), torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_16, buf0, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del primals_16
        # Source Nodes: [l__self___side_mix], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 64, 64, 64), (262144, 1, 4096, 64))
        buf2 = empty_strided_cuda((8, 64, 64, 1), (4096, 64, 1, 32768), torch.float32)
        buf3 = empty_strided_cuda((8, 64, 64, 1), (4096, 64, 1, 32768), torch.float32)
        buf25 = empty_strided_cuda((8, 64, 64, 1), (4096, 1, 64, 64), torch.float32)
        # Source Nodes: [l__self___side_n2_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(primals_17, buf1, primals_3, buf2, buf3, buf25, 32768, 64, grid=grid(32768), stream=stream0)
        buf5 = empty_strided_cuda((8, 64, 64, 64), (262144, 1, 4096, 64), torch.float32)
        # Source Nodes: [l__self___side_n2_norm], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(primals_17, buf1, primals_3, buf2, buf3, buf5, 512, 4096, grid=grid(512, 4096), stream=stream0)
        buf6 = empty_strided_cuda((8, 64, 64, 64), (262144, 1, 4096, 64), torch.float32)
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf5, primals_4, primals_5, buf6, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del primals_5
        # Source Nodes: [l__self___side_ffn_0], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 128, 64, 64), (524288, 1, 8192, 128))
        buf8 = buf7; del buf7  # reuse
        buf9 = empty_strided_cuda((8, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Source Nodes: [l__self___side_ffn_0, l__self___side_ffn_1], Original ATen: [aten.convolution, aten.gelu]
        triton_poi_fused_convolution_gelu_4.run(buf8, primals_7, buf9, 4194304, grid=grid(4194304), stream=stream0)
        del primals_7
        buf10 = empty_strided_cuda((8, 128, 66, 66), (557568, 1, 8448, 128), torch.float32)
        # Source Nodes: [l__self___side_ffn_2], Original ATen: [aten.reflection_pad2d]
        triton_poi_fused_reflection_pad2d_5.run(buf9, buf10, 4460544, grid=grid(4460544), stream=stream0)
        # Source Nodes: [l__self___side_ffn_2], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf11, (8, 128, 64, 64), (524288, 1, 8192, 128))
        buf12 = empty_strided_cuda((8, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Source Nodes: [l__self___side_ffn_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf11, buf12, 4194304, grid=grid(4194304), stream=stream0)
        # Source Nodes: [l__self___side_ffn_4], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 64, 64, 64), (262144, 1, 4096, 64))
        buf14 = buf13; del buf13  # reuse
        # Source Nodes: [h, l__self___side_ffn_4, l__self___side_mix, t], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_add_convolution_7.run(buf14, primals_17, buf1, primals_3, primals_10, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del primals_10
        del primals_17
        del primals_3
        buf15 = empty_strided_cuda((8, 64, 66, 66), (278784, 1, 4224, 64), torch.float32)
        # Source Nodes: [l__self___side_post_0], Original ATen: [aten.reflection_pad2d]
        triton_poi_fused_reflection_pad2d_8.run(buf14, buf15, 2230272, grid=grid(2230272), stream=stream0)
        # Source Nodes: [l__self___side_post_0], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf16, (8, 64, 64, 64), (262144, 1, 4096, 64))
        # Source Nodes: [l__self___side_post_1], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 64, 64), (262144, 1, 4096, 64))
        buf18 = buf3; del buf3  # reuse
        buf19 = buf2; del buf2  # reuse
        buf24 = empty_strided_cuda((8, 64, 64, 1), (4096, 1, 64, 64), torch.float32)
        # Source Nodes: [getattr_l__self___side_post___2___norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_9.run(buf17, primals_13, buf18, buf19, buf24, 32768, 64, grid=grid(32768), stream=stream0)
        buf21 = buf1; del buf1  # reuse
        # Source Nodes: [getattr_l__self___side_post___2___norm], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_10.run(buf17, primals_13, buf18, buf19, buf21, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del buf18
        del buf19
        del primals_13
        buf22 = buf17; del buf17  # reuse
        buf23 = empty_strided_cuda((8, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Source Nodes: [add_2, contiguous_1, l__self___side_post_3, mul], Original ATen: [aten.add, aten.clone, aten.gelu, aten.mul]
        triton_poi_fused_add_clone_gelu_mul_11.run(buf21, primals_14, primals_15, primals_18, primals_1, buf22, buf23, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del primals_15
        del primals_18
    return (buf23, primals_1, primals_2, primals_4, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, buf0, buf5, buf6, buf8, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf21, buf22, buf24, buf25, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
