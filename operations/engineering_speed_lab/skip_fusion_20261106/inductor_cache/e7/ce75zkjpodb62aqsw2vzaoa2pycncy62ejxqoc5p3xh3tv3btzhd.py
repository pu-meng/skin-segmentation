
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


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/il/cilsv4hnb3rg5lk5jbv3mw4xrt2cpq3ejftfez5a6co5vanhodzk.py
# Source Nodes: [l__self___side_post_3], Original ATen: [aten.gelu, aten.mul, aten.native_layer_norm_backward, aten.sum]
# l__self___side_post_3 => add_8, erf_2, mul_10, mul_11, mul_12
triton_red_fused_gelu_mul_native_layer_norm_backward_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_mul_native_layer_norm_backward_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp14 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 * tmp2
        tmp4 = 0.7071067811865476
        tmp5 = tmp1 * tmp4
        tmp6 = libdevice.erf(tmp5)
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp8 * tmp2
        tmp17 = tmp1 * tmp1
        tmp18 = -0.5
        tmp19 = tmp17 * tmp18
        tmp20 = tl_math.exp(tmp19)
        tmp21 = 0.3989422804014327
        tmp22 = tmp20 * tmp21
        tmp23 = tmp1 * tmp22
        tmp24 = tmp16 + tmp23
        tmp25 = tmp15 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask, tmp28, _tmp27)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/gy/cgypq26ubektgupv6x2uleqmspcmoejfn6qnj32fa6olu44plgso.py
# Source Nodes: [l__self___side_post_3], Original ATen: [aten.gelu, aten.mul, aten.sum]
# l__self___side_post_3 => add_8, erf_2, mul_10, mul_11, mul_12
triton_per_fused_gelu_mul_sum_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_mul_sum_1', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/ka/cka3f4bu4nnpnudnvkw4dtmf2e2pcvs22isrby6lf5uotcxfe5jj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (64*x3)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (x5 + (64*r2) + (4096*x4) + (262144*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp2 * tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/vr/cvrnf4vy4vnk4dpv63qlmyymiiyxqwv562lhm34gvpmxyxaufc5x.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    y0 = yindex % 64
    y1 = (yindex // 64)
    x2 = xindex % 64
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x3) + (4096*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (64*y0) + (4096*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x5 + (4096*y4)), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x3 + (64*y4)), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (y0 + (64*x5) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x3 + (64*y4)), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = 0.7071067811865476
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp4
    tmp13 = -0.5
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = 0.3989422804014327
    tmp17 = tmp15 * tmp16
    tmp18 = tmp4 * tmp17
    tmp19 = tmp11 + tmp18
    tmp20 = tmp3 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = 64.0
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 - tmp25
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 - tmp29
    tmp31 = tmp0 * tmp30
    tl.store(out_ptr1 + (x2 + (64*y0) + (4096*x3) + (262144*y1)), tmp31, ymask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/tc/ctcy5rie2ijvwd5jw4cstqbvey2f6enz63ohpbp2fsfq7rfpd7j4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x2 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(((r3 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r3 + (128*x0)) // 4096)) + (524288*x2) + (r3 % 64)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1 + (64*r3) + (8192*x0) + (524288*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr3 + ((64*x1) + (4096*(r3 % 64)) + (262144*((r3 + (128*x0)) // 4096)) + (524288*x2) + (((r3 + (128*x0)) // 64) % 64)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp2 * tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp25, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/3u/c3uavs5j7mszcr3tbp4q6yse23tl5jlg5ufgbhzdpx5a4ynywkwj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/7j/c7jbvkb5ce3zikiylbw4bfqwvcx3icvkmfm537dbwyratvfuiz5v.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/mb/cmb5tvyeulo75hzp7emgr7n5w5bnbzjjqx7opvs47ygegznj6mjm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*(((r2 + (128*x1)) // 64) % 64)) + (4096*(r2 % 64)) + (262144*((r2 + (128*x1)) // 4096))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/jx/cjx6nmsxaj77lcifwjikxgzd33kgr2euhhp22ikqketc4voksqpp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/dy/cdyy6v5rifmvaj4t63k4dobgsnbivhq3ccymjcopr3ose4e5vj77.py
# Source Nodes: [], Original ATen: [aten.reflection_pad2d_backward]

triton_poi_fused_reflection_pad2d_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_backward_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
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
    x3 = (xindex // 64)
    x2 = xindex % 64
    y0 = yindex % 64
    y1 = (yindex // 64)
    x5 = xindex
    y4 = yindex
    tmp0 = 1 + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 65, tl.int64)
    tmp4 = tmp0 <= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 1 + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 <= tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (4288 + y0 + (64*x2) + (4224*x3) + (278784*y1)), tmp10 & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x3
    tmp15 = tl.full([1, 1], 1, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tmp14 <= tmp15
    tmp18 = tmp16 & tmp17
    tmp19 = x2
    tmp20 = tmp19 >= tmp15
    tmp21 = tmp19 <= tmp15
    tmp22 = tmp20 & tmp21
    tmp23 = tmp18 & tmp22
    tmp24 = tl.load(in_ptr0 + (4288 + y0 + ((-4224)*x3) + ((-64)*x2) + (278784*y1)), tmp23 & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tmp13 + tmp26
    tmp28 = tmp18 & tmp9
    tmp29 = tl.load(in_ptr0 + (4288 + y0 + ((-4224)*x3) + (64*x2) + (278784*y1)), tmp28 & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tmp27 + tmp31
    tmp33 = tl.full([1, 1], 62, tl.int64)
    tmp34 = tmp19 >= tmp33
    tmp35 = tmp19 <= tmp33
    tmp36 = tmp34 & tmp35
    tmp37 = tmp18 & tmp36
    tmp38 = tl.load(in_ptr0 + (12352 + y0 + ((-4224)*x3) + ((-64)*x2) + (278784*y1)), tmp37 & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tmp32 + tmp40
    tmp42 = tmp5 & tmp22
    tmp43 = tl.load(in_ptr0 + (4288 + y0 + ((-64)*x2) + (4224*x3) + (278784*y1)), tmp42 & ymask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp41 + tmp45
    tmp47 = tmp5 & tmp36
    tmp48 = tl.load(in_ptr0 + (12352 + y0 + ((-64)*x2) + (4224*x3) + (278784*y1)), tmp47 & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp46 + tmp50
    tmp52 = tmp14 >= tmp33
    tmp53 = tmp14 <= tmp33
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp22
    tmp56 = tl.load(in_ptr0 + (536512 + y0 + ((-4224)*x3) + ((-64)*x2) + (278784*y1)), tmp55 & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp51 + tmp58
    tmp60 = tmp54 & tmp9
    tmp61 = tl.load(in_ptr0 + (536512 + y0 + ((-4224)*x3) + (64*x2) + (278784*y1)), tmp60 & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp59 + tmp63
    tmp65 = tmp54 & tmp36
    tmp66 = tl.load(in_ptr0 + (544576 + y0 + ((-4224)*x3) + ((-64)*x2) + (278784*y1)), tmp65 & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp64 + tmp68
    tl.store(out_ptr0 + (x5 + (4096*y4)), tmp69, ymask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/uu/cuudviivil27upndbbiejqkhmfikyway4ns46emh7om5ayfmi6pz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_10', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/ub/cub5ewmulcu2trwh2kqlfbelsxon55cbvyqigix7vj7ka23vdnpg.py
# Source Nodes: [l__self___side_ffn_3], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
# l__self___side_ffn_3 => add_4, erf_1, mul_6
triton_poi_fused_convolution_backward_gelu_gelu_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_gelu_gelu_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/n5/cn5qjj4dlslk3fnplx3tw2z2udtl4mxf5ndbm6oj4gyq7wqt4wol.py
# Source Nodes: [], Original ATen: [aten.reflection_pad2d_backward]

triton_poi_fused_reflection_pad2d_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_backward_12', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 128
    y1 = (yindex // 128)
    x5 = xindex
    y4 = yindex
    tmp0 = 1 + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 65, tl.int64)
    tmp4 = tmp0 <= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 1 + x3
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 <= tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (8576 + y0 + (128*x3) + (8448*x2) + (557568*y1)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x2
    tmp15 = tl.full([1, 1], 1, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tmp14 <= tmp15
    tmp18 = tmp16 & tmp17
    tmp19 = x3
    tmp20 = tmp19 >= tmp15
    tmp21 = tmp19 <= tmp15
    tmp22 = tmp20 & tmp21
    tmp23 = tmp18 & tmp22
    tmp24 = tl.load(in_ptr0 + (8576 + y0 + ((-8448)*x2) + ((-128)*x3) + (557568*y1)), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tmp13 + tmp26
    tmp28 = tmp18 & tmp9
    tmp29 = tl.load(in_ptr0 + (8576 + y0 + ((-8448)*x2) + (128*x3) + (557568*y1)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tmp27 + tmp31
    tmp33 = tl.full([1, 1], 62, tl.int64)
    tmp34 = tmp19 >= tmp33
    tmp35 = tmp19 <= tmp33
    tmp36 = tmp34 & tmp35
    tmp37 = tmp18 & tmp36
    tmp38 = tl.load(in_ptr0 + (24704 + y0 + ((-8448)*x2) + ((-128)*x3) + (557568*y1)), tmp37, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tmp32 + tmp40
    tmp42 = tmp5 & tmp22
    tmp43 = tl.load(in_ptr0 + (8576 + y0 + ((-128)*x3) + (8448*x2) + (557568*y1)), tmp42, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp41 + tmp45
    tmp47 = tmp5 & tmp36
    tmp48 = tl.load(in_ptr0 + (24704 + y0 + ((-128)*x3) + (8448*x2) + (557568*y1)), tmp47, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp46 + tmp50
    tmp52 = tmp14 >= tmp33
    tmp53 = tmp14 <= tmp33
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp22
    tmp56 = tl.load(in_ptr0 + (1073024 + y0 + ((-8448)*x2) + ((-128)*x3) + (557568*y1)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp51 + tmp58
    tmp60 = tmp54 & tmp9
    tmp61 = tl.load(in_ptr0 + (1073024 + y0 + ((-8448)*x2) + (128*x3) + (557568*y1)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp59 + tmp63
    tmp65 = tmp54 & tmp36
    tmp66 = tl.load(in_ptr0 + (1089152 + y0 + ((-8448)*x2) + ((-128)*x3) + (557568*y1)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp64 + tmp68
    tl.store(out_ptr0 + (x5 + (4096*y4)), tmp69, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/wu/cwuuwpkawm6awejd4s2rthgetowdgbkm7eihtxroyszbtkfbglyy.py
# Source Nodes: [l__self___side_ffn_1], Original ATen: [aten.gelu, aten.gelu_backward]
# l__self___side_ffn_1 => add_3, erf, mul_3
triton_poi_fused_gelu_gelu_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_13', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3 + (64*x2) + (4096*y4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (128*x5) + (524288*y1)), None, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(out_ptr0 + (x5 + (4096*y4)), tmp17, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/2n/c2n2e6wpw57o5mycpjuvcp6c3ngvcovdpw4vr2k6fzdaqx4z5pum.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_14', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/wb/cwbpv4bfmtjbblf4npwi3nbylkh32daffvka72xahfybpkgvt33j.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_15', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/4u/c4uucjoc7cj2urxbwqrunq4u5pr5uk5epif6kdgv6fwa5vkqw6az.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_16', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp7 = tl.load(in_ptr2 + (x3 + (64*r1) + (4096*x2) + (262144*x4)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/iw/ciwypeylbt5zhesdz57oh3utwr54ne3otw7mfgiacu57yud7fuu7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_17', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((64*x1) + (4096*(r2 % 64)) + (262144*((r2 + (128*x0)) // 4096)) + (((r2 + (128*x0)) // 64) % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/qc/cqce2y43gowm7wtt233ob7fc42jbxikwbjpbdkrcnomvybqbrrls.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_18', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/skip_fusion_20261106/inductor_cache/4z/c4zrhq3rmeykgqot427hi45sfw6v4cbnv7gpezurp6nyt2khsfm2.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '71289734499090ef6d56db411758fec09be6b197bd9c453d82346a1fa6f1185c'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 64)
    y0 = yindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x5 + (4096*y4)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + (64*x2) + (4096*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0 + (64*x5) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x5 + (4096*y1)), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3 + (64*y0) + (4096*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x5 + (4096*y1)), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 64.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (4096*y4)), tmp14, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, mul, clone_1, convolution_1, mul_4, _unsafe_index_1, convolution_2, mul_7, add_5, _unsafe_index_3, convolution_4, mul_8, clone_3, div, div_1, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_2, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_6, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_8, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_9, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_12, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(mul, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(clone_1, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(mul_4, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(_unsafe_index_1, (8, 128, 66, 66), (557568, 1, 8448, 128))
    assert_size_stride(convolution_2, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(mul_7, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(add_5, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(_unsafe_index_3, (8, 64, 66, 66), (278784, 1, 4224, 64))
    assert_size_stride(convolution_4, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(mul_8, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(clone_3, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(div, (8, 64, 64, 1), (4096, 1, 64, 64))
    assert_size_stride(div_1, (8, 64, 64, 1), (4096, 1, 64, 64))
    assert_size_stride(tangents_1, (8, 64, 64, 64), (262144, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 64, 1, 1, 256), (16384, 256, 16384, 16384, 1), torch.float32)
        buf8 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
        # Source Nodes: [l__self___side_post_3], Original ATen: [aten.gelu, aten.mul, aten.native_layer_norm_backward, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_gelu_mul_native_layer_norm_backward_sum_0.run(tangents_1, clone_3, primals_1, buf0, buf8, 16384, 128, grid=grid(16384), stream=stream0)
        buf1 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Source Nodes: [l__self___side_post_3], Original ATen: [aten.gelu, aten.mul, aten.sum]
        triton_per_fused_gelu_mul_sum_1.run(buf0, buf1, 64, 256, grid=grid(64), stream=stream0)
        buf2 = empty_strided_cuda((8, 64, 64, 1), (4096, 64, 1, 32768), torch.float32)
        buf3 = empty_strided_cuda((8, 64, 64, 1), (4096, 64, 1, 32768), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(tangents_1, primals_1, clone_3, primals_14, mul_8, buf2, buf3, 32768, 64, grid=grid(32768), stream=stream0)
        buf10 = empty_strided_cuda((8, 64, 64, 64), (262144, 1, 64, 4096), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_3.run(div, tangents_1, primals_1, clone_3, primals_14, buf2, mul_8, buf3, buf10, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del div
        del primals_14
        buf5 = reinterpret_tensor(buf0, (64, 4, 64), (64, 4096, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_4.run(tangents_1, primals_1, clone_3, mul_8, buf5, 16384, 128, grid=grid(16384), stream=stream0)
        del clone_3
        del mul_8
        del primals_1
        buf6 = empty_strided_cuda((64, 4), (1, 64), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf5, buf6, 256, 64, grid=grid(256), stream=stream0)
        del buf5
        buf7 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf6, buf7, 64, 4, grid=grid(64), stream=stream0)
        buf9 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_gelu_mul_sum_1.run(buf8, buf9, 64, 256, grid=grid(64), stream=stream0)
        buf11 = reinterpret_tensor(buf8, (64, 256), (1, 64), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_7.run(buf10, buf11, 16384, 128, grid=grid(16384), stream=stream0)
        buf12 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_8.run(buf11, buf12, 64, 256, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf13 = aten.convolution_backward.default(buf10, convolution_4, primals_12, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf10
        del convolution_4
        del primals_12
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf16 = aten.convolution_backward.default(buf14, _unsafe_index_3, primals_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
        del _unsafe_index_3
        del primals_11
        buf17 = buf16[0]
        buf18 = buf16[1]
        del buf16
        buf19 = reinterpret_tensor(buf14, (8, 64, 64, 64), (262144, 4096, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.reflection_pad2d_backward]
        triton_poi_fused_reflection_pad2d_backward_9.run(buf17, buf19, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del buf17
        buf20 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_10.run(buf19, buf20, 256, 8192, grid=grid(256), stream=stream0)
        buf21 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf20, buf21, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf22 = aten.convolution_backward.default(buf19, mul_7, primals_9, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_7
        del primals_9
        buf23 = buf22[0]
        buf24 = buf22[1]
        del buf22
        buf25 = buf23; del buf23  # reuse
        # Source Nodes: [l__self___side_ffn_3], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_11.run(buf25, convolution_2, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_2
        # Source Nodes: [l__self___side_ffn_3], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf26 = aten.convolution_backward.default(buf25, _unsafe_index_1, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 128, [True, True, False])
        del _unsafe_index_1
        del primals_8
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = reinterpret_tensor(buf25, (8, 128, 64, 64), (524288, 4096, 1, 64), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.reflection_pad2d_backward]
        triton_poi_fused_reflection_pad2d_backward_12.run(buf27, buf29, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del buf27
        buf30 = empty_strided_cuda((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Source Nodes: [l__self___side_ffn_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_13.run(buf29, convolution_1, buf30, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del buf29
        del convolution_1
        buf31 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_14.run(buf30, buf31, 512, 8192, grid=grid(512), stream=stream0)
        buf32 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_15.run(buf31, buf32, 128, 4, grid=grid(128), stream=stream0)
        del buf31
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf33 = aten.convolution_backward.default(buf30, clone_1, primals_6, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf30
        del clone_1
        del primals_6
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf36 = buf3; del buf3  # reuse
        buf37 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_16.run(buf34, primals_4, mul, buf36, buf37, 32768, 64, grid=grid(32768), stream=stream0)
        buf38 = reinterpret_tensor(buf11, (64, 256), (256, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf34, mul, buf38, 16384, 128, grid=grid(16384), stream=stream0)
        buf39 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_gelu_mul_sum_1.run(buf38, buf39, 64, 256, grid=grid(64), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (64, 256), (1, 64), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_18.run(buf34, buf40, 16384, 128, grid=grid(16384), stream=stream0)
        buf41 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_convolution_backward_8.run(buf40, buf41, 64, 256, grid=grid(64), stream=stream0)
        del buf40
        buf42 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf42, div_1, buf34, primals_4, buf36, mul, buf37, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del buf34
        del buf36
        del buf37
        del div_1
        del mul
        del primals_4
        buf43 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_10.run(buf42, buf43, 256, 8192, grid=grid(256), stream=stream0)
        buf44 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf43, buf44, 64, 4, grid=grid(64), stream=stream0)
        del buf43
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf45 = aten.convolution_backward.default(buf42, primals_16, primals_2, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_16
        del primals_2
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
    return (buf1, buf47, buf44, buf39, buf41, buf35, buf32, buf28, buf24, buf21, buf18, buf15, buf12, buf7, buf9, buf46, buf42, tangents_1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    mul_4 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    _unsafe_index_1 = rand_strided((8, 128, 66, 66), (557568, 1, 8448, 128), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    add_5 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    _unsafe_index_3 = rand_strided((8, 64, 66, 66), (278784, 1, 4224, 64), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    mul_8 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    clone_3 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 64, 64, 1), (4096, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 64, 64, 1), (4096, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_4, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, mul, clone_1, convolution_1, mul_4, _unsafe_index_1, convolution_2, mul_7, add_5, _unsafe_index_3, convolution_4, mul_8, clone_3, div, div_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
