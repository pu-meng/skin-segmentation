
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
