
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
