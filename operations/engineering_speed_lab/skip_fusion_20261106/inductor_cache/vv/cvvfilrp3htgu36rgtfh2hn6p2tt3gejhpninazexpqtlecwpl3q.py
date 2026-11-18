
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
