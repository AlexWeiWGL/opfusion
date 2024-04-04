import itertools

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd


def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel_248(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    # Block indexing
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures

    elements_per_feature: tl.constexpr = 32 // bits

    # Load parameters
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx + (outfeatures * (row_idx // elements_per_feature))),
        None,
    )

    wf_weights = (row_idx % elements_per_feature) * bits

    wf_zeros = (col_idx % elements_per_feature) * bits

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + (col_idx + (outfeatures * groups)), None).to(
        tl.float32
    )

    # Unpack weights
    weights = qweights >> wf_weights  # bit shift qweight

    weights = weights & maxq

    # Unpack zeros
    qzero_ncols: tl.constexpr = outfeatures // elements_per_feature
    qzeros = tl.load(
        qzeros_ptr + ((qzero_ncols * groups) + (col_idx // elements_per_feature)),
        None,
        eviction_policy="evict_last",
    )
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq

    # Dequantize
    zeros = zeros + 1
    weights = weights - zeros
    weights = weights.to(tl.float32)
    weights = scales * weights

    tl.store(out_ptr + (x_index), weights, mask=xmask)


def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """

    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]

    out = torch.empty((infeatures, outfeatures), device="cuda", dtype=torch.float16)
    numels = out.numel()
    maxq = 2**bits - 1 if maxq is None else maxq
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731

    dequant_kernel_248[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        numels,
        maxq=maxq,
        bits=bits,
        outfeatures=outfeatures,
        num_groups=num_groups,
    )
    return out


def quant_matmul_248_stream(
    layers, input, qweight, scales, qzeros, g_idx, bits, maxq=None, transpose=True, stream1=torch.cuda.current_stream(),
        stream2=torch.cuda.current_stream(),
):
    events = [torch.cuda.Event() for _ in range(layers)]

    for i in range(layers):
        if i == 0:
            with torch.cuda.stream(stream1):
                W = dequant248(qweight[i], scales[i], qzeros[i], g_idx, bits, maxq=maxq)
                events[0].record()
            with torch.cuda.stream(stream2):
                events[0].wait(stream2)
                if transpose:
                    out = torch.matmul(input[i], W.t())
                else:
                    out = torch.matmul(input[i], W)
        else:
            with torch.cuda.stream(stream1):
                events[i-1].wait(stream1)
                W = dequant248(qweight[i], scales[i], qzeros[i], g_idx, bits, maxq=maxq)
                events[i].record()

            with torch.cuda.stream(stream2):
                events[i].wait(stream2)
                if transpose:
                    out = torch.matmul(input[i], W.t())
                else:
                    out = torch.matmul(input[i], W)
    return out

def quant_matmul_248(
    input, qweight, scales, qzeros, g_idx, bits, maxq=None, transpose=True
):

    W = dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq)

    if transpose:
        out = input @ W.t()
    else:
        out = input @ W
    return out

class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul_248(
                grad_output, qweight, scales, qzeros, g_idx, bits, maxq, transpose=True
            )
        return grad_input, None, None, None, None, None, None
