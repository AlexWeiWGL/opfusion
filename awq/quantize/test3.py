import torch

import triton
import triton.language as tl
import awq_inference_engine
from matplotlib import pyplot as plt

import dequant_gptq
import naive_dequant_gptq

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    mid = tl.program_id(0)
    nid = tl.program_id(1)
    # Starting row + BLOCK_SIZE_M more rows

    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Starting col + BLOCK_SIZE_N more columns
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + a_rows[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(K//BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = c.to(tl.float16)

    # C's block's offsets
    c_ptrs = a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptr+ c_ptrs, c)


# def gemm(a, b):
#     c = torch.empty([M, N], device=a.device, dtype=a.dtype)
#     grid = lambda META: (
#         triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']),
#     )
#     gemm_kernel[grid](a, b, c, M, N, K)
#     return c

@triton.jit
def _zp_dequant_kernel(
    Q, Out,
    scales_ptr, zeros_ptr,
    stride_qk, stride_qn,
    stride_ok, stride_on,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Dequant qweight to output matrix.
    Q is of shape (K//8, N) int32
    Out is of shape (K, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    zeros is of shape (G, N//8) int32
    """
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    gid = pid_k // groupsize

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # pointers
    offs_q = (pid_k // 8) * stride_qk + offs_n * stride_qn
    offs_scales = gid * stride_scales_g + offs_n * stride_scales_n
    offs_zeros = gid * stride_zeros_g + (offs_n // 8) * stride_zeros_n

    # shifter
    shifter = (pid_k % 8) * 4
    zeros_shifter = (offs_n % 8) * 4

    # load
    weight = tl.load(Q + offs_q)
    scales = tl.load(scales_ptr + offs_scales)
    zeros = tl.load(zeros_ptr + offs_zeros).to(dtype=tl.int32)

    # unpack weight and zeros
    weight = (weight >> shifter) & 0xF
    zeros = (zeros >> zeros_shifter) & 0xF
    zeros = (zeros + 1)

    # dequant weight
    weight = (weight - zeros) * scales

    # store the result
    offs_o = pid_k * stride_ok + offs_n * stride_on
    tl.store(Out + offs_o, weight)

def torch_w4a16_matmul(x, qweight, scales, qzeros, group_size, sym=False):
    # unpack qweight
    qweight = torch.repeat_interleave(qweight, dim=0, repeats=8)  #(K//8, N) -> (K, N)
    K = qweight.shape[0]
    shifter = torch.arange(0, K, device=qweight.device, dtype=torch.int32).reshape(-1, 1) #(K, 1)
    shifter = (shifter % 8) * 4
    qweight = (qweight >> shifter) & 0xF
    # unpack qzeros and scales
    if sym:
        qzeros = 8
    else:
        qzeros = torch.repeat_interleave(qzeros, dim=1, repeats=8) #(K/g, N/8) -> (K/g, N)
        N = qzeros.shape[1]
        shifter = torch.arange(0, N, device=qzeros.device, dtype=torch.int32).reshape(1, -1) #(1, N)
        shifter = (shifter % 8) * 4
        qzeros = (qzeros >> shifter) & 0xF
        qzeros = qzeros + 1
        qzeros = torch.repeat_interleave(qzeros, dim=0, repeats=group_size) #(K/g, N) -> (K, N)
    scales = torch.repeat_interleave(scales, dim=0, repeats=group_size) #(K/g, N) -> (K, N)
    # dequant and matmul
    weight = (qweight - qzeros) * scales
    output = torch.matmul(x, weight.to(x.dtype))
    return output


def w4a16_matmul(x, w, qweight, scales, qzeros, group_size):
    block_size_n=128
    K = x.shape[1]
    N = qweight.shape[1]

    # shape constraints
    assert x.shape[-1] == (qweight.shape[0] * 8), "Incompatible dimensions"
    assert x.shape[-1] == w.shape[0], "Incompatible dimensions"
    assert w.shape[-1] == qweight.shape[-1], "Incompatible dimensions"
    assert K % group_size == 0, "K must be a multiple of group size"
    assert N % block_size_n == 0, "N must be a multiple of block_size_n"

    grid = (K, N // block_size_n)

    # dequant qweight to w

    _zp_dequant_kernel[grid](
        qweight, w,
        scales, qzeros,
        qweight.stride(0), qweight.stride(1),
        w.stride(0), w.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
        block_size_n,
        num_warps=2, num_stages=4,
    )
    c = torch.matmul(x, w)
    return c


N, K = 4096, 4096
batch_size = [64*i for i in range(10, 51)]
inputs = [torch.randn((M, K), device='cuda', dtype=torch.float16) for M in batch_size]
input_layers = []
layers = 10

for M in batch_size:
    for j in range(layers):
      input_layers.append(torch.randn((M, K), device='cuda', dtype=torch.float16))



def evaluate_kernel(inputs):
    def record_time(func):
        def wrapper(*args, **kwargs):
            for _ in range(10):
                func(*args, **kwargs)

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(batch_size))]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(batch_size))]

            for i in range(len(inputs)):
                new_args = list(args)
                new_args[0] = inputs[i]
                start_events[i].record()
                func(*new_args, **kwargs)
                end_events[i].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            return times
        return wrapper
    return record_time


def evaluate_layers_kernel(inputs, layers):
    def record_layer_time(func):
        def wrapper(*args, **kwargs):
            for j in range(10):
                new_args = list(args)
                new_args[1] = inputs[j * layers: (j + 1) * layers]
                func(*new_args, **kwargs)

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(batch_size))]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(batch_size))]

            for i in range(len(batch_size)):
                new_args = list(args)
                new_args[1] = inputs[i*layers: (i+1)*layers]
                start_events[i].record()
                func(*new_args, **kwargs)
                end_events[i].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            return times
        return wrapper
    return record_layer_time

@evaluate_kernel(inputs=inputs)
def gptq_stream_matmul(a, qweight, scales, qzeros, group_size, stream1, stream2):
    g_idx = torch.tensor([i//group_size for i in range(4096)], device='cuda')
    naive_dequant_gptq.quant_matmul_248_stream(a, qweight, scales, qzeros, g_idx, 4, stream1=stream1, stream2=stream2)


@evaluate_kernel(inputs=inputs)
def gptq_matmul(a, qweight, scales, qzeros, group_size):
    g_idx = torch.tensor([i//group_size for i in range(4096)], device='cuda')
    dequant_gptq.quant_matmul_248(a, qweight, scales, qzeros, g_idx, 4)

@evaluate_kernel(inputs=inputs)
def gptq_matmul_withoutSyn(a, ref_weight, qweight, scales, qzeros, group_size, stream1, stream2):
    g_idx = torch.tensor([i // group_size for i in range(4096)], device='cuda')
    naive_dequant_gptq.quant_matmul_248_stream_withoutSyn(a, ref_weight, qweight, scales, qzeros, g_idx, 4, stream1=stream1, stream2=stream2)


@evaluate_kernel(inputs=inputs)
def triton_matmul(a,ref_weight, qweight, scales, qzeros, group_size, stream1, stream2):
    block_size_n = 128
    K = a.shape[1]
    N = qweight.shape[1]

    # shape constraints
    assert a.shape[-1] == (qweight.shape[0] * 8), "Incompatible dimensions"
    assert a.shape[-1] == ref_weight.shape[0], "Incompatible dimensions"
    assert ref_weight.shape[-1] == qweight.shape[-1], "Incompatible dimensions"
    assert K % group_size == 0, "K must be a multiple of group size"
    assert N % block_size_n == 0, "N must be a multiple of block_size_n"

    grid = (K, N // block_size_n)

    with torch.cuda.stream(stream1):
        _zp_dequant_kernel[grid](
            qweight, ref_weight,
            scales, qzeros,
            qweight.stride(0), qweight.stride(1),
            ref_weight.stride(0), ref_weight.stride(1),
            scales.stride(0), scales.stride(1),
            qzeros.stride(0), qzeros.stride(1),
            group_size,
            block_size_n,
            num_warps=2, num_stages=4
        )

    with torch.cuda.stream(stream2):
        torch.matmul(a, ref_weight)
    stream2.wait_stream(stream1)
    torch.cuda.synchronize()

@evaluate_layers_kernel(inputs=input_layers, layers=layers)
def gptq_stream_matmul_layers(layers, a, qweight, scales, qzeros, group_size, stream1, stream2):

    g_idx = torch.tensor([i // group_size for i in range(4096)], device='cuda')
    dequant_gptq.quant_matmul_248_stream(layers, a, qweight, scales, qzeros, g_idx, 4, stream1=stream1, stream2=stream2)


@evaluate_layers_kernel(inputs=input_layers, layers=layers)
def gptq_matmul_layers(layers, a, qweight, scales, qzeros, group_size):
    for i in range(layers):
        dequant_gptq.quant_matmul_248(a, qweight, scales, qzeros, group_size, 4)

# @evaluate_kernel(10)
# def gptq_dequant_kernel(qweight, scales, qzeros, g_idx, bits, maxq=None):
#     naive_dequant_gptq.dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq)


@evaluate_kernel(inputs=inputs)
def fp16_kernel(mat_inputs, weight):
    torch.matmul(mat_inputs, weight)
    torch.cuda.synchronize()

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 2000, 4096, 4096
    group_size = 128

    qweights = []
    scaless = []
    qzeross = []

    for j in range(layers):
        qweights.append(
            torch.randint(low=-2147483648, high=2147483647, size=(K // 8, N), device='cuda', dtype=torch.int32))
        scaless.append(torch.randn((K // group_size, N), device='cuda', dtype=torch.float16))
        qzeross.append(
            torch.randint(low=-2147483648, high=2147483647, size=(K // group_size, N // 8), device='cuda',
                          dtype=torch.int32))

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    qweight = torch.randint(low=-2147483648, high=2147483647, size=(K // 8, N), device='cuda', dtype=torch.int32)
    scales = torch.randn((K // group_size, N), device='cuda', dtype=torch.float16)
    qzeros = torch.randint(low=-2147483648, high=2147483647, size=(K // group_size, N // 8), device='cuda',
                           dtype=torch.int32)
    ref_weight = torch.randn((K, N), device='cuda', dtype=torch.float16)

    stream1 = torch.cuda.Stream('cuda')
    stream2 = torch.cuda.current_stream('cuda')
    stream3 = torch.cuda.Stream('cuda')
    g_idx = torch.tensor([i // group_size for i in range(4096)], device='cuda')

    # gptq_matmul_result = gptq_matmul(a, qweight, scales, qzeros, group_size)
    # gptq_stream_matmul_result = gptq_stream_matmul(a, qweight, scales, qzeros, group_size, stream1, stream2)
    # gptq_matmul_withoutSyn_result = gptq_matmul_withoutSyn(a, ref_weight, qweight, scales, qzeros, group_size, stream3, stream2)
    # fp16_result = fp16_kernel(a, ref_weight)
    gptq_matmul_result = gptq_stream_matmul_layers(layers,input_layers, qweights, scaless, qzeross, group_size, stream1, stream2)

    print("gptq matmul: ", gptq_matmul_result)
    # print("gptq stream matmul", gptq_stream_matmul_result)
    # print("gptq matmul withoutSyn:", gptq_matmul_withoutSyn_result)
    # # print("gptq dequant kernel", gptq_dequant_kernel(qweight, scales, qzeros, g_idx, 4))
    # print("fp16 kernel", fp16_result)
    # fig, ax = plt.subplots()
    # ax.plot(batch_size, gptq_matmul_result, color='red', linestyle='-', label='gptq matmul')
    # ax.plot(batch_size, gptq_stream_matmul_result, color='blue', linestyle='-', label='gptq matmul with stream')
    # ax.plot(batch_size, gptq_matmul_withoutSyn_result, color='black', linestyle='-', label='gptq matmul without synchronize')
    # ax.plot(batch_size, fp16_result, color='green', linestyle='-', label='fp16')
    # ax.legend()
    # plt.show()
