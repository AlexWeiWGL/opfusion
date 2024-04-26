import torch

import triton
import triton.language as tl
import awq_inference_engine
import dequant_gptq
import time


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    mid = tl.program_id(0)
    nid = tl.program_id(1)
    # Starting row + BLOCK_SIZE_M more rows

    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Starting col + BLOCK_SIZE_N more columns
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + a_rows[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(K // BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = c.to(tl.float16)

    # C's block's offsets
    c_ptrs = a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptr + c_ptrs, c)


def gemm(a, b):
    c = torch.empty([M, N], device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    gemm_kernel[grid](a, b, c, M, N, K)
    return c


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
    qweight = torch.repeat_interleave(qweight, dim=0, repeats=8)  # (K//8, N) -> (K, N)
    K = qweight.shape[0]
    shifter = torch.arange(0, K, device=qweight.device, dtype=torch.int32).reshape(-1, 1)  # (K, 1)
    shifter = (shifter % 8) * 4
    qweight = (qweight >> shifter) & 0xF
    # unpack qzeros and scales
    if sym:
        qzeros = 8
    else:
        qzeros = torch.repeat_interleave(qzeros, dim=1, repeats=8)  # (K/g, N/8) -> (K/g, N)
        N = qzeros.shape[1]
        shifter = torch.arange(0, N, device=qzeros.device, dtype=torch.int32).reshape(1, -1)  # (1, N)
        shifter = (shifter % 8) * 4
        qzeros = (qzeros >> shifter) & 0xF
        qzeros = qzeros + 1
        qzeros = torch.repeat_interleave(qzeros, dim=0, repeats=group_size)  # (K/g, N) -> (K, N)
    scales = torch.repeat_interleave(scales, dim=0, repeats=group_size)  # (K/g, N) -> (K, N)
    # dequant and matmul
    weight = (qweight - qzeros) * scales
    output = torch.matmul(x, weight.to(x.dtype))
    return output


def w4a16_matmul(x, w, qweight, scales, qzeros, group_size):
    block_size_n = 128
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


def gptq_stream_matmul(layers, a, qweight, scales, qzeros, group_size, stream1, stream2):

    g_idx = torch.tensor([i // group_size for i in range(4096)], device='cuda')
    dequant_gptq.quant_matmul_248_stream(layers, a, qweight, scales, qzeros, g_idx, 4, stream1=stream1, stream2=stream2)


def gptq_matmul(layers, a, qweight, scales, qzeros, group_size):
    g_idx = torch.tensor([i // group_size for i in range(4096)], device='cuda')

    for i in range(layers):
        dequant_gptq.quant_matmul_248(a[i], qweight[i], scales[i], qzeros[i], g_idx, 4)

def torch_matmul(layers, a, qweight):
    for z in range(layers):
        torch.matmul(a[z], qweight[z])
    return 0

def triton_matmul(a, ref_weight, qweight, scales, qzeros, group_size, stream1, stream2):
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


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 1000, 4096, 4096
    group_size = 128

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    qweight = torch.randint(low=-2147483648, high=2147483647, size=(K // 8, N), device='cuda', dtype=torch.int32)
    scales = torch.randn((K // group_size, N), device='cuda', dtype=torch.float16)
    qzeros = torch.randint(low=-2147483648, high=2147483647, size=(K // group_size, N // 8), device='cuda',
                           dtype=torch.int32)
    ref_weight = torch.randn((K, N), device='cuda', dtype=torch.float16)

    stream1 = torch.cuda.Stream('cuda')
    stream2 = torch.cuda.current_stream('cuda')

    # 生成一个模拟用的网络，模拟pipeline
    layers = 50
    inputs = []
    qweights = []
    scaless = []
    qzeross = []
    ref_weights = []
    for i in range(layers):
        inputs.append(torch.randn((M, K), device='cuda', dtype=torch.float16))
        qweights.append(
            torch.randint(low=-2147483648, high=2147483647, size=(K // 8, N), device='cuda', dtype=torch.int32))
        scaless.append(torch.randn((K // group_size, N), device='cuda', dtype=torch.float16))
        qzeross.append(torch.randint(low=-2147483648, high=2147483647, size=(K // group_size, N // 8), device='cuda',
                                     dtype=torch.int32))
        ref_weights.append(torch.randn((K, N), device='cuda', dtype=torch.float16))

    # print(qweight.short())
    # print(qzeros.to(torch.float16))
    #
    # print("Zeropoint quantization.")
    # awq_output = awq_inference_engine.gemm_forward_cuda_new(a, qweight.short(), scales, qzeros)
    # triton_output = w4a16_matmul(a, ref_weight, qweight.to(torch.int32), scales, qzeros, group_size, sym=False)
    # torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros.to(torch.int32), group_size, sym=False)
    # print(f"triton_output={triton_output}")
    # print(f"torch_output={torch_output}")
    # print(f"awq_output={awq_output}")
    # print(f'The maximum difference between torch and triton is {torch.max(torch.abs(awq_output - triton_output))}')

    # benchmark
    print(f"\nBenchmark with bs={M}.")
    print("torch zp:", triton.testing.do_bench(lambda: torch_w4a16_matmul(a, qweight, scales, qzeros, group_size)))
    print("triton zp:", triton.testing.do_bench(
        lambda: triton_matmul(a, ref_weight, qweight, scales, qzeros, group_size, stream1, stream2)))
    print("fp16", triton.testing.do_bench(lambda: torch.matmul(a, ref_weight)))
    print("gptq_stream",
          triton.testing.do_bench(lambda: gptq_stream_matmul(layers, inputs, qweights, scaless, qzeross, group_size, stream1, stream2)))
    print("GPTQ", triton.testing.do_bench(lambda: gptq_matmul(layers, inputs, qweights, scaless, qzeross, group_size)))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M'],
            x_vals=[128 * i for i in range(10, 31)],
            line_arg='provider',
            line_vals=[ 'fp16', 'gptq', 'gptq_stream'],
            line_names=['FP16', 'GPTQ', 'GPTQ-Stream'],
            styles=[('blue', '-'), ('red', '-'), ('green', '-')],
            ylabel = 'TIME(ms)',
            plot_name='different matmal methods',
            args={},
        )
    )

    def benckmark(M, provider):
        ms = 1
        max_ms = 1
        min_ms = 1

        layers = 50
        inputs = []
        qweights = []
        scaless = []
        qzeross = []

        for j in range(layers):
            inputs.append(torch.randn((M, K), device='cuda', dtype=torch.float16))
            qweights.append(
                torch.randint(low=-2147483648, high=2147483647, size=(K // 8, N), device='cuda', dtype=torch.int32))
            scaless.append(torch.randn((K // group_size, N), device='cuda', dtype=torch.float16))
            qzeross.append(
                torch.randint(low=-2147483648, high=2147483647, size=(K // group_size, N // 8), device='cuda',
                              dtype=torch.int32))

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'gptq_stream':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: gptq_stream_matmul(layers, inputs, qweights, scaless, qzeross, group_size, stream1, stream2), quantiles=quantiles)
        if provider == 'gptq':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: gptq_matmul(layers, inputs, qweights, scaless, qzeross, group_size), quantiles=quantiles
            )
        if provider == 'fp16':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_matmul(layers, inputs, ref_weights), quantiles=quantiles
            )
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return ms, min_ms, max_ms
    benckmark.run(show_plots=True, print_data=True)

#
#
