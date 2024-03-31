import torch

import triton
import triton.language as tl


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
    zeros = tl.load(zeros_ptr + offs_zeros)

    # unpack weight and zeros
    weight = (weight >> shifter) & 0xF
    zeros = (zeros >> zeros_shifter) & 0xF
    zeros = (zeros + 1)

    # dequant weight
    weight = (weight - zeros) * scales

    # store the result
    offs_o = pid_k * stride_ok + offs_n * stride_on
    tl.store(Out + offs_o, weight)

@triton.jit
def _sym_dequant_kernel(
    Q, Out,
    scales_ptr,
    ZERO,
    stride_qk, stride_qn,
    stride_ok, stride_on,
    stride_scales_g, stride_scales_n,
    groupsize,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Dequant qweight to output matrix.
    Q is of shape (K//8, N) int32
    Out is of shape (K, N) float16
    scales is of shape (G, N) float16, where G is K // groupsize
    ZERO is 8, where 2 ** (bits-1) = 8
    """
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    gid = pid_k // groupsize

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # pointers
    offs_q = (pid_k // 8) * stride_qk + offs_n * stride_qn
    offs_scales = gid * stride_scales_g + offs_n * stride_scales_n

    # shifter
    shifter = (pid_k % 8) * 4

    # load
    weight = tl.load(Q + offs_q)
    scales = tl.load(scales_ptr + offs_scales)

    # unpack weight and zeros
    weight = (weight >> shifter) & 0xF

    # dequant weight
    weight = (weight - ZERO) * scales

    # store the result
    offs_o = pid_k * stride_ok + offs_n * stride_on
    tl.store(Out + offs_o, weight)


import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(group_A, group_B):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    )

    return group_C


group_m = [1024, 512, 256, 128]
group_n = [1024, 512, 256, 128]
group_k = [1024, 512, 256, 128]
group_A = []
group_B = []
assert len(group_m) == len(group_n)
assert len(group_n) == len(group_k)
group_size = len(group_m)
for i in range(group_size):
    M = group_m[i]
    N = group_n[i]
    K = group_k[i]
    A = torch.rand((M, K), device="cuda", dtype=torch.float16)
    B = torch.rand((K, N), device="cuda", dtype=torch.float16)
    group_A.append(A)
    group_B.append(B)

tri_out = group_gemm_fn(group_A, group_B)
ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
for i in range(group_size):
    assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)


# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
    )


def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['N'],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'],
        # label name for the lines
        line_names=["cuBLAS", "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(N, provider):
    group_size = 4
    group_A = []
    group_B = []
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((N, N), device="cuda", dtype=torch.float16)
        B = torch.rand((N, N), device="cuda", dtype=torch.float16)
        C = torch.empty((N, N), device="cuda", dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [N, N, N]
        g_lds += [N, N, N]

    d_a_ptrs = torch.tensor(A_addrs, device="cuda")
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
    return ms, max_ms, min_ms


benchmark.run(show_plots=True, print_data=True)


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

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 1, 4096, 4096
    group_size = 128

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    qweight = torch.randint(low=-2147483648, high=2147483647, size=(K//8, N), device='cuda', dtype=torch.int32)
    scales = torch.randn((K//group_size, N), device='cuda', dtype=torch.float16)
    qzeros = torch.randint(low=-2147483648, high=2147483647, size=(K//group_size, N//8), device='cuda', dtype=torch.int32)
    ref_weight = torch.randn((K, N), device='cuda', dtype=torch.float16)

    print("Zeropoint quantization.")
    triton_output = w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size, sym=False)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=False)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

    print("\nSymmetric quantization.")
    triton_output = w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size, sym=True)
    torch_output = torch_w4a16_matmul(a, qweight, scales, qzeros, group_size, sym=True)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}')

    # benchmark
    print(f"\nBenchmark with bs={M}.")
    print("fp16:", triton.testing.do_bench(lambda: torch.matmul(a, ref_weight)))
    print("torch zp:", triton.testing.do_bench(lambda: torch_w4a16_matmul(a, qweight, scales, qzeros, group_size)))
    print("triton zp:", triton.testing.do_bench(lambda: w4a16_matmul(a, ref_weight, qweight, scales, qzeros, group_size)))

