"""Custom Triton kernels for the scratch model. Linux/CUDA only (WSL here);
model.py imports this lazily so the Windows-side visualizer never touches it.

First (and so far only) kernel: fused LayerNorm forward.

Why: our hand-written LayerNorm runs as ~6 separate CUDA kernels (mean,
subtract, var, sqrt, divide, scale+shift), each paying a kernel launch and a
full read+write of the tensor through VRAM. The model calls LayerNorm 17
times per decode step — ~100 launches per generated token for normalization
alone, on a decode path we measured to be launch-overhead-bound. Fusing it
means ONE kernel: one read, statistics in registers, one write.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(
    X,                      # *input:  (M, N), row-major
    Y,                      # *output: (M, N)
    G,                      # *gamma:  (N,)
    B,                      # *beta:   (N,)
    stride,                 # elements between consecutive rows of X/Y
    N,                      # row width (n_embd)
    eps,
    BLOCK_N: tl.constexpr,  # power of 2 >= N; how many columns one program holds
):
    """One Triton 'program' = one row (one token's vector), whole row in registers.

    This is the block-level programming model: no threads, no warps in sight —
    we say 'load this masked slice, reduce it, store it' and Triton decides how
    the hardware does it.
    """
    row = tl.program_id(0)                       # which token am I?
    cols = tl.arange(0, BLOCK_N)                 # my column indices (padded)
    mask = cols < N                              # BLOCK_N is padded to a pow2

    # ONE read from VRAM. Compute in fp32 regardless of storage dtype:
    # variance in fp16 loses precision fast.
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # The entire LayerNorm math, register-resident — this is the "fusion":
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)         # zero the padding lanes
    var = tl.sum(diff * diff, axis=0) / N        # biased, matching our eager LN
    x_hat = diff / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)

    # ONE write back, in the output's storage dtype.
    tl.store(Y + row * stride + cols, (x_hat * g + b).to(Y.dtype.element_ty), mask=mask)


def fused_layernorm(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Drop-in replacement for our eager LayerNorm.forward (inference, CUDA)."""
    shape = x.shape
    x2 = x.contiguous().view(-1, shape[-1])      # (M, N): every token is a row
    M, N = x2.shape
    y = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(N)
    _layernorm_fwd[(M,)](                        # grid: one program per row
        x2, y, gamma, beta,
        x2.stride(0), N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 2048 else 8,
    )
    return y.view(shape)
