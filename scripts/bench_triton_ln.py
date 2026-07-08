"""Benchmark the fused Triton LayerNorm: correct first, then fast, then end-to-end.

Three sections:
  1. CORRECTNESS — max abs diff vs our eager LayerNorm across shapes/dtypes.
     A wrong kernel that's fast is worthless; this gate comes first.
  2. KERNEL MICROBENCH — eager hand-LN (~6 kernels) vs torch F.layer_norm
     (PyTorch's own fused kernel — the honest yardstick) vs our Triton kernel,
     across row counts M (M=1 is one decode token; M=8192 is training-scale).
  3. END-TO-END — 250-token cached greedy decode of the 25M model with
     LayerNorm.use_fused off vs on. Our decode path is launch-bound and calls
     LayerNorm 17x per step (~100 launches/token just for LN), so this is
     where a fused kernel should show up in a real latency number.

Run in WSL:
  cd /mnt/c/Users/logot/Downloads/AIlearn && ~/vllm-env/bin/python -m scripts.bench_triton_ln
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.inference import load_model
from src.kernels import fused_layernorm
from src.model import LayerNorm

REPO = Path(__file__).resolve().parents[1]
WARMUP, ITERS = 50, 200


def time_fn(fn) -> float:
    """Median microseconds per call (CUDA events)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    # --- 1. correctness ---------------------------------------------------
    print("correctness vs eager LayerNorm:")
    ln = LayerNorm(512).cuda().eval()
    with torch.no_grad():
        ln.gamma.normal_(1.0, 0.2)  # non-trivial params so scale/shift is tested
        ln.beta.normal_(0.0, 0.2)
        worst = {}
        for shape in [(1, 1, 512), (4, 256, 512), (32, 256, 512)]:
            for dtype in (torch.float32, torch.float16):
                x = torch.randn(*shape, device="cuda", dtype=dtype) * 3 + 1
                ref = ln(x.float())  # eager fp32 = ground truth
                out = fused_layernorm(x, ln.gamma, ln.beta, ln.eps).float()
                worst[dtype] = max(worst.get(dtype, 0), (ref - out).abs().max().item())
        for dtype, diff in worst.items():
            print(f"  {str(dtype):<15} max abs diff {diff:.2e}")
        assert worst[torch.float32] < 1e-5, "fp32 kernel mismatch — fix before benchmarking"

    # --- 2. kernel microbench ----------------------------------------------
    N = 512
    print(f"\nkernel microbench (N={N}, fp32, median us):")
    print(f"{'M rows':>8} {'eager us':>10} {'torch us':>10} {'triton us':>10} "
          f"{'vs eager':>9} {'vs torch':>9}")
    w = torch.randn(N, device="cuda").abs() + 0.5
    b = torch.randn(N, device="cuda")
    for M in (1, 32, 256, 8192):
        x = torch.randn(M, N, device="cuda")
        t_eager = time_fn(lambda: ln(x))
        t_torch = time_fn(lambda: F.layer_norm(x, (N,), w, b, 1e-5))
        t_triton = time_fn(lambda: fused_layernorm(x, w, b, 1e-5))
        print(f"{M:>8} {t_eager:>10.1f} {t_torch:>10.1f} {t_triton:>10.1f} "
              f"{t_eager / t_triton:>8.1f}x {t_torch / t_triton:>8.1f}x")

    # --- 3. end-to-end: cached decode with fused LN off vs on --------------
    print("\nend-to-end: 250-token cached greedy decode, 25M model, CUDA:")
    model, tok, cfg = load_model(REPO / "checkpoints" / "gpt-md.pt")
    model.cuda().eval()
    ids = tok.encode("ROMEO:")
    n_new = cfg["block_size"] - len(ids)

    @torch.no_grad()
    def decode() -> list[int]:
        out = list(ids)
        logits, caches = model.forward_step(torch.tensor([out], device="cuda"), None, 0)
        for t in range(n_new):
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            logits, caches = model.forward_step(
                torch.tensor([[nxt]], device="cuda"), caches, len(out) - 1
            )
        return out

    results = {}
    for fused in (False, True):
        LayerNorm.use_fused = fused
        decode()  # warmup (incl. Triton JIT compile on first fused call)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results[fused] = decode()
        torch.cuda.synchronize()
        results[f"time_{fused}"] = time.perf_counter() - t0
    LayerNorm.use_fused = False  # leave the global as we found it

    same = results[False] == results[True]
    off, on = results["time_False"], results["time_True"]
    print(f"  greedy tokens identical: {same}")
    print(f"  eager LN : {off:.2f}s  ({off / n_new * 1e3:.2f} ms/tok)")
    print(f"  fused LN : {on:.2f}s  ({on / n_new * 1e3:.2f} ms/tok)")
    print(f"  end-to-end speedup: {off / on:.2f}x "
          f"(17 LN calls/step went from ~6 kernels each to 1)")


if __name__ == "__main__":
    main()
