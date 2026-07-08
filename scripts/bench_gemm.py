"""Isolated GEMM microbenchmark: INT4 (GPTQ) kernels vs fp16, same shapes.

Why: our end-to-end serving numbers are overhead-bound, so quantization's
speed effect is invisible there. This times the matrix multiplies THEMSELVES —
the real INT4 QuantLinear modules from the quantized checkpoint vs the fp16
Conv1D modules from the original — across batch sizes M (rows = tokens being
processed at once: M=1 is single-stream decode, big M is prefill/batched).

Also includes one Llama-7B-class shape (K=4096, N=11008) timed at fp16 vs
fp32: if the M=1 time scales with weight BYTES (2x bytes -> ~2x time), the
GEMM is memory-bound — the regime where reading 4.5 bits/weight instead of 16
(a proper W4A16 kernel) must win roughly proportionally.

Method: CUDA events around each call, 50 warmup + 200 timed iters, medians.

Run in WSL:
  cd /mnt/c/Users/logot/Downloads/AIlearn && ~/vllm-env/bin/python -m scripts.bench_gemm
"""

from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
FP16_DIR = REPO / "checkpoints" / "hf-gpt2-md"
INT4_DIR = REPO / "checkpoints" / "hf-gpt2-md-gptq"

M_VALUES = [1, 8, 32, 256, 2048]
WARMUP, ITERS = 50, 200


def time_module(mod, x: torch.Tensor) -> float:
    """Median microseconds per forward call."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        mod(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start.record()
        mod(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    from transformers import AutoModelForCausalLM

    assert torch.cuda.is_available()
    fp16 = AutoModelForCausalLM.from_pretrained(FP16_DIR, dtype=torch.float16).cuda().eval()

    # Force an optimized fused-dequant kernel; gptqmodel's auto pick for these
    # small shapes is TorchAtenLinear (naive full dequant + matmul), which
    # would benchmark the fallback, not the technique.
    from gptqmodel import BACKEND, GPTQModel

    try:
        int4 = GPTQModel.load(str(INT4_DIR), backend=BACKEND.TRITON).model.cuda().eval()
        backend_note = "TRITON (forced)"
    except Exception as e:
        print(f"triton backend unavailable ({type(e).__name__}: {e}); using auto")
        int4 = AutoModelForCausalLM.from_pretrained(INT4_DIR).cuda().eval()
        backend_note = "auto"

    # the three matrix shapes in one of our transformer blocks
    layers = {
        "c_attn 512->1536": "transformer.h.0.attn.c_attn",
        "c_fc   512->2048": "transformer.h.0.mlp.c_fc",
        "c_proj 2048->512": "transformer.h.0.mlp.c_proj",
    }
    fp16_mods = dict(fp16.named_modules())
    int4_mods = dict(int4.named_modules())
    print(f"int4 kernel class: {type(int4_mods[next(iter(layers.values()))]).__name__} "
          f"[{backend_note}]\n")

    print(f"{'layer':<18} {'M':>5} {'fp16 us':>9} {'int4 us':>9} {'int4/fp16':>10}")
    with torch.no_grad():
        for label, name in layers.items():
            k = fp16_mods[name].weight.shape[0]  # Conv1D weight is (in, out)
            for m in M_VALUES:
                x = torch.randn(m, k, device="cuda", dtype=torch.float16)
                t_fp16 = time_module(fp16_mods[name], x)
                t_int4 = time_module(int4_mods[name], x)
                print(f"{label:<18} {m:>5} {t_fp16:>9.1f} {t_int4:>9.1f} {t_int4 / t_fp16:>9.2f}x")

    # --- the regime where W4A16 pays: a Llama-7B-class layer at decode (M=1).
    # If time scales with weight BYTES (fp32 vs fp16), the GEMM is memory-bound,
    # so a proper 4.5-bit kernel would win ~proportionally (~3.6x vs fp16).
    print("\nLlama-7B-class layer (K=4096, N=11008), fp16 vs fp32 = bytes test:")
    K, N = 4096, 11008
    with torch.no_grad():
        for m in (1, 32):
            x16 = torch.randn(m, K, device="cuda", dtype=torch.float16)
            x32 = x16.float()
            lin16 = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
            lin32 = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float32)
            t16 = time_module(lin16, x16)
            t32 = time_module(lin32, x32)
            gbps16 = (K * N * 2 / 2**30) / (t16 / 1e6)
            print(f"  M={m:<4} fp16 {t16:7.1f} us ({gbps16:5.0f} GB/s eff.) | "
                  f"fp32 {t32:7.1f} us | fp32/fp16 = {t32 / t16:.2f}x "
                  f"(2.0x => fully memory-bound)")


if __name__ == "__main__":
    main()
