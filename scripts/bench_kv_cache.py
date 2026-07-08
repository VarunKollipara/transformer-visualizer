"""Measure the KV cache: correctness first, then per-token latency vs context.

Two claims to verify about GPT.generate_cached vs GPT.generate:

  1. CORRECTNESS: caching is a pure reorganization of the same math — greedy
     decoding must produce IDENTICAL tokens, and the last-position logits must
     match to float noise at every step. If they don't, the cache is a bug,
     not an optimization.

  2. SCALING: uncached per-token cost grows with context length T (the whole
     prefix is recomputed every step: O(T) MLP work + O(T^2) attention);
     cached per-token cost should stay ~flat (one token through the model +
     one attention row against the cache).

We measure per-token latency in position buckets on both CPU and GPU for the
25M model. Expect the textbook curve on CPU (compute-bound). On GPU expect the
gap to shrink: our Python-loop-over-heads model launches hundreds of tiny
kernels per step, so small-batch steps are launch-overhead-bound — the same
overhead-vs-compute regime story as every other measurement in this project.

Run in WSL:
  cd /mnt/c/Users/logot/Downloads/AIlearn && ~/vllm-env/bin/python -m scripts.bench_kv_cache
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.inference import load_model

REPO = Path(__file__).resolve().parents[1]
PROMPT = "ROMEO:"
BUCKETS = [(17, 64), (65, 128), (129, 192), (193, 256)]  # context length ranges


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_uncached(model, ids: list[int], n_new: int, device: str) -> tuple[list[int], list[float], list[torch.Tensor]]:
    """Greedy decode recomputing the full prefix each step; time each step."""
    ids = list(ids)
    times, logit_snaps = [], []
    for _ in range(n_new):
        sync(device); t0 = time.perf_counter()
        idx = torch.tensor([ids], device=device)
        logits, _ = model(idx)
        last = logits[0, -1]
        sync(device); times.append(time.perf_counter() - t0)
        logit_snaps.append(last.float().cpu())
        ids.append(int(last.argmax()))
    return ids, times, logit_snaps


@torch.no_grad()
def greedy_cached(model, ids: list[int], n_new: int, device: str) -> tuple[list[int], list[float], list[torch.Tensor], float]:
    """Greedy decode with the KV cache; time prefill and each decode step."""
    ids = list(ids)
    sync(device); t0 = time.perf_counter()
    idx = torch.tensor([ids], device=device)
    logits, caches = model.forward_step(idx, None, pos_start=0)
    sync(device); prefill_s = time.perf_counter() - t0

    times, logit_snaps = [], []
    for t in range(n_new):
        last = logits[0, -1]
        logit_snaps.append(last.float().cpu())
        next_id = int(last.argmax())
        ids.append(next_id)
        sync(device); t0 = time.perf_counter()
        step = torch.tensor([[next_id]], device=device)
        logits, caches = model.forward_step(step, caches, pos_start=len(ids) - 1)
        sync(device); times.append(time.perf_counter() - t0)
    return ids, times, logit_snaps, prefill_s


def bucket_means(times: list[float], t0_len: int) -> dict[str, float]:
    """Mean ms per token, bucketed by the context length at that step."""
    out = {}
    for lo, hi in BUCKETS:
        vals = [
            t for i, t in enumerate(times)
            if lo <= t0_len + i + 1 <= hi  # context length when producing token i
        ]
        if vals:
            out[f"T={lo}-{hi}"] = sum(vals) / len(vals) * 1e3
    return out


def run(device: str) -> dict:
    model, tok, cfg = load_model(REPO / "checkpoints" / "gpt-md.pt")
    model.to(device).eval()
    ids = tok.encode(PROMPT)
    n_new = cfg["block_size"] - len(ids)  # fill the whole context: 250 tokens

    ids_u, times_u, logits_u = greedy_uncached(model, ids, n_new, device)
    ids_c, times_c, logits_c, prefill_s = greedy_cached(model, ids, n_new, device)

    # --- correctness ---
    assert ids_u == ids_c, "cached greedy diverged from uncached — cache bug!"
    worst = max((a - b).abs().max().item() for a, b in zip(logits_u, logits_c))

    kv_bytes = 2 * cfg["num_layers"] * cfg["n_embd"] * 4  # K+V, fp32, per token
    result = {
        "device": device,
        "tokens_match": True,
        "max_logit_diff": worst,
        "prefill_ms": prefill_s * 1e3,
        "uncached_ms_per_tok": bucket_means(times_u, len(ids)),
        "cached_ms_per_tok": bucket_means(times_c, len(ids)),
        "uncached_total_s": sum(times_u),
        "cached_total_s": sum(times_c) + prefill_s,
        "kv_cache_bytes_per_token": kv_bytes,
        "kv_cache_mib_full_context": kv_bytes * cfg["block_size"] / 2**20,
    }

    print(f"\n=== {device} ===  (greedy tokens identical: True, "
          f"max logit diff {worst:.2e})")
    print(f"{'context bucket':<14} {'uncached ms/tok':>16} {'cached ms/tok':>14} {'speedup':>9}")
    for key in result["uncached_ms_per_tok"]:
        u = result["uncached_ms_per_tok"][key]
        c = result["cached_ms_per_tok"][key]
        print(f"{key:<14} {u:>16.2f} {c:>14.2f} {u / c:>8.1f}x")
    print(f"total for {n_new} tokens: uncached {result['uncached_total_s']:.2f}s | "
          f"cached {result['cached_total_s']:.2f}s (incl. {prefill_s * 1e3:.1f}ms prefill) | "
          f"{result['uncached_total_s'] / result['cached_total_s']:.1f}x")
    print(f"KV cache size: {kv_bytes / 1024:.0f} KiB/token, "
          f"{result['kv_cache_mib_full_context']:.1f} MiB at full context (B=1, fp32)")
    return result


def main() -> None:
    results = [run("cpu")]
    if torch.cuda.is_available():
        results.append(run("cuda"))
    dest = REPO / "benchmarks" / "kv-cache.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
