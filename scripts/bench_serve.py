"""Benchmark harness for the vLLM server: latency, throughput, memory.

Measures, per concurrency level:
  - TTFT   (time to first token)  — prefill + queueing, what "responsive" means
  - ITL    (inter-token latency)  — decode-phase gap between streamed tokens
  - throughput                    — aggregate output tokens/sec across streams
  - GPU memory                    — nvidia-smi snapshot (note: vLLM PRE-allocates
                                    its KV-cache pool, so this reflects the pool
                                    size, not live usage)

Method notes (the parts that make numbers trustworthy):
  - Streaming API with client-side perf_counter timestamps per SSE token event.
    ITL therefore includes detokenize+HTTP overhead — that's intentional; it's
    what a real client experiences.
  - Fixed prompt_len/gen_len with ignore_eos=True so every request does exactly
    the same work; otherwise early-EOS requests would flatter the numbers.
  - Prompts sampled from the training corpus at random offsets (deterministic
    seed) so prefill sees realistic text, not one cached prompt.
  - Warmup requests before each level, excluded from stats.
  - Percentiles (p50/p99), not just means — tails are where scheduling pain lives.

Run inside WSL (client next to server, avoiding the Windows<->WSL NAT hop):
  ~/vllm-env/bin/python scripts/bench_serve.py --label fp32-baseline

Results land in benchmarks/<label>.json plus a printed markdown table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import aiohttp

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "input.txt"


def pctl(values: list[float], p: float) -> float:
    """Nearest-rank percentile; fine at our sample sizes."""
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def gpu_memory_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def sample_prompts(n: int, prompt_len: int, seed: int) -> list[str]:
    text = CORPUS.read_text(encoding="utf-8")
    rng = random.Random(seed)
    return [
        text[i : i + prompt_len]
        for i in (rng.randrange(0, len(text) - prompt_len) for _ in range(n))
    ]


async def one_request(
    session: aiohttp.ClientSession, url: str, model: str, prompt: str, gen_len: int
) -> dict:
    """Stream one completion; timestamp every token event."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_len,
        "temperature": 1.0,
        "ignore_eos": True,            # fixed work per request
        "skip_special_tokens": False,  # '\n' is a real character here
        "stream": True,
    }
    times: list[float] = []
    n_tokens = 0
    start = time.perf_counter()
    async with session.post(url, json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk["choices"] and chunk["choices"][0].get("text"):
                times.append(time.perf_counter())
                n_tokens += 1
    return {
        "ttft": times[0] - start if times else float("nan"),
        "itls": [b - a for a, b in zip(times, times[1:])],
        "n_tokens": n_tokens,
        "start": start,
        "end": times[-1] if times else start,
    }


async def run_level(
    base: str, model: str, concurrency: int, num_requests: int,
    prompt_len: int, gen_len: int, seed: int,
) -> dict:
    url = f"{base}/v1/completions"
    prompts = sample_prompts(num_requests, prompt_len, seed)
    sem = asyncio.Semaphore(concurrency)  # closed-loop load: keep exactly N in flight

    async with aiohttp.ClientSession() as session:
        # warmup (not measured): let the scheduler/caches settle
        await asyncio.gather(*(
            one_request(session, url, model, p, gen_len)
            for p in sample_prompts(min(concurrency, 8), prompt_len, seed + 1)
        ))

        async def bounded(p: str) -> dict:
            async with sem:
                return await one_request(session, url, model, p, gen_len)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(bounded(p) for p in prompts))
        wall = time.perf_counter() - t0

    ttfts = [r["ttft"] for r in results]
    itls = [x for r in results for x in r["itls"]]
    total_tokens = sum(r["n_tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_s": round(wall, 3),
        "ttft_ms": {"p50": pctl(ttfts, 50) * 1e3, "p99": pctl(ttfts, 99) * 1e3},
        "itl_ms": {
            "mean": statistics.mean(itls) * 1e3,
            "p50": pctl(itls, 50) * 1e3,
            "p99": pctl(itls, 99) * 1e3,
        },
        "output_tok_per_s": total_tokens / wall,
        "req_per_s": num_requests / wall,
        "gpu_mem_mib": gpu_memory_mib(),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. fp32-baseline; names the output file")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="char-gpt")
    ap.add_argument("--concurrency", default="1,8,32,128")
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--gen-len", type=int, default=48)  # 16 + 48 = 64 = n_positions
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    levels = [int(c) for c in args.concurrency.split(",")]
    runs = []
    for c in levels:
        n = min(512, max(64, c * 8))  # enough requests to see steady state + tails
        r = await run_level(args.base, args.model, c, n, args.prompt_len, args.gen_len, args.seed)
        runs.append(r)
        print(
            f"conc={c:>4}  n={n:>3}  ttft p50/p99 = {r['ttft_ms']['p50']:6.1f}/{r['ttft_ms']['p99']:6.1f} ms  "
            f"itl p50/p99 = {r['itl_ms']['p50']:5.2f}/{r['itl_ms']['p99']:5.2f} ms  "
            f"tok/s = {r['output_tok_per_s']:8.0f}  mem = {r['gpu_mem_mib']} MiB",
            flush=True,
        )

    out = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workload": {
            "prompt_len": args.prompt_len, "gen_len": args.gen_len,
            "temperature": 1.0, "seed": args.seed,
        },
        "runs": runs,
    }
    dest = REPO / "benchmarks" / f"{args.label}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")

    # markdown table for OPTIMIZATION_LOG.md
    print(f"\n| conc | TTFT p50 (ms) | TTFT p99 | ITL p50 (ms) | ITL p99 | out tok/s | req/s |")
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        print(
            f"| {r['concurrency']} | {r['ttft_ms']['p50']:.1f} | {r['ttft_ms']['p99']:.1f} "
            f"| {r['itl_ms']['p50']:.2f} | {r['itl_ms']['p99']:.2f} "
            f"| {r['output_tok_per_s']:.0f} | {r['req_per_s']:.1f} |"
        )


if __name__ == "__main__":
    asyncio.run(main())
