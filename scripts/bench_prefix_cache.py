"""Prefix caching: measure vLLM reusing KV cache blocks across requests.

The idea: vLLM hashes the KV cache in fixed 16-token blocks. If a new request's
prompt starts with token-blocks it has already computed (a shared system
prompt, a document being asked many questions), those blocks are reused and
prefill skips them — the request only pays prefill for its unique tail. This
is KV-cache *reuse* (vs. our scratch implementation, which is KV-cache
*existence*).

Design: three phases, same prompt length (208 tokens) and gen length (16):
  A. shared  — every request = one fixed 192-char prefix + unique 16-char tail
  B. unique  — every request = a distinct 208-char corpus window
Run A and B against a server with prefix caching ON (vLLM V1 default), read
the server's own hit counters from /metrics before and after each phase, and
compare TTFT. Then rerun A against a server started with
--no-enable-prefix-caching to attribute any TTFT difference.

Run in WSL:
  ~/vllm-env/bin/python scripts/bench_prefix_cache.py --label prefix-on
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import urllib.request
from pathlib import Path

import aiohttp

from scripts.bench_serve import one_request  # reuse the streaming TTFT measurement

REPO = Path(__file__).resolve().parents[1]
N = 64
PREFIX_LEN = 192   # 12 full 16-token KV blocks — the cacheable part
TAIL_LEN = 16
GEN_LEN = 16
SEED = 7


def prefix_cache_counters(base: str) -> dict[str, float]:
    """Sum the server's prefix-cache hit/query counters from /metrics."""
    with urllib.request.urlopen(f"{base.rstrip('/')}/metrics", timeout=10) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or "prefix_cache" not in line:
            continue
        name, _, value = line.rpartition(" ")
        key = "hits" if "hit" in name else "queries" if "quer" in name else name
        out[key] = out.get(key, 0.0) + float(value)
    return out


def make_prompts() -> dict[str, list[str]]:
    text = (REPO / "data" / "input.txt").read_text(encoding="utf-8")
    rng = random.Random(SEED)
    prefix = text[10_000 : 10_000 + PREFIX_LEN]
    shared = [
        prefix + text[i : i + TAIL_LEN]
        for i in (rng.randrange(0, len(text) - TAIL_LEN) for _ in range(N))
    ]
    unique = [
        text[i : i + PREFIX_LEN + TAIL_LEN]
        for i in (rng.randrange(0, len(text) - PREFIX_LEN - TAIL_LEN) for _ in range(N))
    ]
    return {"shared": shared, "unique": unique}


async def run_phase(base: str, prompts: list[str]) -> dict:
    url = f"{base}/v1/completions"
    before = prefix_cache_counters(base)
    async with aiohttp.ClientSession() as session:
        # sequential (concurrency 1): isolate prefill cost in TTFT
        results = [await one_request(session, url, "char-gpt", p, GEN_LEN) for p in prompts]
    after = prefix_cache_counters(base)
    ttfts = sorted(r["ttft"] * 1e3 for r in results)
    return {
        "ttft_ms_p50": ttfts[len(ttfts) // 2],
        "ttft_ms_mean": statistics.mean(ttfts),
        "cache_queries": after.get("queries", 0) - before.get("queries", 0),
        "cache_hits": after.get("hits", 0) - before.get("hits", 0),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. prefix-on / prefix-off")
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args()

    prompts = make_prompts()
    out = {"label": args.label}
    for phase in ("shared", "unique"):
        r = await run_phase(args.base, prompts[phase])
        out[phase] = r
        hit_rate = r["cache_hits"] / r["cache_queries"] if r["cache_queries"] else 0.0
        print(f"{args.label:>10} | {phase:<6} prefixes | TTFT p50 {r['ttft_ms_p50']:6.2f} ms "
              f"(mean {r['ttft_ms_mean']:6.2f}) | prefix-cache hit rate "
              f"{hit_rate:6.1%} ({r['cache_hits']:.0f}/{r['cache_queries']:.0f} tokens)")

    dest = REPO / "benchmarks" / f"prefix-cache-{args.label}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
