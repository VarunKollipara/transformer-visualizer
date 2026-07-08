# Optimization Log

Running record of the inference-optimization extension: what we tried, why, the
exact numbers, and the gotchas. This is the interview talking-point reference.

**The model:** from-scratch char-level GPT (618,817 params: `n_embd=128`,
4 heads, 3 layers, `block_size=64`, vocab 65), trained on tinyshakespeare to
val loss 1.65. Architecturally identical to GPT-2, just tiny.

**Hardware:** RTX 3070 Laptop (8 GB, Ampere, CUDA 13.0 driver), Windows 11 +
WSL2/Ubuntu for the Linux-only serving stack.

---

## Step 1 — Serve with vLLM

### Decision: vLLM over SGLang

- vLLM supports the `GPT2LMHeadModel` architecture out of the box; SGLang's
  supported-model list targets modern chat-model families (Llama/Qwen/Mistral)
  and does not include GPT-2. For a hand-built GPT-2-shaped model, that's
  decisive.
- vLLM also has the larger ecosystem and is the more common baseline in job
  postings; SGLang remains an option later for a side-by-side on an
  off-the-shelf small model.

### Step 1a (done 2026-07-07): port the checkpoint to HF GPT-2 format

vLLM doesn't execute user `nn.Module`s — it re-implements each architecture
with its own kernels and loads only HF-format weights. So serving our model
means porting weights into the GPT-2 format. Script:
[scripts/export_hf.py](scripts/export_hf.py) → `checkpoints/hf-gpt2/`.

Gotchas hit (all three are classic porting failure modes):

1. **Conv1D transpose.** HF GPT-2 stores linears as `Conv1D`, weight shape
   `(in, out)` — the transpose of `nn.Linear`'s `(out, in)`. Every matrix
   needs a `.T`.
2. **Fused QKV head order.** Our 4 separate per-head Q/K/V `Linear`s stack
   into one `c_attn` matrix; row order must be head-major to match GPT-2's
   split-heads reshape (and it matches our own `torch.cat` over heads).
3. **lm_head bias.** Ours has one; GPT-2's format doesn't. Folded it into
   `ln_f.beta` by solving `W @ delta = b` via pseudoinverse — exact because
   `W` (65×128) has full row rank. Residual: < 1e-5. Zero behavior change.
4. (Minor) tokenizers' Oniguruma regex rejects the `(?s)` inline flag; used
   `[\s\S]` to make the char-split pre-tokenizer match `\n`.

Also built a HF fast tokenizer wrapping the 65-char vocab (WordLevel +
split-every-char pre-tokenizer + Fuse decoder).

**Parity results (port is exact):**

| Check | Result |
|---|---|
| Max abs logit diff, scratch vs HF (B=4; T=64, 17, 1) | **8.58e-06** (fp32 noise) |
| Same, after reload from disk | 8.58e-06 |
| Greedy generation, 100 tokens | identical token-for-token |
| Tokenizer round-trip + agreement with CharTokenizer | exact |
| Param count | 619,904 = 618,817 − 65 (folded lm_head bias) + 1,152 (GPT-2's mandatory QKV biases, set to 0) |

Config notes: `activation_function="relu"` (we trained with ReLU, not GELU),
`tie_word_embeddings=False` (we trained wte and lm_head separately),
all dropout 0, `n_positions=64`, `eos_token_id=0` (`'\n'`; benchmarks will use
`ignore_eos=True` so this never affects measurements).

### Step 1b (done 2026-07-07): vLLM server running in WSL2, exact parity

Environment: WSL2 Ubuntu 26.04, uv-managed Python 3.12 venv at `~/vllm-env`
(system Python 3.14 is newer than vLLM supports), **vLLM 0.24.0**,
torch 2.11.0+cu130, CUDA visible in WSL via the Windows driver.

Serve command (from WSL):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 ~/vllm-env/bin/vllm serve \
  /mnt/c/Users/logot/Downloads/AIlearn/checkpoints/hf-gpt2 \
  --served-model-name char-gpt --dtype float32 \
  --max-model-len 64 --gpu-memory-utilization 0.25 --port 8000
```

The untied-embeddings risk didn't materialize: vLLM 0.24's `gpt2.py` builds a
separate `ParallelLMHead` and only ties when `config.tie_word_embeddings` is
true (verified in the installed source before serving).

**End-to-end parity ([scripts/verify_vllm.py](scripts/verify_vllm.py)):** 58/58
greedy tokens from prompt `"ROMEO:"` identical between transformers-on-CPU and
vLLM's GPT-2 kernels on GPU, both fp32. (58 = 64 positions − 6 prompt tokens;
absolute positional embeddings mean vLLM can't slide the window the way our
scratch `generate()` does.)

Gotchas hit standing up the server (each was a startup crash or wrong output):

1. **No C compiler in fresh WSL.** vLLM's torch.compile → Triton JIT needs
   gcc to build its CUDA glue module. Fix: `apt-get install gcc` (as root via
   `wsl -u root`).
2. **FlashInfer JIT wants the full CUDA toolkit (`nvcc`)**, which WSL doesn't
   get from the Windows driver (driver ≠ toolkit). Its fused sampler is
   irrelevant at vocab=65, so disabled it: `VLLM_USE_FLASHINFER_SAMPLER=0`.
   vLLM falls back to native torch sampling.
3. **Detokenizer silently deleted every newline.** We marked `'\n'` as
   EOS/BOS, and the completions API defaults to `skip_special_tokens=True` on
   output. Content tokens were perfect; text was newline-free. Char-level
   models where a real character doubles as EOS must send
   `"skip_special_tokens": false` per request (harness will).
4. `torch.compile` cache persists (`~/.cache/vllm/torch_compile_cache`), so
   first startup ≈ minutes, later startups much faster.

---

## Step 2 — Benchmark harness + first measured lever (done 2026-07-07)

Harness: [scripts/bench_serve.py](scripts/bench_serve.py). Streaming client
with per-token client-side timestamps; closed-loop concurrency sweep; fixed
work per request (`prompt_len=16`, `gen_len=48` → exactly the 64-token
context; `ignore_eos=True`); prompts sampled from the corpus with a fixed
seed; warmup excluded; p50/p99 reported. Client runs inside WSL next to the
server to avoid the Windows↔WSL NAT hop. Results are JSON in `benchmarks/`.

### fp32 baseline vs fp16 (only change: `--dtype`)

| conc | TTFT p50 ms (32→16) | ITL p50 ms (32→16) | out tok/s (32→16) | GPU MiB (32→16) |
|---|---|---|---|---|
| 1   | 6.5 → 8.0    | 1.40 → 1.84  | **598 → 462**      | 2590 → 2594 |
| 8   | 14.1 → 18.4  | 1.72 → 2.66  | 3636 → 2562        | 2590 → 2594 |
| 32  | 30.6 → 32.3  | 1.99 → 2.56  | 11189 → 9085       | 2590 → 2594 |
| 128 | 114.3 → 109.3| 3.60 → 3.79  | 14977 → 15151      | 2592 → 2596 |

Full data: `benchmarks/fp32-baseline.json`, `benchmarks/fp16.json`.

### Findings (the interview story)

1. **The workload is overhead-bound, and the numbers prove it.** A 619K-param
   forward step is microseconds of math, yet single-stream ITL is ~1.4 ms —
   the time goes to scheduler ticks, detokenization, SSE/HTTP. Consequence:
   **fp16 helped nothing and hurt low concurrency** (−23% single-stream
   throughput, likely extra casts/different kernel selection), reaching parity
   only at conc=128 where batching amortizes overhead. Compute/memory-BW
   optimizations can't speed up a stack whose bottleneck is neither. Measure
   first — this run *disproves* the default "fp16 = faster" assumption for
   this regime.
2. **Continuous batching works as advertised**: 598 → ~15,000 tok/s (25×) from
   conc 1→128, while per-token latency only degrades ~2.5×. The cost shows up
   in **TTFT p99: 10 ms → 385 ms** — queueing delay, the latency/throughput
   tradeoff in one number.
3. **GPU memory is flat across load and dtype** — it's the pre-allocated paged
   KV pool (25% of 8 GB), not live usage; the 2.5 MB of weights are invisible
   at this scale. nvidia-smi measures the pool, not the demand.
4. **Quality**: fp16 greedy output identical to fp32 for all 58 checkable
   tokens — at vocab=65 the argmax margin dwarfs fp16 noise (would not hold
   for a 32K-vocab model).

**Implication for Step 3 (quantization), flagged in Step 1:** this data
confirms INT4/INT8 on the 619K model would show nothing (same overhead-bound
regime). To make quantization measurable we need a model big enough to be
compute/memory-bound — train a ~10–25M-param variant and/or benchmark a small
open model (e.g. Qwen2.5-0.5B) with this same harness. Decide at checkpoint.

---

## Step 2b — Scale-up: 25M-param model, trained on GPU (done 2026-07-07)

Decision at checkpoint: scale up our own model (Option A) rather than switch
to an off-the-shelf one.

**Training** ([scripts/train_scaled.py](scripts/train_scaled.py)): `n_embd=512`,
8 heads, 8 layers, `block_size=256` → **25,405,505 params** (40× the original).
New ingredients the bigger model needed: dropout 0.2 (added to
[src/model.py](src/model.py) with default 0.0 — old checkpoint untouched),
weight decay 0.1 on 2-D params only, LR warmup 200 steps + cosine 3e-4 → 3e-5,
grad clip 1.0, TF32 matmuls. 5,000 steps × 32×256 tokens = 23 min on the
3070 (WSL). **Val loss 1.49 vs 1.65** for the small model; train/val gap ~0.31
(dropout held off memorization at ~35 corpus epochs). Checkpoint:
`checkpoints/gpt-md.pt`, curve in `checkpoints/history-md.json`.

**Port + serve**: `export_hf.py --ckpt gpt-md.pt --out hf-gpt2-md` worked
unchanged (mapping is shape-generic) — logit diff 8.1e-06, greedy 100/100.
Served parity: **250/250 greedy tokens** exact (fp32, `--max-model-len 256`).

### fp32 vs fp16 at 25M params (same harness, short 16/48 and long 64/192)

| workload | conc | out tok/s fp32 → fp16 | ITL p50 ms fp32 → fp16 |
|---|---|---|---|
| short | 1   | 621 → 516   | 1.37 → 1.99 |
| short | 128 | 19,364 → 17,945 | 2.68 → 2.99 |
| long  | 1   | 727 → 563   | 1.29 → 1.78 |
| long  | 128 | **25,338 → 25,357** (tie) | 3.16 → 2.88 |

Full data: `benchmarks/md-fp32*.json`, `benchmarks/md-fp16*.json`.

### Findings

1. **Still overhead-bound at 25M.** Single-stream throughput is the same as
   the 619K model (621 vs 598 tok/s) — the model grew 40× and per-token
   latency didn't move. Sanity math: 25M params × 2 B (fp16) = 50 MB; one
   decode step streams that in ~0.13 ms at this GPU's bandwidth, still buried
   under the ~1.4 ms serving-stack floor. Weight streaming won't dominate
   until weights reach hundreds of MB (≈ 200M+ params). fp16 accordingly
   still loses at low concurrency and only ties at conc=128/long.
2. **fp16 greedy diverges at the FIRST token at 25M** (vs identical for 58
   tokens at 619K). More params + tighter logit margins → one fp16 rounding
   flip → entirely different (but not visibly worse) trajectory. Takeaway:
   **string comparison stops being a quality metric at scale; Step 3 must add
   a perplexity eval** (mean cross-entropy on held-out corpus via the server
   or local forward passes).
3. Unexplained artifact, logged honestly: at conc=128/short the 25M fp32
   server outran the 619K one (19.4k vs 15.0k tok/s). Same harness, same
   workload; likely scheduler/window-size (max-model-len 64 vs 256)
   interaction rather than the model itself. Worth a profile if it ever
   matters; not load-bearing for our conclusions.

### Consequence for Step 3 (quantization) — decided by this data

End-to-end serving throughput cannot resolve quantization effects in our
regime, even at 25M. So Step 3 measures what the technique actually changes:
(a) **weight memory** (fp16 vs INT4 checkpoint + loaded footprint),
(b) **quality** via perplexity on held-out corpus (new eval),
(c) **isolated GEMM microbenchmark** — time the quantized matmul kernel vs the
fp16 one directly, outside the serving stack, where the difference is visible.
Optionally: one end-to-end AWQ/GPTQ run on a genuinely memory-bound open model
(e.g. Qwen2.5-0.5B) with the same harness, as the "here's where it pays off"
chart. Server currently up: 25M fp16 on :8000.

---

## Step 3 — GPTQ INT4 quantization (done 2026-07-07)

Tooling: **gptqmodel 7.1.0** (AutoGPTQ successor; supports the GPT-2/Conv1D
architecture, which AWQ tooling does not — that decided GPTQ vs AWQ).
Quantized with bits=4, group_size=128, 256×256-char calibration windows from
the TRAIN split only ([scripts/quantize_gptq.py](scripts/quantize_gptq.py)) →
`checkpoints/hf-gpt2-md-gptq`. New quality instrument:
[scripts/eval_ppl.py](scripts/eval_ppl.py) (perplexity on held-out val split;
also has `--fake-rtn-bits` for a round-to-nearest baseline). Kernel
microbenchmark: [scripts/bench_gemm.py](scripts/bench_gemm.py).

### Results

**Memory (the real win):** weights 48.5 MiB (fp16-equiv) → **13.1 MiB
(3.71×)**; theory says 16/4.5 bits ≈ 3.6×, plus fp16 embeddings/lm_head —
numbers check out. Peak eval VRAM 291 → 255 MiB (delta ≈ weight savings).

**Quality (perplexity, val split, 110,925 predictions):**

| variant | CE nats/char | ppl |
|---|---|---|
| fp32 | 1.4819 | 4.401 |
| fp16 | 1.4819 | 4.401 |
| RTN INT4 (fake-quant) | 1.4824 | 4.404 |
| **GPTQ INT4** | **1.4817** | **4.400** |

GPTQ ≈ fp16 (−0.0002 nats = noise), and better than RTN — error compensation
measurably works, though this well-regularized model is easy to quantize
(4-bit nearly free even naively).

**Isolated kernel bench (median µs, CUDA events, 200 iters):** with the fused
Triton kernel (`TritonV2Linear`), INT4 is **1.0–3.9× SLOWER** than fp16
cuBLAS at our layer shapes (512/2048 wide) across M=1..2048. Explanation: our
weight matrices are 0.4–2 MiB — they sit in L2 cache, so nothing is
memory-bound and dequant work is pure overhead. The **bytes test** at a
Llama-7B shape (K=4096, N=11008): fp32/fp16 time = 1.72–2.05× ≈ 2× (time
scales with weight bytes ⇒ memory-bound; fp16 sustains ~358 GB/s ≈ 80% of the
3070's bandwidth). In THAT regime a 4.5-bit kernel wins ~proportionally
(~2–3× realistic) — which is exactly why the industry quantizes 7B+ models
and why our 25M model cannot show it end-to-end.

### Gotchas

1. **Check which kernel you're benchmarking.** gptqmodel auto-selected
   `TorchAtenLinear` (naive full-dequant fallback) — 6–8× slower than fp16 and
   ~2× slower than the Triton kernel. First run measured the fallback, not the
   technique; forced `backend=BACKEND.TRITON` for the real comparison.
2. Randomizing packed quant buffers to fabricate a synthetic large QuantLinear
   corrupts `g_idx` (group-index map) → device-side asserts. Replaced with the
   fp16-vs-fp32 bytes test, which proves the regime claim without forged data.
3. `transformers` needs `optimum` installed to load GPTQ checkpoints.
4. CUDA stderr (device asserts) and stdout interleave misleadingly in logs —
   asserts appeared "before" a table that had already completed.

### The interview narrative this step produces

"I quantized a model to INT4 with GPTQ, proved quality was preserved with a
perplexity eval I built (including an RTN ablation showing GPTQ's error
compensation helps), measured a 3.7× memory reduction, then showed with an
isolated kernel benchmark WHY it can't speed up this model (L2-resident
weights) and quantified the memory-bound regime where it does (bytes test at
7B shapes, ~80% bandwidth utilization)."

---

## Step 4 — KV cache (done 2026-07-07)

Two halves: implement KV caching from scratch in our own model (cache
*existence*), then measure vLLM's prefix caching (cache *reuse*).

### 4a. From-scratch KV cache in src/model.py

Added an inference-only incremental path (`Head.forward_step` →
`GPT.generate_cached`): cache each layer's K/V per head; each decode step
feeds ONLY the new token and attends its single query against the cache.
Causal masking generalizes to a `tril(..., diagonal=T_past)` offset so the
same code does prefill (S=prompt) and decode (S=1). Absolute positions cap
the cache at `block_size` — same constraint vLLM enforces (LEARN Q4).

**Correctness (the non-negotiable):** cached and uncached greedy produce
identical tokens for the full 250-token run on both CPU and GPU; per-step
logits match to 3–6e-06. KV memory: 32 KiB/token, 8 MiB at full context
(B=1, fp32) — 2·layers·n_embd·4 B.

**Latency vs context ([scripts/bench_kv_cache.py](scripts/bench_kv_cache.py)),
25M model, greedy 250 tokens:**

| context | CPU uncached → cached (speedup) | GPU uncached → cached |
|---|---|---|
| T=17–64   | 22.1 → 10.2 ms/tok (2.2×) | 21.9 → 21.1 (1.0×) |
| T=65–128  | 33.5 → 11.0 ms/tok (3.1×) | 19.7 → 21.2 (0.9×) |
| T=129–192 | 50.0 → 11.7 ms/tok (4.3×) | 19.7 → 21.5 (0.9×) |
| T=193–256 | 64.1 → 12.8 ms/tok (5.0×) | 19.8 → 21.2 (0.9×) |
| **total** | **10.7 s → 2.9 s (3.7×)** | 5.4 s → 5.4 s (1.0×) |

CPU is the textbook chart: uncached grows linearly with context, cached stays
flat, speedup widens with T — O(T²)→O(T) made visible. GPU is the project's
recurring lesson: our Python-loop-over-heads model launches hundreds of tiny
kernels per step, so BOTH paths are launch-bound at ~20 ms/tok and the saved
compute was already free. Cross-reference: vLLM serves this same model at
~1.4 ms/tok — ~15× over our naive loop — which quantifies what a real serving
stack's fused kernels + scheduler are worth.

### 4b. vLLM prefix caching (KV reuse across requests)

[scripts/bench_prefix_cache.py](scripts/bench_prefix_cache.py): 64 sequential
requests, 208-token prompts, either sharing a 192-token prefix or all-unique;
hit counters read from the server's /metrics; rerun with
`--no-enable-prefix-caching` for attribution.

| config | TTFT p50 shared | TTFT p50 unique | hit rate shared |
|---|---|---|---|
| prefix caching ON (default) | 8.13 ms | 8.28 ms | **90.9%** |
| prefix caching OFF | 8.01 ms | 7.79 ms | — |

The hit rate is exactly predictable: KV blocks are 16 tokens, so 12 blocks =
192 of 208 prompt tokens are reusable; first request is cold →
(64·192 − 192)/13312 = 90.9%. **TTFT didn't improve** because a 208-token
prefill on a 25M model costs well under a millisecond — there was nothing to
save (attribution run confirms). On a 7B model with a 2K system prompt, the
same 90% hit rate is the difference between ~200 ms and ~20 ms TTFT; the
mechanism we verified is the one that delivers that.

Data: `benchmarks/kv-cache.json`, `benchmarks/prefix-cache-*.json`.

### Open flag for Step 3 (quantization) — raised early on purpose

AWQ/GPTQ tooling and 4-bit kernels (Marlin/ExLlama) target Llama-class models
with dims in the thousands. At 618K params / `n_embd=128`, INT4 quantization
would show no speedup (everything is launch overhead at this size) and the
tooling largely doesn't support GPT-2-tiny. Honest options when we get there:
(a) train a scaled-up variant (~10–25M params — well within the 3070's reach)
so quantization numbers mean something, and/or (b) demonstrate AWQ/GPTQ on a
small open model (e.g. Qwen2.5-0.5B) with the same harness. Decide at the
Step 2→3 checkpoint.
