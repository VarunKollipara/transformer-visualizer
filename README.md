# Transformer Visualizer

> A GPT-style transformer built from scratch in PyTorch, visualized in the
> browser, then optimized and benchmarked with vLLM, KV caching, GPTQ, and Triton.

**[Live Demo](https://transformer-visualizer-silk.vercel.app/)** · **[Optimization Deep Dive](OPTIMIZATION_LOG.md)**

`PyTorch` · `Triton` · `vLLM` · `ONNX` · `Next.js` · `TypeScript` · `GPTQ`

A GPT-style transformer built **layer by layer from scratch** in PyTorch (no
`nn.Transformer`, no `nn.MultiheadAttention`), an interactive web visualizer
of its internals, and — the second act — a full **inference-optimization
pass**: the same model served through vLLM, benchmarked, quantized, KV-cached,
and given a hand-written Triton kernel, with every claim backed by a script
and a JSON of measurements.

Two rules throughout: understand every piece, and **measure before optimizing**.

## Act 1 — the model and the visualizer

- Character-level GPT built by hand: embeddings, single attention head →
  multi-head, MLP, LayerNorm, residuals, stacked blocks ([src/model.py](src/model.py)).
- Trained on tinyshakespeare; a 619K-param version powers the website via
  in-browser ONNX inference (no backend).
- Interactive visualizer (Next.js): live generation with attention heatmaps,
  per-token probability bars, temperature playground, embedding map, and a
  plain-language "how does an AI actually work?" walkthrough built from
  [NOTES.md](NOTES.md).

## Act 2 — inference optimization (the systems story)

A 25M-param variant (512d, 8 heads, 8 layers, 256 context; trained in 23 min
on an RTX 3070) run through the full serving-optimization toolchain. Highlights,
all reproducible from [scripts/](scripts) with data in [benchmarks/](benchmarks):

| What | Result |
|---|---|
| **Serve via vLLM** — port the scratch checkpoint to HF GPT-2 format (Conv1D transpose, fused QKV, lm_head bias folded into ln_f via pseudoinverse) | logit parity 8e-06; greedy generations identical through the whole serving stack (250/250 tokens) |
| **Benchmark harness** — streaming TTFT / inter-token latency / throughput / memory, concurrency sweep | continuous batching: 25× throughput (620 → ~15–25k tok/s) for ~2.5× per-token latency; TTFT p99 shows the queueing cost |
| **fp16 vs fp32** | *no speedup* — measured, then explained: the workload is overhead-bound (~1.4 ms serving-stack floor vs ~0.13 ms of weight streaming) |
| **GPTQ INT4 quantization** | **3.7× smaller** (48.5 → 13.1 MiB); perplexity 4.400 vs 4.401 fp16 (quality preserved, with an RTN ablation showing GPTQ's error compensation); kernel microbench + a bytes test showing exactly which regime INT4 wins in (~80% bandwidth-bound at 7B shapes) |
| **KV cache, from scratch** — incremental `forward_step` with per-head K/V cache in my own model | bit-faithful (identical greedy tokens), **3.7× on CPU** with the textbook O(T²)→O(T) curve; 1.0× on GPU — correctly predicted: the naive loop is launch-bound |
| **vLLM prefix caching** | hit rate **90.9%**, matching the 16-token block math to the token; TTFT attribution run with the feature disabled |
| **Triton kernel** — fused LayerNorm (one program per token, fp32 stats) | correct to 1.4e-06; 3–4× vs my eager 6-kernel path, honestly benchmarked at 0.6–0.9× vs torch's own fused kernel; end-to-end 1.05× = Amdahl's law with my own numbers |

The full lab notebook — every decision, number, and gotcha —
is [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md). The concept log for
non-experts is [NOTES.md](NOTES.md); [LEARN.md](LEARN.md) is a 20-question
self-quiz distilled from the work.

The through-line: at small scale almost every famous optimization "fails" —
and each time, the harness identifies the bottleneck category
(overhead / compute / memory bandwidth), proves it with a targeted
experiment, and quantifies the regime where the technique pays.

## Project layout

```
src/            model (from scratch), tokenizer, inference, Triton kernels
scripts/        training, HF export, vLLM verification, benchmark harnesses,
                GPTQ quantization, perplexity eval
benchmarks/     measurement JSONs (every table above is reproducible)
web/            Next.js visualizer (static export, in-browser ONNX inference)
NOTES.md        plain-language concept log (feeds the website's explainer)
OPTIMIZATION_LOG.md   the systems lab notebook: what/why/numbers/gotchas
LEARN.md        checkpoint Q&A — the understanding audit
```

## Running it

Python side (Windows or Linux; [uv](https://docs.astral.sh/uv/) manages everything):

```bash
uv sync
uv run python -m scripts.train           # train the small model (CPU, ~15 min)
uv run python -m scripts.export_hf       # port a checkpoint to HF GPT-2 format (+ parity test)
```

Serving/optimization side (Linux or WSL2 + NVIDIA GPU):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve checkpoints/hf-gpt2-md \
  --served-model-name char-gpt --dtype float16 --max-model-len 256 \
  --gpu-memory-utilization 0.25
python -m scripts.verify_vllm --dir checkpoints/hf-gpt2-md --max-tokens 250
python -m scripts.bench_serve --label my-run
```

Web visualizer:

```bash
cd web && npm install && npm run dev
```
