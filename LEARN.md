# LEARN — Checkpoint Questions & Answers

Every checkpoint question from the inference-optimization work, with its
answer. Varun: try to answer from memory first, then read. These get quizzed
later.

---

## Step 1a — Porting the model to HF GPT-2 format

### Q1. Why can't vLLM just run your `model.py`, and what does it actually take from `checkpoints/hf-gpt2/`?

**Answer:** vLLM never executes your model code. Its speed comes from
re-implementing each architecture it supports with its own GPU kernels (paged
attention, fused QKV) and a scheduler that batches many requests continuously —
none of which works on an arbitrary `nn.Module` it can't see inside. So the
only things it takes from the folder are **data, not code**:

- `config.json` — names a recognized architecture (`GPT2LMHeadModel`) plus its
  dimensions (`n_embd=128`, 3 layers, 4 heads, `n_positions=64`, ReLU, untied
  embeddings). This tells vLLM *which of its own implementations* to build.
- `model.safetensors` — the learned weights, poured into that implementation.
- tokenizer files — so the server can turn request text into IDs and back.

"Serving your model" therefore meant proving your model *is* a GPT-2 in shape
and porting the weights into that format — then proving parity.

### Q2. The ported model has 1,152 *more* parameters than yours and 65 *fewer*, yet produces identical outputs. Where did each number come from?

**Answer:**

- **+1,152**: HF GPT-2's fused QKV layer (`c_attn`) always has a bias — there's
  no option to omit it. Our attention was trained with `bias=False`, so the
  port fills those slots with zeros: 3 layers × (3 × 128) = 1,152 zeros. Zeros
  added to every projection change nothing.
- **−65**: our `lm_head` had a bias (one per vocab entry, 65); GPT-2's format
  has no slot for it. We *folded* it into the final LayerNorm's shift `beta`:
  since `logits = W·(ln_f output) + b`, adding a correction `δ` to beta where
  `W·δ = b` produces identical logits with the bias gone. An exact `δ` exists
  because `W` is 65×128 — 128 unknowns, only 65 equations, full row rank.

The lesson under both: a model is not one canonical set of tensors — many
different parameterizations compute the same function, and format conversion
lives in that equivalence space. But you never trust it without a parity test
(ours: max logit diff 8.6e-06, greedy generations identical for 100 tokens).

---

## Step 1b — Serving with vLLM

### Q3. The server's first output had every word right but every newline missing. What happened, and why does this bite char-level models specifically?

**Answer:** The *model* generated the newline tokens correctly (greedy tokens
matched the local run exactly) — the *detokenizer* deleted them. We declared
`'\n'` as the tokenizer's EOS/BOS special token, and OpenAI-style APIs default
to `skip_special_tokens=True` when turning output tokens back into text, so
anything marked "special" is stripped. In a BPE model, special tokens
(`<|endoftext|>`) are artificial and never part of real text, so stripping is
harmless. In a char-level model the EOS *is* a real character that carries
meaning, so the default silently corrupts output. Fix: send
`"skip_special_tokens": false` with each request. General lesson: a serving
stack has layers (tokenize → schedule → model → sample → detokenize), and
correctness bugs can live in any of them — which is why the parity test
compares final *text*, end to end.

### Q4. Your scratch `generate()` can produce unlimited text, but the server maxes out at 58 tokens for a 6-char prompt. Why?

**Answer:** Our model uses **learned absolute positional embeddings** — a
64-row table, one row per position. Position 65 simply has no embedding. The
scratch loop dodges this by sliding the window (`idx[:, -block_size:]`),
re-labeling the last 64 characters as positions 0..63 each step. vLLM doesn't
do that: with paged KV cache, each cached key/value was computed *with its
position baked in*, so you can't cheaply re-label positions — the server
enforces `prompt + max_tokens ≤ max_model_len` instead. This is also a preview
of why modern models use *relative* position schemes (RoPE, ALiBi): they make
longer-than-trained contexts and cache manipulation far more tractable —
directly relevant when we get to KV-cache experiments in step 4.

### Q5. The model file is 2.5 MB. Why did vLLM still reserve ~2 GB of GPU memory at startup?

**Answer:** By design. vLLM pre-allocates a fraction of VRAM
(`--gpu-memory-utilization`, default 0.9, we set 0.25) at startup and carves it
into fixed-size **pages** for the KV cache — the PagedAttention idea. Instead
of reserving one contiguous max-length buffer per request (mostly padding
waste), pages are handed to requests on demand and recycled, like an OS
managing RAM. Pre-allocating up front means no allocation stalls or
fragmentation at serving time, and the KV-cache pool — not the weights — is
what limits how many requests can be in flight at once. For our tiny model the
pool is comically oversized; for a 7B model on this same 8 GB GPU it would be
the scarcest resource in the system.

---

## Step 2 — Benchmarking

### Q6. fp16 is the most basic "make it faster" switch, yet it made single-stream throughput 23% WORSE and changed nothing at high concurrency. Why?

**Answer:** Because this workload is **overhead-bound**, not compute- or
memory-bound. Our 619K-param forward step is microseconds of arithmetic, but
each decode step also pays fixed costs — scheduler tick, kernel launches,
detokenization, SSE/HTTP — totaling ~1.4 ms. fp16 shrinks the arithmetic and
weight traffic (which were already negligible) while adding dtype-cast work
and different kernel selections, so it can even lose slightly. The general
skill: identify the bottleneck's *category* first (compute, memory bandwidth,
or overhead), because each category has a disjoint set of remedies. fp16/
quantization attack compute and memory bandwidth; they cannot attack overhead.
For big models (where weights are GBs and every step streams them from VRAM),
the same switch is a genuine ~2x — the technique isn't wrong, the regime was.

### Q7. At concurrency 128, TTFT p99 exploded to 385 ms while ITL p99 only reached ~16 ms. Why do the two latencies degrade so differently?

**Answer:** They live in different places. TTFT includes **queueing**: with
128 requests in flight, a new arrival may wait many scheduler rounds before
being admitted to a batch, and p99 captures the unluckiest arrivals. ITL is
measured only *after* admission, when the request is inside the continuous
batch — each decode step serves all active requests at once, so individual
gaps stay small. This split is why serving dashboards track TTFT and ITL
separately: they're controlled by different mechanisms (admission/scheduling
vs. batch step time) and fixed by different knobs.

### Q8. GPU memory read ~2,590 MiB at every concurrency level and both dtypes. What is that number actually measuring, and why doesn't it move?

**Answer:** It's the KV-cache **pool**, not usage. At startup vLLM grabs
`gpu_memory_utilization` × VRAM (we set 0.25 × 8 GB) and carves it into fixed
pages (PagedAttention); requests borrow and return pages from this pool at
runtime. nvidia-smi sees only the one big allocation, so load doesn't move it.
And dtype doesn't visibly move it because our weights (2.5 MB in fp32, 1.25 MB
in fp16) are rounding error next to the pool. On a real deployment the pool is
the resource that determines max concurrent context — which is why KV-cache
compression/eviction (step 4) is a whole optimization field.

---

## Step 2b — Scaling up (619K → 25M params)

### Q9. The model got 40× bigger, yet single-stream generation speed didn't change (598 → 621 tok/s). Why not?

**Answer:** Because the per-step cost floor isn't the model — it's the
serving stack (~1.4 ms of scheduler/kernel-launch/detokenize/HTTP per decode
step). Sanity math: at fp16 a 25M-param model's weights are 50 MB; one decode
step has to stream them from VRAM once, which takes ~0.13 ms at ~400 GB/s.
0.13 ms hiding under a 1.4 ms floor is invisible. The model would need
weights of *hundreds* of MB (≈200M+ params) before weight-streaming becomes
the thing you feel per token. This is the concept of **arithmetic intensity /
regime**: knowing which resource saturates first tells you which
optimizations can possibly matter.

### Q10. fp16 greedy output matched fp32 for all 58 tokens on the small model, but diverged at the FIRST character on the 25M model — while looking just as coherent. What happened, and is fp16 "lower quality" here?

**Answer:** Greedy decoding takes an argmax over logits each step. Divergence
happens when fp16 rounding (~3 decimal digits) flips which logit is biggest —
i.e., when the top-2 margin is smaller than the rounding noise. The tiny
model's distributions were blunt (65-way choice, huge margins); the 25M model
is sharper and more nuanced, so near-ties are common and a flip happens
almost immediately. After one flip the *context* differs, so every subsequent
token differs — trajectory divergence, compounding like chaos, not
degradation. It is NOT evidence of quality loss: both texts are equally
plausible samples. The right quality metric is **perplexity** (average
cross-entropy on held-out text), which scores the whole probability
distribution rather than one unlucky argmax — we add it in Step 3.

### Q11. On the long workload (64+192) at conc=128, throughput hit ~25,300 tok/s vs ~19,400 for the short workload (16+48). Longer sequences are more work — why is throughput HIGHER?

**Answer:** Because throughput here counts *output tokens per second*, and the
long workload spends more of its wall-clock in the efficient part of the
pipeline. Every request pays fixed per-request costs — HTTP setup, admission,
prefill scheduling, final detokenize/teardown. A 48-token request pays that
tax every 48 tokens (420 req/s of churn!); a 192-token request amortizes the
same tax over 4× more output. Deeper decode batches also stay fuller between
arrivals. Same reason bulk transfers beat many small ones on any system:
fixed costs amortize. The flip side is visible in the same table: long-run
TTFT p99 ballooned to ~790 ms — sustained full batches make new arrivals wait
longer for admission.

---

## Step 3 — GPTQ INT4 quantization

### Q12. GPTQ and RTN both store the same 4 bits per weight. What does GPTQ do differently, why does it need calibration data when RTN doesn't, and what did our measurements show?

**Answer:** RTN rounds every weight to its nearest 4-bit level independently —
it minimizes *weight* error. GPTQ minimizes *output* error (`Wx`), which is
what actually matters: it quantizes each weight matrix one column at a time,
and after rounding each column it adjusts the not-yet-quantized columns to
cancel the output error the rounding just introduced. Deciding how much an
error in one input channel matters — and how other channels can compensate —
requires knowing the input statistics, i.e. the Hessian H = XXᵀ estimated
from a few hundred **calibration** samples pushed through the layer. That's
the entire reason GPTQ needs data and RTN doesn't. Our numbers: fp16 ppl
4.401, RTN 4.404, GPTQ 4.400 — GPTQ fully recovers fp16 quality and beats
RTN, though our heavily-regularized model is easy to quantize (even RTN
barely hurts). Never calibrate on your eval split.

### Q13. The INT4 kernel was 1–4× SLOWER than fp16 in our microbenchmark — even after forcing the fused Triton kernel. Why, and what flips the sign?

**Answer:** W4A16 kernels win by moving fewer bytes: they read 4.5 bits per
weight instead of 16, dequantizing on the fly. That only helps if the GEMM is
**memory-bound** — limited by streaming weights from VRAM. Our layers are 512
and 2048 wide: 0.4–2 MiB per matrix, which fits in the GPU's L2 cache. Weights
barely travel, so there's no traffic to save, and the dequant instructions are
pure added cost; a plain cuBLAS fp16 GEMM wins. The sign flips when weight
matrices are tens of MB (K,N in the thousands — any 1B+ model) and M is small
(decode): then every step streams every weight from VRAM, bandwidth is the
wall, and 3.6× fewer bytes ≈ 2–3× faster. Bonus lesson from the same run:
gptqmodel silently defaulted to its naive fallback kernel (TorchAtenLinear),
which was 2× slower again — always print WHICH kernel you're benchmarking.

### Q14. What is the "bytes test," and what did it prove without any INT4 code at all?

**Answer:** At the Llama-7B layer shape (4096×11008) with M=1, we timed the
same GEMM in fp16 and fp32. fp32 moves exactly 2× the weight bytes. If time
is set by bandwidth, fp32 should take ~2× as long — and it did (1.72–2.05×),
with fp16 sustaining ~358 GB/s ≈ 80% of the 3070's theoretical bandwidth.
That's a controlled experiment proving the GEMM is memory-bound at that
shape: time scales with bytes moved, not with FLOPs (which are identical in
both cases... same multiply count — the FLOPs don't change, only the bytes).
Therefore a kernel reading 4.5 bits/weight must win roughly in proportion to
bytes saved — no INT4 kernel needed to establish the regime. The general
method: when you can't measure a technique directly, measure the *scaling
law* that governs it.

---

## Step 4 — KV cache

### Q15. Why is it safe to cache K and V forever, but never Q? What property of causal attention makes the cache valid?

**Answer:** In causal attention, token t's key and value are functions of the
token's own embedding and the layer inputs at positions ≤ t — all frozen
history. Nothing that happens later can change them, so once computed they're
valid for the rest of the generation (bit-identical, which our test confirmed
at 3e-06 logit noise over 250 steps). A *query*, by contrast, is only used at
the moment its token is the one doing the looking — after that step the row
of attention it produced is already folded into the output, so storing q buys
nothing. One more subtlety: with learned absolute positions, each cached k/v
has its position baked in, so the cache cannot "slide" — it dies at
block_size (the same reason vLLM enforced prompt+gen ≤ max_model_len in Q4).
Cost of all this: 2 · layers · n_embd · bytes per token — 32 KiB/token for
our 25M model, ~500 KiB/token for a 7B — which is why the KV pool, not
weights, is what limits concurrency at scale.

### Q16. The same KV cache gave a 5× speedup on CPU but exactly 1.0× on GPU. Reconcile those results.

**Answer:** The cache removes *compute* (recomputing the prefix every step:
O(T) MLP work and O(T²) attention). It does nothing about *per-step fixed
costs*. On CPU, compute is the bottleneck — so removing it shows the textbook
curve: uncached cost grows with context (22→64 ms/tok), cached stays flat
(~11 ms/tok), speedup widens with T (2.2×→5.0×). On GPU, our hand-written
model launches hundreds of tiny kernels per step (a Python loop over 8 heads
× 8 layers × 3 projections...), each launch costing microseconds while the
math inside costs nanoseconds — both paths are launch-bound at ~20 ms/tok, so
removing the math changes nothing. Same optimization, opposite outcomes,
entirely decided by which resource was scarce. (And vLLM serving this exact
model at ~1.4 ms/tok shows what fixing the launch problem — fused kernels,
no Python in the loop — is worth: ~15×.)

### Q17. The prefix-cache hit rate came out at exactly 90.9%. Derive that number — and why didn't TTFT improve even at 90.9% reuse?

**Answer:** vLLM caches KV in 16-token blocks and can only reuse *complete*
blocks. Our shared prefix is 192 tokens = exactly 12 blocks; each 208-token
prompt therefore has 192 reusable tokens. The first request is cold (0 hits),
the other 63 hit all 192: (64·192 − 192) / (64·208) = 12096/13312 = 90.9%.
The unique-prefix control shows 0.0%, and disabling the feature reproduces
the same TTFT — so the counter measures real reuse, not noise. TTFT didn't
move because what caching saves is prefill *compute*, and prefilling 208
tokens through a 25M model takes well under a millisecond of an ~8 ms TTFT
that is mostly HTTP/scheduling overhead. The mechanism is real and verified;
the regime (again) has nothing for it to save. On a 7B model with a 2K-token
system prompt, this same mechanism is a 5–10× TTFT win.

---

## Step 5 — Fused LayerNorm in Triton

### Q18. In the Triton kernel, what is a "program," why must BLOCK_N be a power of 2 with a mask, and why do we compute statistics in fp32 even for fp16 inputs?

**Answer:** A *program* is one instance of the kernel function; we launch a
grid of them (one per token row) and each sees only its `tl.program_id`.
Inside a program you operate on whole blocks — `tl.load` a slice, `tl.sum`
it — and Triton maps that to threads/warps/coalesced memory for you; that's
the block-level abstraction that makes 30 lines competitive with CUDA.
Block shapes must be powers of 2 because Triton tiles them onto hardware
lanes; our row width 512 happens to be one already, but the general pattern
is pad-to-pow2 and `mask = cols < N` so the padding lanes load 0 and store
nothing. Statistics go through fp32 because variance is a sum of squares of
small differences — in fp16 (10 mantissa bits) catastrophic cancellation
corrupts it. Our fp16 output differs from the fp32 reference by ~2e-3, which
is exactly one fp16 rounding step at those magnitudes — storage precision,
not kernel error (the fp32 path agrees to 1.4e-06).

### Q19. Our kernel beat the eager LayerNorm 3–4× but LOST to torch's F.layer_norm at small M (0.6×). Why isn't that a failure, and what's the lesson about baselines?

**Answer:** The 3–4× over eager is real and expected: eager is ~6 kernels
with ~6 memory round-trips; ours is 1. But PyTorch *also ships* a fused
LayerNorm (F.layer_norm) — years of tuning, C++ launch path. At M=1 both
kernels finish in microseconds and the difference is pure launch machinery:
our Python wrapper + Triton dispatch costs ~15 µs more per call. At M=8192,
where actual work dominates, we're at 0.9× — near parity. Lesson: the honest
baseline for a custom kernel is the *best existing fused implementation*, not
the naive path. Custom kernels earn their keep when no fused version exists
for your exact op combination (e.g. FlashAttention before it was everywhere,
or our GPTQ dequant+GEMM) — not when re-deriving a standard op. Writing one
correct kernel teaches the skill; knowing when NOT to write one is the
judgment.

### Q20. LayerNorm got 3× faster but the model only decoded 1.05× faster. Do the arithmetic that explains this, and say what the real fix would be.

**Answer:** Amdahl's law with our own numbers: eager LN ≈ 113 µs × 17 calls
≈ 1.9 ms of a ~20 ms decode step (~10%). Fusing cuts it to ~0.7 ms, saving
~1.2 ms — predicted step time 19.9 → ~18.8 ms; measured 19.0. A 3× speedup
on a 10% slice is a 1.05× whole. The other 19 ms is the remaining ~250 tiny
kernel launches per step: the Python loop over 8 heads × 8 layers, each with
separate q/k/v projections, cats, softmaxes. No single fused op fixes a
death-by-a-thousand-launches architecture — the real remedy is structural:
fuse q/k/v into one matmul, batch all heads into one tensor op, keep Python
out of the decode loop. That's precisely what vLLM's GPT-2 implementation
does, and we measured what it's worth: 1.4 ms/tok vs our 19 ms — 14×. The
optimization ladder goes: algorithm (KV cache) → structure (batched/fused
ops) → kernels (Triton) — in that order of leverage.
