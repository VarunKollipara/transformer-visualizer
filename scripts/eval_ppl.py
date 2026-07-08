"""Perplexity on the held-out validation split — the quality yardstick.

Why this exists (Step 3): once models are big enough that fp16/INT4 rounding
flips argmaxes, comparing generated *strings* tells you nothing (trajectories
diverge without being worse). Perplexity scores the whole probability
distribution against real held-out text: mean cross-entropy per character,
exponentiated. Lower = the model finds the true text less surprising. This is
the metric every quantization paper reports, and it lets us put fp32, fp16,
and INT4 on one honest axis.

Method: encode the val split (last 10% of the corpus — same split as
training), cut into NON-overlapping 256-token windows, and average the loss
over every predicted position. Non-overlapping windows slightly understate
quality (early tokens in each window have little context) but apply equally
to every variant, and what we need is the *delta* between variants.

Run in WSL (GPU):
  ~/vllm-env/bin/python -m scripts.eval_ppl --model checkpoints/hf-gpt2-md --dtype float16
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/hf-gpt2-md")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16", "auto"],
                    help="'auto' lets the checkpoint decide (use for quantized models)")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fake-rtn-bits", type=int, default=0,
                    help="if >0, round-to-nearest fake-quantize the transformer "
                         "matrices to this many bits (group=128) before eval — "
                         "the naive baseline GPTQ is supposed to beat")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "auto": "auto"}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(REPO / args.model, dtype=dtype)
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(REPO / args.model)

    if args.fake_rtn_bits:
        # Round-to-nearest, symmetric, one scale per 128 input weights — the
        # same layers and grouping GPTQ uses, minus the error compensation.
        # Quantize -> dequantize in place: quality effect without INT4 kernels.
        from transformers.pytorch_utils import Conv1D

        qmax = 2 ** (args.fake_rtn_bits - 1) - 1  # e.g. 7 for 4-bit signed
        with torch.no_grad():
            for name, mod in model.named_modules():
                if not isinstance(mod, Conv1D):
                    continue  # embeddings/LN/lm_head stay fp16, like GPTQ
                w = mod.weight  # (in, out); group along the INPUT dim
                n_in, n_out = w.shape
                g = w.view(n_in // 128, 128, n_out)                    # (G, 128, out)
                scale = g.abs().amax(dim=1, keepdim=True) / qmax       # (G, 1, out)
                scale = torch.clamp(scale, min=1e-8)
                g.copy_((g / scale).round().clamp(-qmax - 1, qmax) * scale)

    # Same 90/10 split as src/data.train_val_split, so no training text leaks in.
    text = (REPO / "data" / "input.txt").read_text(encoding="utf-8")
    val_text = text[int(0.9 * len(text)):]
    ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    n_windows = len(ids) // args.window
    windows = ids[: n_windows * args.window].view(n_windows, args.window)  # (N, T)

    total_nll, total_preds = 0.0, 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_windows, args.batch):
            batch = windows[i : i + args.batch].to(device)          # (B, T)
            logits = model(batch).logits                            # (B, T, V)
            # positions 0..T-2 predict targets 1..T-1
            nll = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_preds += batch[:, 1:].numel()

    ce = total_nll / total_preds  # nats per character
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    tag = f"{args.dtype}, fake-RTN int{args.fake_rtn_bits}" if args.fake_rtn_bits else args.dtype
    print(f"model            : {args.model} ({tag})")
    print(f"val characters   : {total_preds:,} predicted positions")
    print(f"cross-entropy    : {ce:.4f} nats/char  ({ce / math.log(2):.4f} bits/char)")
    print(f"perplexity       : {math.exp(ce):.3f}")
    print(f"peak GPU memory  : {mem:.0f} MiB   |  eval time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
