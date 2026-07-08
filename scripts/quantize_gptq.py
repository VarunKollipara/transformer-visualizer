"""Quantize the 25M model to INT4 with GPTQ (weights-only, W4A16).

What GPTQ does, in one paragraph: for each weight matrix, run a few hundred
calibration samples through the layer to estimate which input channels carry
big activations (the Hessian H = X X^T). Then quantize the matrix one column
at a time; after rounding each column to 4 bits, adjust the not-yet-quantized
columns to cancel the output error the rounding just caused, weighted by H.
It minimizes error in the layer's OUTPUTS, not its weights — which is why it
beats round-to-nearest at the same bit width.

Config choices:
  - bits=4, group_size=128: one fp16 scale per 128 input weights — the
    standard W4 recipe; real cost ~4.5 bits/weight, ~3.6x smaller than fp16.
  - Calibration = 256 random 256-char windows from the TRAIN split. Never
    calibrate on the val split — the perplexity eval must stay untouched.
  - Only the transformer matrices (c_attn/c_proj/c_fc) are quantized;
    embeddings, LayerNorms, and lm_head stay fp16 (standard practice — they're
    small and disproportionately sensitive).

Run in WSL (GPU):
  cd /mnt/c/Users/logot/Downloads/AIlearn && ~/vllm-env/bin/python -m scripts.quantize_gptq
"""

from __future__ import annotations

import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "checkpoints" / "hf-gpt2-md"
DST = REPO / "checkpoints" / "hf-gpt2-md-gptq"

N_SAMPLES = 256
SAMPLE_LEN = 256
SEED = 1337


def calibration_texts() -> list[str]:
    """Random windows from the train split (first 90%), same split as training."""
    text = (REPO / "data" / "input.txt").read_text(encoding="utf-8")
    train_text = text[: int(0.9 * len(text))]
    rng = random.Random(SEED)
    return [
        train_text[i : i + SAMPLE_LEN]
        for i in (rng.randrange(0, len(train_text) - SAMPLE_LEN) for _ in range(N_SAMPLES))
    ]


def main() -> None:
    from gptqmodel import GPTQModel, QuantizeConfig

    cfg = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(str(SRC), cfg)
    model.quantize(calibration_texts(), batch_size=8)
    model.save(str(DST))
    print(f"saved INT4 model -> {DST}")

    fp32 = sum(f.stat().st_size for f in SRC.glob("*.safetensors"))
    int4 = sum(f.stat().st_size for f in DST.glob("*.safetensors"))
    # our export dir stores fp32; the fair serving comparison is against fp16
    print(f"weights on disk  : fp32 {fp32 / 2**20:.1f} MiB | fp16-equiv {fp32 / 2 / 2**20:.1f} MiB "
          f"| int4 {int4 / 2**20:.1f} MiB ({fp32 / 2 / int4:.2f}x smaller than fp16)")


if __name__ == "__main__":
    main()
