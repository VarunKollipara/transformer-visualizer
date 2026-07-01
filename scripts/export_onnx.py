"""Export the trained char-GPT to ONNX for client-side (in-browser) inference.

The deployed site has no backend: the three *live* endpoints (forward/logits/
generate) run the model in the browser via onnxruntime-web. This script writes
`web/public/model/gpt.onnx` with a forward that returns BOTH:
  - logits     (1, T, vocab)          -> next-char scores (logits / generate)
  - attention  (layers, heads, 1, T, T) -> per-head weights (forward heatmap)

Sequence length T is a dynamic axis (1..block_size). Also verifies the ONNX
outputs match PyTorch before saving.

Run:  uv run python -m scripts.export_onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src import inference

OUT = Path(__file__).resolve().parents[1] / "web" / "public" / "model"
OUT_PATH = OUT / "gpt.onnx"


class ExportGPT(nn.Module):
    """Wrap GPT so its forward also returns the attention each head computed."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor):
        logits, _ = self.model(idx)  # (1, T, V); fills each head's .att as a side effect
        att = torch.stack(
            [
                torch.stack([h.att for h in blk.attn.heads], dim=0)  # (H, 1, T, T)
                for blk in self.model.blocks
            ],
            dim=0,
        )  # (L, H, 1, T, T)
        return logits, att


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = inference.load_model()
    wrapper = ExportGPT(model).eval()

    T = 16  # any 1..block_size; the axis is exported dynamic
    dummy = torch.randint(0, tok.vocab_size, (1, T), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy,),
        OUT_PATH.as_posix(),
        input_names=["idx"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "idx": {1: "T"},
            "logits": {1: "T"},
            "attention": {3: "T", 4: "T"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    # The exporter may split weights into a sidecar .onnx.data file; inline them
    # into a single self-contained .onnx so the browser fetches just one file.
    import onnx

    m = onnx.load(OUT_PATH.as_posix())  # loads any external data
    onnx.save_model(m, OUT_PATH.as_posix(), save_as_external_data=False)
    sidecar = OUT_PATH.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"  wrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes, single file)")

    # ── correctness check: ONNX vs PyTorch on a real prompt ──
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (skip verify: onnxruntime not installed — `uv add --dev onnxruntime`)")
        return

    ids = tok.encode("To be, or not to b")[-cfg["block_size"] :]
    x = np.array([ids], dtype=np.int64)

    with torch.no_grad():
        pt_logits, pt_att = wrapper(torch.tensor(x))
    sess = ort.InferenceSession(OUT_PATH.as_posix(), providers=["CPUExecutionProvider"])
    on_logits, on_att = sess.run(["logits", "attention"], {"idx": x})

    dl = float(np.abs(pt_logits.numpy() - on_logits).max())
    da = float(np.abs(pt_att.numpy() - on_att).max())
    print(f"  verify: max|Δlogits|={dl:.2e}  max|Δattention|={da:.2e}")
    assert dl < 1e-3 and da < 1e-4, "ONNX output diverges from PyTorch!"
    # sanity: the model should still love 'e' after '...to b'
    top = tok.itos[int(on_logits[0, -1].argmax())]
    print(f"  argmax next char after 'To be, or not to b' = {top!r}  (expect 'e')")


if __name__ == "__main__":
    main()
