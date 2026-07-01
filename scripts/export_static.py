"""Bake the non-live API responses to static JSON for a backend-free deploy.

Three of the six endpoints don't need live inference — they're the same every
time: model/vocab info, the token-embedding PCA map, and the training history.
We precompute them here into web/public/data/*.json so the deployed site can
fetch them as plain static files (no FastAPI server).

The live endpoints (forward / logits / generate) run client-side via ONNX and
are NOT produced here.

Run:  uv run python -m scripts.export_static
"""

from __future__ import annotations

import json
from pathlib import Path

from src import inference

OUT = Path(__file__).resolve().parents[1] / "web" / "public" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = inference.load_model()

    # /api/info  ->  {config, vocab, vocab_size}
    info = {"config": cfg, "vocab": tok.chars, "vocab_size": tok.vocab_size}
    (OUT / "info.json").write_text(json.dumps(info))

    # /api/embeddings  ->  {points: [...]}   (PCA of the token-embedding table)
    (OUT / "embeddings.json").write_text(json.dumps(inference.embeddings_2d(model, tok)))

    # /api/training  ->  {history, samples, config}  (the saved loss curve + samples)
    (OUT / "training.json").write_text(json.dumps(inference.load_history()))

    for f in ("info.json", "embeddings.json", "training.json"):
        p = OUT / f
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
