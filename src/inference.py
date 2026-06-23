"""Load a trained GPT checkpoint and run it while exposing its internals.

This is the bridge between the model and the visualizer. Beyond generating text,
it returns the things we want to *show*: attention weights, the next-character
probability distribution at each position, and token info.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GPT
from src.tokenizer import CharTokenizer

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
CKPT_PATH = CKPT_DIR / "gpt.pt"
HISTORY_PATH = CKPT_DIR / "history.json"


def load_model(ckpt_path: Path = CKPT_PATH) -> tuple[GPT, CharTokenizer, dict]:
    """Rebuild the model + tokenizer from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    tok = CharTokenizer.from_chars(ckpt["chars"])
    model = GPT(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        block_size=cfg["block_size"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # inference mode
    return model, tok, cfg


def _collect_attention(model: GPT) -> list[list[list[list[float]]]]:
    """Read the attention weights each head saved during the last forward pass.

    Returns a nested list indexed [layer][head] -> (T, T) matrix, so the frontend
    can render one heatmap per head.
    """
    layers = []
    for block in model.blocks:
        heads = [head.att[0].tolist() for head in block.attn.heads]  # each (T, T)
        layers.append(heads)
    return layers


def _topk(probs: torch.Tensor, tok: CharTokenizer, k: int) -> list[dict[str, Any]]:
    """Top-k characters from a probability vector, as [{char, prob}, ...]."""
    vals, idxs = torch.topk(probs, k)
    return [{"char": tok.itos[i.item()], "prob": v.item()} for v, i in zip(vals, idxs)]


@torch.no_grad()
def forward_with_internals(
    model: GPT, tok: CharTokenizer, text: str, top_k: int = 10
) -> dict[str, Any]:
    """Run one forward pass over `text` and return predictions + attention.

    Powers the attention heatmap and the per-position probability bars.
    """
    if text == "":
        text = "\n"
    ids = tok.encode(text)[-model.block_size :]  # clip to the context window
    idx = torch.tensor([ids], dtype=torch.long)  # (1, T)
    logits, _ = model(idx)                        # (1, T, V); also fills head.att
    probs = F.softmax(logits[0], dim=-1)          # (T, V)

    positions = [
        {"char": tok.itos[ids[t]], "topk": _topk(probs[t], tok, top_k)}
        for t in range(len(ids))
    ]
    return {
        "tokens": [tok.itos[i] for i in ids],
        "positions": positions,
        "attention": _collect_attention(model),  # [layer][head] -> (T, T)
        "num_layers": len(model.blocks),
        "num_heads": len(model.blocks[0].attn.heads),
    }


@torch.no_grad()
def generate(
    model: GPT,
    tok: CharTokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Autoregressive generation, returning each step's choice and the top-k it
    sampled from. `temperature` sharpens (<1) or flattens (>1) the distribution.
    """
    if prompt == "":
        prompt = "\n"
    ids = tok.encode(prompt)
    start = len(ids)
    steps = []
    for _ in range(max_new_tokens):
        idx = torch.tensor([ids[-model.block_size :]], dtype=torch.long)
        logits, _ = model(idx)
        logits = logits[0, -1, :] / max(temperature, 1e-6)  # (V,) last position
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        steps.append({"char": tok.itos[next_id], "topk": _topk(probs, tok, top_k)})
        ids.append(next_id)
    return {
        "prompt": prompt,
        "generated": tok.decode(ids[start:]),
        "steps": steps,
    }


@torch.no_grad()
def next_logits(
    model: GPT, tok: CharTokenizer, text: str, top_k: int = 8
) -> dict[str, Any]:
    """Return the raw (pre-softmax) logits for the most likely next characters.

    Powers the softmax/temperature playground: the user starts from the model's
    real scores, then reshapes them client-side.
    """
    if text == "":
        text = "\n"
    ids = tok.encode(text)[-model.block_size :]
    idx = torch.tensor([ids], dtype=torch.long)
    logits, _ = model(idx)
    last = logits[0, -1, :]  # (V,) raw scores for the next character
    vals, idxs = torch.topk(last, top_k)
    return {
        "context": text,
        "candidates": [
            {"char": tok.itos[int(i.item())], "logit": float(v.item())}
            for v, i in zip(vals, idxs)
        ],
    }


@torch.no_grad()
def embeddings_2d(model: GPT, tok: CharTokenizer) -> dict[str, Any]:
    """Project the learned token-embedding table down to 2D (via PCA) for a map.

    Each token is a point; tokens the model treats similarly land near each other.
    """
    weight = model.token_emb.table.detach().numpy()      # (V, C)
    centered = weight - weight.mean(axis=0, keepdims=True)
    # PCA: the top-2 right singular directions capture the most variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T                          # (V, 2)

    # scale to a tidy [-1, 1] box for plotting
    span = np.abs(coords).max(axis=0)
    span[span == 0] = 1.0
    coords = coords / span

    points = []
    for i, ch in enumerate(tok.chars):
        if ch == " ":
            group = "space"
        elif ch == "\n":
            group = "newline"
        elif ch.isalpha():
            group = "upper" if ch.isupper() else "lower"
        elif ch.isdigit():
            group = "digit"
        else:
            group = "punct"
        points.append(
            {"id": i, "char": ch, "x": float(coords[i, 0]), "y": float(coords[i, 1]), "group": group}
        )
    return {"points": points}


def load_history() -> dict[str, Any]:
    """Read the saved training history (loss curve + sample generations)."""
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)
    return json.loads(HISTORY_PATH.read_text())
