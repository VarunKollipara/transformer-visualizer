"""Load the corpus and serve it as batches of (input, target) training pairs.

Pipeline:
    raw text  --encode-->  one long 1-D tensor of IDs  --split-->  train / val
    then get_batch() carves out random (B, T) windows on demand.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.tokenizer import CharTokenizer

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "input.txt"


def load_corpus() -> tuple[CharTokenizer, torch.Tensor]:
    """Read the corpus, build the tokenizer, and encode the whole thing once."""
    text = DATA_PATH.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    # The entire corpus as one long sequence of IDs.
    # dtype=torch.long (64-bit int) because IDs are indices, and the layers that
    # consume them (embeddings) require integer (long) indices.
    data = torch.tensor(tok.encode(text), dtype=torch.long)  # (N,)
    return tok, data


def train_val_split(data: torch.Tensor, frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the stream into a training part and a held-out validation part.

    We keep the last 10% unseen during training so we can later check whether the
    model has actually *learned the language* vs. just *memorized the training
    text*. If train loss keeps falling but val loss doesn't, that's overfitting.
    """
    n = int(frac * len(data))
    train_data = data[:n]   # first 90%
    val_data = data[n:]     # last 10%
    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carve out one random batch of (input, target) windows.

    Returns:
        x : (batch_size, block_size)  the input windows
        y : (batch_size, block_size)  the same windows shifted right by one
    """
    # Pick `batch_size` random starting indices into the stream. The upper bound
    # is len(data) - block_size so that a full window (and its +1 shift) always
    # fits without running off the end.
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)  # (B,)

    # For each start i: x is data[i : i+block_size], y is the same slice shifted
    # by one. torch.stack glues the list of 1-D windows into a 2-D (B, T) grid.
    x = torch.stack([data[i : i + block_size] for i in ix])              # (B, T)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])     # (B, T)
    return x, y


if __name__ == "__main__":
    # --- demo (boilerplate): show one tiny batch and the predictions inside it ---
    torch.manual_seed(1337)  # reproducible randomness, so you see the same batch I describe

    tok, data = load_corpus()
    train_data, val_data = train_val_split(data)
    print(f"corpus: {len(data):,} tokens  |  train: {len(train_data):,}  val: {len(val_data):,}")

    block_size, batch_size = 8, 4
    xb, yb = get_batch(train_data, block_size, batch_size)
    print(f"\nx shape: {tuple(xb.shape)}   y shape: {tuple(yb.shape)}   (B, T)")
    print(f"\nx (input IDs):\n{xb}")
    print(f"\ny (target IDs = x shifted by 1):\n{yb}")

    # Unpack the predictions hidden in the FIRST sequence of the batch:
    print(f"\nThe {block_size} predictions inside sequence 0:")
    for t in range(block_size):
        context = xb[0, : t + 1]
        target = yb[0, t]
        print(f"  after {tok.decode(context.tolist())!r:<12} -> predict {tok.decode([target.item()])!r}")
