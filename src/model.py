"""The transformer model, built one layer at a time.

We grow this file across Phase 2:
    [x] TokenEmbedding      ID  -> learnable vector
    [x] PositionalEmbedding position -> learnable vector
    [ ] Head (self-attention)
    [ ] MultiHead, MLP, Block, LayerNorm, residuals
    [ ] the full GPT

Nothing here uses nn.Transformer / nn.MultiheadAttention — every layer is built
from scratch so we understand all of it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """A learnable lookup table: token ID -> a vector of length n_embd.

    This is exactly what nn.Embedding does internally — a (vocab_size, n_embd)
    matrix of parameters that you index with the token IDs.
    """

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()  # required: registers this as an nn.Module
        # One learnable row (length n_embd) per token. nn.Parameter tells PyTorch
        # "these numbers are dials — track gradients and update them in training."
        # Initialized from a standard normal so they start small and varied.
        self.table = nn.Parameter(torch.randn(vocab_size, n_embd))  # (V, C)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) integer token IDs.
        # Indexing a (V, C) table with a (B, T) index tensor replaces each ID by
        # its row -> (B, T, C). That's the whole embedding operation.
        return self.table[idx]  # (B, T, C)


class PositionalEmbedding(nn.Module):
    """A learnable lookup table: position index (0..block_size-1) -> a vector.

    Same machinery as TokenEmbedding, but indexed by *where* a token sits rather
    than *what* it is. Added to the token embedding so each token's vector
    carries both its identity and its location.
    """

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        # One learnable row per position. block_size is the most positions we
        # ever need, because the model never sees more than block_size tokens.
        self.table = nn.Parameter(torch.randn(block_size, n_embd))  # (T_max, C)

    def forward(self, T: int) -> torch.Tensor:
        # Return the position vectors for positions 0..T-1 -> (T, C).
        # (T may be <= block_size; we just take the first T rows.)
        return self.table[:T]  # (T, C)


if __name__ == "__main__":
    # --- demo (boilerplate): token + positional embeddings on one batch ---
    from src.data import get_batch, load_corpus, train_val_split

    torch.manual_seed(1337)

    tok, data = load_corpus()
    train_data, _ = train_val_split(data)

    block_size, batch_size, n_embd = 8, 4, 16
    xb, _ = get_batch(train_data, block_size, batch_size)
    B, T = xb.shape

    tok_emb_layer = TokenEmbedding(tok.vocab_size, n_embd)
    pos_emb_layer = PositionalEmbedding(block_size, n_embd)

    tok_emb = tok_emb_layer(xb)   # (B, T, C) — WHAT each token is
    pos_emb = pos_emb_layer(T)    # (T, C)    — WHERE each position is
    h = tok_emb + pos_emb         # (B, T, C) — broadcast-add position into every batch row

    print(f"x        : {tuple(xb.shape)}   (B, T) IDs")
    print(f"tok_emb  : {tuple(tok_emb.shape)}   (B, T, C)")
    print(f"pos_emb  : {tuple(pos_emb.shape)}      (T, C)  -> broadcasts over the {B} batch rows")
    print(f"h = sum  : {tuple(h.shape)}   (B, T, C)  the model's input representation")

    # Payoff: a token that repeats at two positions should now differ.
    seq = xb[0]
    repeats = [(i, j) for i in range(T) for j in range(i + 1, T) if seq[i] == seq[j]]
    if repeats:
        i, j = repeats[0]
        ch = tok.decode([seq[i].item()])
        print(f"\nIn sequence 0, {ch!r} appears at positions {i} and {j}:")
        print(f"  token-only vectors identical?     {torch.equal(tok_emb[0, i], tok_emb[0, j])}")
        print(f"  after adding position, identical? {torch.equal(h[0, i], h[0, j])}")
