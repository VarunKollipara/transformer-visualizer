"""The transformer model, built one layer at a time.

We grow this file across Phase 2:
    [x] TokenEmbedding      ID  -> learnable vector
    [x] PositionalEmbedding position -> learnable vector
    [~] Head (self-attention)   <- YOU implement forward()
    [ ] MultiHead, MLP, Block, LayerNorm, residuals
    [ ] the full GPT

Nothing here uses nn.Transformer / nn.MultiheadAttention — every layer is built
from scratch so we understand all of it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Head(nn.Module):
    """One self-attention head.

    Each token emits a query and a key (to decide *who to attend to*) and a value
    (the info it hands over). A token's output is the causally-masked, softmax-
    weighted blend of the values of itself and the tokens before it.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        # Three learned linear projections: each maps a length-C token vector to a
        # length-head_size vector. bias=False keeps them pure matrix multiplies.
        self.key = nn.Linear(n_embd, head_size, bias=False)     # h -> k
        self.query = nn.Linear(n_embd, head_size, bias=False)   # h -> q
        self.value = nn.Linear(n_embd, head_size, bias=False)   # h -> v

        # Causal mask: a lower-triangular matrix of 1s. tril[i, j] == 1 means
        # "token i is allowed to attend to token j" (j <= i). Stored as a buffer
        # (fixed, not a learned parameter) so it moves with the model to GPU etc.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.att: torch.Tensor | None = None  # last attention weights, for inspection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  the per-token vectors (token + positional embedding)
        B, T, C = x.shape

        # STEP 1: every token makes a query, a key, and a value.
        # (just multiply x by the three learned matrices we set up in __init__)
        k = self.key(x)      # (B, T, head_size)  "what I am"
        q = self.query(x)    # (B, T, head_size)  "what I'm looking for"
        v = self.value(x)    # (B, T, head_size)  "the info I'll give"

        # STEP 2: compare every query to every key -> a relevance score for each
        # (query token, key token) pair. The transpose lines the head_size dims up
        # so the matrix multiply does the dot products. Scaling by 1/sqrt(head_size)
        # keeps the scores from getting too big (which would make softmax too spiky).
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)

        # STEP 3: block the future. Wherever the mask is 0 (a future token), set the
        # score to -inf, so after softmax it becomes 0% attention.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)

        # STEP 4: turn each row of scores into percentages that add up to 1.
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        self.att = wei  # save the attention weights so we can inspect/visualize them

        # STEP 5: blend each token's view of the values using those percentages.
        out = wei @ v  # (B, T, head_size)
        return out


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

    # --- self-attention head (runs once YOU implement Head.forward) ---
    head_size = 8
    head = Head(n_embd, head_size, block_size)
    att_out = head(h)  # (B, T, head_size)

    print(f"\nhead output: {tuple(att_out.shape)}   (B, T, head_size)")
    print(f"attention weights for sequence 0 (rows=query token, cols=key token):")
    print(head.att[0].round(decimals=2))

    # Checks: weights are a valid causal attention pattern.
    rows_sum_to_1 = torch.allclose(head.att.sum(dim=-1), torch.ones(B, T))
    upper = head.att[0].triu(diagonal=1)  # the "future" region (above the diagonal)
    future_is_zero = torch.allclose(upper, torch.zeros_like(upper))
    assert rows_sum_to_1, "each attention row must sum to 1 (did you softmax over dim=-1?)"
    assert future_is_zero, "tokens must not attend to the future (did you apply the causal mask?)"
    print("\nOK: attention rows sum to 1 and the future is masked")
