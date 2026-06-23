"""The transformer model, built one layer at a time.

We grow this file across Phase 2:
    [x] TokenEmbedding      ID  -> learnable vector
    [x] PositionalEmbedding position -> learnable vector
    [x] Head (self-attention)
    [x] MiniGPT             minimal end-to-end model that can train + generate
    [x] MultiHeadAttention  several heads in parallel
    [x] FeedForward         per-token MLP (the "thinking" step)
    [x] LayerNorm           per-token normalization (built by hand)
    [x] Block               (LayerNorm -> attention -> +) then (LayerNorm -> MLP -> +)
    [x] GPT                 embeddings -> stacked Blocks -> lm_head

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


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and mixed.

    One head learns a single way of looking back; multiple heads let the model
    track several kinds of relationships at once (e.g. the previous character vs.
    the start of the word). We split the n_embd width into `num_heads` smaller
    heads of size n_embd // num_heads, run them in parallel, concatenate their
    outputs back to width n_embd, then apply a linear projection to mix them.
    """

    def __init__(self, n_embd: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        head_size = n_embd // num_heads
        # nn.ModuleList registers each head so its parameters are tracked/trained.
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)  # lets the heads' outputs mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run every head on the same input x, concatenate along the channel dim.
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)                                       # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    """Per-token MLP: each token processes its own gathered info independently.

    "Attention is communication; the MLP is computation." No token mixing happens
    here — the same little network is applied to every position on its own.
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        # nn.Sequential just chains the layers in order. Widen 4x, bend with a
        # ReLU nonlinearity (max(0, x)), then project back to n_embd.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand to a wider hidden layer
            nn.ReLU(),                       # nonlinearity (without it, the two
                                             # Linears would collapse into one)
            nn.Linear(4 * n_embd, n_embd),  # project back to n_embd
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, n_embd) -> (B, T, n_embd)


class LayerNorm(nn.Module):
    """Normalize each token's vector to mean 0 / variance 1, then rescale.

    Built by hand. For each token (the last dim, C), subtract the mean and divide
    by the standard deviation, so its features sit at a stable scale. Two learnable
    parameters per channel — gamma (scale) and beta (shift) — let the model adjust
    or undo that normalization if it helps. Keeping activations well-scaled is what
    makes deep stacks train stably.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps  # tiny constant so we never divide by zero
        self.gamma = nn.Parameter(torch.ones(n_embd))   # learnable scale, starts at 1
        self.beta = nn.Parameter(torch.zeros(n_embd))   # learnable shift, starts at 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)                 # (B, T, 1) per-token mean
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # (B, T, 1) per-token variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)    # normalize across channels
        return self.gamma * x_norm + self.beta              # scale and shift


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (MLP).

    Two important details:
      - Residual connections: x = x + sublayer(x). Each sublayer *adds a
        refinement* rather than replacing x, giving gradients a clean path back
        through the whole stack (this is what makes depth trainable).
      - Pre-norm: we LayerNorm the input *before* each sublayer (modern GPT style,
        more stable than normalizing after).
    """

    def __init__(self, n_embd: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # communicate, then add back (residual)
        x = x + self.ffn(self.ln2(x))   # compute, then add back (residual)
        return x


class MiniGPT(nn.Module):
    """The smallest end-to-end model that can actually train and generate.

    Pipeline:  IDs -> (token + position embeddings) -> one attention head
               -> a linear "language-model head" -> logits over the vocab.
    Given targets, it also computes the cross-entropy loss.
    """

    def __init__(self, vocab_size: int, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = TokenEmbedding(vocab_size, n_embd)
        self.pos_emb = PositionalEmbedding(block_size, n_embd)
        self.head = Head(n_embd, head_size, block_size)
        # The prediction head: maps each token's context vector to one score
        # ("logit") per possible next character. vocab_size scores per position.
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx, targets: (B, T) integer IDs
        B, T = idx.shape

        h = self.token_emb(idx) + self.pos_emb(T)  # (B, T, C)   what + where
        h = self.head(h)                           # (B, T, head_size)  context-aware
        logits = self.lm_head(h)                   # (B, T, vocab_size) raw next-char scores

        if targets is None:
            return logits, None

        # Cross-entropy expects (N, vocab) logits and (N,) targets, so flatten the
        # batch and time dims into one long list of predictions. cross_entropy does
        # softmax + the -log(prob of the true char) you derived by hand, averaged.
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive generation: the predict -> sample -> append loop."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]      # crop to last block_size tokens
            logits, _ = self(idx_cond)                 # (B, T, V)
            logits = logits[:, -1, :]                  # only the LAST position -> (B, V)
            probs = F.softmax(logits, dim=-1)          # (B, V) next-char distribution
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1) sample one
            idx = torch.cat([idx, next_id], dim=1)     # append, then loop
        return idx


class GPT(nn.Module):
    """The upgraded model: embeddings -> stacked transformer Blocks -> lm_head.

    Same forward/generate contract as MiniGPT, but the single attention head is
    replaced by `num_layers` Blocks (each = multi-head attention + MLP, with
    residuals and LayerNorm), and a final LayerNorm sits before the lm_head.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = TokenEmbedding(vocab_size, n_embd)
        self.pos_emb = PositionalEmbedding(block_size, n_embd)
        # nn.Sequential chains the blocks; each preserves (B, T, n_embd).
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads, block_size) for _ in range(num_layers)]
        )
        self.ln_f = LayerNorm(n_embd)               # final norm before prediction
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        h = self.token_emb(idx) + self.pos_emb(T)  # (B, T, C)
        h = self.blocks(h)                         # (B, T, C)  the deep stack
        h = self.ln_f(h)                           # (B, T, C)
        logits = self.lm_head(h)                   # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive generation (identical loop to MiniGPT.generate)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    # --- demo (boilerplate): token + positional embeddings on one batch ---
    from src.data import get_batch, load_corpus, train_val_split

    torch.manual_seed(1337)

    tok, data = load_corpus()
    train_data, _ = train_val_split(data)

    block_size, batch_size, n_embd = 8, 4, 16
    xb, yb = get_batch(train_data, block_size, batch_size)
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

    # --- multi-head attention: same width, split across heads ---
    num_heads = 4
    mha = MultiHeadAttention(n_embd, num_heads, block_size)
    mha_out = mha(h)
    print(
        f"\nmulti-head ({num_heads} heads x size {n_embd // num_heads}) "
        f"output: {tuple(mha_out.shape)}   (B, T, n_embd)"
    )

    # --- feed-forward (per-token MLP) sanity check ---
    ff = FeedForward(n_embd)
    ff_out = ff(mha_out)
    print(f"feed-forward output: {tuple(ff_out.shape)}   (B, T, n_embd)")

    # --- the full minimal model: loss + generation ---
    import math

    model = MiniGPT(tok.vocab_size, n_embd, head_size, block_size)
    logits, loss = model(xb, yb)
    print(f"\nMiniGPT logits: {tuple(logits.shape)}   (B, T, vocab_size)")
    print(f"initial loss  : {loss.item():.3f}   (expected ~ln(65) = {math.log(tok.vocab_size):.3f})")

    # Generate 200 chars from a single newline (ID 0). Untrained -> gibberish.
    start = torch.zeros((1, 1), dtype=torch.long)  # (1, 1) a single newline token
    out_ids = model.generate(start, max_new_tokens=200)[0].tolist()
    print("\n--- untrained sample (should be gibberish) ---")
    print(tok.decode(out_ids))

    # --- the upgraded GPT (multi-head + MLP + stacked blocks) builds + runs ---
    gpt = GPT(tok.vocab_size, n_embd=64, num_heads=4, num_layers=3, block_size=block_size)
    _, gpt_loss = gpt(xb, yb)
    print(
        f"\nGPT params: {sum(p.numel() for p in gpt.parameters()):,}   "
        f"initial loss: {gpt_loss.item():.3f}   (vs MiniGPT's ~22.7K params)"
    )
