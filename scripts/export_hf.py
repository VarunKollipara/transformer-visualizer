"""Export the from-scratch GPT checkpoint to Hugging Face GPT-2 format.

Why this exists: vLLM (and every serving framework like it) does not run
arbitrary nn.Modules. It re-implements each supported architecture with its own
kernels (paged attention, fused QKV, continuous batching) and only loads
*weights* from a Hugging Face-format folder: config.json + safetensors +
tokenizer. Our model is architecturally GPT-2 — learned positional embeddings,
pre-LN blocks, causal multi-head attention, 4x-wide ReLU MLP — so we can port
the weights exactly and let vLLM serve it via its GPT-2 implementation.

Three gotchas this script handles:

  1. Conv1D transpose. HF GPT-2 stores its linear layers as `Conv1D` whose
     weight is (in_features, out_features) — the TRANSPOSE of nn.Linear's
     (out_features, in_features). Every ported matrix below gets a .T.

  2. Fused QKV, head-major. Our MultiHeadAttention has 4 separate Heads, each
     with its own query/key/value Linear of shape (head_size, n_embd). GPT-2
     uses ONE fused c_attn producing [Q | K | V], each n_embd wide, and later
     splits that n_embd into (num_heads, head_size) — head 0 first. So we stack
     our per-head weights in head order, exactly matching our torch.cat in
     MultiHeadAttention.forward.

  3. lm_head bias fold. Our lm_head is Linear(n_embd, vocab, bias=True); HF
     GPT-2's lm_head has NO bias. Dropping the bias would change the logits.
     Instead we fold it into the final LayerNorm's beta:

         logits = W @ (gamma * x_hat + beta) + b
                = W @ (gamma * x_hat + (beta + delta))   where  W @ delta = b

     W is (65, 128) with full row rank, so an exact delta exists (via
     pseudoinverse). Same logits, one fewer parameter tensor. ln_f feeds only
     the lm_head, so nothing else is affected.

After exporting, this script verifies parity: the HF model must produce the
same logits as the scratch model (max abs diff ~ float32 noise), and greedy
generations must match.

Run:  uv run python -m scripts.export_hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference import load_model

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def build_hf_state_dict(model) -> dict[str, torch.Tensor]:
    """Map our scratch GPT's parameters onto HF GPT-2's parameter names/layouts."""
    sd: dict[str, torch.Tensor] = {}

    sd["transformer.wte.weight"] = model.token_emb.table  # (V, C)
    sd["transformer.wpe.weight"] = model.pos_emb.table    # (T_max, C)

    for i, block in enumerate(model.blocks):
        p = f"transformer.h.{i}"

        sd[f"{p}.ln_1.weight"] = block.ln1.gamma  # (C,)
        sd[f"{p}.ln_1.bias"] = block.ln1.beta     # (C,)

        # Stack per-head Linears into the fused QKV matrix. Each head's
        # query.weight is (head_size, C); concatenating over heads gives
        # (C, C) in nn.Linear layout with rows in head-major order — the same
        # order our torch.cat over head outputs uses, and the same order
        # GPT-2's split-into-heads reshape expects.
        q = torch.cat([h.query.weight for h in block.attn.heads], dim=0)  # (C, C)
        k = torch.cat([h.key.weight for h in block.attn.heads], dim=0)    # (C, C)
        v = torch.cat([h.value.weight for h in block.attn.heads], dim=0)  # (C, C)
        qkv = torch.cat([q, k, v], dim=0)          # (3C, C) nn.Linear layout
        sd[f"{p}.attn.c_attn.weight"] = qkv.T      # (C, 3C) Conv1D layout
        sd[f"{p}.attn.c_attn.bias"] = torch.zeros(qkv.shape[0])  # ours has no QKV bias

        sd[f"{p}.attn.c_proj.weight"] = block.attn.proj.weight.T  # (C, C)
        sd[f"{p}.attn.c_proj.bias"] = block.attn.proj.bias        # (C,)

        sd[f"{p}.ln_2.weight"] = block.ln2.gamma
        sd[f"{p}.ln_2.bias"] = block.ln2.beta

        # FeedForward: net[0] expands C -> 4C, net[2] projects 4C -> C.
        sd[f"{p}.mlp.c_fc.weight"] = block.ffn.net[0].weight.T    # (C, 4C)
        sd[f"{p}.mlp.c_fc.bias"] = block.ffn.net[0].bias          # (4C,)
        sd[f"{p}.mlp.c_proj.weight"] = block.ffn.net[2].weight.T  # (4C, C)
        sd[f"{p}.mlp.c_proj.bias"] = block.ffn.net[2].bias        # (C,)

    # Fold the lm_head bias into ln_f.beta (see module docstring for the math).
    W = model.lm_head.weight  # (V, C)
    b = model.lm_head.bias    # (V,)
    delta = torch.linalg.pinv(W) @ b  # (C,) minimal-norm solution of W @ delta = b
    residual = (W @ delta - b).abs().max().item()
    assert residual < 1e-5, f"bias fold not exact: residual {residual:.2e}"

    sd["transformer.ln_f.weight"] = model.ln_f.gamma
    sd["transformer.ln_f.bias"] = model.ln_f.beta + delta
    sd["lm_head.weight"] = W  # untied from wte (we trained them separately)

    return {k: v.detach().clone() for k, v in sd.items()}


def build_hf_tokenizer(chars: list[str], model_max_length: int):
    """Wrap our 65-char vocabulary as a HF fast tokenizer.

    WordLevel with a split-every-character pre-tokenizer is exactly a char
    tokenizer; the Fuse decoder concatenates tokens back without inserting
    separators, keeping decode(encode(x)) == x.
    """
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {ch: i for i, ch in enumerate(chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token=None))
    # [\s\S] = any char INCLUDING newline ('.' alone would skip our '\n' token).
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()

    # The model was trained with no EOS concept; '\n' (id 0) is the closest
    # natural stop. Benchmarks will pass ignore_eos=True and a fixed max_tokens
    # anyway, so this choice never affects measurements.
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="\n",
        eos_token="\n",
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
    )


def export(ckpt_path: Path, export_dir: Path) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    model, tok, cfg = load_model(ckpt_path)

    config = GPT2Config(
        vocab_size=cfg["vocab_size"],       # 65
        n_positions=cfg["block_size"],      # 64
        n_embd=cfg["n_embd"],               # 128
        n_layer=cfg["num_layers"],          # 3
        n_head=cfg["num_heads"],            # 4
        activation_function="relu",         # we used ReLU, not GPT-2's GELU
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,          # we trained wte and lm_head separately
        bos_token_id=0,                     # '\n'
        eos_token_id=0,
    )
    hf_model = GPT2LMHeadModel(config)

    sd = build_hf_state_dict(model)
    missing, unexpected = hf_model.load_state_dict(sd, strict=False)
    # Non-persistent buffers (causal-mask 'bias') may show as missing; nothing
    # else should be. Unexpected keys would mean our mapping is wrong.
    real_missing = [k for k in missing if not k.endswith(("attn.bias", "attn.masked_bias"))]
    assert not real_missing, f"missing weights: {real_missing}"
    assert not unexpected, f"unexpected weights: {unexpected}"
    hf_model.eval()

    hf_tok = build_hf_tokenizer(tok.chars, model_max_length=cfg["block_size"])

    # --- parity checks: the port is worthless unless it's provably the same model ---

    # 1. Tokenizer round-trip + agreement with CharTokenizer.
    sample = "First Citizen:\nWe are accounted poor citizens."
    assert hf_tok.decode(hf_tok.encode(sample)) == sample, "tokenizer round-trip failed"
    assert hf_tok.encode(sample) == tok.encode(sample), "tokenizer disagrees with CharTokenizer"

    # 2. Logit parity on random batches, full and partial context lengths.
    torch.manual_seed(0)
    worst = 0.0
    with torch.no_grad():
        for T in (cfg["block_size"], 17, 1):
            idx = torch.randint(0, cfg["vocab_size"], (4, T))  # (B, T)
            ours, _ = model(idx)                # (B, T, V)
            theirs = hf_model(idx).logits       # (B, T, V)
            worst = max(worst, (ours - theirs).abs().max().item())
    assert worst < 1e-4, f"logit mismatch: max abs diff {worst:.2e}"

    # 3. Greedy generations must match token-for-token.
    prompt_ids = tok.encode("ROMEO:")
    a = list(prompt_ids)
    b_ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(100):
            ctx = torch.tensor([a[-cfg["block_size"]:]])
            a.append(int(model(ctx)[0][0, -1].argmax()))
            ctx = torch.tensor([b_ids[-cfg["block_size"]:]])
            b_ids.append(int(hf_model(ctx).logits[0, -1].argmax()))
    greedy_match = a == b_ids

    export_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(export_dir)
    hf_tok.save_pretrained(export_dir)

    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"exported to        : {export_dir}")
    print(f"parameters         : {n_params:,}")
    print(f"max logit diff     : {worst:.2e}   (float32 noise is ~1e-6..1e-5)")
    print(f"greedy match (100) : {greedy_match}")
    print(f"sample             : {tok.decode(a)!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gpt.pt", help="checkpoint filename in checkpoints/")
    ap.add_argument("--out", default="hf-gpt2", help="output dir name in checkpoints/")
    args = ap.parse_args()
    export(CKPT_DIR / args.ckpt, CKPT_DIR / args.out)
