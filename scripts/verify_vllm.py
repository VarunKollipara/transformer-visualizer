"""Verify the vLLM server is faithfully serving our model.

The strongest cheap end-to-end check: greedy decoding is deterministic, so the
server (vLLM's own GPT-2 kernels, on GPU) must reproduce the exact same tokens
as transformers running our exported checkpoint locally (CPU). Any porting or
serving bug — wrong weight layout, tied embeddings, wrong dtype, tokenizer
drift — shows up as divergence within a few characters.

Context math: the prompt "ROMEO:" is 6 tokens and n_positions=64, so the server
can generate at most 58 tokens (absolute positional embeddings can't slide).

Run (Windows side, server running in WSL):  uv run python -m scripts.verify_vllm
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import torch

SERVER = "http://localhost:8000"
PROMPT = "ROMEO:"

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default="checkpoints/hf-gpt2", help="exported model folder")
_ap.add_argument("--max-tokens", type=int, default=58, help="n_positions minus prompt length")
ARGS = _ap.parse_args()
MAX_TOKENS = ARGS.max_tokens


def local_greedy() -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(ARGS.dir)
    tok = AutoTokenizer.from_pretrained(ARGS.dir)
    model.eval()

    ids = tok.encode(PROMPT)
    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            logits = model(torch.tensor([ids])).logits  # (1, T, V)
            ids.append(int(logits[0, -1].argmax()))
    return tok.decode(ids[len(tok.encode(PROMPT)):])


def server_greedy() -> str:
    body = json.dumps(
        {
            "model": "char-gpt",
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,       # greedy
            "ignore_eos": True,     # '\n' is nominally EOS; we want all 58 tokens
            # '\n' is marked special (it's our EOS), and the API strips special
            # tokens from output text by default — which would delete every
            # newline the model generates. Char-level models need this off.
            "skip_special_tokens": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main() -> None:
    local = local_greedy()
    served = server_greedy()
    print(f"prompt          : {PROMPT!r}")
    print(f"local  (CPU fp32): {local!r}")
    print(f"served (GPU)     : {served!r}")   # dtype = whatever the server was launched with
    if local == served:
        print(f"\nMATCH: all {MAX_TOKENS} greedy tokens identical through the serving stack")
    else:
        diverge = next(
            (i for i, (a, b) in enumerate(zip(local, served)) if a != b), min(len(local), len(served))
        )
        print(f"\nDIVERGED at generated char {diverge}: local {local[diverge:diverge+5]!r} vs served {served[diverge:diverge+5]!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
