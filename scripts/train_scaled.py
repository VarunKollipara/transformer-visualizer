"""Train the scaled-up GPT (~25M params) on GPU, for the optimization phase.

Same skeleton as scripts/train.py (Phase 3), with the ingredients a 40x-bigger
model needs:

  - GPU device placement (first GPU-trained model in this project).
  - TF32 matmuls: Ampere tensor cores run fp32 matmuls ~10x faster at slightly
    reduced mantissa precision — free speed for training.
  - Dropout 0.2: at ~35 passes over a 1.1 MB corpus, a 25M model can memorize
    the text; dropout forces general patterns instead. (Phase-3 model was too
    small and undertrained to need it.)
  - Weight decay on 2-D matrices only: shrinks weights toward zero so they stay
    only as big as the data justifies. Norm gains/shifts and biases are
    scale-setters, not pattern-memorizers — convention is to exempt them.
  - LR warmup + cosine decay: fresh Adam has garbage moment estimates for the
    first steps (full LR there can blow up a bigger model), and cosine decay
    lands training in a flatter minimum with small careful steps at the end.
  - Gradient clipping at 1.0: cheap insurance against a single bad batch
    producing a huge step.

Run inside WSL (needs the CUDA torch):
  cd /mnt/c/Users/logot/Downloads/AIlearn && ~/vllm-env/bin/python -m scripts.train_scaled
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from src.data import get_batch, load_corpus, train_val_split
from src.model import GPT

# --- config: ~25M params (12 * n_layers * n_embd^2 = 12 * 8 * 512^2) ---
name = "md"           # checkpoints/gpt-md.pt, history-md.json
block_size = 256      # 4x the small model: makes KV-cache experiments meaningful
batch_size = 32
n_embd = 512
num_heads = 8
num_layers = 8
dropout = 0.2
max_steps = 5000
eval_interval = 250
eval_iters = 40
learning_rate = 3e-4  # peak LR (bigger model -> smaller LR than the 1e-3 before)
min_lr = 3e-5         # cosine decays to this
warmup_steps = 200
weight_decay = 0.1
grad_clip = 1.0
seed = 1337

torch.manual_seed(seed)
torch.set_float32_matmul_precision("high")  # allow TF32 on Ampere tensor cores

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

# --- data (the whole encoded corpus fits in VRAM; index batches right there) ---
tok, data = load_corpus()
train_data, val_data = train_val_split(data)
train_data, val_data = train_data.to(device), val_data.to(device)

# --- model ---
model = GPT(tok.vocab_size, n_embd, num_heads, num_layers, block_size, dropout).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {n_params:,}")

# --- optimizer: weight decay for matrices, none for 1-D params (norms, biases) ---
decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
)


def lr_at(step: int) -> float:
    """Linear warmup to peak, then cosine decay to min_lr."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)  # 0 -> 1
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out = {}
    model.eval()  # crucial now: turns dropout OFF for an honest readout
    for split, d in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(d, block_size, batch_size)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --- training loop ---
history: list[dict] = []
samples: list[dict] = []
sample_seed = torch.zeros((1, 1), dtype=torch.long, device=device)
t0 = time.time()

for step in range(max_steps + 1):
    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(
            f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
            f"| lr {lr_at(step):.2e} | {elapsed:6.0f}s", flush=True,
        )
        history.append({"step": step, "train": losses["train"], "val": losses["val"]})
        model.eval()  # sample without dropout
        sample_ids = model.generate(sample_seed, max_new_tokens=160)[0].tolist()
        model.train()
        samples.append({"step": step, "text": tok.decode(sample_ids)})

    # set this step's learning rate on every param group
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)

    xb, yb = get_batch(train_data, block_size, batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# --- save ---
ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / f"gpt-{name}.pt"
torch.save(
    {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {
            "vocab_size": tok.vocab_size,
            "n_embd": n_embd,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "block_size": block_size,
            "dropout": dropout,  # recorded for provenance; inference passes 0.0
        },
        "chars": tok.chars,
    },
    ckpt_path,
)
print(f"saved checkpoint -> {ckpt_path}")

(ckpt_dir / f"history-{name}.json").write_text(
    json.dumps(
        {
            "history": history,
            "samples": samples,
            "config": {
                "n_embd": n_embd, "num_heads": num_heads, "num_layers": num_layers,
                "block_size": block_size, "max_steps": max_steps, "params": n_params,
            },
        }
    )
)
print(f"saved training history -> {ckpt_dir / f'history-{name}.json'}")

model.eval()
print("\n--- trained sample ---")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tok.decode(model.generate(start, max_new_tokens=500)[0].tolist()))
