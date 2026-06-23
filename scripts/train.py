"""Phase 3: train MiniGPT, watch the loss fall, then generate a sample.

Run:  uv run python -m scripts.train

You implement the 4 TODO lines in the training loop (the forward -> backward ->
step you learned in Phase 0). Everything else is set up for you.
"""

from __future__ import annotations

import torch

from src.data import get_batch, load_corpus, train_val_split
from src.model import GPT

# --- config (bigger than the minimal model; runs in ~10-15 min on CPU) ---
block_size = 64       # context length (how many chars the model sees at once)
batch_size = 32       # sequences processed in parallel per step
n_embd = 128          # embedding width (C)
num_heads = 4         # attention heads per block (head size = n_embd // num_heads)
num_layers = 3        # number of stacked transformer blocks
max_steps = 3000      # how many gradient steps to take
eval_interval = 300   # how often to measure & print loss
eval_iters = 50       # how many batches to average when measuring loss
learning_rate = 1e-3  # step size for the optimizer
seed = 1337

torch.manual_seed(seed)

# --- data ---
tok, data = load_corpus()
train_data, val_data = train_val_split(data)

# --- model + optimizer ---
model = GPT(tok.vocab_size, n_embd, num_heads, num_layers, block_size)
print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """Average loss over several batches of train and val data (honest readout)."""
    out = {}
    model.eval()  # eval mode (matters once we add dropout/etc. later)
    for name, d in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(d, block_size, batch_size)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()  # back to train mode
    return out


# --- training loop ---
history: list[dict] = []   # loss curve over time
samples: list[dict] = []   # what the model generates at each checkpoint
sample_seed = torch.zeros((1, 1), dtype=torch.long)  # generate from a newline

for step in range(max_steps + 1):
    # every so often, measure and report progress on train + val
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
        history.append({"step": step, "train": losses["train"], "val": losses["val"]})
        # capture a short generation so we can *watch* the model learn to write
        sample_ids = model.generate(sample_seed, max_new_tokens=160)[0].tolist()
        samples.append({"step": step, "text": tok.decode(sample_ids)})

    # grab one fresh batch of training data
    xb, yb = get_batch(train_data, block_size, batch_size)

    # 1) FORWARD: run the model to get predictions and how wrong they are (loss).
    logits, loss = model(xb, yb)

    # 2) RESET: clear gradients from the previous step (PyTorch accumulates them
    #    by default, so without this they'd pile up across steps).
    optimizer.zero_grad(set_to_none=True)

    # 3) BACKWARD: backprop — fill in the gradient of the loss for every parameter.
    loss.backward()

    # 4) STEP: the optimizer nudges every parameter one step downhill. THIS is the
    #    line that actually changes the model. Everything above just decides how.
    optimizer.step()


# --- save a checkpoint so the API / visualizer can load the trained model ---
from pathlib import Path

ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / "gpt.pt"
torch.save(
    {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": tok.vocab_size,
            "n_embd": n_embd,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "block_size": block_size,
        },
        "chars": tok.chars,
    },
    ckpt_path,
)
print(f"saved checkpoint -> {ckpt_path}")

# Save the training history + sample generations for the "watch it learn" viz.
import json

(ckpt_dir / "history.json").write_text(
    json.dumps(
        {
            "history": history,
            "samples": samples,
            "config": {
                "n_embd": n_embd,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "block_size": block_size,
                "max_steps": max_steps,
                "params": sum(p.numel() for p in model.parameters()),
            },
        }
    )
)
print(f"saved training history -> {ckpt_dir / 'history.json'}")


# --- generate a sample from the trained model ---
print("\n--- trained sample ---")
start = torch.zeros((1, 1), dtype=torch.long)  # a single newline to start
print(tok.decode(model.generate(start, max_new_tokens=500)[0].tolist()))
