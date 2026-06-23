"""FastAPI backend for the visualizer.

Serves the trained model's internals to the frontend:
  GET  /api/info      model + vocabulary info
  POST /api/forward   run one forward pass over text -> attention + probabilities
  POST /api/generate  autoregressive generation -> each step's choice + top-k

Run with:  uv run uvicorn src.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import inference

app = FastAPI(title="Transformer Visualizer API")

# The Next.js dev server runs on a different origin; allow it to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model():
    """Load the model once, lazily, so the server can start before training ends."""
    try:
        return inference.load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No checkpoint yet. Run `uv run python -m scripts.train` first.",
        )


class ForwardRequest(BaseModel):
    text: str
    top_k: int = 10


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 10


@app.get("/api/info")
def info():
    _, tok, cfg = get_model()
    return {"config": cfg, "vocab": tok.chars, "vocab_size": tok.vocab_size}


@app.post("/api/forward")
def forward(req: ForwardRequest):
    model, tok, _ = get_model()
    return inference.forward_with_internals(model, tok, req.text, req.top_k)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    model, tok, _ = get_model()
    return inference.generate(
        model, tok, req.prompt, req.max_new_tokens, req.temperature, req.top_k
    )


class LogitsRequest(BaseModel):
    text: str
    top_k: int = 8


@app.post("/api/logits")
def logits(req: LogitsRequest):
    model, tok, _ = get_model()
    return inference.next_logits(model, tok, req.text, req.top_k)


@app.get("/api/embeddings")
def embeddings():
    model, tok, _ = get_model()
    return inference.embeddings_2d(model, tok)


@app.get("/api/training")
def training():
    try:
        return inference.load_history()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No training history yet. Run `uv run python -m scripts.train`.",
        )
