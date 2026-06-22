# Transformer Visualizer

A small GPT-style transformer (character-level), built from scratch in PyTorch,
with an interactive web app that visualizes its internals — attention, token
probabilities, embeddings — and a plain-language explainer of how it all works.

Two goals: **ship a polished interactive visualizer**, and **understand every
piece of it**.

## Status

Phase 0 — setup & foundations. See [the roadmap](CLAUDE.md) for the full plan
and [NOTES.md](NOTES.md) for the plain-language concept log.

## Project layout

```
src/        Python: tokenizer, model, training, inference API (built incrementally)
data/       Training corpora (small text files)
NOTES.md    Running glossary / concept log — written for non-experts
CLAUDE.md   Project memory & teaching contract
```

## Environment

Managed with [uv](https://docs.astral.sh/uv/). uv handles the Python version
(pinned in `.python-version`) and the virtual environment for you.

```bash
uv sync                 # create the venv and install dependencies
uv run python -V        # run anything inside the env
```
