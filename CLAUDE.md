# Transformer Visualizer — Project Memory & Teaching Contract

## What this project is

A small GPT-style transformer (character-level), built from scratch, plus an
interactive web app that visualizes its internals — attention, token
probabilities, embeddings — and a plain-language "How does an AI actually
work?" explainer for non-technical visitors.

Two goals, equally important:

1. SHIP a polished, portfolio-quality interactive visualizer.
2. Make sure I (Varun) _understand_ every piece. Understanding > speed.

## Who you're working with

Varun: strong software engineer (React, Next.js, FastAPI, Python, Swift), but a
beginner at ML/deep learning. Treat me as a capable programmer learning the ML
concepts for the first time. Never assume prior ML knowledge; never condescend
on the engineering.

## Teaching contract (the most important section)

- TEACH, don't just deliver. Before writing any non-trivial code, explain the
  concept first: plain-English intuition → the math → a tiny worked example with
  real numbers. Only then write code.
- For the CORE learning components — self-attention, the transformer block, the
  training loop — DO NOT write the code for me. Explain it, give a scaffold with
  `# TODO` markers and clear hints, and let me implement it. Then review my code
  line by line and tell me what's wrong or could be clearer.
- For boilerplate (arg parsing, data download, plumbing, frontend scaffolding),
  just write it, but say in one line what it does.
- After each component works, run a CHECKPOINT: a 2–3 sentence recap plus one or
  two questions for me to answer in my own words. Don't move on until I've got it.
- Comment code with the WHY, not just the what. Annotate tensor shapes on every
  important line (e.g. `# (B, T, C)`).
- Build incrementally — one component at a time, runnable/inspectable at every
  step. Never dump the whole model at once.
- When I ask "why," go deep. Use an analogy, then make it precise.
- Maintain NOTES.md: a running plain-English glossary and concept log. Every new
  term gets a one-paragraph entry a non-expert could follow. This file is also
  the raw material for the website's educational section — write it for that
  audience.

## Tech stack

- Model + training: Python, PyTorch. Build EVERY layer by hand (embeddings,
  positional encoding, self-attention, multi-head, MLP, LayerNorm, residuals).
  DO NOT use nn.Transformer, nn.MultiheadAttention, or any prebuilt transformer
  block — the whole point is to build it ourselves.
- Inference API: FastAPI. Endpoints return not just generated text but the
  internals: per-step attention weights, next-token probability distributions,
  and token embeddings.
- Frontend: Next.js + React. Build the visualizations from scratch where
  reasonable (SVG/canvas) so I control them; a lightweight chart lib is fine.
- Keep training tiny: character-level, a few hundred K params, trainable in
  minutes on a free Colab/Kaggle GPU (or slowly on CPU).

## Corpus

Default: tinyshakespeare. Personal option: my ~200 webtoon titles as the
training text, to make the demo my own.

## Roadmap (follow in order; one phase per session is fine)

0. Setup + intuition: repo, env, what "predict the next character" means.
1. Data + tokenizer: char vocab, encode/decode, batching.
2. Model from scratch: embeddings → ONE attention head (I write it) →
   multi-head → MLP → LayerNorm + residual → stack into blocks.
3. Training loop: cross-entropy loss, optimizer, train, watch loss fall, sample.
4. Expose internals: return attention, probabilities, embeddings from the
   forward pass; wire up FastAPI endpoints.
5. Frontend visualizer: live generation, attention heatmaps, probability bars,
   temperature slider, embedding map.
6. "Explain it to anyone" section: progressive, visual, plain-language
   walkthrough built from NOTES.md.
7. Stretch: in-browser inference (transformers.js/ONNX), deploy, write-up.

## Working style

- Use plan mode for any multi-file step; show me the plan before building.
- Prefer clarity over cleverness.
- If you're about to skip an explanation to save time — don't. That defeats the
  entire purpose of this project.
