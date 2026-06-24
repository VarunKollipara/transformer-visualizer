# HANDOFF — Transformer Visualizer

A from-scratch character-level GPT + an **interactive "movie"** that explains how an
AI (ChatGPT-style) works end to end. This doc is for continuing the project in a
fresh chat. Read it fully before touching code.

- **Repo:** `C:\Users\logot\Downloads\AIlearn` · git branch `main` (commit regularly; pushed to a private GitHub repo).
- **User:** Varun — strong software engineer (React/Next/FastAPI/Python), **ML beginner**. Teach ML concepts; don't condescend on engineering.
- **Today's date context in-session drifted; convert relative dates to absolute when writing memory.**

---

## 1. What exists (the stack)

### Python (model + API) — repo root, run with `uv`
- `src/tokenizer.py` — char-level tokenizer (65-char vocab from tinyshakespeare). `CharTokenizer`, `.from_chars()`.
- `src/data.py` — load corpus, train/val split, `get_batch` (B,T) batches.
- `src/model.py` — **GPT built by hand**: `TokenEmbedding`, `PositionalEmbedding`, `Head` (self-attention), `MultiHeadAttention`, `FeedForward`, `LayerNorm`, `Block`, `GPT`, plus a minimal `MiniGPT`. No `nn.Transformer`.
- `scripts/train.py` — trains and writes `checkpoints/gpt.pt` + `checkpoints/history.json` (loss curve + sample text per step). Config: `block_size=64, n_embd=128, num_heads=4, num_layers=3`, ~**619K params**, ~3000 steps, val loss ≈ **1.66**. ~10 min on CPU.
- `src/inference.py` — loads checkpoint; `forward_with_internals`, `generate`, `embeddings_2d` (PCA), `next_logits`, `load_history`.
- `src/api.py` — **FastAPI**. Endpoints (CORS open): `GET /api/info`, `POST /api/forward`, `POST /api/generate`, `GET /api/embeddings`, `GET /api/training`, `POST /api/logits`. Loads the model lazily.
- `data/input.txt` — tinyshakespeare (downloaded via `scripts/download_data.py`).
- `NOTES.md` — plain-language concept log (glossary) for the educational content. `CLAUDE.md` — teaching contract.

### Frontend — `web/` (Next.js 16, React 19, Tailwind v4, `motion`/Framer Motion)
- **Heads up:** `web/AGENTS.md` says this Next.js has breaking changes — check `web/node_modules/next/dist/docs/` before writing Next-specific code. (`'use client'` etc. still standard.)
- `web/src/components/Experience.tsx` — **the controller.** A fullscreen, **no-scroll**, strictly-linear sequence of scenes (an interactive movie). You advance by interacting; never scroll. Each scene id has its **own** entrance/exit transition (`VARIANTS` map). `AnimatePresence mode="wait"`.
- `web/src/components/scenes/*` — the scenes. `ui.tsx` (SceneTitle/SceneText), `Continue.tsx`, `types.ts` (`SceneProps = {onNext, onBack, restart}`).
- `web/src/lib/api.ts` — typed client for the FastAPI endpoints. `web/src/lib/example.ts` — `EXAMPLE_LINE = "To be, or not to be, that is the question"` (the example carried through the whole movie).
- `web/.env.local` — `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`.

### Current scene order (in `Experience.tsx` SCENES)
1. **opening** (`OpeningScene`) — 4 phases, one continuous central morph: **invite (glowing line) → tokens (numbers appear) → vectors (bars grow) → dims (the same vectors collapse to 1/2/128 numbers via a toggle)**.
2. **position-query** (`PositionQueryScene`) — 3 phases, central morph: **order (positioned tiles) → qkv (the "T" tile slides to center, emits Query/Key/Value) → attention (it expands into the full line, lit with REAL attention weights; click any char)**.
3. **mlp** (`MLPScene`) — widen→ReLU→narrow; interactive ReLU.
4. **norm-residual** (`NormResidualScene`) — residual + LayerNorm (normalize toggle).
5. **blocks** (`BlocksScene`) — stack the tower (±blocks).
6. **softmax** (`SoftmaxScene`) — real logits → probabilities + temperature slider.
7. **training** (`TrainingScene`) — loss-curve scrubber (uses `/api/training`), noise→Shakespeare.
8. **generate** (`GenerateScene`) — finale: prompt + temperature, live autoregressive reveal + per-char distribution, "this is ChatGPT" close, ↺ start over.

Scenes 1 & 2 are the **continuous central-morph showpieces**. Scenes 3–8 are deep-dives, each with its **own bespoke entrance transition** (not a central morph — they're abstract objects that don't share a morph target; the user approved this).

---

## 2. How to run / verify

```bash
# Python env is uv-managed. From repo root:
uv run python -m scripts.train          # (re)train -> checkpoints/gpt.pt + history.json (~10 min, CPU). Needed once.
uv run uvicorn src.api:app --port 8000  # backend (loads checkpoint lazily)

# Frontend, from web/:
npm run dev          # dev server on :3000
npm run build        # typecheck + lint + prod build (DO THIS to catch errors)
```
- Shell is **PowerShell** (primary) on Windows; a Bash tool also exists. `uv` is the Python entrypoint (no system python).
- The page expects the backend at `127.0.0.1:8000`; each scene degrades gracefully if it's down.

### ⚠️ Verification gotchas (these cost us hours — read!)
1. **The preview dev server gets CORRUPTED by repeated hot-reloads through broken states.** Symptoms: scene advancement freezes, screenshots hang, code that is actually correct looks broken. **FIX: stop and restart the preview server fresh** (`preview_stop` then `preview_start`) before concluding anything is broken. Re-test after a clean restart.
2. **`preview_screenshot` is flaky/often hangs in this environment; `preview_eval` is reliable.** Use `preview_eval` to read DOM state (headings, button text, element counts) to verify scenes when screenshots time out.
3. To drive the linear flow in eval: click the opening line via `document.querySelector('[role=button]')`, then repeatedly click `document.querySelector('button.rounded-full.bg-indigo-600')` (the Continue button) until the finale ("Finally — the model writes"). Full path = ~11 Continue clicks.

---

## 3. Animation rules learned the HARD way (do not violate)

- **Never animate a `motion` element's own `backgroundColor`/`borderColor`/border on a `layout` container via `animate`** — it can hang the `AnimatePresence mode="wait"` scene exit (the scene never finishes exiting → frozen). Use a **CSS** transition for colors (`transition-colors duration-500` + Tailwind color classes). This is exactly how the opening's outline-box fade is done now.
- **Do not put multiple `AnimatePresence mode="wait"` blocks reacting to the same state in one component** — they deadlock (stale content from prior phases lingers; renderer hangs). **Pattern that works:** ONE `AnimatePresence` only for the element that must morph; render everything else with plain conditionals + enter-only `motion` (no `exit`, no `AnimatePresence`). The opening and position-query scenes follow this.
- **Central morph technique (reliable):** merge adjacent scenes into ONE component with phase state; the shared central object is a `motion` element with `layout` and a **stable `key`** so it morphs between phases (like the opening letters→tiles, or the "T" tile → centered token → line). Avoid cross-scene `layoutId` shared elements (fragile here).
- **Motion taste (Emil Kowalski):** mostly <300ms, `ease-out` for entrances, spring for interactive bits, animate transform/opacity, respect `prefers-reduced-motion`. Pacing should be **a touch slow** so morphs are watchable (user asked for this).

---

## 4. Direction & taste (from saved memory — honor these)

Memory dir: `C:\Users\logot\.claude\projects\C--Users-logot-Downloads-AIlearn\memory\` (`MEMORY.md` is the index).
- **project-vision** — interactive, click-to-learn explainer of how ChatGPT works end to end; our char-GPT is the live engine. Portfolio piece.
- **cinematic-direction** — it's an interactive **movie**: fullscreen no-scroll scene state machine, **strictly linear**, **mandatory inline deep-dives covering EVERYTHING incl. internals**, **full continuity** (carry the "To be…" example, morph the same elements), slightly slow pacing.
- **design-aesthetic** — education-focused **LIGHT** palette (warm paper, serif headings via Lora), NOT dark/neon. Semantic accent colors: tokens=sky, embeddings=violet, attention=teal, training=rose, generation=indigo. **Maximize visualizations.** Dual audience (beginner + knowledgeable).
- **deploy-last** — Vercel/deployment is the **final** step; polish everything first.
- **code-writing-style** — write full, working code and explain it; **no TODO scaffolds** (overrides CLAUDE.md's scaffold rule).

User feedback pattern: very particular about **motion feel** — has rejected several approaches (top "signature chip" was rejected; wanted full **central morphs** where one screen's object morphs into the next). Confirm feel before mass-replicating an animation idea.

---

## 5. What's next (open work)

- **More central morphs (optional):** the user liked the opening & position→attention continuous morphs. They asked to "do them all," but the internal scenes (mlp/norm/blocks/softmax/training) are abstract and don't share a morph object, so they currently use bespoke entrance transitions instead (accepted). If pushing further, merge specific adjacent pairs into phase-machines (see §3 technique) — one at a time, verify with a fresh server each time.
- **Mobile responsiveness pass** — it's desktop-first; tune scenes for narrow screens.
- **Cleanup:** several files are now unused (build-valid but dead): `scenes/WhyDimensionsScene.tsx`, `scenes/AttentionScene.tsx`, `scenes/PositionalScene.tsx`, `scenes/QKVScene.tsx`, `scenes/HeroScene.tsx`, `scenes/TokenizeScene.tsx`, `scenes/EmbeddingsScene.tsx`, `scenes/OutroScene.tsx`, and the old scroll-era components (`Section`, `SectionNav`, `FlowConnector`, `Reveal`, `Concept`, `Aside`, `GoDeeper`, `ExampleChips`, `PipelineDiagram`, `TokenizationScene`, `CorpusSample`, `lib/glossary.ts`). Some logic was repurposed (`TrainingViz`, `QKVDiagram` are still used). Delete dead ones when convenient.
- **Final step (per memory):** in-browser ONNX inference (remove backend dependency) + Vercel deploy. **Do this last.**

## 6. Recent commits (for orientation)
- `9cb7845` opening: fold "why 128 numbers" in as a 4th phase (vectors collapse to 1/2/128 in place); refactor off AnimatePresence.
- `68512c7` opening: fix outline-box snap on first transition (CSS color fade).
- `d5dff7c` Q/K/V → attention central morph (token expands into the full line).
- `b7b2ad5` positional → Q/K/V central morph (merged scene).
- `f4581b7` per-scene bespoke entrance/exit transitions.

Run `git log --oneline -25` for the rest. Everything is committed; tree should be clean.
