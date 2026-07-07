# HANDOFF — Transformer Visualizer

A from-scratch character-level GPT + an **interactive "movie"** that explains how an
AI (ChatGPT-style) works end to end. This doc is for continuing the project in a
fresh chat. Read it fully before touching code.

- **Status: SHIPPED.** The full explainer (17 slides) is built, polished,
  reviewed for accuracy, made responsive, and **deployed to Vercel as a static,
  backend-free site** (client-side ONNX inference). The originally planned
  roadmap (phases 0–7) is complete. See §5 for what each step delivered and the
  short list of optional future polish.
- **Repo:** `C:\Users\logot\Downloads\AIlearn` · git branch `main` (commit regularly; pushed to `github.com/VarunKollipara/transformer-visualizer`, branch `main`).
- **User:** Varun — strong software engineer (React/Next/FastAPI/Python), **ML beginner**. Teach ML concepts; don't condescend on engineering.
- **Dates drift in-session; convert relative dates to absolute when writing memory.**

---

## 1. What exists (the stack)

### Python (model + API) — repo root, run with `uv`
- `src/tokenizer.py` — char-level tokenizer (65-char vocab from tinyshakespeare). `CharTokenizer`, `.from_chars()`.
- `src/data.py` — load corpus, train/val split, `get_batch` (B,T) batches.
- `src/model.py` — **GPT built by hand**: `TokenEmbedding`, `PositionalEmbedding` (learnable), `Head` (scaled dot-product self-attention with causal mask), `MultiHeadAttention`, `FeedForward` (Linear 128→512 → ReLU → 512→128), `LayerNorm` (by hand, with γ/β), `Block` (**pre-norm**: `x = x + attn(LN(x))` then `x = x + ffn(LN(x))`), `GPT`, plus a minimal `MiniGPT`. No `nn.Transformer`.
- `scripts/train.py` — trains, writes `checkpoints/gpt.pt` + `checkpoints/history.json` (loss curve + sample text per step). Config: `block_size=64, n_embd=128, num_heads=4, num_layers=3`, ~**619K params**, ~3000 steps, val loss ≈ **1.65**. ~10 min on CPU.
- `src/inference.py` — loads checkpoint; `forward_with_internals`, `generate`, `embeddings_2d` (PCA), `next_logits`, `load_history`.
- `src/api.py` — **FastAPI** (CORS open): `GET /api/info`, `POST /api/forward`, `POST /api/generate`, `GET /api/embeddings`, `GET /api/training`, `POST /api/logits`. Loads the model lazily.
- `data/input.txt` — tinyshakespeare. `NOTES.md` — plain-language concept log / glossary (the source of truth for the teaching copy; technically accurate, covers scaling, pre-norm, γ/β, etc.). `CLAUDE.md` — teaching contract.

### Frontend — `web/` (Next.js 16, React 19, Tailwind v4, `motion`/Framer Motion)
- **Heads up:** `web/AGENTS.md` — this Next.js has breaking changes; check `web/node_modules/next/dist/docs/` before Next-specific code. Its **SWC JSX transform** also has a whitespace quirk (see §3).
- `web/src/components/Experience.tsx` — **the controller.** Owns the current **slide** index. A `SCENES` manifest lists each scene + its ordered `phases` ({id,label}); these flatten into one **17-slide list** (`SLIDES`). Renders the current scene `Component` with a `phase` prop; `onNext`/`onBack` step the flat slide list. `AnimatePresence mode="wait"` is **keyed by scene id**, so the scene entrance/exit fires only at a *scene* boundary — within-scene slide (phase) changes keep the component mounted, which is what lets the morphs run.
- `web/src/components/scenes/*` — the scenes (see §1 order). `ui.tsx` (SceneTitle/SceneText), `Continue.tsx`, `types.ts` (`SceneProps = {phase, onNext, onBack, restart}` — `phase` is controlled by Experience).
- `web/src/components/TrainingViz.tsx` — loss-curve + scrubber; takes `onSample(text,step)` + `hideSample` props so the Finale can lift the sample text out and carry it.
- `web/src/lib/api.ts` — typed client: the 3 static endpoints fetch `/data/*.json`; the 3 live ones lazy-import `web/src/lib/onnx.ts` (client-side inference). `web/src/lib/example.ts` — `EXAMPLE_LINE = "To be, or not to be, that is the question"` (carried through the whole movie).
- `web/next.config.ts` — `output: "export"` (static export → `web/out`). `web/scripts/copy-ort.mjs` — copies the onnx wasm to `public/ort` on pre-dev/build. `web/.env.local` (`NEXT_PUBLIC_API_URL`) is **obsolete** — nothing reads it anymore.
- **Baked/deploy assets:** `web/public/data/{info,embeddings,training}.json` (committed), `web/public/model/gpt.onnx` (3.2MB, committed), `web/public/ort/*` (wasm, gitignored — regenerated at build).
- **Live component set is self-contained** (dead scroll-era cluster was deleted): `Experience`, `TrainingViz`, `scenes/{OpeningScene, PositionQueryScene, BlockScene, FinaleScene, Continue, ui, types}`, `lib/{api, example, onnx}`. Nothing else.

### Current structure — 4 scenes, 17 slides (in `Experience.tsx` SCENES)
1. **opening** (`OpeningScene`) — 4 slides: **invite → tokens → vectors → dims**. One continuous central morph: the row of characters morphs line → numbered tiles → tiles+mini-vectors → vectors collapse to 1/2/128 via a toggle.
2. **position-query** (`PositionQueryScene`) — 6 slides: **order → qkv → attention → think → widen → relu**. This is **both halves of a transformer block** (attention + MLP), merged so an element carries across every boundary: the "T" tile morphs order→qkv→attention; then the actual **T/o/b tiles fly out of the attention line** into the per-token MLP demo (think); the **T tile then threads** as an anchor into widen & relu.
3. **block** (`BlockScene`) — 5 slides: **residual → layernorm → stack → logits → softmax**. The **same token vector threads** residual (`x′`, emerald) → layernorm (normalized, amber) → the **base of the tower** (stack). Then the **prediction-bars panel** grows stack→logits and reshapes logits→softmax (recolor + temperature).
4. **finale** (`FinaleScene`) — 2 slides: **training → generate**. The **model's text canvas is the carried element**: the training scrubber's sample (surfaced from `TrainingViz` via `onSample`) becomes the live autoregressive generation output. Ends with "this exact loop becomes ChatGPT" + ↺ start over.

**Progress bar** (top-right, in `Experience`): **one clickable dot per slide**, grouped by scene with a small gap. Clicking any dot jumps straight to that slide (jumping remounts scene / sets the phase). This replaced the old "strictly linear, no jumping" rule (Varun asked for it).

Every scene boundary and every within-scene phase change is now an **element-carry morph** or a deliberate directional enter — there are no plain crossfades left.

---

## 2. How to run / verify

**The frontend is now fully standalone — no backend needed at runtime.** The
three "static" endpoints are baked to `web/public/data/*.json`; the three "live"
ones (forward/logits/generate) run the exported model **in the browser** via
onnxruntime-web. `src/api.py` (FastAPI) still exists but is only used to
*regenerate* those baked assets, not at runtime.

```bash
# Frontend, from web/ — this is all you need to run the app:
npm run dev          # dev server on :3000 (predev copies the onnx wasm to public/ort)
npm run build        # typecheck + lint + STATIC EXPORT -> web/out (prebuild copies the wasm)

# Python (uv, from repo root) — only to (re)train or (re)bake assets:
uv run python -m scripts.train           # (re)train -> checkpoints/gpt.pt + history.json (~10 min CPU)
uv run python -m scripts.export_static   # -> web/public/data/{info,embeddings,training}.json
uv run python -m scripts.export_onnx     # -> web/public/model/gpt.onnx (verifies vs PyTorch)
# (re-run the two export_* scripts after any retrain so the site matches the model)
```
- Shell is **PowerShell** (primary) on Windows; a Bash tool also exists. `uv` is the Python entrypoint.
- **Client-side inference:** `web/src/lib/onnx.ts` loads `/model/gpt.onnx` (3.2MB, single file) + vocab and reproduces the old API response shapes. Uses the CPU-wasm build (`onnxruntime-web/wasm`, single-threaded), wasm self-hosted from `/public/ort` (copied from node_modules at pre-dev/build, gitignored). Experience prewarms the model on the first Continue. First inference downloads ~17MB (wasm+model), then cached.
- **QA:** just `npm run dev` and drive the dots; no server to start. All 17 slides render with real data offline.

### ⚠️ Verification gotchas (read — these cost hours)
1. **Restart the preview server fresh before judging anything "broken."** Repeated hot-reloads through broken states corrupt the dev server (frozen advancement, code that's correct looks broken). `preview_start` may **reuse** a stale server — call `preview_stop` then `preview_start` to force a clean one.
2. **`preview_screenshot` is slower than the transitions** (~0.55–0.75 s), so you catch rest frames, not mid-morph. Verify structure/rest-state with `preview_eval` (DOM reads); you cannot reliably observe animation smoothness from here — reason about it or ask Varun.
3. **Navigation in eval:** the fastest way is the progress dots — `[...document.querySelectorAll('header button[aria-label^="Slide"]')][i].click()` jumps to slide i (0-based). The in-scene Continue button is still `button.rounded-full.bg-indigo-600`. The invite line is `[role=button]`. Finale heading is "Finally — you make it write".
4. **Overflow:** slides are tuned to fit ~**892px** content height. Shorter preview windows (e.g. 659px) will scroll the tall slides (ReLU especially). That's a **responsive-pass** concern, not a bug — check `document.querySelector('main .overflow-y-auto')` scrollHeight vs clientHeight, and note the window height before concluding overflow.

---

## 3. Animation rules learned the HARD way (do not violate)

- **`AnimatePresence mode="popLayout"` for any morph where siblings EXIT.** When some elements leave (e.g. 38 of the 41 tiles on attention→think), default AnimatePresence keeps them occupying layout space *during* their exit, so the survivors drift toward an intermediate spot and then **snap** to their final position when the space collapses. `mode="popLayout"` (with a `relative` parent) pulls exiting elements out of flow immediately → survivors glide straight home. This fixed the attention→think / think→widen snap.
- **A carried element must keep the SAME size across phases** — animate only its *position*. Size changes (a bar area that shrinks, a card that appears in one phase, padding that grows) distort during a layout morph. Bars in carried elements use **`scaleY` only** (fixed-height area, no per-bar `layout`); phase-specific extras (captions, buttons, cards) live **outside** the morphing element so they don't inflate its box. (See BlockScene's carried vector.)
- **Never animate a `motion` element's own `backgroundColor`/`borderColor`/border on a `layout` container via `animate`** — it can hang the `mode="wait"` exit. Use a **CSS** transition for colors (`transition-colors duration-500` + Tailwind classes).
- **One `AnimatePresence` per morph.** Don't stack multiple `mode="wait"` blocks reacting to the same state in one component — they deadlock. Pattern: ONE AnimatePresence for the morphing element; everything else is plain conditionals + enter-only `motion`.
- **Central-morph technique (reliable):** merge adjacent scenes into ONE component with a phase machine; the shared object is a `motion` element with `layout` + a **stable key** so it morphs between phases. Avoid cross-scene `layoutId` (fragile here) — merge instead.
- **Scene transitions are unified** (`SCENE_VARIANT` in Experience): one subtle fade + slight scale + rise, same for every scene, so a boundary reads as a coherent "advance" rather than a per-scene slide-then-jump. The morphs carry the interest; keep scene-level motion subtle.
- **Motion taste (Emil Kowalski):** mostly <300 ms for UI bits, `ease-out` entrances, spring for interactive, animate transform/opacity, a **touch slow** so morphs are watchable. Respect `prefers-reduced-motion` (still TODO in places).

### JSX whitespace gotcha (this Next's SWC)
This Next.js's SWC transform **trims the leading space of a multi-line text node that follows an inline `</strong>` / `</em>`**, mashing words in the render ("gatherhints", "block= attention"). Fix with an explicit `{" "}` after the closing tag. If you add/edit copy with inline bold/italic that wraps to the next source line, add the `{" "}`.

---

## 4. Direction & taste (from saved memory — honor these)

Memory dir: `C:\Users\logot\.claude\projects\C--Users-logot-Downloads-AIlearn\memory\` (`MEMORY.md` is the index).
- **project-vision** — interactive, click-to-learn explainer of how ChatGPT works end to end; our char-GPT is the live engine. Portfolio piece.
- **cinematic-direction** — an interactive **movie**: fullscreen scene state machine, **mandatory inline deep-dives covering EVERYTHING incl. internals**, **full continuity** (carry the "To be…" example, morph the same elements). Slightly slow pacing. *(Updated: the progress bar is now clickable / jumpable, reversing the original "strictly linear, no jumping" rule.)*
- **design-aesthetic** — education-focused **LIGHT** palette (warm paper, serif headings via Lora), NOT dark/neon. Accent colors: tokens=sky, embeddings=violet, attention=teal, training=rose/amber, generation=indigo. **Maximize visualizations.** Dual audience.
- **deploy-last** — Vercel/deployment is the **final** step; polish first.
- **code-writing-style** — write full, working code and explain it; **no TODO scaffolds**.

User is very particular about **motion feel** (rejected a "signature chip"; wanted full central morphs; flagged a drift-then-snap that popLayout fixed). Confirm feel before mass-replicating an animation idea.

---

## 5. What was done (the full arc) + optional future work

The roadmap is **complete**. What each of the final three steps delivered:

1. ✅ **Teaching / accuracy review** — reviewed all copy against `src/model.py`; content was already accurate, added 4 precision tweaks (attention scaling, pre-norm ordering note, ReLU "it all collapses", char-vs-subword tokenization note). NOTES.md was already accurate.
2. ✅ **Mobile / responsive pass** — scope: keep desktop no-scroll; **allow vertical scroll on phones**; kill horizontal overflow. Audited all 17 slides at 375px; fixed the only two horizontal-overflow spots (header title hidden below `sm` + dots centered; MLP widen bar row narrower/tighter below `sm`). Attention line wraps, grids/controls stack, SVGs scale. Tall slides just scroll on mobile (intended).
3. ✅ **Deploy** — the site is **backend-free and live on Vercel**:
   - **Part A:** `scripts/export_static.py` bakes info/embeddings/training → `web/public/data/*.json`; `api.ts` fetches those statically.
   - **Part B:** `scripts/export_onnx.py` exports the char-GPT → single-file `web/public/model/gpt.onnx` (3.2MB), verified vs PyTorch (Δ≈1e-6). `web/src/lib/onnx.ts` runs it in-browser (onnxruntime-web, CPU wasm) reproducing the exact API shapes; `api.ts` routes forward/logits/generate there.
   - **Part C:** `next.config.ts` `output:"export"` → static `web/out`. All 3 live endpoints verified working **with no backend** (attention weights, logits, streaming generation).
   - **Vercel setup that worked:** import the GitHub repo → **Root Directory = `web`**. If the Next.js framework preset auto-detects, defaults are fine; the static export also serves fine as a plain folder (Build `npm run build`, Output `out`). No env vars, no server. `web/public/{data,model}` committed; `web/public/ort` (wasm) copied at build. To redeploy the model after a **retrain**, re-run `export_static` + `export_onnx`, commit the new `data/*.json` + `gpt.onnx`, push.

Also done along the way: merged 8 scenes → 4 (element-carry morphs at every boundary), per-slide clickable progress bar, unified scene transitions, popLayout tile-morph fix, removed the dead scroll-era component cluster, fixed the SWC whitespace mashes.

### Optional future polish (nothing blocking)
- **`prefers-reduced-motion`** — not yet respected everywhere; a pass to honor it would be good for accessibility.
- **Short-viewport fit** — the ReLU slide scrolls below ~800px window height (no-scroll target is ~892px); tighten if desired.
- **First-load UX** — first inference pulls ~17MB (wasm+model); a small "warming up the model…" indicator on the attention/generate slides would smooth the wait beyond the existing background prewarm.
- **Delete the FastAPI backend?** `src/api.py` is now unused at runtime (kept only as a reference / alt way to regenerate assets). Could remove if you want a pure static repo.

---

## 6. Recent commits (for orientation)
Run `git log --oneline -30` for the full arc. Highlights (newest first):
- `20e4786` deploy: static export (`output: export`) for a host-agnostic build.
- `5cee686` deploy: prewarm the in-browser model on first advance.
- `8f6f5dc` deploy part B/C: client-side ONNX inference, backend fully removed.
- `60d280f` deploy part B1: export char-GPT to ONNX (verified vs PyTorch).
- `dc73be1` deploy part A: bake info/embeddings/training to static JSON.
- `9faf38f` responsive: fix horizontal overflow at mobile widths.
- `de82783` accuracy review: 4 precision tweaks to the teaching copy.
- `73a299a` fix missing spaces after inline bold/italic (SWC whitespace).
- `74cdd00` polish pass: fit the training slide + remove dead code.
- `580333a` fix attention→think tile morph: `popLayout` so survivors don't snap.
- `1d6c1c8` merge training + generate into one Finale scene (text carry).
- `1136930` merge norm/residual + blocks into one Block scene (vector carry).
- `832b6c4` per-slide progress bar (lift phase state to Experience).

Everything is committed; tree should be clean.
