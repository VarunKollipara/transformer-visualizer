"use client";

import { AnimatePresence, motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

type Phase = "cost" | "batching" | "cache" | "quantize" | "measure";

// Series colors, validated (dataviz six checks, light surface):
// indigo = the optimized path, amber = the naive path. ΔE(CVD) ≈ 136.
const INDIGO = "#4f46e5";
const AMBER = "#d97706";

/* ────────────────────────── measured data ──────────────────────────
   Every number below is a real measurement from this repo's benchmarks/
   (25M-param model, RTX 3070, vLLM 0.24 / PyTorch). Not illustrative. */

// bench_serve.py, md-fp32: output tokens/sec by concurrent users
const BATCH = [
  { users: 1, toks: 621 },
  { users: 8, toks: 4233 },
  { users: 32, toks: 11943 },
  { users: 128, toks: 19364 },
];

// bench_kv_cache.py, CPU: ms per token by context length
const KV = {
  ctx: [64, 128, 192, 256],
  uncached: [22.1, 33.5, 50.0, 64.1],
  cached: [10.2, 11.0, 11.7, 12.8],
};

// quantize_gptq.py + eval_ppl.py
const MEM = [
  { label: "16-bit", mib: 48.5, color: "#a5b4fc" }, // indigo-300
  { label: "4-bit", mib: 13.1, color: INDIGO },
];

const EASE = [0.22, 1, 0.36, 1] as const;
// One shared timing for every layout (position/size) morph in this scene, so
// the carried stage reads as one object gliding between slides.
const LAYOUT_T = { duration: 0.7, ease: EASE };

/* ────────────────────────── charts ────────────────────────── */

// Rounded-top bar as a path, grown via scaleY from the baseline.
function Bar({
  x, y, w, h, fill, delay, title,
}: {
  x: number; y: number; w: number; h: number; fill: string; delay: number; title: string;
}) {
  const r = Math.min(4, w / 2, h);
  const d = `M ${x} ${y + h} L ${x} ${y + r} Q ${x} ${y} ${x + r} ${y} L ${x + w - r} ${y} Q ${x + w} ${y} ${x + w} ${y + r} L ${x + w} ${y + h} Z`;
  return (
    <motion.path
      d={d}
      fill={fill}
      style={{ transformBox: "fill-box", transformOrigin: "bottom" }}
      initial={{ scaleY: 0 }}
      animate={{ scaleY: 1 }}
      transition={{ delay, duration: 0.6, ease: EASE }}
    >
      <title>{title}</title>
    </motion.path>
  );
}

// Throughput vs concurrent users — the continuous-batching payoff.
function BatchingChart() {
  const W = 560, H = 250, L = 52, R = 16, T = 26, B = 40;
  const max = 20000;
  const iw = W - L - R, ih = H - T - B;
  const bw = 34;
  const xAt = (i: number) => L + (iw / BATCH.length) * (i + 0.5) - bw / 2;
  const yAt = (v: number) => T + ih * (1 - v / max);
  const grid = [5000, 10000, 15000, 20000];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img"
      aria-label="Bar chart: output tokens per second versus concurrent users. 1 user: 621. 8: 4,233. 32: 11,943. 128: 19,364.">
      {grid.map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={yAt(v)} y2={yAt(v)} stroke="#e7e5e4" strokeWidth={1} />
          <text x={L - 8} y={yAt(v) + 3.5} textAnchor="end" fontSize={10.5} fill="#a8a29e">
            {v / 1000}k
          </text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={T + ih} y2={T + ih} stroke="#d6d3d1" strokeWidth={1} />
      {BATCH.map((b, i) => (
        <g key={b.users}>
          <Bar x={xAt(i)} y={yAt(b.toks)} w={bw} h={ih - (yAt(b.toks) - T)} fill={INDIGO}
            delay={0.25 + i * 0.14} title={`${b.users} users: ${b.toks.toLocaleString()} tokens/s`} />
          <motion.text
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ delay: 0.65 + i * 0.14, duration: 0.35 }}
            x={xAt(i) + bw / 2} y={yAt(b.toks) - 7} textAnchor="middle"
            fontSize={12} fontWeight={600} fill="#44403c"
          >
            {b.toks >= 1000 ? `${(b.toks / 1000).toFixed(b.toks < 10000 ? 1 : 0)}k` : b.toks}
          </motion.text>
          <text x={xAt(i) + bw / 2} y={T + ih + 16} textAnchor="middle" fontSize={11.5} fill="#78716c">
            {b.users}
          </text>
        </g>
      ))}
      <text x={L + iw / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="#a8a29e">
        people served at once
      </text>
      <text x={14} y={T - 10} fontSize={11} fill="#a8a29e">characters written per second (total)</text>
    </svg>
  );
}

// ms per token vs context length, with and without the KV cache.
function KvChart() {
  const W = 560, H = 260, L = 46, R = 128, T = 24, B = 40;
  const maxY = 70;
  const iw = W - L - R, ih = H - T - B;
  const xAt = (c: number) => L + iw * ((c - KV.ctx[0]) / (KV.ctx[KV.ctx.length - 1] - KV.ctx[0]));
  const yAt = (v: number) => T + ih * (1 - v / maxY);
  const path = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"} ${xAt(KV.ctx[i])} ${yAt(v)}`).join(" ");
  const series = [
    { name: "without cache", vals: KV.uncached, color: AMBER, end: "64 ms" },
    { name: "with cache", vals: KV.cached, color: INDIGO, end: "13 ms" },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img"
      aria-label="Line chart: milliseconds per character versus text length, with and without the KV cache. Without: grows 22 to 64 ms. With: stays flat around 10 to 13 ms.">
      {[20, 40, 60].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={yAt(v)} y2={yAt(v)} stroke="#e7e5e4" strokeWidth={1} />
          <text x={L - 8} y={yAt(v) + 3.5} textAnchor="end" fontSize={10.5} fill="#a8a29e">{v}</text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={T + ih} y2={T + ih} stroke="#d6d3d1" strokeWidth={1} />
      {KV.ctx.map((c) => (
        <text key={c} x={xAt(c)} y={T + ih + 16} textAnchor="middle" fontSize={11.5} fill="#78716c">
          {c}
        </text>
      ))}
      <text x={L + iw / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="#a8a29e">
        characters of text so far
      </text>
      <text x={12} y={T - 8} fontSize={11} fill="#a8a29e">ms to write the next character</text>
      {series.map((s, si) => (
        <g key={s.name}>
          <motion.path
            d={path(s.vals)} fill="none" stroke={s.color} strokeWidth={2} strokeLinecap="round"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
            transition={{ delay: 0.3 + si * 0.55, duration: 0.9, ease: "easeInOut" }}
          />
          {s.vals.map((v, i) => (
            <motion.circle
              key={i} cx={xAt(KV.ctx[i])} cy={yAt(v)} r={4} fill="#fff"
              stroke={s.color} strokeWidth={2}
              initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.35 + si * 0.55 + i * 0.18, duration: 0.25 }}
            >
              <title>{`${s.name} — ${KV.ctx[i]} chars: ${v} ms/char`}</title>
            </motion.circle>
          ))}
          <motion.text
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ delay: 1.15 + si * 0.55, duration: 0.35 }}
            x={xAt(KV.ctx[KV.ctx.length - 1]) + 10} y={yAt(s.vals[s.vals.length - 1]) + 4}
            fontSize={11.5} fontWeight={600} fill="#57534e"
          >
            <tspan fill={s.color}>●</tspan> {s.name}
          </motion.text>
        </g>
      ))}
    </svg>
  );
}

// Memory footprint before/after 4-bit quantization.
function MemoryChart() {
  const W = 560, H = 120, L = 64, R = 90;
  const iw = W - L - R;
  const max = 50;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img"
      aria-label="Bar chart: model size. 16-bit weights: 48.5 mebibytes. 4-bit weights: 13.1 mebibytes.">
      {MEM.map((m, i) => {
        const y = 22 + i * 44;
        const w = iw * (m.mib / max);
        return (
          <g key={m.label}>
            <text x={L - 10} y={y + 14} textAnchor="end" fontSize={12} fontWeight={600} fill="#57534e">
              {m.label}
            </text>
            <motion.rect
              x={L} y={y} height={20} rx={4} fill={m.color}
              initial={{ width: 0 }} animate={{ width: w }}
              transition={{ delay: 0.3 + i * 0.25, duration: 0.7, ease: EASE }}
            >
              <title>{`${m.label} weights: ${m.mib} MiB`}</title>
            </motion.rect>
            <motion.text
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              transition={{ delay: 0.85 + i * 0.25, duration: 0.35 }}
              x={L + w + 8} y={y + 14} fontSize={12} fontWeight={600} fill="#44403c"
            >
              {m.mib} MiB
            </motion.text>
          </g>
        );
      })}
    </svg>
  );
}

// Phase-1 loop: characters appear one by one; each one pulses "the whole model
// ran". Loops forever — it's the cost intuition the rest of the scene attacks.
function CostLoop() {
  const chars = "the whole model runs for every single character".split("");
  const step = 0.12;
  const total = chars.length * step + 1.6;
  return (
    <div className="mx-auto mt-5 max-w-xl rounded-xl border border-stone-200 bg-stone-50 p-5 text-left">
      <div className="font-mono text-[15px] leading-relaxed text-stone-800">
        {chars.map((c, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 1, 1] }}
            transition={{
              delay: i * step, duration: 0.3,
              repeat: Infinity, repeatDelay: total - 0.3,
            }}
          >
            {c}
          </motion.span>
        ))}
      </div>
      <motion.div
        className="mt-3 flex items-center gap-2 text-xs text-stone-400"
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
      >
        <span className="inline-block h-2 w-2 rounded-full bg-indigo-500" />
        25 million multiplications… per character
      </motion.div>
    </div>
  );
}

/* ────────────────────────── the carried stage ──────────────────────────
   The scene's protagonist: THE MODEL as a physical object that persists
   across all five slides (never keyed by phase) and accumulates each trick —
   the crowd fans in (batching), a memory docks on (KV cache), the box itself
   shrinks (quantization), and finally it earns its "measured" badge. All
   morphs are layout animations on the same shared timing. */
function ModelStage({ phase }: { phase: Phase }) {
  const crowd = phase !== "cost";                                 // trick 1
  const memory = phase === "cache" || phase === "quantize" || phase === "measure"; // trick 2
  const small = phase === "quantize" || phase === "measure";      // trick 3
  const badge = phase === "measure";

  return (
    <motion.div
      layout
      transition={{ layout: LAYOUT_T }}
      className="mx-auto mt-7 flex min-h-[110px] max-w-xl items-center justify-center gap-3 sm:gap-4"
    >
      {/* the people sending requests */}
      <div className="flex flex-col items-end gap-1.5">
        <motion.div
          layout
          transition={{ layout: LAYOUT_T }}
          className="rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-700"
        >
          you
        </motion.div>
        <AnimatePresence mode="popLayout">
          {crowd &&
            [0, 1, 2].map((i) => (
              <motion.div
                key={`u${i}`}
                layout
                initial={{ opacity: 0, x: -14 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -14 }}
                transition={{ delay: 0.2 + i * 0.12, duration: 0.4, ease: EASE, layout: LAYOUT_T }}
                className="h-5 w-5 rounded-full border border-stone-300 bg-white"
              >
                <span className="sr-only">another person&apos;s request</span>
              </motion.div>
            ))}
        </AnimatePresence>
      </div>

      {/* requests flowing in */}
      <motion.span
        layout
        animate={{ opacity: [0.35, 1, 0.35] }}
        transition={{
          opacity: { duration: 1.4, repeat: Infinity, ease: "easeInOut" },
          layout: LAYOUT_T,
        }}
        className="text-stone-400"
        aria-hidden
      >
        →
      </motion.span>

      {/* THE MODEL — the object every slide operates on */}
      <motion.div
        layout
        transition={{ layout: LAYOUT_T }}
        className={`relative rounded-xl border-2 border-indigo-200 bg-white text-center shadow-sm ${
          small ? "px-3 py-2" : "px-6 py-4"
        }`}
      >
        <motion.div layout="position" className={`font-semibold text-stone-800 ${small ? "text-xs" : "text-sm"}`}>
          the model
        </motion.div>
        <div className={`mt-0.5 font-mono text-stone-400 ${small ? "text-[9px]" : "text-[11px]"}`}>
          25M weights ·{" "}
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.span
              key={small ? "4bit" : "16bit"}
              layout
              initial={{ opacity: 0, y: 7 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -7 }}
              transition={{ duration: 0.35, ease: EASE }}
              className={`inline-block font-semibold ${small ? "text-indigo-600" : "text-stone-500"}`}
            >
              {small ? "4-bit · 13.1 MiB" : "16-bit · 48.5 MiB"}
            </motion.span>
          </AnimatePresence>
        </div>
        {/* the heartbeat: it's always running */}
        <motion.span
          animate={{ opacity: [0.3, 1, 0.3], scale: [0.85, 1, 0.85] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-indigo-500"
          aria-hidden
        />
        {/* the final slide's stamp */}
        <AnimatePresence>
          {badge && (
            <motion.span
              key="badge"
              initial={{ opacity: 0, scale: 0.6, rotate: -8 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0, scale: 0.6 }}
              transition={{ delay: 0.35, duration: 0.4, ease: EASE }}
              className="absolute -right-4 -top-3 rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700"
            >
              ⏱ measured
            </motion.span>
          )}
        </AnimatePresence>
      </motion.div>

      {/* the memory it gains in trick 2 */}
      <AnimatePresence mode="popLayout">
        {memory && (
          <motion.div
            key="kv"
            layout
            initial={{ opacity: 0, x: 18 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 18 }}
            transition={{ duration: 0.5, ease: EASE, layout: LAYOUT_T }}
            className="flex items-center gap-2"
          >
            <span className="text-stone-400" aria-hidden>⇄</span>
            <div
              className={`rounded-lg border border-dashed border-indigo-300 bg-indigo-50/60 text-center ${
                small ? "px-2 py-1.5" : "px-3 py-2"
              }`}
            >
              <div className={`font-semibold text-indigo-700 ${small ? "text-[10px]" : "text-xs"}`}>memory</div>
              <div className={`text-indigo-400 ${small ? "text-[8px]" : "text-[10px]"}`}>
                everything it already read
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ────────────────────────── the scene ────────────────────────── */

// A chart panel with a consistent entrance.
function Panel({ children, delay = 0.25 }: { children: React.ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.55, ease: EASE }}
      className="mx-auto mt-5 max-w-xl rounded-xl border border-stone-200 bg-white p-4"
    >
      {children}
    </motion.div>
  );
}

// Epilogue: the systems act. Everything the visitor just learned (the loop, the
// re-reading, the weights) becomes a performance problem — and every claim on
// screen is a measurement from this repo's benchmark harness. The title/copy
// and charts are keyed per phase; the ModelStage between them is NOT — it's
// the carried element that glides and grows through the whole scene.
export default function SpeedScene({ phase: phaseProp, onNext, restart }: SceneProps) {
  const phase = phaseProp as Phase;

  return (
    <div className="text-center">
      {/* per-phase title + intro (keyed: fresh entrance each slide) */}
      <motion.div key={`t-${phase}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "cost" && (
          <>
            <SceneTitle>Epilogue: why isn&apos;t ChatGPT slow?</SceneTitle>
            <SceneText>
              Think about what you just did. Every single character took a{" "}
              <strong className="text-stone-700">full pass through the model</strong> —
              and real assistants do this with billions of parameters, for millions
              of people, all at once. Making that fast is its own engineering craft:{" "}
              <strong className="text-stone-700">inference optimization</strong>.
              We tried the industry&apos;s three big tricks on this very model — and
              measured everything.
            </SceneText>
          </>
        )}
        {phase === "batching" && (
          <>
            <SceneTitle>Trick 1 — serve everyone in one pass</SceneTitle>
            <SceneText>
              One person&apos;s next-character request barely wakes the GPU up. So
              inference servers like <strong className="text-stone-700">vLLM</strong>{" "}
              stack many people&apos;s requests into each pass —{" "}
              <strong className="text-stone-700">continuous batching</strong>
              {" — "}and new arrivals hop in mid-flight. Here&apos;s our model,
              measured:
            </SceneText>
          </>
        )}
        {phase === "cache" && (
          <>
            <SceneTitle>Trick 2 — remember what you already read</SceneTitle>
            <SceneText>
              Remember the loop: to write character #201, the model re-reads all 200
              before it — recomputing every key and value you saw in the attention
              scene — then keeps <em>one</em> row of answers. But frozen history never
              changes. So the model grows a{" "}
              <strong className="text-stone-700">memory</strong>: the{" "}
              <strong className="text-stone-700">KV cache</strong> keeps those keys
              and values, and each new character only pays for itself. We built it
              into this model and measured:
            </SceneText>
          </>
        )}
        {phase === "quantize" && (
          <>
            <SceneTitle>Trick 3 — store every weight in 4 bits</SceneTitle>
            <SceneText>
              Each of the model&apos;s 25 million learned numbers normally takes 16
              bits of memory. <strong className="text-stone-700">Quantization</strong>{" "}
              rounds each one to a 4-bit code — just 16 levels — chosen cleverly (an
              algorithm called <strong className="text-stone-700">GPTQ</strong>{" "}
              adjusts neighbouring weights to cancel each rounding error). Watch the
              model itself shrink:
            </SceneText>
          </>
        )}
        {phase === "measure" && (
          <>
            <SceneTitle>The twist: half of it did nothing</SceneTitle>
            <SceneText>
              There it is — shared, remembering, 3.7× smaller. But honest results
              from our own benchmark harness: on a model this small, some famous
              tricks simply don&apos;t pay.
            </SceneText>
          </>
        )}
      </motion.div>

      {/* THE CARRIED ELEMENT — persists and morphs across every slide */}
      <ModelStage phase={phase} />

      {/* per-phase evidence below the stage (keyed: fresh entrance each slide) */}
      <motion.div key={`c-${phase}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.45 }}>
        {phase === "cost" && (
          <>
            <CostLoop />
            <Continue onClick={onNext} label="Trick 1: share the ride" delay={0.5} />
          </>
        )}

        {phase === "batching" && (
          <>
            <Panel>
              <BatchingChart />
            </Panel>
            <SceneText delay={0.45} className="mt-4 text-[15px]">
              31× more text per second — and each individual person only waits about
              twice as long per character (1.4 → 2.7&nbsp;ms). Sharing the ride is
              nearly free.
            </SceneText>
            <Continue onClick={onNext} label="Trick 2: stop re-reading" delay={0.55} />
          </>
        )}

        {phase === "cache" && (
          <>
            <Panel>
              <KvChart />
            </Panel>
            <SceneText delay={0.45} className="mt-4 text-[15px]">
              Same math, reorganized — the output is{" "}
              <strong className="text-stone-700">identical to the last bit</strong>,
              but the cost per character stops growing with the length of the text.
              This is why a long chat doesn&apos;t get slower with every message.
            </SceneText>
            <Continue onClick={onNext} label="Trick 3: shrink the numbers" delay={0.55} />
          </>
        )}

        {phase === "quantize" && (
          <>
            <Panel>
              <MemoryChart />
            </Panel>
            <SceneText delay={0.45} className="mt-4 text-[15px]">
              3.7× smaller — and the writing got{" "}
              <strong className="text-stone-700">no worse at all</strong>
              {" (we scored it on held-out Shakespeare: 4.401 before, 4.400 after — a tie). That's "}
              how big models fit on small chips, and on giant ones it makes them
              faster too: less memory to drag around.
            </SceneText>
            <Continue onClick={onNext} label="One last twist" delay={0.55} />
          </>
        )}

        {phase === "measure" && (
          <>
            <motion.ul
              initial="hidden"
              animate="show"
              variants={{ show: { transition: { staggerChildren: 0.15, delayChildren: 0.4 } } }}
              className="mx-auto mt-5 max-w-xl space-y-3 text-left"
            >
              {[
                ["Half-precision math (fp16)", "made our model 17% slower — nothing it speeds up was the bottleneck."],
                ["The 4-bit kernel", "ran 4× slower than 16-bit here… our weights already fit in the GPU's cache. At ChatGPT scale, the same kernel wins 2–3×."],
                ["The KV cache on GPU", "1.0× — our hand-written loop was bottlenecked by overhead, not math. On CPU: 3.7×."],
              ].map(([head, rest], i) => (
                <motion.li
                  key={i}
                  variants={{ hidden: { opacity: 0, x: -12 }, show: { opacity: 1, x: 0, transition: { duration: 0.45, ease: EASE } } }}
                  className="rounded-lg border border-stone-200 bg-stone-50 px-4 py-3 text-[15px] leading-relaxed text-stone-600"
                >
                  <strong className="text-stone-800">{head}</strong> {rest}
                </motion.li>
              ))}
            </motion.ul>
            <SceneText delay={0.9} className="mt-6 text-[15px]">
              That&apos;s the real lesson of the craft:{" "}
              <strong className="text-stone-700">measure first, optimize second</strong>.
              Every optimization attacks one scarce resource — compute, memory
              traffic, or overhead — and does nothing unless that resource is the
              bottleneck. The numbers on these slides came from the same little model
              you drove a minute ago.
            </SceneText>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2, duration: 0.4 }}>
              <button onClick={restart} className="mt-6 text-sm font-medium text-indigo-600 hover:text-indigo-500">
                ↺ start the whole story over
              </button>
            </motion.div>
          </>
        )}
      </motion.div>
    </div>
  );
}
