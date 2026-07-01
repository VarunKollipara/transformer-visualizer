"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { getLogits, type Candidate } from "@/lib/api";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

// ── residual / layernorm data ──
// x and the layer's small edit, so x' = x + edit is only a nudge away from x.
const X = [0.55, 0.35, 0.7, 0.45, 0.6, 0.4, 0.5, 0.65];
const EDIT = [0.08, -0.06, 0.05, 0.1, -0.04, 0.07, -0.05, 0.06];
const XP = X.map((v, j) => v + EDIT[j]); // x' = x + layer(x) — the carried token vector
const MEAN = XP.reduce((a, b) => a + b, 0) / XP.length;
const STD = Math.sqrt(XP.reduce((a, b) => a + (b - MEAN) ** 2, 0) / XP.length) || 1;
const normFrac = (v: number) => 0.52 + ((v - MEAN) / STD) * 0.09; // tidy, even band

// ── softmax / logits data ──
function softmax(logits: number[], t: number): number[] {
  const temp = Math.max(t, 0.01);
  const s = logits.map((l) => l / temp);
  const m = Math.max(...s);
  const e = s.map((x) => Math.exp(x - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}
const FALLBACK: Candidate[] = [
  { char: "e", logit: 9.6 }, { char: "u", logit: 7.0 }, { char: "r", logit: 7.0 },
  { char: "l", logit: 6.9 }, { char: "a", logit: 6.7 }, { char: "i", logit: 6.7 },
  { char: "y", logit: 6.1 }, { char: "o", logit: 6.0 },
];

type Phase = "residual" | "layernorm" | "stack" | "logits" | "softmax";

// One shared timing for every layout (position/size) morph in this scene, so
// elements move together instead of at different speeds.
const LAYOUT_T = { duration: 0.6, ease: [0.22, 1, 0.36, 1] as const };

// Both halves of "how a token's vector becomes a prediction": the two tricks
// that make depth work (residual, LayerNorm), then stacking blocks and reading
// off the answer (logits, softmax). The SAME token vector is the carried thread:
// x' (residual) → normalized (LayerNorm) → the vector at the base of the tower.
export default function BlockScene({ phase: phaseProp, onNext }: SceneProps) {
  const phase = phaseProp as Phase;
  const [normalized, setNormalized] = useState(false);
  const [layers, setLayers] = useState(3);
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [temp, setTemp] = useState(1);

  useEffect(() => {
    getLogits("To be, or not to b", 8)
      .then((d) => setCands(d.candidates))
      .catch(() => {});
  }, []);

  const data = cands ?? FALLBACK;
  const logits = useMemo(() => data.map((c) => c.logit), [data]);
  const probs = useMemo(() => softmax(logits, temp), [logits, temp]);
  const lo = Math.min(...logits);
  const hi = Math.max(...logits);
  const logitH = (l: number) => 0.14 + 0.86 * (hi === lo ? 0.5 : (l - lo) / (hi - lo));
  const maxProb = Math.max(...probs);

  const ln = phase === "layernorm";
  const onChart = phase === "logits" || phase === "softmax";
  const showPanel = phase === "stack" || onChart;
  const showVec = phase === "residual" || ln || phase === "stack";
  const vecFrac = (v: number) => (ln && normalized ? normFrac(v) : v);

  return (
    <div className="text-center">
      {/* title + intro, keyed by phase */}
      <motion.div key={`t-${phase}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "residual" && (
          <>
            <SceneTitle>Trick 1: add, don&apos;t overwrite</SceneTitle>
            <SceneText>
              We&apos;re about to stack many of these layers. The first thing that
              makes that safe: every layer <strong className="text-stone-700">adds a
              small edit</strong> to the vector instead of rewriting it from scratch.
            </SceneText>
          </>
        )}
        {ln && (
          <>
            <SceneTitle>Trick 2: keep the numbers tidy</SceneTitle>
            <SceneText>
              That very vector now passes through{" "}
              <strong className="text-stone-700">LayerNorm</strong> (
              <strong className="text-stone-700">layer normalization</strong>), which
              rescales its numbers back to a steady, even range — so values don&apos;t
              snowball as they pass through layer after layer.
            </SceneText>
          </>
        )}
        {phase === "stack" && (
          <>
            <SceneTitle>Stack it into a tower</SceneTitle>
            <SceneText>
              One <strong className="text-stone-700">block</strong>{" "}
              = attention + an MLP, each wrapped in the residual and LayerNorm you just saw. Stack the
              blocks and that token&apos;s vector — the one entering at the bottom —
              gets refined again and again as it rises, until the model commits to a
              guess for the next character.
            </SceneText>
          </>
        )}
        {phase === "logits" && (
          <>
            <SceneTitle>That guess is a score for every character</SceneTitle>
            <SceneText>
              The tower&apos;s output is turned into one number — a{" "}
              <strong className="text-stone-700">logit</strong> — for each of the 65
              possible next characters. A logit is just a raw, unbounded score (it can
              be negative or positive): higher means the model favours that character
              more.
            </SceneText>
          </>
        )}
        {phase === "softmax" && (
          <>
            <SceneTitle>Softmax turns those scores into probabilities</SceneTitle>
            <SceneText>
              Logits don&apos;t add up to anything yet.{" "}
              <strong className="text-stone-700">Softmax</strong> fixes that in two
              moves: <strong className="text-stone-700">exponentiate</strong> every
              score (e<sup>x</sup> — makes them all positive and stretches the gaps
              apart), then <strong className="text-stone-700">divide by the total</strong>{" "}
              so they sum to 100%.
            </SceneText>
          </>
        )}
      </motion.div>

      {/* chart caption (chart phases only) */}
      {onChart && (
        <motion.p key={`cap-${phase}`} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }} className="mt-4 text-[13px] text-stone-400">
          {phase === "logits"
            ? "the model's raw scores — taller = more favoured"
            : "exaggerate the gaps, then normalise → probabilities"}
        </motion.p>
      )}

      {/* ── prediction bars panel: shared across stack/logits/softmax ── */}
      {showPanel && (
        <motion.div
          layout
          transition={{ layout: LAYOUT_T }}
          className={`mx-auto transition-colors duration-500 ${
            onChart
              ? "mt-3 max-w-lg border border-transparent bg-transparent px-0 py-0"
              : "mt-6 w-fit rounded-xl border border-stone-200 bg-white px-3 pb-1.5 pt-2"
          }`}
        >
          <motion.div layout className={`flex items-end justify-center ${onChart ? "gap-2 sm:gap-3" : "gap-[3px]"}`}>
            {data.map((c, i) => {
              const h = phase === "softmax" ? probs[i] / (maxProb || 1) : logitH(c.logit);
              return (
                <motion.div
                  key={c.char}
                  layout
                  transition={{ layout: LAYOUT_T }}
                  className={`flex flex-col items-center ${onChart ? "flex-1" : "w-3"}`}
                >
                  {onChart && (
                    <motion.span
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.25, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                      className="mb-1 font-mono text-[11px] text-stone-500"
                    >
                      {phase === "logits" ? c.logit.toFixed(1) : `${(probs[i] * 100).toFixed(0)}%`}
                    </motion.span>
                  )}
                  <div className={`w-full ${onChart ? "h-40 max-w-9" : "h-7 max-w-2.5"}`}>
                    <motion.div
                      animate={{ scaleY: Math.max(0.02, h) }}
                      transition={{ type: "spring", stiffness: 200, damping: 24 }}
                      style={{ transformOrigin: "bottom", height: "100%" }}
                      className={`mx-auto w-full rounded-t-md transition-colors duration-500 ${
                        phase === "softmax" ? "bg-indigo-500" : "bg-stone-300"
                      }`}
                    />
                  </div>
                  <motion.span
                    layout="position"
                    className={`mt-1.5 font-mono font-semibold text-stone-800 ${onChart ? "text-sm" : "text-[10px]"}`}
                  >
                    {show(c.char)}
                  </motion.span>
                </motion.div>
              );
            })}
          </motion.div>
        </motion.div>
      )}

      {/* ── STACK part A: caption + tower of blocks (base is the carried vector) ── */}
      {phase === "stack" && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}>
          <p className="mt-1.5 text-[11px] text-stone-400">
            next-character prediction — one bar per possible character
          </p>
          <span className="mt-1 block text-stone-300">↑</span>
          <div className="mx-auto flex w-full max-w-xs flex-col items-stretch gap-1.5">
            {Array.from({ length: layers }, (_, i) => layers - 1 - i).map((n) => (
              <motion.div
                key={n}
                layout
                initial={{ opacity: 0, scale: 0.9, y: 8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 26 }}
                className="rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-left"
              >
                <div className="text-sm font-semibold text-teal-800">Block {n + 1}</div>
                <div className="font-mono text-xs text-teal-600">attention → MLP</div>
              </motion.div>
            ))}
            <span className="text-stone-300">↑</span>
          </div>
        </motion.div>
      )}

      {/* ── RESIDUAL: the diagram + the inputs that form x' ── */}
      {phase === "residual" && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}>
          <div className="mx-auto mt-7 max-w-xl rounded-xl border border-stone-200 bg-white p-5">
            <div className="flex items-center justify-center gap-2 font-mono text-sm">
              <span className="rounded bg-stone-100 px-2 py-1">x</span>
              <span className="text-stone-300">→</span>
              <span className="rounded bg-indigo-100 px-2 py-1 text-indigo-800">layer</span>
              <span className="text-stone-300">→</span>
              <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-800">＋</span>
              <span className="text-stone-300">→</span>
              <span className="rounded bg-stone-100 px-2 py-1">x′</span>
            </div>
            <div className="relative mx-auto mt-1 h-5 w-64">
              <svg viewBox="0 0 256 24" className="w-full">
                <path d="M20 2 Q128 26 236 2" fill="none" stroke="#34d399" strokeWidth="2" strokeDasharray="3 3" />
              </svg>
              <span className="absolute left-1/2 top-3 -translate-x-1/2 text-[11px] text-emerald-600">
                original x carried straight across
              </span>
            </div>
            <div className="mt-6 flex items-end justify-center gap-6">
              <div className="flex flex-col items-center gap-1">
                <div className="flex h-16 items-end gap-[3px]">
                  {X.map((v, j) => (
                    <span key={j} style={{ height: `${v * 56 + 2}px` }} className="w-2 rounded-sm bg-stone-300" />
                  ))}
                </div>
                <span className="text-xs text-stone-400">x</span>
              </div>
              <span className="mb-5 text-stone-300">+</span>
              <div className="flex flex-col items-center gap-1">
                <div className="flex h-16 items-end gap-[3px]">
                  {EDIT.map((v, j) => (
                    <span key={j} style={{ height: `${Math.abs(v) * 56 + 2}px`, alignSelf: v < 0 ? "flex-start" : "flex-end" }} className="w-2 rounded-sm bg-indigo-400" />
                  ))}
                </div>
                <span className="text-xs text-stone-400">layer(x) — a small edit</span>
              </div>
              <span className="mb-5 text-stone-300">=</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* ── carried vector: x' (residual) → normalized (layernorm) → tower base (stack) ──
          Kept the SAME size in every phase (bars use scaleY only, no per-bar
          layout) so the morph is purely positional — it glides between slides
          rather than stretching. The LayerNorm caption/button sit outside the
          morphing element so they don't inflate its box. */}
      {showVec && (
        <div className={`mx-auto ${phase === "stack" ? "mt-1" : "mt-5"}`}>
          {ln && (
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.35 }} className="mb-3 text-sm text-stone-500">
              The very vector we just built — some numbers run higher than others.
              Press normalise:
            </motion.p>
          )}
          <motion.div layout transition={{ layout: LAYOUT_T }} className="mx-auto w-fit">
            <div className="flex h-14 items-end justify-center gap-[4px]">
              {XP.map((v, j) => (
                <motion.div
                  key={j}
                  animate={{ scaleY: Math.max(0.06, vecFrac(v)) }}
                  transition={{ type: "spring", stiffness: 280, damping: 22 }}
                  style={{ transformOrigin: "bottom", height: "100%" }}
                  className={`w-3 rounded-sm transition-colors duration-500 ${phase === "residual" ? "bg-emerald-400" : "bg-amber-400"}`}
                />
              ))}
            </div>
            <span className="mt-1 block font-mono text-[11px] text-stone-400">
              {phase === "residual" ? "x′ = x + layer(x)" : phase === "stack" ? "a token's vector, entering the stack" : "one token's vector"}
            </span>
          </motion.div>
          {ln && (
            <motion.button
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.35 }}
              onClick={() => setNormalized((n) => !n)}
              className="mt-4 rounded-full bg-stone-800 px-4 py-1.5 text-sm font-medium text-white hover:bg-stone-700"
            >
              {normalized ? "reset" : "normalise"}
            </motion.button>
          )}
        </div>
      )}

      {/* ── STACK part B: the block-count slider (below the base vector) ── */}
      {phase === "stack" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2, duration: 0.4 }}>
          <div className="mt-5 flex items-center justify-center gap-3">
            <button onClick={() => setLayers((l) => Math.max(1, l - 1))} className="h-8 w-8 rounded-full border border-stone-300 text-lg text-stone-600 hover:bg-stone-100">−</button>
            <span className="w-24 font-mono text-sm text-stone-700">{layers} block{layers > 1 ? "s" : ""}</span>
            <button onClick={() => setLayers((l) => Math.min(8, l + 1))} className="h-8 w-8 rounded-full border border-stone-300 text-lg text-stone-600 hover:bg-stone-100">+</button>
          </div>
          <SceneText delay={0.1} className="mt-4 text-[15px]">
            More blocks = more refinement. Our model uses{" "}
            <strong className="text-stone-700">3</strong>. GPT-3 stacks{" "}
            <strong className="text-stone-700">96</strong> — same idea, only taller.
          </SceneText>
        </motion.div>
      )}

      {/* residual / layernorm explanations */}
      {phase === "residual" && (
        <SceneText delay={0.3} className="mt-5 text-[15px]">
          Because the original is always carried forward, even a layer that learns
          nothing useful can&apos;t wreck the signal — and the learning signal has a
          clear, short path back through every layer.
        </SceneText>
      )}
      {ln && (
        <p className="mx-auto mt-4 max-w-xl text-sm text-stone-500">
          Under the hood it&apos;s two steps: <strong className="text-stone-700">
          subtract the average</strong> (recentre around 0), then{" "}
          <strong className="text-stone-700">divide by the spread</strong> (a
          consistent scale). Same shape, tamed magnitude. Modern GPTs actually do
          this <em>just before</em> each attention and MLP step — that ordering
          (&ldquo;pre-norm&rdquo;) is what keeps a deep stack stable.
        </p>
      )}

      {/* logits-phase note */}
      {phase === "logits" && (
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mt-4 max-w-md text-[15px] text-stone-500"
        >
          Real scores after{" "}
          <span className="font-mono text-stone-600">&ldquo;To be, or not to b&rdquo;</span>.
          It clearly favours <span className="font-mono font-semibold text-stone-700">e</span> —
          but a tall bar isn&apos;t yet a probability.
        </motion.p>
      )}

      {/* softmax-phase: temperature */}
      {phase === "softmax" && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.4 }}>
          <label className="mx-auto mt-5 block max-w-md">
            <span className="mb-1 flex justify-between text-sm text-stone-500">
              <span>Temperature — how boldly it picks</span>
              <span className="font-mono font-semibold text-indigo-700">{temp.toFixed(2)}</span>
            </span>
            <input type="range" min={0.1} max={2} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} className="w-full accent-indigo-600" />
          </label>
          <p className="mx-auto mt-2 max-w-md text-[13px] text-stone-400">
            Temperature divides every logit <em>before</em> softmax.{" "}
            <strong className="text-stone-600">Low</strong> (&lt;1) sharpens the gaps;{" "}
            <strong className="text-stone-600">High</strong>{" "}(&gt;1) levels the field.
          </p>
        </motion.div>
      )}

      {/* advance */}
      {phase === "residual" && <Continue onClick={onNext} label="And the second trick" delay={0.4} />}
      {ln && <Continue onClick={onNext} label="Now stack them into a tower" delay={0.3} />}
      {phase === "stack" && <Continue onClick={onNext} label="Score every possible next character" delay={0.3} />}
      {phase === "logits" && <Continue onClick={onNext} label="Make them real probabilities" delay={0.2} />}
      {phase === "softmax" && <Continue onClick={onNext} label="But first — how did it learn all this?" delay={0.2} />}
    </div>
  );
}
