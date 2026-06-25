"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { getLogits, type Candidate } from "@/lib/api";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

// Plain JS softmax, with the temperature divide folded in, mirroring the model.
function softmax(logits: number[], t: number): number[] {
  const temp = Math.max(t, 0.01);
  const s = logits.map((l) => l / temp); // 1. divide every score by temperature
  const m = Math.max(...s); // (shift for numerical stability — doesn't change the result)
  const e = s.map((x) => Math.exp(x - m)); // 2. exponentiate: makes everything positive, exaggerates gaps
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum); // 3. divide by the total so they sum to 1 (100%)
}

// A deterministic little "final vector" — the strip that rises out of the tower.
const FINAL_VEC = Array.from(
  { length: 16 },
  (_, j) => 0.3 + 0.65 * Math.abs(Math.sin(j * 1.7 + 0.6)),
);

type Phase = "stack" | "logits" | "softmax";

export default function BlocksSoftmaxScene({ onNext }: SceneProps) {
  const [phase, setPhase] = useState<Phase>("stack");
  const [layers, setLayers] = useState(3);
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [temp, setTemp] = useState(1);

  useEffect(() => {
    // the model's REAL scores for what follows "...to b" — it should love "e"
    getLogits("To be, or not to b", 8)
      .then((d) => setCands(d.candidates))
      .catch(() => {});
  }, []);

  const logits = useMemo(() => cands?.map((c) => c.logit) ?? [], [cands]);
  const probs = useMemo(() => softmax(logits, temp), [logits, temp]);

  // Heights share one rule: taller = more favoured. Softmax is monotonic in the
  // logit, so the bars keep their order — only the GAPS between them change.
  const lo = logits.length ? Math.min(...logits) : 0;
  const hi = logits.length ? Math.max(...logits) : 1;
  const logitH = (l: number) => 0.14 + 0.86 * (hi === lo ? 0.5 : (l - lo) / (hi - lo));
  const maxProb = probs.length ? Math.max(...probs) : 1;

  const onChart = phase === "logits" || phase === "softmax";

  return (
    <div className="text-center">
      {/* title + intro, keyed by phase so each swap animates */}
      <motion.div key={phase} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "stack" && (
          <>
            <SceneTitle>Stack it into a tower</SceneTitle>
            <SceneText>
              One <strong className="text-stone-700">block</strong> = attention
              (tokens share context) + an MLP (each token thinks), each wrapped in a
              residual and LayerNorm. Stack the blocks and a token&apos;s vector gets
              refined again and again as it rises.
            </SceneText>
          </>
        )}
        {phase === "logits" && (
          <>
            <SceneTitle>The final vector becomes one score per character</SceneTitle>
            <SceneText>
              At the top of the tower, the last token&apos;s vector is multiplied by
              one more set of learned weights to produce a{" "}
              <strong className="text-stone-700">logit</strong> for every possible
              next character — a raw, unbounded score (it can be negative or
              positive) saying how much the model favours that character.
            </SceneText>
          </>
        )}
        {phase === "softmax" && (
          <>
            <SceneTitle>Softmax turns scores into probabilities</SceneTitle>
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

      {/* ── the persistent morph: the "final vector" strip ──
          It exists in every phase with a stable key + layout, so it glides from
          the top of the tower up to the head of the chart. */}
      <motion.div
        layout
        transition={{ layout: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } }}
        className={`mx-auto flex w-fit flex-col items-center ${onChart ? "mt-6" : "mt-7"}`}
      >
        <div className="flex h-9 items-end gap-[2px] rounded-lg border border-violet-200 bg-violet-50 px-2 py-1.5">
          {FINAL_VEC.map((h, j) => (
            <motion.span
              key={j}
              initial={{ scaleY: 0 }}
              animate={{ scaleY: h }}
              transition={{ delay: j * 0.015, type: "spring", stiffness: 260, damping: 22 }}
              style={{ transformOrigin: "bottom", height: "100%" }}
              className="block w-1 rounded-sm bg-violet-400"
            />
          ))}
        </div>
        <span className="mt-1 text-[11px] text-violet-500">
          final vector{onChart ? "" : " (128 numbers)"}
        </span>
      </motion.div>

      {/* ── STACK phase: the tower below the vector ── */}
      {phase === "stack" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
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
            <div className="rounded-md bg-stone-200 px-3 py-1.5 text-sm text-stone-600">
              token + position vectors
            </div>
          </div>

          <div className="mt-5 flex items-center justify-center gap-3">
            <button
              onClick={() => setLayers((l) => Math.max(1, l - 1))}
              className="h-8 w-8 rounded-full border border-stone-300 text-lg text-stone-600 hover:bg-stone-100"
            >
              −
            </button>
            <span className="w-24 font-mono text-sm text-stone-700">
              {layers} block{layers > 1 ? "s" : ""}
            </span>
            <button
              onClick={() => setLayers((l) => Math.min(8, l + 1))}
              className="h-8 w-8 rounded-full border border-stone-300 text-lg text-stone-600 hover:bg-stone-100"
            >
              +
            </button>
          </div>

          <SceneText delay={0.1} className="mt-5 text-[15px]">
            More blocks = more refinement and richer patterns. Our model uses{" "}
            <strong className="text-stone-700">3</strong>. GPT-3 stacks{" "}
            <strong className="text-stone-700">96</strong>. The idea is identical —
            only the height changes.
          </SceneText>
        </motion.div>
      )}

      {/* ── LOGITS / SOFTMAX phases: the bars ──
          These persist across both phases (stable keys), so the SAME bar reshapes
          from a logit height into a probability height — the central morph. */}
      {onChart && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
          <p className="mt-3 text-[13px] text-stone-400">
            {phase === "logits"
              ? "× output weights →  one score per character"
              : "÷ exaggerate & normalise →  probabilities"}
          </p>

          {cands ? (
            <div className="mx-auto mt-3 flex h-44 max-w-lg items-end justify-center gap-2 sm:gap-3">
              {cands.map((c, i) => {
                const h = phase === "logits" ? logitH(c.logit) : probs[i] / (maxProb || 1);
                return (
                  <div key={c.char} className="flex h-full flex-1 flex-col items-center justify-end">
                    {/* value above: logit -> percentage */}
                    <span className="mb-1 font-mono text-[11px] text-stone-500">
                      {phase === "logits"
                        ? c.logit.toFixed(1)
                        : `${(probs[i] * 100).toFixed(0)}%`}
                    </span>
                    <div className="w-full max-w-9 flex-1">
                      <motion.div
                        animate={{ scaleY: Math.max(0.02, h) }}
                        transition={{ type: "spring", stiffness: 200, damping: 24 }}
                        style={{ transformOrigin: "bottom", height: "100%" }}
                        className={`w-full rounded-t-md transition-colors duration-500 ${
                          phase === "logits" ? "bg-stone-300" : "bg-indigo-500"
                        }`}
                      />
                    </div>
                    <span className="mt-1.5 font-mono text-sm font-semibold text-stone-800">
                      {show(c.char)}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="mt-6 text-sm text-stone-400">
              (Start the backend to see the model&apos;s real scores.)
            </p>
          )}

          {/* logits-phase note */}
          {phase === "logits" && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="mx-auto mt-4 max-w-md text-[15px] text-stone-500"
            >
              These are the model&apos;s real scores after{" "}
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
                <input
                  type="range"
                  min={0.1}
                  max={2}
                  step={0.05}
                  value={temp}
                  onChange={(e) => setTemp(+e.target.value)}
                  className="w-full accent-indigo-600"
                />
              </label>
              <p className="mx-auto mt-2 max-w-md text-[13px] text-stone-400">
                Temperature divides every logit <em>before</em> softmax.{" "}
                <strong className="text-stone-600">Low</strong> (&lt;1) sharpens the
                gaps — the leader runs away with it (safe, repetitive).{" "}
                <strong className="text-stone-600">High</strong> (&gt;1) flattens them —
                the field levels out (varied, riskier).
              </p>
            </motion.div>
          )}
        </motion.div>
      )}

      {phase === "stack" && (
        <Continue onClick={() => setPhase("logits")} label="Turn the final vector into a guess" delay={0.3} />
      )}
      {phase === "logits" && (
        <Continue onClick={() => setPhase("softmax")} label="Make them real probabilities" delay={0.2} />
      )}
      {phase === "softmax" && (
        <Continue onClick={onNext} label="But first — how did it learn all this?" delay={0.2} />
      )}
    </div>
  );
}
