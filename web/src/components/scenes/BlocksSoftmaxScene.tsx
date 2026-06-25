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
  const e = s.map((x) => Math.exp(x - m)); // 2. exponentiate: positive + exaggerates gaps
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum); // 3. divide by the total so they sum to 1 (100%)
}

// Used only if the backend is offline, so the visual still tells the story.
const FALLBACK: Candidate[] = [
  { char: "e", logit: 9.6 }, { char: "u", logit: 7.0 }, { char: "r", logit: 7.0 },
  { char: "l", logit: 6.9 }, { char: "a", logit: 6.7 }, { char: "i", logit: 6.7 },
  { char: "y", logit: 6.1 }, { char: "o", logit: 6.0 },
];

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

  const data = cands ?? FALLBACK;
  const logits = useMemo(() => data.map((c) => c.logit), [data]);
  const probs = useMemo(() => softmax(logits, temp), [logits, temp]);

  // One height rule across phases: taller = more favoured. Softmax is monotonic
  // in the logit, so the bars keep their order — only the GAPS change.
  const lo = Math.min(...logits);
  const hi = Math.max(...logits);
  const logitH = (l: number) => 0.14 + 0.86 * (hi === lo ? 0.5 : (l - lo) / (hi - lo));
  const maxProb = Math.max(...probs);

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
              refined again and again as it rises — until, at the very top, the model
              commits to a guess for the next character.
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

      {/* ── THE MORPH ──
          The candidate bars ARE the persistent object. In the stack phase they're
          a small "prediction" panel on top of the tower; on Continue the SAME bars
          (stable keys) grow into the full chart, then reshape into probabilities.
          Nested layout (panel → row → column) is the proven opening-scene pattern;
          the bars themselves use scaleY only (no layout) to avoid distortion. */}
      <motion.div
        layout
        transition={{ layout: { duration: 0.75, ease: [0.22, 1, 0.36, 1] } }}
        className={`mx-auto transition-colors duration-500 ${
          onChart
            ? "mt-3 max-w-lg border border-transparent bg-transparent px-0 py-0"
            : "mt-7 w-fit rounded-xl border border-stone-200 bg-white px-3 pb-1.5 pt-2"
        }`}
      >
        <motion.div layout className={`flex items-end justify-center ${onChart ? "gap-2 sm:gap-3" : "gap-[3px]"}`}>
          {data.map((c, i) => {
            const h = phase === "softmax" ? probs[i] / (maxProb || 1) : logitH(c.logit);
            return (
              <motion.div
                key={c.char}
                layout
                transition={{ layout: { duration: 0.75, ease: [0.22, 1, 0.36, 1] } }}
                className={`flex flex-col items-center ${onChart ? "flex-1" : "w-3"}`}
              >
                {/* value above: logit -> percentage (chart phases only) */}
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

      {/* ── STACK phase: the label + tower beneath the prediction panel ── */}
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

      {phase === "stack" && (
        <Continue onClick={() => setPhase("logits")} label="Score every possible next character" delay={0.3} />
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
