"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { getLogits, type Candidate } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

function softmax(logits: number[], t: number): number[] {
  const temp = Math.max(t, 0.01);
  const s = logits.map((l) => l / temp);
  const m = Math.max(...s);
  const e = s.map((x) => Math.exp(x - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}

export default function SoftmaxScene({ onNext }: SceneProps) {
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [temp, setTemp] = useState(1);

  useEffect(() => {
    // the model's real scores for what follows "...to be"
    getLogits("To be, or not to b", 6)
      .then((d) => setCands(d.candidates))
      .catch(() => {});
  }, []);

  const probs = useMemo(
    () => (cands ? softmax(cands.map((c) => c.logit), temp) : []),
    [cands, temp],
  );
  const maxLogit = cands ? Math.max(...cands.map((c) => c.logit)) : 1;

  return (
    <div className="text-center">
      <SceneTitle>From scores to a single choice</SceneTitle>
      <SceneText>
        After all the blocks, the model outputs one raw score — a{" "}
        <strong className="text-stone-700">logit</strong> — for every possible
        next character. <strong className="text-stone-700">Softmax</strong> turns
        those into probabilities that add to 100%. (Real scores after &ldquo;To be,
        or not to b&rdquo;.)
      </SceneText>

      {cands && (
        <div className="mx-auto mt-6 max-w-lg space-y-2">
          <div className="grid grid-cols-[2rem_1fr_1.3fr_3rem] gap-3 text-xs text-stone-400">
            <span>next</span>
            <span>logit (raw score)</span>
            <span>probability</span>
            <span className="text-right">%</span>
          </div>
          {cands.map((c, i) => (
            <div key={i} className="grid grid-cols-[2rem_1fr_1.3fr_3rem] items-center gap-3">
              <span className="font-mono text-sm font-semibold text-stone-800">
                {show(c.char)}
              </span>
              <div className="h-3 overflow-hidden rounded bg-stone-100">
                <div
                  className="h-full rounded bg-stone-300"
                  style={{ width: `${(c.logit / maxLogit) * 100}%` }}
                />
              </div>
              <div className="h-4 overflow-hidden rounded bg-stone-100">
                <motion.div
                  className="h-full rounded bg-indigo-500"
                  animate={{ width: `${probs[i] * 100}%` }}
                  transition={{ duration: 0.25 }}
                />
              </div>
              <span className="text-right font-mono text-xs text-stone-500">
                {(probs[i] * 100).toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      )}

      <label className="mx-auto mt-7 block max-w-lg">
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
        <span className="mt-1 block text-xs text-stone-400">
          Low → one character dominates (safe, repetitive). High → the field
          levels out (varied, riskier).
        </span>
      </label>

      <Continue onClick={onNext} label="But first — how did it learn all this?" delay={0.2} />
    </div>
  );
}
