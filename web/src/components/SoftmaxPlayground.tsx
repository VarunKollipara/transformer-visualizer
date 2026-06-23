"use client";

import { useEffect, useMemo, useState } from "react";
import { getLogits, type Candidate } from "@/lib/api";

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

// softmax with temperature: p_i = exp(l_i / T) / sum_j exp(l_j / T)
function softmax(logits: number[], temp: number): number[] {
  const t = Math.max(temp, 0.01);
  const scaled = logits.map((l) => l / t);
  const m = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export default function SoftmaxPlayground() {
  const [context, setContext] = useState("To be or not to b");
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [temp, setTemp] = useState(1.0);
  const [error, setError] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => {
      getLogits(context.slice(-64) || "\n", 7)
        .then((d) => {
          setCands(d.candidates);
          setError(false);
        })
        .catch(() => setError(true));
    }, 250);
    return () => clearTimeout(t);
  }, [context]);

  const logits = useMemo(() => cands?.map((c) => c.logit) ?? [], [cands]);
  const probs = useMemo(() => softmax(logits, temp), [logits, temp]);

  const setLogit = (i: number, v: number) =>
    setCands((cs) =>
      cs ? cs.map((c, j) => (j === i ? { ...c, logit: v } : c)) : cs,
    );

  const lo = Math.min(...logits, 0) - 2;
  const hi = Math.max(...logits, 1) + 4;

  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
      <label className="block">
        <span className="mb-1 block text-sm text-stone-500">
          Context (the model&apos;s real scores for what comes next)
        </span>
        <input
          value={context}
          onChange={(e) => setContext(e.target.value)}
          className="w-full rounded-lg border border-stone-300 bg-stone-50 px-4 py-2.5 font-mono text-stone-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
        />
      </label>

      {error && (
        <p className="mt-3 text-sm text-amber-700">
          Backend not reachable — start the API server (see banner at top).
        </p>
      )}

      <label className="mt-4 block">
        <span className="mb-1 flex justify-between text-sm text-stone-500">
          <span>Temperature</span>
          <span className="font-mono font-semibold text-indigo-700">
            {temp.toFixed(2)}
          </span>
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

      {cands && (
        <div className="mt-5 space-y-2.5">
          <div className="grid grid-cols-[2rem_1fr_1.4fr_3rem] items-center gap-3 text-xs text-stone-400">
            <span>char</span>
            <span>logit (raw score — drag)</span>
            <span>probability after softmax</span>
            <span className="text-right">%</span>
          </div>
          {cands.map((c, i) => (
            <div
              key={i}
              className="grid grid-cols-[2rem_1fr_1.4fr_3rem] items-center gap-3"
            >
              <span className="text-center font-mono text-sm font-semibold text-stone-800">
                {show(c.char)}
              </span>
              <input
                type="range"
                min={lo}
                max={hi}
                step={0.1}
                value={c.logit}
                onChange={(e) => setLogit(i, +e.target.value)}
                className="w-full accent-stone-400"
              />
              <div className="h-4 overflow-hidden rounded bg-stone-100">
                <div
                  className="h-full rounded bg-indigo-500 transition-[width] duration-150"
                  style={{ width: `${probs[i] * 100}%` }}
                />
              </div>
              <span className="text-right font-mono text-xs text-stone-500">
                {(probs[i] * 100).toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="mt-5 text-sm text-stone-500">
        <strong className="text-stone-700">Logits</strong> are the raw scores the
        model outputs. <strong className="text-stone-700">Softmax</strong> turns
        them into probabilities that sum to 100%. Drag{" "}
        <strong className="text-stone-700">temperature</strong> low and one
        character takes almost all the probability (safe, repetitive); drag it
        high and the field levels out (varied, riskier). Nudge a logit and watch
        its probability respond.
      </p>
    </div>
  );
}
