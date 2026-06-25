"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { generate, type Step } from "@/lib/api";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch);

export default function GenerateScene({ restart }: SceneProps) {
  const [prompt, setPrompt] = useState("ROMEO:");
  const [temperature, setTemperature] = useState(0.8);
  const [steps, setSteps] = useState<Step[] | null>(null);
  const [revealed, setRevealed] = useState(0);
  const [loading, setLoading] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const run = async () => {
    setLoading(true);
    setSteps(null);
    if (timer.current) clearInterval(timer.current);
    try {
      const res = await generate(prompt, 220, temperature, 6);
      setSteps(res.steps);
      setRevealed(0);
    } catch {
      // backend offline; leave steps null
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!steps) return;
    timer.current = setInterval(() => {
      setRevealed((r) => {
        if (r >= steps.length) {
          if (timer.current) clearInterval(timer.current);
          return r;
        }
        return r + 1;
      });
    }, 38);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [steps]);

  const current = steps && revealed > 0 ? steps[revealed - 1] : null;
  const max = current ? Math.max(...current.topk.map((t) => t.prob), 0.0001) : 1;
  const done = steps !== null && revealed >= steps.length;

  return (
    <div className="text-center">
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-2xl font-semibold text-stone-900 sm:text-3xl"
      >
        Finally — the model writes
      </motion.h2>
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto mt-2 max-w-xl text-stone-500"
      >
        The model turns its scores into probabilities, picks the next character
        (<strong className="text-stone-700">temperature</strong> sets how boldly),
        adds it, and repeats. Give it a prompt and watch the loop run.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto mt-6 flex max-w-xl flex-col gap-3 sm:flex-row sm:items-end"
      >
        <label className="flex-1 text-left">
          <span className="mb-1 block text-sm text-stone-500">Prompt</span>
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full rounded-lg border border-stone-300 bg-stone-50 px-4 py-2.5 font-mono text-stone-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
          />
        </label>
        <button
          onClick={run}
          disabled={loading}
          className="rounded-lg bg-indigo-600 px-5 py-2.5 font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50"
        >
          {loading ? "Running…" : "Run the model"}
        </button>
      </motion.div>

      <motion.label
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto mt-3 block max-w-xl text-left"
      >
        <span className="mb-1 flex justify-between text-sm text-stone-500">
          <span>Temperature</span>
          <span className="font-mono font-semibold text-indigo-700">
            {temperature.toFixed(2)}
          </span>
        </span>
        <input
          type="range"
          min={0.1}
          max={1.5}
          step={0.05}
          value={temperature}
          onChange={(e) => setTemperature(+e.target.value)}
          className="w-full accent-indigo-600"
        />
      </motion.label>

      {steps && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mt-5 grid max-w-2xl gap-4 text-left sm:grid-cols-[1fr_12rem]"
        >
          <div className="min-h-36 whitespace-pre-wrap rounded-xl border border-stone-200 bg-stone-50 p-4 font-mono text-sm leading-relaxed text-stone-800">
            <span className="font-semibold text-indigo-700">{prompt}</span>
            {steps.slice(0, revealed).map((s) => s.char).join("")}
            {!done && <span className="animate-pulse text-indigo-500">▋</span>}
          </div>
          {current && (
            <div>
              <p className="mb-2 text-xs text-stone-400">picking this character</p>
              <div className="space-y-1">
                {current.topk.map((t, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="w-5 text-right font-mono text-xs text-stone-600">
                      {show(t.char)}
                    </span>
                    <div className="h-3 flex-1 overflow-hidden rounded bg-stone-200">
                      <div
                        className="h-full bg-indigo-500"
                        style={{ width: `${(t.prob / max) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}

      {done && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mx-auto mt-6 max-w-xl"
        >
          <p className="text-[15px] leading-relaxed text-stone-600">
            That&apos;s the whole machine. Scale it up — far more text, billions of
            parameters, a much longer memory — and this exact loop becomes ChatGPT.
          </p>
          <button
            onClick={restart}
            className="mt-4 text-sm font-medium text-indigo-600 hover:text-indigo-500"
          >
            ↺ start over
          </button>
        </motion.div>
      )}
    </div>
  );
}
