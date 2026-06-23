"use client";

import { useEffect, useRef, useState } from "react";
import { generate, type Step } from "@/lib/api";

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

export default function GenerationDemo() {
  const [prompt, setPrompt] = useState("ROMEO:");
  const [temperature, setTemperature] = useState(0.8);
  const [steps, setSteps] = useState<Step[] | null>(null);
  const [revealed, setRevealed] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const run = async () => {
    setLoading(true);
    setError(false);
    setSteps(null);
    if (timer.current) clearInterval(timer.current);
    try {
      const res = await generate(prompt, 240, temperature, 8);
      setSteps(res.steps);
      setRevealed(0);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  // Reveal characters one at a time to visualize the predict→append loop.
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
    }, 35);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [steps]);

  const current = steps && revealed > 0 ? steps[revealed - 1] : null;
  const max = current ? Math.max(...current.topk.map((t) => t.prob), 0.0001) : 1;

  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
        <label className="flex-1">
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
          {loading ? "Generating…" : "Generate"}
        </button>
      </div>

      <label className="mt-4 block">
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
        <span className="mt-1 block text-xs text-stone-400">
          Lower = safer &amp; more repetitive · higher = more varied &amp; more
          mistakes
        </span>
      </label>

      {error && (
        <p className="mt-3 text-sm text-amber-700">
          Backend not reachable — start the API server (see banner at top).
        </p>
      )}

      {steps && (
        <div className="mt-5 grid gap-4 sm:grid-cols-[1fr_auto]">
          <div className="min-h-40 whitespace-pre-wrap rounded-xl border border-stone-200 bg-stone-50 p-4 font-mono text-sm leading-relaxed text-stone-800">
            <span className="font-semibold text-indigo-700">{prompt}</span>
            {steps
              .slice(0, revealed)
              .map((s) => s.char)
              .join("")}
            {revealed < steps.length && (
              <span className="animate-pulse text-indigo-500">▋</span>
            )}
          </div>

          {current && (
            <div className="w-full sm:w-56">
              <p className="mb-2 text-xs text-stone-400">
                distribution for this character
              </p>
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
        </div>
      )}
    </div>
  );
}
