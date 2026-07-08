"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import TrainingViz from "@/components/TrainingViz";
import { generate, type Step } from "@/lib/api";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch);

type Phase = "training" | "generate";

// The finale, in two slides that share one object: the model's GENERATED TEXT.
// In `training` you scrub from noise to Shakespeare and that text lives in a
// mono panel; advancing to `generate`, the loss curve recedes and the SAME panel
// becomes the live canvas you drive with a prompt.
export default function FinaleScene({ phase: phaseProp, onNext, restart }: SceneProps) {
  const phase = phaseProp as Phase;

  // training: the current scrubber sample, surfaced from TrainingViz
  const [trainSample, setTrainSample] = useState("");
  const [trainStep, setTrainStep] = useState(0);
  const onSample = useCallback((text: string, step: number) => {
    setTrainSample(text);
    setTrainStep(step);
  }, []);

  // generate: the autoregressive loop
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
      {/* title + intro, keyed by phase */}
      <motion.div key={`t-${phase}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "training" ? (
          <>
            <SceneTitle>How did it learn all this?</SceneTitle>
            <SceneText>
              Everything started as random numbers. Training repeats four steps
              thousands of times — <strong className="text-stone-700">predict</strong>,
              measure the error (the <strong className="text-stone-700">loss</strong>),
              compute how to fix every parameter (<strong className="text-stone-700">
              backprop</strong>), nudge them all a hair. Drag the scrubber to watch
              noise become Shakespeare:
            </SceneText>
          </>
        ) : (
          <>
            <SceneTitle>Finally — you make it write</SceneTitle>
            <SceneText>
              Same model, same trick that filled the panel below — now you drive it.
              It turns its scores into probabilities, picks the next character
              (<strong className="text-stone-700">temperature</strong> sets how
              boldly), adds it, and repeats. Give it a prompt:
            </SceneText>
          </>
        )}
      </motion.div>

      {/* training-only: the loss-curve scrubber (its sample is lifted out below) */}
      {phase === "training" && (
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mt-5 max-w-2xl text-left"
        >
          <TrainingViz onSample={onSample} hideSample />
        </motion.div>
      )}

      {/* generate-only: prompt + temperature controls */}
      {phase === "generate" && (
        <>
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
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
            transition={{ delay: 0.3, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto mt-3 block max-w-xl text-left"
          >
            <span className="mb-1 flex justify-between text-sm text-stone-500">
              <span>Temperature</span>
              <span className="font-mono font-semibold text-indigo-700">{temperature.toFixed(2)}</span>
            </span>
            <input type="range" min={0.1} max={1.5} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} className="w-full accent-indigo-600" />
          </motion.label>
        </>
      )}

      {/* ── the carried element: the model's text canvas ──
          scrubber sample in `training`, live generation in `generate`. */}
      <motion.div layout transition={{ layout: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } }} className="mx-auto mt-4 max-w-2xl">
        <div className="min-h-28 whitespace-pre-wrap rounded-xl border border-stone-200 bg-stone-50 p-4 text-left font-mono text-[13px] leading-relaxed text-stone-800">
          {phase === "training" ? (
            trainSample || <span className="text-stone-400">loading the training run…</span>
          ) : steps ? (
            <>
              <span className="font-semibold text-indigo-700">{prompt}</span>
              {steps.slice(0, revealed).map((s) => s.char).join("")}
              {!done && <span className="animate-pulse text-indigo-500">▋</span>}
            </>
          ) : (
            <span className="text-stone-400">Press &ldquo;Run the model&rdquo; and watch the loop write, character by character…</span>
          )}
        </div>
        {phase === "training" && (
          <p className="mt-1 text-left text-xs text-stone-400">
            what the model writes at step {trainStep.toLocaleString()} — drag the scrubber above
          </p>
        )}
        {phase === "generate" && current && (
          <div className="mx-auto mt-3 max-w-xs text-left">
            <p className="mb-2 text-xs text-stone-400">picking this character</p>
            <div className="space-y-1">
              {current.topk.map((t, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-5 text-right font-mono text-xs text-stone-600">{show(t.char)}</span>
                  <div className="h-3 flex-1 overflow-hidden rounded bg-stone-200">
                    <div className="h-full bg-indigo-500" style={{ width: `${(t.prob / max) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* training-only: overfitting note + advance */}
      {phase === "training" && (
        <>
          <SceneText delay={0.2} className="mt-4 text-[15px]">
            Both the training and the held-out (never-seen) loss fall together — so
            it&apos;s genuinely learning the patterns, not just memorising.
          </SceneText>
          <Continue onClick={onNext} label="Now you make it write" delay={0.3} />
        </>
      )}

      {/* generate-only: the closing line, now a handoff into the epilogue */}
      {phase === "generate" && done && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="mx-auto mt-6 max-w-xl">
          <p className="text-[15px] leading-relaxed text-stone-600">
            That&apos;s the whole machine. Scale it up — far more text, billions of
            parameters, a much longer memory — and this exact loop becomes ChatGPT.
            One puzzle left: this loop looks <em>expensive</em>&hellip;
          </p>
          <Continue onClick={onNext} label="Epilogue: why isn't ChatGPT slow?" delay={0.15} />
          <div>
            <button onClick={restart} className="mt-4 text-sm font-medium text-stone-400 transition hover:text-stone-600">
              ↺ start over
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
}
