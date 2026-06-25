"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

// Directional enter so each phase change feels deliberate, not a plain crossfade.
const enter = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] as const },
};

// x and the layer's small edit, so x' = x + edit is only a nudge away from x.
const X = [0.55, 0.35, 0.7, 0.45, 0.6, 0.4, 0.5, 0.65];
const EDIT = [0.08, -0.06, 0.05, 0.1, -0.04, 0.07, -0.05, 0.06];

const RAW = [0.9, 0.2, 1.4, 0.5, 1.1, 0.3, 0.8, 1.6, 0.4, 1.0];
const NORM = [0.55, 0.35, 0.75, 0.45, 0.65, 0.4, 0.52, 0.82, 0.42, 0.6];

type Phase = "residual" | "layernorm";

export default function NormResidualScene({ onNext }: SceneProps) {
  const [phase, setPhase] = useState<Phase>("residual");
  const [normalized, setNormalized] = useState(false);
  const bars = normalized ? NORM : RAW;

  return (
    <div className="text-center">
      <motion.div key={`t-${phase}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "residual" ? (
          <>
            <SceneTitle>Trick 1: add, don&apos;t overwrite</SceneTitle>
            <SceneText>
              We&apos;re about to stack many of these layers. The first thing that
              makes that safe: every layer <strong className="text-stone-700">adds a
              small edit</strong> to the vector instead of rewriting it from scratch.
            </SceneText>
          </>
        ) : (
          <>
            <SceneTitle>Trick 2: keep the numbers tidy</SceneTitle>
            <SceneText>
              The second: after each step, <strong className="text-stone-700">
              LayerNorm</strong> (<strong className="text-stone-700">layer
              normalization</strong>) rescales a token&apos;s numbers back to a steady,
              even range — so values don&apos;t snowball as they pass through layer
              after layer.
            </SceneText>
          </>
        )}
      </motion.div>

      {/* ── RESIDUAL phase ── */}
      {phase === "residual" && (
        <motion.div key="residual" {...enter}>
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

            {/* concrete: x, the small edit, and x' = x + edit */}
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
              <div className="flex flex-col items-center gap-1">
                <div className="flex h-16 items-end gap-[3px]">
                  {X.map((v, j) => (
                    <span key={j} style={{ height: `${(v + EDIT[j]) * 56 + 2}px` }} className="w-2 rounded-sm bg-emerald-400" />
                  ))}
                </div>
                <span className="text-xs text-stone-400">x′ = x + layer(x)</span>
              </div>
            </div>
          </div>

          <SceneText delay={0.3} className="mt-5 text-[15px]">
            Because the original is always carried forward, even a layer that learns
            nothing useful can&apos;t wreck the signal — and the learning signal has a
            clear, short path back through every layer. That&apos;s what lets a stack
            go dozens of layers deep without falling apart.
          </SceneText>

          <Continue onClick={() => setPhase("layernorm")} label="And the second trick" delay={0.4} />
        </motion.div>
      )}

      {/* ── LAYERNORM phase ── */}
      {phase === "layernorm" && (
        <motion.div key="layernorm" {...enter}>
          <div className="mx-auto mt-7 max-w-xl rounded-xl border border-stone-200 bg-white p-5">
            <p className="mb-3 text-sm text-stone-500">
              One token&apos;s vector. Some numbers are big, some small, all over the
              place. Press normalise:
            </p>
            <div className="flex h-20 items-end justify-center gap-[4px]">
              {bars.map((v, j) => (
                <motion.span
                  key={j}
                  animate={{ height: `${v * 44 + 2}px` }}
                  transition={{ type: "spring", stiffness: 280, damping: 22 }}
                  className="w-3 rounded-sm bg-amber-400"
                />
              ))}
            </div>
            <button
              onClick={() => setNormalized((n) => !n)}
              className="mt-4 rounded-full bg-stone-800 px-4 py-1.5 text-sm font-medium text-white hover:bg-stone-700"
            >
              {normalized ? "reset" : "normalise"}
            </button>
            <p className="mt-4 text-sm text-stone-500">
              Under the hood it&apos;s two steps: <strong className="text-stone-700">
              subtract the average</strong> of the vector (recentre it around 0), then{" "}
              <strong className="text-stone-700">divide by the spread</strong> (so the
              values have a consistent scale). Same shape, tamed magnitude.
            </p>
          </div>

          <SceneText delay={0.3} className="mt-5 text-[15px]">
            Residual connections keep the signal flowing; LayerNorm keeps it from
            blowing up. Together they&apos;re the quiet plumbing that makes a deep
            stack trainable at all.
          </SceneText>

          <Continue onClick={onNext} label="Stack it into a tower" delay={0.4} />
        </motion.div>
      )}
    </div>
  );
}
