"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const RAW = [0.9, 0.2, 1.4, 0.5, 1.1, 0.3, 0.8, 1.6, 0.4, 1.0];
const NORM = [0.55, 0.35, 0.75, 0.45, 0.65, 0.4, 0.52, 0.82, 0.42, 0.6];

export default function NormResidualScene({ onNext }: SceneProps) {
  const [normalized, setNormalized] = useState(false);
  const bars = normalized ? NORM : RAW;

  return (
    <div className="text-center">
      <SceneTitle>What keeps a deep stack stable</SceneTitle>
      <SceneText>
        We&apos;re about to stack many of these layers. Two small tricks make that
        possible.
      </SceneText>

      {/* residual */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="mx-auto mt-7 max-w-xl rounded-xl border border-stone-200 bg-white p-4"
      >
        <p className="mb-3 text-sm font-semibold text-stone-700">
          1. Residual connections
        </p>
        <div className="flex items-center justify-center gap-2 font-mono text-sm">
          <span className="rounded bg-stone-100 px-2 py-1">x</span>
          <span className="text-stone-300">→</span>
          <span className="rounded bg-indigo-100 px-2 py-1 text-indigo-800">layer</span>
          <span className="text-stone-300">→</span>
          <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-800">＋</span>
          <span className="text-stone-300">→</span>
          <span className="rounded bg-stone-100 px-2 py-1">x′</span>
        </div>
        <div className="relative mx-auto mt-1 h-4 w-64">
          <svg viewBox="0 0 256 24" className="w-full">
            <path d="M20 2 Q128 26 210 2" fill="none" stroke="#34d399" strokeWidth="2" strokeDasharray="3 3" />
          </svg>
          <span className="absolute left-1/2 top-3 -translate-x-1/2 text-[11px] text-emerald-600">
            original carried forward
          </span>
        </div>
        <p className="mt-3 text-sm text-stone-500">
          <span className="font-mono">x = x + layer(x)</span> — every layer{" "}
          <em>adds a small refinement</em> instead of replacing everything. That
          keeps a clear path for learning signals to flow back, so the network can
          be many layers deep without falling apart.
        </p>
      </motion.div>

      {/* layernorm */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.4 }}
        className="mx-auto mt-5 max-w-xl rounded-xl border border-stone-200 bg-white p-4"
      >
        <p className="mb-3 text-sm font-semibold text-stone-700">2. LayerNorm</p>
        <div className="flex h-16 items-end justify-center gap-[3px]">
          {bars.map((v, j) => (
            <motion.span
              key={j}
              animate={{ height: `${v * 38 + 2}px` }}
              transition={{ type: "spring", stiffness: 280, damping: 22 }}
              className="w-2.5 rounded-sm bg-amber-400"
            />
          ))}
        </div>
        <button
          onClick={() => setNormalized((n) => !n)}
          className="mt-3 rounded-full bg-stone-800 px-4 py-1.5 text-sm font-medium text-white hover:bg-stone-700"
        >
          {normalized ? "reset" : "normalize"}
        </button>
        <p className="mt-3 text-sm text-stone-500">
          LayerNorm rescales each token&apos;s numbers to a steady, even range, so
          values don&apos;t snowball as they pass through layer after layer.
        </p>
      </motion.div>

      <Continue onClick={onNext} label="Stack it into a tower" delay={0.6} />
    </div>
  );
}
