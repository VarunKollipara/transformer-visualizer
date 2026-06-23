"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

function Bars({ values, color = "bg-indigo-400", from = 0 }: { values: number[]; color?: string; from?: number }) {
  return (
    <div className="flex h-12 items-end gap-[2px]">
      {values.map((v, j) => (
        <motion.span
          key={j}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: Math.max(0.02, v) }}
          transition={{ delay: from + j * 0.02, type: "spring", stiffness: 280, damping: 20 }}
          style={{ transformOrigin: "bottom", height: "100%" }}
          className={`block w-1.5 rounded-sm ${color}`}
        />
      ))}
    </div>
  );
}

const IN = [0.5, 0.8, 0.3, 0.6, 0.4, 0.7, 0.45, 0.55];
const HIDDEN = Array.from({ length: 24 }, (_, j) => 0.25 + 0.7 * Math.abs(Math.sin(j * 1.3)));
const OUT = [0.6, 0.4, 0.7, 0.35, 0.65, 0.5, 0.45, 0.6];

// signed values for the ReLU demo
const SIGNED = [0.7, -0.4, 0.5, -0.8, 0.3, 0.9, -0.2, 0.6, -0.6, 0.4];

export default function MLPScene({ onNext }: SceneProps) {
  const [relu, setRelu] = useState(false);

  return (
    <div className="text-center">
      <SceneTitle>Then each token thinks for itself</SceneTitle>
      <SceneText>
        Attention <em>moved information between</em> tokens. Now each token
        processes what it gathered, on its own, through a small two-layer network
        — the <strong className="text-stone-700">MLP</strong>. It widens the
        vector, bends it, and shrinks it back.
      </SceneText>

      <div className="mt-7 flex items-end justify-center gap-3">
        <div className="flex flex-col items-center gap-1">
          <Bars values={IN} />
          <span className="text-xs text-stone-400">vector (128)</span>
        </div>
        <span className="mb-5 text-stone-300">→</span>
        <div className="flex flex-col items-center gap-1">
          <Bars values={HIDDEN} color="bg-violet-400" from={0.2} />
          <span className="text-xs text-stone-400">widen ×4 + ReLU</span>
        </div>
        <span className="mb-5 text-stone-300">→</span>
        <div className="flex flex-col items-center gap-1">
          <Bars values={OUT} from={0.7} />
          <span className="text-xs text-stone-400">back to 128</span>
        </div>
      </div>

      <div className="mx-auto mt-8 max-w-md rounded-xl border border-stone-200 bg-white p-4">
        <p className="mb-3 text-sm text-stone-500">
          The bend is <strong className="text-stone-700">ReLU</strong> ={" "}
          <span className="font-mono">max(0, x)</span> — keep positives, zero out
          negatives. Try it:
        </p>
        <div className="flex h-16 items-center justify-center gap-[3px]">
          {SIGNED.map((v, j) => {
            const shown = relu ? Math.max(0, v) : v;
            return (
              <motion.span
                key={j}
                animate={{ height: `${Math.abs(shown) * 50 + 2}px` }}
                transition={{ type: "spring", stiffness: 300, damping: 22 }}
                className={`w-2 rounded-sm ${shown < 0 ? "bg-rose-400 self-start" : "bg-emerald-400 self-end"}`}
                style={{ alignSelf: shown < 0 ? "flex-start" : "flex-end" }}
              />
            );
          })}
        </div>
        <button
          onClick={() => setRelu((r) => !r)}
          className="mt-3 rounded-full bg-stone-800 px-4 py-1.5 text-sm font-medium text-white hover:bg-stone-700"
        >
          {relu ? "reset" : "apply ReLU"}
        </button>
        <p className="mt-3 text-xs text-stone-400">
          Without this bend, stacking layers would collapse into a single straight
          line — the model could only learn the simplest relationships.
        </p>
      </div>

      <Continue onClick={onNext} label="What keeps it all stable" delay={0.3} />
    </div>
  );
}
