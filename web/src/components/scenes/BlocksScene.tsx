"use client";

import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

export default function BlocksScene({ onNext }: SceneProps) {
  const [layers, setLayers] = useState(3);

  return (
    <div className="text-center">
      <SceneTitle>Stack it into a tower</SceneTitle>
      <SceneText>
        One <strong className="text-stone-700">block</strong> = attention (tokens
        share context) + an MLP (each token thinks), each wrapped in a residual
        and LayerNorm. Stack the blocks and the token&apos;s vector gets refined
        again and again as it rises.
      </SceneText>

      <div className="mx-auto mt-6 flex w-full max-w-xs flex-col items-stretch gap-1.5">
        <div className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold text-white">
          next-character prediction
        </div>
        <span className="text-stone-300">↑</span>
        <AnimatePresence mode="popLayout">
          {Array.from({ length: layers }, (_, i) => layers - 1 - i).map((n) => (
            <motion.div
              key={n}
              layout
              initial={{ opacity: 0, scale: 0.9, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: "spring", stiffness: 300, damping: 26 }}
              className="rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-left"
            >
              <div className="text-sm font-semibold text-teal-800">Block {n + 1}</div>
              <div className="font-mono text-xs text-teal-600">attention → MLP</div>
            </motion.div>
          ))}
        </AnimatePresence>
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
        <span className="w-24 font-mono text-sm text-stone-700">{layers} block{layers > 1 ? "s" : ""}</span>
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

      <Continue onClick={onNext} label="Turn the final vector into a guess" delay={0.3} />
    </div>
  );
}
