"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const ORDER_A = ["dog", "bites", "man"];
const ORDER_B = ["man", "bites", "dog"];

export default function PositionalScene({ onNext }: SceneProps) {
  const [swapped, setSwapped] = useState(false);
  const words = swapped ? ORDER_B : ORDER_A;

  return (
    <div className="text-center">
      <SceneTitle>Order changes everything</SceneTitle>
      <SceneText>
        Embeddings tell the model <em>what</em> each token is — but not{" "}
        <em>where</em> it sits. Yet the same words in a different order mean
        opposite things:
      </SceneText>

      <div className="mt-6 flex items-center justify-center gap-2 font-mono text-xl">
        {words.map((w, i) => (
          <motion.span
            key={w}
            layout
            transition={{ type: "spring", stiffness: 240, damping: 26 }}
            className={`rounded-lg px-3 py-1.5 ${
              w === "dog"
                ? "bg-violet-100 text-violet-800"
                : w === "man"
                  ? "bg-amber-100 text-amber-800"
                  : "bg-stone-100 text-stone-600"
            }`}
          >
            {w}
          </motion.span>
        ))}
      </div>
      <button
        onClick={() => setSwapped((s) => !s)}
        className="mt-3 text-sm font-medium text-indigo-600 hover:text-indigo-500"
      >
        ⇄ swap dog and man
      </button>
      <div className="mx-auto mt-2 min-h-[1.5rem] max-w-md">
        <AnimatePresence mode="wait">
          <motion.p
            key={swapped ? "b" : "a"}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-sm text-stone-500"
          >
            {swapped ? "Now the man is the victim." : "Here the dog is the biter."}
          </motion.p>
        </AnimatePresence>
      </div>

      <SceneText delay={0.1} className="mt-8">
        So before anything else, the model <strong className="text-stone-700">adds
        a position vector</strong> to each token&apos;s embedding — a learned
        signal for &ldquo;I&apos;m the 1st token,&rdquo; &ldquo;the 2nd,&rdquo; and
        so on. Same letter, different spot → different vector:
      </SceneText>

      <div className="mt-5 flex items-center justify-center gap-2">
        {["t", "o", "b", "e"].map((ch, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + i * 0.12 }}
            className="flex flex-col items-center gap-1"
          >
            <div className="rounded-md border border-sky-200 bg-sky-50 px-2 py-1 font-mono text-sm text-stone-800">
              {ch}
            </div>
            <span className="text-stone-300">+</span>
            <motion.div
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + i * 0.12, type: "spring", stiffness: 300, damping: 20 }}
              className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 font-mono text-xs text-amber-700"
            >
              pos {i}
            </motion.div>
          </motion.div>
        ))}
      </div>

      <Continue onClick={onNext} label="Now they can look at each other" delay={1.3} />
    </div>
  );
}
