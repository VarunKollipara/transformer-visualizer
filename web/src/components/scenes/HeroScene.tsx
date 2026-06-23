"use client";

import { motion } from "motion/react";
import { EXAMPLE_LINE } from "@/lib/example";
import type { SceneProps } from "./types";

// The landing scene: one sentence + the Shakespeare text, with a single line
// glowing to invite the click that starts the journey.
const LINES = [
  "HAMLET:",
  EXAMPLE_LINE,
  "Whether 'tis nobler in the mind to suffer",
  "The slings and arrows of outrageous fortune,",
];

export default function HeroScene({ onNext }: SceneProps) {
  return (
    <div className="text-center">
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="font-display text-3xl font-semibold tracking-tight text-stone-900 sm:text-4xl"
      >
        This is how ChatGPT actually works.
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4, duration: 0.5 }}
        className="mt-3 text-stone-500"
      >
        It all begins with text. Click the glowing line to begin.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 0.6, ease: "easeOut" }}
        className="mx-auto mt-12 flex max-w-xl flex-col gap-1.5 text-left font-mono text-[15px]"
      >
        {LINES.map((line) => {
          if (line !== EXAMPLE_LINE) {
            return (
              <p key={line} className="select-none px-4 py-2 text-stone-300">
                {line}
              </p>
            );
          }
          return (
            <motion.button
              key={line}
              onClick={onNext}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.99 }}
              animate={{
                boxShadow: [
                  "0 0 0 0 rgba(99,102,241,0.0)",
                  "0 0 0 5px rgba(99,102,241,0.14)",
                  "0 0 0 0 rgba(99,102,241,0.0)",
                ],
              }}
              transition={{
                boxShadow: { duration: 2.2, repeat: Infinity, ease: "easeInOut" },
              }}
              className="rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-2 text-left text-indigo-900"
            >
              {line}
            </motion.button>
          );
        })}
      </motion.div>
    </div>
  );
}
