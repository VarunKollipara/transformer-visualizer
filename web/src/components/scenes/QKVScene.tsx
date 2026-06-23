"use client";

import { motion } from "motion/react";
import QKVDiagram from "@/components/QKVDiagram";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

export default function QKVScene({ onNext }: SceneProps) {
  return (
    <div className="text-center">
      <SceneTitle>How a token decides what to look at</SceneTitle>
      <SceneText>
        Each token produces three little vectors. Think of them as a token
        walking into a room and:
      </SceneText>

      <div className="mx-auto mt-6 max-w-xl">
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            ["Query", "what I'm looking for", "teal"],
            ["Key", "what I can offer", "sky"],
            ["Value", "the info I'll hand over", "violet"],
          ].map(([t, d, c], i) => (
            <motion.div
              key={t}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.15, duration: 0.4 }}
              className={`rounded-xl border p-3 ${
                c === "teal"
                  ? "border-teal-200 bg-teal-50 text-teal-800"
                  : c === "sky"
                    ? "border-sky-200 bg-sky-50 text-sky-800"
                    : "border-violet-200 bg-violet-50 text-violet-800"
              }`}
            >
              <div className="font-semibold">{t}</div>
              <div className="mt-0.5 text-sm opacity-75">{d}</div>
            </motion.div>
          ))}
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="mx-auto mt-6 max-w-xl"
      >
        <QKVDiagram />
      </motion.div>

      <SceneText delay={0.9} className="mt-5 text-[15px]">
        A token compares its <strong className="text-teal-700">query</strong> to
        every other token&apos;s <strong className="text-sky-700">key</strong>{" "}
        (just multiply and add — a dot product) to get a relevance score. Divide
        by √(size) to keep it steady, softmax the scores into percentages, then
        blend everyone&apos;s <strong className="text-violet-700">values</strong>{" "}
        by those percentages. That blend is the token&apos;s new, context-aware
        vector.
      </SceneText>

      <Continue onClick={onNext} label="See it on the real model" delay={1.1} />
    </div>
  );
}
