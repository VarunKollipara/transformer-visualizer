"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { getInfo } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch);

export default function TokenizeScene({ onNext }: SceneProps) {
  const [vocab, setVocab] = useState<string[] | null>(null);

  useEffect(() => {
    getInfo()
      .then((i) => setVocab(i.vocab))
      .catch(() => {});
  }, []);

  const stoi = useMemo(() => {
    const m = new Map<string, number>();
    vocab?.forEach((c, i) => m.set(c, i));
    return m;
  }, [vocab]);

  const chars = [...EXAMPLE_LINE];
  const settle = chars.length * 0.03 + 0.3;

  return (
    <div className="text-center">
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-2xl font-semibold text-stone-900 sm:text-3xl"
      >
        First, the text becomes numbers
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mx-auto mt-2 max-w-lg text-stone-500"
      >
        A model can&apos;t read letters — only numbers. Each character is swapped
        for its spot in a fixed list of 65 characters. This is{" "}
        <strong className="text-stone-700">tokenization</strong>.
      </motion.p>

      {/* the original line, then each character springs into a numbered tile */}
      <motion.div
        className="mx-auto mt-9 flex max-w-2xl flex-wrap justify-center gap-1.5"
        variants={{ show: { transition: { staggerChildren: 0.03, delayChildren: 0.3 } } }}
        initial="hidden"
        animate="show"
      >
        {chars.map((ch, i) => (
          <motion.div
            key={i}
            variants={{
              hidden: { opacity: 0, y: 18, scale: 0.4 },
              show: { opacity: 1, y: 0, scale: 1 },
            }}
            transition={{ type: "spring", stiffness: 500, damping: 24 }}
            className="flex flex-col items-center rounded-lg border border-sky-200 bg-sky-50 px-2 py-1.5"
          >
            <span className="font-mono text-lg leading-none text-stone-800">
              {show(ch)}
            </span>
            <span className="mt-1 font-mono text-xs leading-none text-sky-600">
              {stoi.get(ch) ?? "?"}
            </span>
          </motion.div>
        ))}
      </motion.div>

      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: settle, duration: 0.4 }}
        className="mx-auto mt-9 max-w-lg text-sm text-stone-500"
      >
        Even the <span className="font-mono text-stone-700">space</span> and{" "}
        <span className="font-mono text-stone-700">comma</span> get their own
        number. The model now sees this line as a list of{" "}
        <strong className="text-stone-700">{chars.length} numbers</strong> —
        nothing more.
      </motion.p>

      <Continue onClick={onNext} label="Now give them meaning" delay={settle + 0.2} />
    </div>
  );
}
