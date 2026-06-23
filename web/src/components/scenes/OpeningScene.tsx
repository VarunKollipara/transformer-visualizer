"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { getInfo } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import type { SceneProps } from "./types";

const display = (ch: string) => (ch === " " ? "␣" : ch);
const CONTEXT_BELOW = [
  "Whether 'tis nobler in the mind to suffer",
  "The slings and arrows of outrageous fortune,",
];

// Hero + tokenization in one scene. Clicking the line doesn't cut to a new
// screen — the SAME letter elements slide up and grow their token number beneath
// them, via motion's shared layout animation.
export default function OpeningScene({ onNext }: SceneProps) {
  const [vocab, setVocab] = useState<string[] | null>(null);
  const [phase, setPhase] = useState<"invite" | "tokens">("invite");

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
  const tokens = phase === "tokens";

  return (
    <div className="text-center">
      <motion.h1
        layout="position"
        className="font-display text-3xl font-semibold tracking-tight text-stone-900 sm:text-4xl"
      >
        This is how ChatGPT actually works.
      </motion.h1>

      <AnimatePresence mode="popLayout">
        {!tokens && (
          <motion.p
            key="sub"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
            className="mt-3 text-stone-500"
          >
            Trained on nothing but the complete works of Shakespeare.
          </motion.p>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!tokens && (
          <motion.p
            key="ctx-above"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.45 }}
            exit={{ opacity: 0 }}
            className="mx-auto mt-10 max-w-xl text-left font-mono text-[15px] text-stone-400"
          >
            HAMLET:
          </motion.p>
        )}
      </AnimatePresence>

      {/* the persistent characters — these morph from line to tokens */}
      <motion.div
        layout
        role={tokens ? undefined : "button"}
        onClick={() => !tokens && setPhase("tokens")}
        animate={
          tokens
            ? { boxShadow: "0 0 0 0 rgba(0,0,0,0)" }
            : {
                boxShadow: [
                  "0 0 0 0 rgba(99,102,241,0.0)",
                  "0 0 0 5px rgba(99,102,241,0.14)",
                  "0 0 0 0 rgba(99,102,241,0.0)",
                ],
              }
        }
        transition={
          tokens
            ? { layout: { type: "spring", stiffness: 340, damping: 34 } }
            : { boxShadow: { duration: 2.2, repeat: Infinity, ease: "easeInOut" } }
        }
        className={
          tokens
            ? "mx-auto mt-6 flex max-w-2xl flex-wrap justify-center gap-1.5"
            : "mx-auto mt-2 flex max-w-xl cursor-pointer flex-wrap justify-start rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-2"
        }
      >
        {chars.map((ch, i) => (
          <motion.div
            key={i}
            layout
            transition={{ type: "spring", stiffness: 380, damping: 30 }}
            whileHover={tokens ? { scale: 1.12 } : undefined}
            className={
              tokens
                ? "flex flex-col items-center rounded-lg border border-sky-200 bg-sky-50 px-2 py-1.5"
                : "px-0 py-0"
            }
          >
            <motion.span
              layout="position"
              className={
                tokens
                  ? "font-mono text-lg leading-none text-stone-800"
                  : "font-mono text-[15px] leading-none text-indigo-900"
              }
            >
              {tokens ? display(ch) : ch}
            </motion.span>
            {tokens && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.3 }}
                className="mt-1 font-mono text-[11px] leading-none text-sky-600"
              >
                {stoi.get(ch) ?? "?"}
              </motion.span>
            )}
          </motion.div>
        ))}
      </motion.div>

      <AnimatePresence>
        {!tokens && (
          <motion.div key="ctx-below" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            {CONTEXT_BELOW.map((l) => (
              <p
                key={l}
                className="mx-auto mt-2 max-w-xl text-left font-mono text-[15px] text-stone-300"
              >
                {l}
              </p>
            ))}
            <p className="mt-7 text-sm text-stone-400">
              Click the glowing line to begin.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {tokens && (
          <motion.div
            key="explain"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            <p className="mx-auto mt-7 max-w-lg text-stone-500">
              Each letter just slid into place above its number. A model can&apos;t
              read letters — only numbers — so every character is swapped for its
              spot in a fixed list of 65. This is{" "}
              <strong className="text-stone-700">tokenization</strong>.
            </p>
            <p className="mx-auto mt-2 max-w-lg text-sm text-stone-400">
              Even the space and comma get their own number. This whole line is now
              just {chars.length} numbers — nothing more. (Hover a tile.)
            </p>
            <Continue onClick={onNext} label="Now give them meaning" delay={0.5} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
