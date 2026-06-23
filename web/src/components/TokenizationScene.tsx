"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { getInfo } from "@/lib/api";

// Famous lines from the training corpus. Clicking one animates its tokenization.
const LINES = [
  "To be, or not to be, that is the question",
  "But soft! what light through yonder window breaks?",
  "Now is the winter of our discontent",
  "Friends, Romans, countrymen, lend me your ears",
];

const STATS: [string, string][] = [
  ["1,115,394", "characters of text"],
  ["65", "unique characters"],
  ["0", "facts about the world"],
];

const show = (ch: string) => (ch === " " ? "␣" : ch);

export default function TokenizationScene() {
  const [vocab, setVocab] = useState<string[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const reduce = useReducedMotion();

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

  return (
    <div className="overflow-hidden rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
      <AnimatePresence mode="wait" initial={false}>
        {selected === null ? (
          <motion.div
            key="list"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            <p className="mb-3 text-sm text-stone-500">
              A few lines from the training text.{" "}
              <span className="font-medium text-stone-700">Click one</span> to
              watch it become the numbers the model actually reads.
            </p>
            <div className="space-y-1.5">
              {LINES.map((line) => (
                <motion.button
                  key={line}
                  onClick={() => setSelected(line)}
                  whileHover={reduce ? undefined : { scale: 1.01 }}
                  whileTap={reduce ? undefined : { scale: 0.995 }}
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  className="group flex w-full items-center justify-between gap-3 rounded-lg border border-stone-200 px-4 py-2.5 text-left font-mono text-[15px] text-stone-700 hover:border-sky-300 hover:bg-sky-50/50"
                >
                  <span>{line}</span>
                  <span className="shrink-0 text-xs text-stone-300 transition group-hover:text-sky-500">
                    tokenize →
                  </span>
                </motion.button>
              ))}
            </div>
            <div className="mt-4 grid grid-cols-3 gap-3 text-center">
              {STATS.map(([n, label]) => (
                <div key={label} className="rounded-xl bg-amber-50 px-2 py-3">
                  <div className="font-display text-2xl font-semibold text-amber-700">
                    {n}
                  </div>
                  <div className="mt-1 text-xs text-stone-500">{label}</div>
                </div>
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="detail"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            <button
              onClick={() => setSelected(null)}
              className="mb-4 inline-flex items-center gap-1 text-sm text-stone-500 hover:text-stone-800"
            >
              ← back to the text
            </button>
            <p className="mb-4 font-mono text-sm text-stone-400">{selected}</p>

            <motion.div
              className="flex flex-wrap gap-1.5"
              variants={{
                show: { transition: { staggerChildren: reduce ? 0 : 0.022 } },
              }}
              initial="hidden"
              animate="show"
            >
              {[...selected].map((ch, i) => {
                const id = stoi.get(ch);
                return (
                  <motion.div
                    key={i}
                    variants={{
                      hidden: { opacity: 0, y: 14, scale: 0.6 },
                      show: { opacity: 1, y: 0, scale: 1 },
                    }}
                    transition={{ type: "spring", stiffness: 500, damping: 26 }}
                    className="flex flex-col items-center rounded-lg border border-sky-200 bg-sky-50 px-2 py-1.5"
                  >
                    <span className="font-mono text-base leading-none text-stone-800">
                      {show(ch)}
                    </span>
                    <span className="mt-1 font-mono text-[11px] leading-none text-sky-600">
                      {id ?? "?"}
                    </span>
                  </motion.div>
                );
              })}
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: reduce ? 0 : 0.35, ease: "easeOut" }}
              className="mt-4 text-sm text-stone-500"
            >
              Every character — even spaces — became a number from the
              model&apos;s 65-character vocabulary. This row of numbers is what
              actually flows into the model.
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
