"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const ORDER_A = ["dog", "bites", "man"];
const ORDER_B = ["man", "bites", "dog"];
const TILES = ["t", "o", "b", "e"];

// Two phases sharing one central object: the token tile "t" stays on stage and
// MORPHS from "a positioned token" into "the token that emits query/key/value".
export default function PositionQueryScene({ onNext }: SceneProps) {
  const [phase, setPhase] = useState<"order" | "qkv">("order");
  const [swapped, setSwapped] = useState(false);
  const words = swapped ? ORDER_B : ORDER_A;
  const tiles = phase === "order" ? TILES : ["t"];

  return (
    <div className="text-center">
      {/* title + intro, keyed by phase (enter-only fade) */}
      <motion.div key={phase} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "order" ? (
          <>
            <SceneTitle>Order changes everything</SceneTitle>
            <SceneText>
              Embeddings say <em>what</em> a token is, not <em>where</em>. The same
              words in a different order mean opposite things:
            </SceneText>
          </>
        ) : (
          <>
            <SceneTitle>How a token decides what to look at</SceneTitle>
            <SceneText>
              Now positioned, this token produces three small vectors — think of it
              walking into a room with:
            </SceneText>
          </>
        )}
      </motion.div>

      {/* dog/man demo — only while explaining order */}
      {phase === "order" && (
        <div>
          <div className="mt-5 flex items-center justify-center gap-2 font-mono text-lg">
            {words.map((w) => (
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
          <button onClick={() => setSwapped((s) => !s)} className="mt-2 text-sm font-medium text-indigo-600 hover:text-indigo-500">
            ⇄ swap dog and man
          </button>
          <p className="mx-auto mt-4 max-w-lg text-[15px] text-stone-500">
            So we add a <strong className="text-stone-700">position</strong> to
            each token&apos;s vector — same letter, different spot, different
            vector. Take the first token:
          </p>
        </div>
      )}

      {/* the token tiles — "t" persists and morphs to centre stage in the qkv phase */}
      <div className="mt-6 flex items-center justify-center gap-2">
        <AnimatePresence>
          {tiles.map((ch) => (
            <motion.div
              key={ch}
              layout
              initial={ch === "t" ? false : { opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ layout: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }, opacity: { duration: 0.3 } }}
              className={`flex flex-col items-center rounded-lg border ${
                phase === "qkv" && ch === "t"
                  ? "border-teal-300 bg-teal-50 px-5 py-3"
                  : "border-sky-200 bg-sky-50 px-2.5 py-1.5"
              }`}
            >
              <motion.span layout="position" className={`font-mono text-stone-800 ${phase === "qkv" && ch === "t" ? "text-2xl" : "text-base"}`}>
                {ch}
              </motion.span>
              {phase === "order" && (
                <span className="mt-1 rounded bg-amber-100 px-1 font-mono text-[10px] text-amber-700">
                  pos {TILES.indexOf(ch)}
                </span>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* query / key / value emanating from that token */}
      {phase === "qkv" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35, duration: 0.4 }}>
          <div className="mx-auto mt-5 grid max-w-md grid-cols-3 gap-3">
            {[
              ["Query", "what I want", "teal"],
              ["Key", "what I offer", "sky"],
              ["Value", "my info", "violet"],
            ].map(([t, d, c], i) => (
              <motion.div
                key={t}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45 + i * 0.12 }}
                className={`rounded-xl border p-2.5 ${
                  c === "teal"
                    ? "border-teal-200 bg-teal-50 text-teal-800"
                    : c === "sky"
                      ? "border-sky-200 bg-sky-50 text-sky-800"
                      : "border-violet-200 bg-violet-50 text-violet-800"
                }`}
              >
                <div className="text-sm font-semibold">{t}</div>
                <div className="text-xs opacity-75">{d}</div>
              </motion.div>
            ))}
          </div>
          <SceneText delay={0.8} className="mt-4 text-[15px]">
            It compares its <strong className="text-teal-700">query</strong> to
            every other token&apos;s <strong className="text-sky-700">key</strong>{" "}
            to score relevance, softmaxes those into percentages, then blends
            everyone&apos;s <strong className="text-violet-700">values</strong>.
            That blend is its new, context-aware vector.
          </SceneText>
        </motion.div>
      )}

      {phase === "order" ? (
        <Continue onClick={() => setPhase("qkv")} label="Now: how it looks around" delay={0.2} />
      ) : (
        <Continue onClick={onNext} label="See it on the real model" delay={1} />
      )}
    </div>
  );
}
