"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { forward, type ForwardResponse } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const ORDER_A = ["dog", "bites", "man"];
const ORDER_B = ["man", "bites", "dog"];
const ORDER_TILES = ["T", "o", "b", "e"];
const LINE = [...EXAMPLE_LINE];
const show = (ch: string) => (ch === " " ? "␣" : ch);

type Phase = "order" | "qkv" | "attention";

// Three phases sharing one central object: the first token tile ("T") stays on
// stage and morphs — positioned token → the token emitting query/key/value →
// the first tile of the whole line, which then lights up with real attention.
export default function PositionQueryScene({ onNext }: SceneProps) {
  const [phase, setPhase] = useState<Phase>("order");
  const [swapped, setSwapped] = useState(false);
  const [data, setData] = useState<ForwardResponse | null>(null);
  const [query, setQuery] = useState<number | null>(null);

  useEffect(() => {
    forward(EXAMPLE_LINE, 5)
      .then((d) => setData(d))
      .catch(() => {});
  }, []);

  const words = swapped ? ORDER_B : ORDER_A;
  const tiles = phase === "order" ? ORDER_TILES : phase === "qkv" ? ["T"] : LINE;

  const row = useMemo(() => {
    if (!data || query === null) return null;
    return data.attention[0][0][query];
  }, [data, query]);
  const rowMax = useMemo(
    () => (row && query !== null ? Math.max(...row.filter((_, j) => j <= query)) : 1),
    [row, query],
  );

  const goAttention = () => {
    setQuery(LINE.length - 1);
    setPhase("attention");
  };

  return (
    <div className="text-center">
      {/* title + intro, keyed by phase */}
      <motion.div key={phase} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "order" && (
          <>
            <SceneTitle>Order changes everything</SceneTitle>
            <SceneText>
              Embeddings say <em>what</em> a token is, not <em>where</em>. The same
              words in a different order mean opposite things:
            </SceneText>
          </>
        )}
        {phase === "qkv" && (
          <>
            <SceneTitle>How a token decides what to look at</SceneTitle>
            <SceneText>Now positioned, this token produces three small vectors:</SceneText>
          </>
        )}
        {phase === "attention" && (
          <>
            <SceneTitle>Now it looks at the others — for real</SceneTitle>
            <SceneText>
              That one token is part of a whole line. Each compares its query to
              the others&apos; keys. <strong className="text-stone-700">Click any
              character</strong> to see what it attends to — brighter = more.
            </SceneText>
          </>
        )}
      </motion.div>

      {/* dog/man demo — order only */}
      {phase === "order" && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}>
          <div className="mt-5 flex items-center justify-center gap-2 font-mono text-lg">
            {words.map((w) => (
              <motion.span key={w} layout transition={{ type: "spring", stiffness: 240, damping: 26 }}
                className={`rounded-lg px-3 py-1.5 ${w === "dog" ? "bg-violet-100 text-violet-800" : w === "man" ? "bg-amber-100 text-amber-800" : "bg-stone-100 text-stone-600"}`}>
                {w}
              </motion.span>
            ))}
          </div>
          <button onClick={() => setSwapped((s) => !s)} className="mt-2 text-sm font-medium text-indigo-600 hover:text-indigo-500">
            ⇄ swap dog and man
          </button>
          <p className="mx-auto mt-4 max-w-lg text-[15px] text-stone-500">
            So we add a <strong className="text-stone-700">position</strong> to each
            token&apos;s vector. Take the first token:
          </p>
        </motion.div>
      )}

      {/* the shared tiles — tile 0 ("T") morphs through all three phases */}
      <div className={`mx-auto mt-6 flex max-w-2xl flex-wrap items-start justify-center ${phase === "attention" ? "gap-1" : "gap-2"}`}>
        <AnimatePresence>
          {tiles.map((ch, i) => {
            const isFirst = i === 0;
            const big = phase === "qkv" && isFirst;
            const w = phase === "attention" && query !== null && i <= query ? (row ? row[i] / (rowMax || 1) : 0) : 0;
            const isQuery = phase === "attention" && i === query;
            return (
              <motion.div
                key={i}
                layout
                initial={isFirst ? false : { opacity: 0, scale: 0.6 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                transition={{ layout: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }, opacity: { duration: 0.3 } }}
                onClick={() => phase === "attention" && setQuery(i)}
                className={`relative flex flex-col items-center overflow-hidden rounded-lg border font-mono ${
                  big ? "border-teal-300 bg-teal-50 px-5 py-3" : "border-sky-200 bg-sky-50"
                } ${phase === "attention" ? `h-9 w-7 cursor-pointer justify-center ${isQuery ? "ring-2 ring-teal-400" : ""}` : "px-2.5 py-1.5"}`}
              >
                {phase === "attention" && (
                  <motion.span className="absolute inset-0 bg-teal-400" initial={false} animate={{ opacity: w }} transition={{ duration: 0.3 }} />
                )}
                <motion.span layout="position" className={`relative text-stone-800 ${big ? "text-2xl" : phase === "attention" ? "text-sm" : "text-base"}`}>
                  {phase === "attention" ? show(ch) : ch}
                </motion.span>
                {phase === "order" && (
                  <span className="relative mt-1 rounded bg-amber-100 px-1 text-[10px] text-amber-700">pos {i}</span>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* query/key/value — qkv only */}
      {phase === "qkv" && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35, duration: 0.4 }}>
          <div className="mx-auto mt-5 grid max-w-md grid-cols-3 gap-3">
            {[
              ["Query", "what I want", "teal"],
              ["Key", "what I offer", "sky"],
              ["Value", "my info", "violet"],
            ].map(([t, d, c]) => (
              <div key={t} className={`rounded-xl border p-2.5 ${c === "teal" ? "border-teal-200 bg-teal-50 text-teal-800" : c === "sky" ? "border-sky-200 bg-sky-50 text-sky-800" : "border-violet-200 bg-violet-50 text-violet-800"}`}>
                <div className="text-sm font-semibold">{t}</div>
                <div className="text-xs opacity-75">{d}</div>
              </div>
            ))}
          </div>
          <SceneText delay={0.6} className="mt-4 text-[15px]">
            It compares its <strong className="text-teal-700">query</strong> to every
            other token&apos;s <strong className="text-sky-700">key</strong>, softmaxes
            those into percentages, then blends everyone&apos;s{" "}
            <strong className="text-violet-700">values</strong>.
          </SceneText>
        </motion.div>
      )}

      {/* attention note */}
      {phase === "attention" && (
        <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.5, ease: [0.22, 1, 0.36, 1] }} className="mx-auto mt-5 max-w-lg text-sm text-stone-400">
          It can only look left — never at the future. This is one of several
          &ldquo;heads&rdquo;, each learning a different pattern.
        </motion.p>
      )}

      {phase === "order" && <Continue onClick={() => setPhase("qkv")} label="Now: how it looks around" delay={0.2} />}
      {phase === "qkv" && <Continue onClick={goAttention} label="See it on the whole line" delay={0.9} />}
      {phase === "attention" && <Continue onClick={onNext} label="Then each token thinks for itself" delay={0.4} />}
    </div>
  );
}
