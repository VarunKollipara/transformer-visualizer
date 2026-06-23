"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { getEmbeddings, type EmbeddingPoint } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch);

// Deterministic pseudo-vector just for the illustration of "one number -> many".
const BARS = 24;
function fakeVector(ch: string): number[] {
  const code = ch.charCodeAt(0);
  return Array.from(
    { length: BARS },
    (_, j) => 0.2 + 0.75 * Math.abs(Math.sin(code * 0.7 * (j + 1))),
  );
}

const W = 520;
const H = 300;
const PAD = 24;
const sx = (x: number) => PAD + ((x + 1) / 2) * (W - 2 * PAD);
const sy = (y: number) => PAD + (1 - (y + 1) / 2) * (H - 2 * PAD);

export default function EmbeddingsScene({ onNext }: SceneProps) {
  const [points, setPoints] = useState<EmbeddingPoint[] | null>(null);

  useEffect(() => {
    getEmbeddings()
      .then((d) => setPoints(d.points))
      .catch(() => {});
  }, []);

  const lineChars = useMemo(() => new Set([...EXAMPLE_LINE]), []);
  const vec = fakeVector("o");

  return (
    <div className="text-center">
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-2xl font-semibold text-stone-900 sm:text-3xl"
      >
        Next, each number becomes meaning
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mx-auto mt-2 max-w-lg text-stone-500"
      >
        A number like <span className="font-mono">53</span> is just a label. The
        model swaps it for a <strong className="text-stone-700">vector</strong> —
        a list of 128 numbers it learns — so it can capture how characters relate.
      </motion.p>

      {/* one token expanding into its vector */}
      <div className="mt-7 flex items-center justify-center gap-3">
        <div className="flex flex-col items-center rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-1.5">
          <span className="font-mono text-lg leading-none text-stone-800">o</span>
          <span className="mt-1 font-mono text-xs leading-none text-sky-600">53</span>
        </div>
        <span className="text-stone-300">→</span>
        <div className="flex h-12 items-end gap-[3px]">
          {vec.map((h, j) => (
            <motion.span
              key={j}
              initial={{ scaleY: 0 }}
              animate={{ scaleY: h }}
              transition={{
                delay: 0.4 + j * 0.02,
                type: "spring",
                stiffness: 320,
                damping: 22,
              }}
              style={{ transformOrigin: "bottom", height: "100%" }}
              className="block w-1.5 rounded-sm bg-violet-400"
            />
          ))}
        </div>
        <span className="font-mono text-xs text-stone-400">128 numbers</span>
      </div>

      {/* the real learned map */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.1 }}
        className="mx-auto mt-7 max-w-lg text-sm text-stone-500"
      >
        Do this for all 65 characters and a map appears — characters the model
        uses similarly drift together. Yours (from this line) are in{" "}
        <span className="font-semibold text-violet-600">violet</span>:
      </motion.p>

      <motion.svg
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.3 }}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto mt-3 w-full max-w-xl rounded-xl bg-stone-50"
      >
        <line x1={sx(-1)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#e7e5e4" />
        <line x1={sx(0)} y1={sy(-1)} x2={sx(0)} y2={sy(1)} stroke="#e7e5e4" />
        {points?.map((p) => {
          const on = lineChars.has(p.char);
          return (
            <text
              key={p.id}
              x={sx(p.x)}
              y={sy(p.y)}
              fontSize={on ? 15 : 11}
              fontWeight={on ? 700 : 400}
              fill={on ? "#7c3aed" : "#d6d3d1"}
              textAnchor="middle"
              dominantBaseline="central"
              className="font-mono"
            >
              {show(p.char)}
            </text>
          );
        })}
      </motion.svg>

      <Continue onClick={onNext} label="Let the tokens talk to each other" delay={1.6} />
    </div>
  );
}
