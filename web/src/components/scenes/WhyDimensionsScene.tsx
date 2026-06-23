"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { getEmbeddings, type EmbeddingPoint } from "@/lib/api";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

const REP = ["T", "e", "a", "o", ".", ",", " ", "q"];
const display = (ch: string) => (ch === " " ? "␣" : ch);

const W = 480;
const sx = (x: number, w = W) => 24 + ((x + 1) / 2) * (w - 48);
const sy = (y: number, h: number) => 20 + (1 - (y + 1) / 2) * (h - 40);

function miniVec(ch: string, n = 12): number[] {
  const code = ch.charCodeAt(0);
  return Array.from(
    { length: n },
    (_, j) => 0.2 + 0.78 * Math.abs(Math.sin(code * 0.7 * (j + 1))),
  );
}

const CAPTIONS: Record<number, React.ReactNode> = {
  1: (
    <>
      <strong className="text-stone-700">One number</strong> = a spot on a line.
      You can only <em>sort</em> tokens. Two unrelated things — a vowel and a
      comma — are forced onto the same axis with nowhere else to go.
    </>
  ),
  2: (
    <>
      <strong className="text-stone-700">Two numbers</strong> = a point on a
      plane. Now tokens can differ in two independent ways at once. Better — but
      still only two kinds of difference.
    </>
  ),
  128: (
    <>
      <strong className="text-stone-700">128 numbers</strong> = 128 independent
      ways to differ. Room to capture vowel-ness, capitalization, punctuation,
      what usually follows… all on separate axes. We can&apos;t draw it, but the
      model uses every one.
    </>
  ),
};

export default function WhyDimensionsScene({ onNext }: SceneProps) {
  const [dims, setDims] = useState<1 | 2 | 128>(1);
  const [points, setPoints] = useState<EmbeddingPoint[] | null>(null);

  useEffect(() => {
    getEmbeddings()
      .then((d) => setPoints(d.points))
      .catch(() => {});
  }, []);

  const reps = useMemo(
    () => (points ? REP.map((c) => points.find((p) => p.char === c)).filter(Boolean) as EmbeddingPoint[] : []),
    [points],
  );

  return (
    <div className="text-center">
      <SceneTitle>Why 128 numbers per token?</SceneTitle>
      <SceneText>
        A token&apos;s vector is just coordinates. How many coordinates you give
        it decides how much it can express. Slide up the dimensions:
      </SceneText>

      <div className="mt-5 inline-flex rounded-full border border-stone-200 bg-white p-1 text-sm">
        {([1, 2, 128] as const).map((d) => (
          <button
            key={d}
            onClick={() => setDims(d)}
            className={`rounded-full px-4 py-1.5 font-medium transition ${
              dims === d ? "bg-violet-600 text-white" : "text-stone-500 hover:text-stone-800"
            }`}
          >
            {d === 1 ? "1 number" : d === 2 ? "2 numbers" : "128 numbers"}
          </button>
        ))}
      </div>

      <div className="mx-auto mt-6 min-h-[260px] max-w-xl">
        <AnimatePresence mode="wait">
          {dims === 1 && (
            <motion.svg
              key="1d"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.4 }}
              viewBox={`0 0 ${W} 90`}
              className="w-full"
            >
              <line x1={sx(-1)} y1={50} x2={sx(1)} y2={50} stroke="#e7e5e4" strokeWidth={2} />
              {reps.map((p) => (
                <g key={p.id}>
                  <circle cx={sx(p.x)} cy={50} r={3} fill="#7c3aed" />
                  <text x={sx(p.x)} y={32} fontSize={14} textAnchor="middle" className="font-mono" fill="#7c3aed">
                    {display(p.char)}
                  </text>
                </g>
              ))}
            </motion.svg>
          )}
          {dims === 2 && (
            <motion.svg
              key="2d"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.4 }}
              viewBox={`0 0 ${W} 240`}
              className="w-full rounded-xl bg-stone-50"
            >
              <line x1={sx(-1)} y1={sy(0, 240)} x2={sx(1)} y2={sy(0, 240)} stroke="#e7e5e4" />
              <line x1={sx(0)} y1={sy(-1, 240)} x2={sx(0)} y2={sy(1, 240)} stroke="#e7e5e4" />
              {reps.map((p) => (
                <text
                  key={p.id}
                  x={sx(p.x)}
                  y={sy(p.y, 240)}
                  fontSize={15}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="font-mono"
                  fill="#7c3aed"
                >
                  {display(p.char)}
                </text>
              ))}
            </motion.svg>
          )}
          {dims === 128 && (
            <motion.div
              key="128d"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.4 }}
              className="space-y-1.5"
            >
              {reps.map((p) => (
                <div key={p.id} className="flex items-center justify-center gap-2">
                  <span className="w-5 text-right font-mono text-sm text-stone-700">
                    {display(p.char)}
                  </span>
                  <div className="flex h-6 items-end gap-[2px]">
                    {miniVec(p.char).map((h, j) => (
                      <motion.span
                        key={j}
                        initial={{ scaleY: 0 }}
                        animate={{ scaleY: h }}
                        transition={{ delay: j * 0.015, type: "spring", stiffness: 300, damping: 20 }}
                        style={{ transformOrigin: "bottom", height: "100%" }}
                        className="block w-[3px] rounded-sm bg-violet-400"
                      />
                    ))}
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="mx-auto mt-4 min-h-[5rem] max-w-xl">
        <AnimatePresence mode="wait">
          <motion.p
            key={dims}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="text-[15px] leading-relaxed text-stone-500"
          >
            {CAPTIONS[dims]}
          </motion.p>
        </AnimatePresence>
      </div>

      <Continue onClick={onNext} label="But position matters too" delay={0.1} />
    </div>
  );
}
