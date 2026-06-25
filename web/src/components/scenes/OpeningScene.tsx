"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { getEmbeddings, getInfo, type EmbeddingPoint } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import type { SceneProps } from "./types";

const display = (ch: string) => (ch === " " ? "␣" : ch);
const CONTEXT_BELOW = [
  "Whether 'tis nobler in the mind to suffer",
  "The slings and arrows of outrageous fortune,",
];

type Phase = "invite" | "tokens" | "vectors" | "dims";

const DIMS_CAPTION: Record<number, React.ReactNode> = {
  1: (
    <>
      <strong className="text-stone-700">1 number</strong> per token — you can only
      line them up. A vowel and a comma get forced onto the same axis.
    </>
  ),
  2: (
    <>
      <strong className="text-stone-700">2 numbers</strong> — a point on a plane, so
      tokens can differ in two ways at once. Better, but still cramped.
    </>
  ),
  128: (
    <>
      <strong className="text-stone-700">128 numbers</strong> — 128 independent ways
      to differ, room to capture vowel-ness, case, punctuation… all at once. We
      can&apos;t draw it, but the model uses every axis.
    </>
  ),
};

// Deterministic mini "vector" purely for the visual (the real one is 128-D).
function miniVec(ch: string): number[] {
  const code = ch.charCodeAt(0);
  return Array.from(
    { length: 7 },
    (_, j) => 0.25 + 0.7 * Math.abs(Math.sin(code * 0.7 * (j + 1))),
  );
}

// Map projection (compact).
const W = 460;
const H = 210;
const PAD = 18;
const sx = (x: number) => PAD + ((x + 1) / 2) * (W - 2 * PAD);
const sy = (y: number) => PAD + (1 - (y + 1) / 2) * (H - 2 * PAD);

export default function OpeningScene({ onNext }: SceneProps) {
  const [vocab, setVocab] = useState<string[] | null>(null);
  const [points, setPoints] = useState<EmbeddingPoint[] | null>(null);
  const [phase, setPhase] = useState<Phase>("invite");
  const [dims, setDims] = useState<1 | 2 | 128>(128);

  useEffect(() => {
    getInfo()
      .then((i) => setVocab(i.vocab))
      .catch(() => {});
    getEmbeddings()
      .then((d) => setPoints(d.points))
      .catch(() => {});
  }, []);

  const stoi = useMemo(() => {
    const m = new Map<string, number>();
    vocab?.forEach((c, i) => m.set(c, i));
    return m;
  }, [vocab]);
  const lineChars = useMemo(() => new Set([...EXAMPLE_LINE]), []);

  const chars = [...EXAMPLE_LINE];
  const tokens = phase !== "invite";
  const vectors = phase === "vectors";
  const showVectors = phase === "vectors" || phase === "dims";

  return (
    <div className="text-center">
      <motion.h1
        layout="position"
        className="font-display text-3xl font-semibold tracking-tight text-stone-900 sm:text-4xl"
      >
        This is how ChatGPT actually works.
      </motion.h1>

      {phase === "invite" && (
        <>
          <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }} className="mt-3 text-stone-500">
            Trained on nothing but the complete works of Shakespeare.
          </motion.p>
          <motion.p initial={{ opacity: 0, y: 8 }} animate={{ opacity: 0.45, y: 0 }} transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }} className="mx-auto mt-10 max-w-xl text-left font-mono text-[15px] text-stone-400">
            HAMLET:
          </motion.p>
        </>
      )}

      {/* the persistent characters — they morph line → tokens → vectors */}
      <motion.div
        layout
        role={phase === "invite" ? "button" : undefined}
        onClick={() => phase === "invite" && setPhase("tokens")}
        animate={
          phase === "invite"
            ? {
                boxShadow: [
                  "0 0 0 0 rgba(99,102,241,0.0)",
                  "0 0 0 5px rgba(99,102,241,0.14)",
                  "0 0 0 0 rgba(99,102,241,0.0)",
                ],
              }
            : { boxShadow: "0 0 0 0 rgba(0,0,0,0)" }
        }
        transition={
          phase === "invite"
            ? { boxShadow: { duration: 2.2, repeat: Infinity, ease: "easeInOut" } }
            : { layout: { duration: 0.9, ease: [0.22, 1, 0.36, 1] } }
        }
        className={
          phase === "invite"
            ? "mx-auto mt-2 flex max-w-xl cursor-pointer flex-wrap justify-start rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-2 transition-colors duration-500"
            : "mx-auto mt-6 flex max-w-2xl flex-wrap justify-center gap-1.5 rounded-lg border border-transparent bg-transparent transition-colors duration-500"
        }
      >
        {chars.map((ch, i) => (
          <motion.div
            key={i}
            layout
            transition={{ layout: { duration: 0.85, ease: [0.22, 1, 0.36, 1], delay: i * 0.008 } }}
            whileHover={tokens ? { scale: 1.12 } : undefined}
            className={
              phase === "invite"
                ? "px-0 py-0"
                : "flex flex-col items-center rounded-lg border border-sky-200 bg-sky-50 px-1.5 py-1.5"
            }
          >
            <motion.span
              layout="position"
              className={`whitespace-pre font-mono leading-none ${
                tokens ? "text-base text-stone-800" : "text-[15px] text-indigo-900"
              }`}
            >
              {tokens ? display(ch) : ch}
            </motion.span>

            {phase === "tokens" && (
              <motion.span
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.75, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="mt-1 font-mono text-[11px] leading-none text-sky-600"
              >
                {stoi.get(ch) ?? "?"}
              </motion.span>
            )}

            {showVectors && (
              <div className="mt-1 flex h-5 items-end gap-[1.5px]">
                {miniVec(ch).map((h, j) => {
                  const dimsBars = dims === 128 ? 7 : dims;
                  const target = phase === "dims" && j >= dimsBars ? 0 : h;
                  return (
                    <motion.span
                      key={j}
                      initial={{ scaleY: 0 }}
                      animate={{ scaleY: target }}
                      transition={{
                        delay: phase === "vectors" ? 0.3 + j * 0.07 : 0,
                        type: "spring",
                        stiffness: 220,
                        damping: 20,
                      }}
                      style={{ transformOrigin: "bottom", height: "100%" }}
                      className="block w-[2px] rounded-sm bg-violet-400"
                    />
                  );
                })}
              </div>
            )}
          </motion.div>
        ))}
      </motion.div>

      {/* invite helper text */}
      {phase === "invite" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}>
          {CONTEXT_BELOW.map((l) => (
            <p key={l} className="mx-auto mt-2 max-w-xl text-left font-mono text-[15px] text-stone-300">
              {l}
            </p>
          ))}
          <p className="mt-7 text-sm text-stone-400">Click the glowing line to begin.</p>
        </motion.div>
      )}

      {/* tokens explanation */}
      {phase === "tokens" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.4 }}>
          <p className="mx-auto mt-7 max-w-lg text-stone-500">
            Each letter slid into place above its number. A model can&apos;t read
            letters — only numbers — so every character is swapped for its spot in a
            fixed list of 65. This is{" "}
            <strong className="text-stone-700">tokenization</strong>.
          </p>
          <Continue onClick={() => setPhase("vectors")} label="Now give them meaning" delay={0.4} />
        </motion.div>
      )}

      {/* vectors explanation + map */}
      {vectors && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.4 }}>
          <p className="mx-auto mt-6 max-w-lg text-stone-500">
            Each number now becomes a{" "}
            <strong className="text-stone-700">vector</strong> — a list of 128 numbers
            the model learns (a few shown). Characters it uses similarly end up with
            similar vectors, forming a map:
          </p>
          {points && (
            <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto mt-3 w-full max-w-md rounded-xl bg-stone-50">
              <line x1={sx(-1)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#e7e5e4" />
              <line x1={sx(0)} y1={sy(-1)} x2={sx(0)} y2={sy(1)} stroke="#e7e5e4" />
              {points.map((p) => {
                const on = lineChars.has(p.char);
                return (
                  <text key={p.id} x={sx(p.x)} y={sy(p.y)} fontSize={on ? 13 : 10} fontWeight={on ? 700 : 400} fill={on ? "#7c3aed" : "#d6d3d1"} textAnchor="middle" dominantBaseline="central" className="font-mono">
                    {display(p.char)}
                  </text>
                );
              })}
            </svg>
          )}
          <Continue onClick={() => setPhase("dims")} label="Why so many numbers?" delay={0.3} />
        </motion.div>
      )}

      {/* dims: the same vectors collapse to 1 / 2 / 128 numbers */}
      {phase === "dims" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <p className="mx-auto mt-6 max-w-lg text-stone-500">
            But why <strong className="text-stone-700">128</strong>? A vector is just
            coordinates — more of them means more ways a token can differ. Collapse the
            same vectors and see what&apos;s lost:
          </p>
          <div className="mt-4 inline-flex rounded-full border border-stone-200 bg-white p-1 text-sm">
            {([1, 2, 128] as const).map((d) => (
              <button
                key={d}
                onClick={() => setDims(d)}
                className={`rounded-full px-4 py-1.5 font-medium transition ${dims === d ? "bg-violet-600 text-white" : "text-stone-500 hover:text-stone-800"}`}
              >
                {d === 1 ? "1 number" : d === 2 ? "2 numbers" : "128 numbers"}
              </button>
            ))}
          </div>
          <p className="mx-auto mt-4 min-h-[3.5rem] max-w-lg text-[15px] text-stone-500">
            {DIMS_CAPTION[dims]}
          </p>
          <Continue onClick={onNext} label="But order matters too" delay={0.1} />
        </motion.div>
      )}
    </div>
  );
}
