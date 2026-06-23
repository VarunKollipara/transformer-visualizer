"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { getEmbeddings, getInfo, type EmbeddingPoint } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import type { SceneProps } from "./types";

const display = (ch: string) => (ch === " " ? "␣" : ch);
const CONTEXT_BELOW = [
  "Whether 'tis nobler in the mind to suffer",
  "The slings and arrows of outrageous fortune,",
];

type Phase = "invite" | "tokens" | "vectors";

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

  return (
    <div className="text-center">
      <motion.h1
        layout="position"
        className="font-display text-3xl font-semibold tracking-tight text-stone-900 sm:text-4xl"
      >
        This is how ChatGPT actually works.
      </motion.h1>

      <AnimatePresence mode="popLayout">
        {phase === "invite" && (
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
        {phase === "invite" && (
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

      {/* the persistent characters — they morph from line → tokens → vectors */}
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
            : { layout: { type: "spring", stiffness: 210, damping: 30 } }
        }
        className={
          phase === "invite"
            ? "mx-auto mt-2 flex max-w-xl cursor-pointer flex-wrap justify-start rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-2"
            : "mx-auto mt-6 flex max-w-2xl flex-wrap justify-center gap-1.5"
        }
      >
        {chars.map((ch, i) => (
          <motion.div
            key={i}
            layout
            transition={{ type: "spring", stiffness: 240, damping: 28 }}
            whileHover={tokens ? { scale: 1.12 } : undefined}
            className={
              phase === "invite"
                ? "px-0 py-0"
                : "flex flex-col items-center rounded-lg border border-sky-200 bg-sky-50 px-1.5 py-1.5"
            }
          >
            <motion.span
              layout="position"
              className={
                tokens
                  ? "font-mono text-base leading-none text-stone-800"
                  : "font-mono text-[15px] leading-none text-indigo-900"
              }
            >
              {tokens ? display(ch) : ch}
            </motion.span>

            {phase === "tokens" && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.3 }}
                className="mt-1 font-mono text-[11px] leading-none text-sky-600"
              >
                {stoi.get(ch) ?? "?"}
              </motion.span>
            )}

            {vectors && (
              <div className="mt-1 flex h-5 items-end gap-[1.5px]">
                {miniVec(ch).map((h, j) => (
                  <motion.span
                    key={j}
                    initial={{ scaleY: 0 }}
                    animate={{ scaleY: h }}
                    transition={{
                      delay: 0.1 + j * 0.03,
                      type: "spring",
                      stiffness: 320,
                      damping: 20,
                    }}
                    style={{ transformOrigin: "bottom", height: "100%" }}
                    className="block w-[2px] rounded-sm bg-violet-400"
                  />
                ))}
              </div>
            )}
          </motion.div>
        ))}
      </motion.div>

      {/* invite helper text */}
      <AnimatePresence>
        {phase === "invite" && (
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

      {/* tokens explanation */}
      <AnimatePresence mode="popLayout">
        {phase === "tokens" && (
          <motion.div
            key="explain-tokens"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            <p className="mx-auto mt-7 max-w-lg text-stone-500">
              Each letter slid into place above its number. A model can&apos;t read
              letters — only numbers — so every character is swapped for its spot
              in a fixed list of 65. This is{" "}
              <strong className="text-stone-700">tokenization</strong>.
            </p>
            <Continue
              onClick={() => setPhase("vectors")}
              label="Now give them meaning"
              delay={0.4}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* vectors explanation + map */}
      <AnimatePresence>
        {vectors && (
          <motion.div
            key="explain-vectors"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.4 }}
          >
            <p className="mx-auto mt-6 max-w-lg text-stone-500">
              Each number now becomes a{" "}
              <strong className="text-stone-700">vector</strong> — a list of 128
              numbers the model learns (a few shown). Characters it uses similarly
              end up with similar vectors, forming a map:
            </p>
            {points && (
              <svg
                viewBox={`0 0 ${W} ${H}`}
                className="mx-auto mt-3 w-full max-w-md rounded-xl bg-stone-50"
              >
                <line x1={sx(-1)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#e7e5e4" />
                <line x1={sx(0)} y1={sy(-1)} x2={sx(0)} y2={sy(1)} stroke="#e7e5e4" />
                {points.map((p) => {
                  const on = lineChars.has(p.char);
                  return (
                    <text
                      key={p.id}
                      x={sx(p.x)}
                      y={sy(p.y)}
                      fontSize={on ? 13 : 10}
                      fontWeight={on ? 700 : 400}
                      fill={on ? "#7c3aed" : "#d6d3d1"}
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="font-mono"
                    >
                      {display(p.char)}
                    </text>
                  );
                })}
              </svg>
            )}
            <Continue onClick={onNext} label="Let the tokens talk to each other" delay={0.3} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
