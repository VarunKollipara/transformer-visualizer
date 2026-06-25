"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

function Bars({ values, color = "bg-indigo-400", from = 0 }: { values: number[]; color?: string; from?: number }) {
  return (
    <div className="flex h-12 items-end gap-[2px]">
      {values.map((v, j) => (
        <motion.span
          key={j}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: Math.max(0.02, v) }}
          transition={{ delay: from + j * 0.02, type: "spring", stiffness: 280, damping: 20 }}
          style={{ transformOrigin: "bottom", height: "100%" }}
          className={`block w-1.5 rounded-sm ${color}`}
        />
      ))}
    </div>
  );
}

const IN = [0.5, 0.8, 0.3, 0.6, 0.4, 0.7, 0.45, 0.55];
const HIDDEN = Array.from({ length: 24 }, (_, j) => 0.25 + 0.7 * Math.abs(Math.sin(j * 1.3)));
const OUT = [0.6, 0.4, 0.7, 0.35, 0.65, 0.5, 0.45, 0.6];

// signed values for the "whole layer" ReLU demo
const SIGNED = [0.7, -0.4, 0.5, -0.8, 0.3, 0.9, -0.2, 0.6, -0.6, 0.4, -0.3, 0.8];

// A single value drawn as a bar above/below a zero line (up = positive/emerald,
// down = negative/rose). Used to make ReLU concrete on ONE number.
function SignedBar({ value, label }: { value: number; label: string }) {
  const half = 34; // px each side of the zero line
  const h = Math.min(Math.abs(value) / 1.5, 1) * half;
  const neg = value < 0;
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative h-[72px] w-10">
        <div className="absolute left-0 top-1/2 h-px w-full bg-stone-300" />
        <motion.div
          animate={{ height: h }}
          transition={{ type: "spring", stiffness: 320, damping: 24 }}
          style={neg ? { top: "50%" } : { bottom: "50%" }}
          className={`absolute left-1/2 w-6 -translate-x-1/2 rounded-sm ${neg ? "bg-rose-400" : "bg-emerald-400"}`}
        />
      </div>
      <span className="font-mono text-sm font-semibold text-stone-700">{value.toFixed(2)}</span>
      <span className="text-xs text-stone-400">{label}</span>
    </div>
  );
}

type Phase = "shape" | "relu";

export default function MLPScene({ onNext }: SceneProps) {
  const [phase, setPhase] = useState<Phase>("shape");
  const [x, setX] = useState(-0.6);
  const [reluAll, setReluAll] = useState(false);
  const y = Math.max(0, x);

  return (
    <div className="text-center">
      <motion.div key={phase} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "shape" ? (
          <>
            <SceneTitle>Then each token thinks for itself</SceneTitle>
            <SceneText>
              Attention <em>moved information between</em> tokens. Now each token
              processes what it gathered, on its own, through a small two-layer
              network — the <strong className="text-stone-700">MLP</strong>. First it{" "}
              <strong className="text-stone-700">widens</strong> the vector to give
              itself room to compute, then <strong className="text-stone-700">shrinks
              </strong> it back to the original size.
            </SceneText>
          </>
        ) : (
          <>
            <SceneTitle>The bend in the middle: ReLU</SceneTitle>
            <SceneText>
              Between widening and shrinking sits one tiny operation,{" "}
              <strong className="text-stone-700">ReLU</strong>. On a single number
              the rule could not be simpler: if it&apos;s negative, output{" "}
              <strong className="text-stone-700">0</strong>; if it&apos;s positive,{" "}
              <strong className="text-stone-700">keep it</strong>. That&apos;s the
              whole thing — <span className="font-mono">max(0, x)</span>.
            </SceneText>
          </>
        )}
      </motion.div>

      {/* ── SHAPE phase ── */}
      {phase === "shape" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
          <div className="mt-8 flex items-end justify-center gap-3">
            <div className="flex flex-col items-center gap-1">
              <Bars values={IN} />
              <span className="text-xs text-stone-400">vector (128)</span>
            </div>
            <span className="mb-5 text-stone-300">→</span>
            <div className="flex flex-col items-center gap-1">
              <Bars values={HIDDEN} color="bg-violet-400" from={0.2} />
              <span className="text-xs text-stone-400">widen ×4 (512)</span>
            </div>
            <span className="mb-5 text-stone-300">→</span>
            <div className="flex flex-col items-center gap-1">
              <Bars values={OUT} from={0.7} />
              <span className="text-xs text-stone-400">back to 128</span>
            </div>
          </div>
          <SceneText delay={0.3} className="mt-7 text-[15px]">
            Widening gives the layer more room to mix and reshape the numbers;
            shrinking brings it back so the next block sees the same size. But there
            has to be a <em>bend</em> in between — that&apos;s next.
          </SceneText>
          <Continue onClick={() => setPhase("relu")} label="See the bend" delay={0.4} />
        </motion.div>
      )}

      {/* ── RELU phase ── */}
      {phase === "relu" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
          {/* one-number interactive */}
          <div className="mx-auto mt-7 max-w-md rounded-xl border border-stone-200 bg-white p-5">
            <div className="flex items-end justify-center gap-8">
              <SignedBar value={x} label="input x" />
              <span className="mb-7 text-2xl text-stone-300">→</span>
              <SignedBar value={y} label="output max(0, x)" />
            </div>
            <label className="mt-5 block">
              <span className="mb-1 flex justify-between text-sm text-stone-500">
                <span>Drag the input below zero</span>
                <span className="font-mono font-semibold text-stone-700">x = {x.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={-1.5}
                max={1.5}
                step={0.05}
                value={x}
                onChange={(e) => setX(+e.target.value)}
                className="w-full accent-emerald-600"
              />
            </label>
            <p className="mt-3 text-sm text-stone-500">
              {x < 0 ? (
                <>x is negative, so the output snaps to <strong className="text-stone-700">0</strong>.</>
              ) : (
                <>x is positive, so it passes straight through <strong className="text-stone-700">unchanged</strong>.</>
              )}
            </p>
          </div>

          {/* whole layer */}
          <div className="mx-auto mt-5 max-w-md rounded-xl border border-stone-200 bg-white p-4">
            <p className="mb-3 text-sm text-stone-500">
              Now apply that exact rule to <em>every</em> number in the hidden layer
              at once — the red (negative) ones get flattened to zero:
            </p>
            <div className="flex h-16 items-center justify-center gap-[3px]">
              {SIGNED.map((v, j) => {
                const shown = reluAll ? Math.max(0, v) : v;
                return (
                  <motion.span
                    key={j}
                    animate={{ height: `${Math.abs(shown) * 50 + 2}px` }}
                    transition={{ type: "spring", stiffness: 300, damping: 22 }}
                    className={`w-2 rounded-sm ${shown < 0 ? "bg-rose-400" : "bg-emerald-400"}`}
                    style={{ alignSelf: shown < 0 ? "flex-start" : "flex-end" }}
                  />
                );
              })}
            </div>
            <button
              onClick={() => setReluAll((r) => !r)}
              className="mt-3 rounded-full bg-stone-800 px-4 py-1.5 text-sm font-medium text-white hover:bg-stone-700"
            >
              {reluAll ? "reset" : "apply ReLU to all"}
            </button>
          </div>

          <p className="mx-auto mt-5 max-w-md text-[15px] text-stone-500">
            Why bother? Without a bend, stacking two layers is just math that
            collapses into a single straight line — the model could only learn the
            simplest relationships. This little kink is what lets depth actually buy
            it something.
          </p>

          <Continue onClick={onNext} label="What keeps it all stable" delay={0.3} />
        </motion.div>
      )}
    </div>
  );
}
