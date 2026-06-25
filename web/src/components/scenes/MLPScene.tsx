"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

// Standard enter motion for each phase's body, so phase changes feel deliberate
// rather than a plain crossfade.
const enter = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] as const },
};

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

// A single value drawn as a bar above/below a zero line (up = positive/emerald,
// down = negative/rose). Makes ReLU concrete on ONE number.
function SignedBar({ value, label }: { value: number; label: string }) {
  const half = 34;
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

// Why the bend matters: the same wavy data, fit by a single straight line
// (can't) vs. a bent, piecewise line that ReLU makes possible (can).
const PTS: [number, number][] = [
  [12, 20], [30, 44], [50, 62], [70, 44], [88, 20],
];
function FitPlot({ bent, title }: { bent: boolean; title: string }) {
  const path = bent ? "M12,20 L50,62 L88,20" : "M8,44 L92,44";
  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg viewBox="0 0 100 72" className="w-36">
        <line x1="8" y1="66" x2="92" y2="66" stroke="#e7e5e4" strokeWidth="1" />
        <motion.path
          d={path}
          fill="none"
          stroke={bent ? "#14b8a6" : "#a8a29e"}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.7, ease: "easeOut", delay: 0.2 }}
        />
        {PTS.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="3" fill="#44403c" />
        ))}
      </svg>
      <span className={`text-xs font-medium ${bent ? "text-teal-700" : "text-stone-500"}`}>{title}</span>
    </div>
  );
}

type Phase = "idea" | "shape" | "relu";

export default function MLPScene({ onNext }: SceneProps) {
  const [phase, setPhase] = useState<Phase>("idea");
  const [x, setX] = useState(-0.6);
  const y = Math.max(0, x);

  return (
    <div className="text-center">
      <motion.div key={`t-${phase}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {phase === "idea" && (
          <>
            <SceneTitle>Then each token thinks for itself</SceneTitle>
            <SceneText>
              Attention just let every token <em>gather</em> hints from the words
              around it. But gathering isn&apos;t the same as understanding. So next,
              each token takes everything it collected and <strong className="text-stone-700">
              works on it alone</strong> — this step, no token looks at any other.
            </SceneText>
          </>
        )}
        {phase === "shape" && (
          <>
            <SceneTitle>Inside: widen, then shrink</SceneTitle>
            <SceneText>
              The MLP does something that looks odd at first. It temporarily{" "}
              <strong className="text-stone-700">widens</strong> the token&apos;s 128
              numbers to 512 — four times as many — then{" "}
              <strong className="text-stone-700">shrinks</strong> them back to 128.
            </SceneText>
          </>
        )}
        {phase === "relu" && (
          <>
            <SceneTitle>Why it needs a bend</SceneTitle>
            <SceneText>
              Widening and shrinking are both <strong className="text-stone-700">
              linear</strong> — only multiplying and adding. And a linear step
              followed by another linear step is still just <em>one</em> linear step.
              Stack a hundred and the whole tower collapses into a single straight
              line: it could only ever draw straight relationships.
            </SceneText>
          </>
        )}
      </motion.div>

      {/* ── IDEA phase ── */}
      {phase === "idea" && (
        <motion.div key="idea" {...enter}>
          <div className="mx-auto mt-8 flex max-w-lg items-start justify-center gap-6">
            {["o", "b", "e"].map((ch, k) => (
              <div key={ch} className="flex flex-col items-center gap-1.5">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-sky-200 bg-sky-50 font-mono text-stone-800">
                  {ch}
                </div>
                <span className="text-stone-300">↓</span>
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 + k * 0.12, type: "spring", stiffness: 280, damping: 22 }}
                  className="rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 font-mono text-xs font-semibold text-violet-700"
                >
                  MLP
                </motion.div>
                <span className="text-stone-300">↓</span>
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-200 bg-emerald-50 font-mono text-stone-800">
                  {ch}
                </div>
              </div>
            ))}
          </div>
          <p className="mx-auto mt-5 max-w-lg text-[15px] text-stone-500">
            That little network is the <strong className="text-stone-700">MLP</strong>{" "}
            (<strong className="text-stone-700">multi-layer perceptron</strong>) — the
            plainest kind of neural net, the first one you&apos;d build in any ML
            course. The <em>same</em> MLP runs on every token separately, like handing
            each one its own small calculator to think with.
          </p>
          <Continue onClick={() => setPhase("shape")} label="Look inside the MLP" delay={0.3} />
        </motion.div>
      )}

      {/* ── SHAPE phase ── */}
      {phase === "shape" && (
        <motion.div key="shape" {...enter}>
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
            Why widen? The extra numbers are <strong className="text-stone-700">scratch
            space</strong> — room to look for useful combinations of features (is this
            a vowel <em>and</em> mid-word <em>and</em> just after a space?). The shrink
            then summarises what it found back into the standard 128 so the next block
            fits. But between the two, one crucial thing has to happen.
          </SceneText>
          <Continue onClick={() => setPhase("relu")} label="The crucial bend" delay={0.4} />
        </motion.div>
      )}

      {/* ── RELU phase ── */}
      {phase === "relu" && (
        <motion.div key="relu" {...enter}>
          {/* the why, made visual */}
          <div className="mx-auto mt-6 flex max-w-lg items-center justify-center gap-8 rounded-xl border border-stone-200 bg-white p-5">
            <FitPlot bent={false} title="only straight lines" />
            <span className="text-2xl text-stone-300">vs</span>
            <FitPlot bent title="bends allowed" />
          </div>
          <p className="mx-auto mt-3 max-w-lg text-[15px] text-stone-500">
            <strong className="text-stone-700">ReLU</strong> (
            <strong className="text-stone-700">rectified linear unit</strong>) is the
            fix: a tiny non-linear kink dropped between the two layers. That one kink
            is what lets a deep stack <em>bend</em> — fitting curved, conditional
            patterns instead of only straight ones.
          </p>

          {/* the rule, on one number */}
          <div className="mx-auto mt-6 max-w-md rounded-xl border border-stone-200 bg-white p-5">
            <p className="mb-4 text-sm text-stone-500">
              The rule itself, on a single number, couldn&apos;t be simpler:{" "}
              <span className="font-mono">max(0, x)</span> — keep positives, zero out
              negatives. Drag it:
            </p>
            <div className="flex items-end justify-center gap-8">
              <SignedBar value={x} label="input x" />
              <span className="mb-7 text-2xl text-stone-300">→</span>
              <SignedBar value={y} label="output max(0, x)" />
            </div>
            <label className="mt-5 block">
              <span className="mb-1 flex justify-between text-sm text-stone-500">
                <span>{x < 0 ? "negative → snaps to 0" : "positive → passes through"}</span>
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
            <p className="mt-4 text-xs text-stone-400">
              Every one of the 512 numbers in the wide layer gets this same treatment.
            </p>
          </div>

          <Continue onClick={onNext} label="What keeps it all stable" delay={0.3} />
        </motion.div>
      )}
    </div>
  );
}
