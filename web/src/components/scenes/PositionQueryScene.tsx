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

// Standard enter motion for a phase's body.
const enter = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] as const },
};

// ── MLP helper visuals ──
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

function SignedBar({ value, label }: { value: number; label: string }) {
  const half = 28;
  const h = Math.min(Math.abs(value) / 1.5, 1) * half;
  const neg = value < 0;
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative h-[60px] w-10">
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

const PTS: [number, number][] = [[12, 20], [30, 44], [50, 62], [70, 44], [88, 20]];
function FitPlot({ bent, title }: { bent: boolean; title: string }) {
  const path = bent ? "M12,20 L50,62 L88,20" : "M8,44 L92,44";
  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg viewBox="0 0 100 72" className="w-28">
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

// One scene covering BOTH halves of a transformer block: attention (tokens
// share context) and the MLP (each token then thinks alone). Phases:
//   order → qkv → attention → think → widen → relu
// The cross-slide morph: the attention-lit LINE of tiles in `attention` stays
// on stage into `think`, then recedes as context while each token drops into
// its own MLP — what used to be a hard cut between two slides.
type Phase = "order" | "qkv" | "attention" | "think" | "widen" | "relu";

export default function PositionQueryScene({ phase: phaseProp, onNext }: SceneProps) {
  const phase = phaseProp as Phase;
  const [swapped, setSwapped] = useState(false);
  const [data, setData] = useState<ForwardResponse | null>(null);
  // preset so the attention phase works even when jumped to directly
  const [query, setQuery] = useState<number | null>(LINE.length - 1);
  const [x, setX] = useState(-0.6);
  const reluY = Math.max(0, x);

  useEffect(() => {
    forward(EXAMPLE_LINE, 5)
      .then((d) => setData(d))
      .catch(() => {});
  }, []);

  const words = swapped ? ORDER_B : ORDER_A;
  // Tiles, keyed so the SAME tile persists across phases and layout-morphs.
  // In `attention` the whole line is keyed by its line index; in `think` we keep
  // only T/o/b (their original line indices 0/1/3) so those exact tiles fly out
  // of the line into the MLP demo; widen/relu keep just T (index 0) as the anchor
  // we follow into the deep-dive.
  const THINK_IDX = [0, 1, 3]; // T, o, b in "To be"
  const tileList: { ch: string; key: number }[] =
    phase === "order"
      ? ORDER_TILES.map((ch, i) => ({ ch, key: i }))
      : phase === "qkv"
        ? [{ ch: "T", key: 0 }]
        : phase === "attention"
          ? LINE.map((ch, i) => ({ ch, key: i }))
          : phase === "think"
            ? THINK_IDX.map((i) => ({ ch: LINE[i], key: i }))
            : [{ ch: "T", key: 0 }]; // widen, relu — the subject token, carried
  const lineMode = phase !== "order" && phase !== "qkv";
  const tilesGap = phase === "attention" ? "gap-1" : phase === "think" ? "gap-7" : "gap-2";

  const row = useMemo(() => {
    if (!data || query === null) return null;
    return data.attention[0][0][query];
  }, [data, query]);
  const rowMax = useMemo(
    () => (row && query !== null ? Math.max(...row.filter((_, j) => j <= query)) : 1),
    [row, query],
  );

  return (
    <div className="text-center">
      {/* title + intro, keyed by phase */}
      <motion.div key={`t-${phase}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
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
        {phase === "think" && (
          <>
            <SceneTitle>Then each token thinks for itself</SceneTitle>
            <SceneText>
              Attention let every token <em>gather</em> hints from the line above.
              But gathering isn&apos;t understanding. So next, each of those same
              tokens takes what it collected and <strong className="text-stone-700">
              works on it alone</strong> — this step, no token looks at any other.
            </SceneText>
          </>
        )}
        {phase === "widen" && (
          <>
            <SceneTitle>Inside the MLP: widen, then shrink</SceneTitle>
            <SceneText>
              That little per-token network is the <strong className="text-stone-700">
              MLP</strong> (<strong className="text-stone-700">multi-layer
              perceptron</strong>). Inside, it temporarily{" "}
              <strong className="text-stone-700">widens</strong> the token&apos;s 128
              numbers to 512, then <strong className="text-stone-700">shrinks</strong>{" "}
              them back to 128.
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

      {/* the shared tiles — the "T" tile morphs through order→qkv→attention; the
          line then condenses so T/o/b fly into the MLP demo, and T alone carries
          into the widen/relu deep-dive. Each tile is a keyed column so it
          layout-morphs to its new home between phases. */}
      {/* mode="popLayout" pulls exiting tiles out of the layout flow at once, so
          the surviving tiles glide straight to their final centered spot in one
          smooth move instead of drifting then snapping when the others clear. */}
      <div className={`relative mx-auto mt-6 flex max-w-2xl flex-wrap items-start justify-center ${tilesGap}`}>
        <AnimatePresence mode="popLayout">
          {tileList.map(({ ch, key }) => {
            const isFirst = key === 0;
            const big = phase === "qkv" && isFirst;
            const w = phase === "attention" && query !== null && key <= query ? (row ? row[key] / (rowMax || 1) : 0) : 0;
            const isQuery = phase === "attention" && key === query;
            return (
              <motion.div
                key={key}
                layout
                initial={isFirst ? false : { opacity: 0, scale: 0.6 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                transition={{ layout: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }, opacity: { duration: 0.4 } }}
                className="flex flex-col items-center"
              >
                {/* the character tile box — the part that travels through the line */}
                <div
                  onClick={() => phase === "attention" && setQuery(key)}
                  className={`relative flex items-center justify-center overflow-hidden rounded-lg border font-mono ${
                    big ? "border-teal-300 bg-teal-50 px-5 py-3" : "border-sky-200 bg-sky-50"
                  } ${lineMode ? `h-9 w-7 ${phase === "attention" ? "cursor-pointer" : ""} ${isQuery ? "ring-2 ring-teal-400" : ""}` : "px-2.5 py-1.5"}`}
                >
                  {phase === "attention" && (
                    <motion.span className="absolute inset-0 bg-teal-400" initial={false} animate={{ opacity: w }} transition={{ duration: 0.3 }} />
                  )}
                  <motion.span layout="position" className={`relative text-stone-800 ${big ? "text-2xl" : lineMode ? "text-sm" : "text-base"}`}>
                    {lineMode ? show(ch) : ch}
                  </motion.span>
                </div>

                {/* order: position label */}
                {phase === "order" && (
                  <span className="mt-1 rounded bg-amber-100 px-1 text-[10px] text-amber-700">pos {key}</span>
                )}

                {/* think: this token drops into its own MLP */}
                {phase === "think" && (
                  <motion.div
                    className="flex flex-col items-center gap-1.5"
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.35, duration: 0.4 }}
                  >
                    <span className="mt-1 text-stone-300">↓</span>
                    <div className="rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 font-mono text-xs font-semibold text-violet-700">MLP</div>
                    <span className="text-stone-300">↓</span>
                    <div className="flex h-9 w-7 items-center justify-center rounded-lg border border-emerald-200 bg-emerald-50 font-mono text-sm text-stone-800">
                      {show(ch)}
                    </div>
                  </motion.div>
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

      {/* think: the carried T/o/b tiles (above) have each dropped into an MLP */}
      {phase === "think" && (
        <motion.div key="think" {...enter}>
          <p className="mx-auto mt-6 max-w-lg text-[15px] text-stone-500">
            The <em>same</em> little network runs on every token separately, like
            handing each one its own small calculator. Let&apos;s open one up.
          </p>
          <Continue onClick={onNext} label="Look inside the MLP" delay={0.3} />
        </motion.div>
      )}

      {/* widen: MLP shape (the carried T tile sits above this) */}
      {phase === "widen" && (
        <motion.div key="widen" {...enter}>
          <p className="mt-3 text-[13px] text-stone-400">↓ inside its MLP</p>
          <div className="mt-4 flex items-end justify-center gap-3">
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
            then summarises what it found back into the standard 128. But between the
            two, one crucial thing has to happen.
          </SceneText>
          <Continue onClick={onNext} label="The crucial bend" delay={0.4} />
        </motion.div>
      )}

      {/* relu (the carried T tile sits above this) */}
      {phase === "relu" && (
        <motion.div key="relu" {...enter}>
          <p className="mt-2 text-[13px] text-stone-400">↓ between the widen and the shrink</p>
          <div className="mx-auto mt-2 flex max-w-lg items-center justify-center gap-6 rounded-xl border border-stone-200 bg-white p-4">
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

          <div className="mx-auto mt-4 max-w-md rounded-xl border border-stone-200 bg-white p-4">
            <p className="mb-3 text-sm text-stone-500">
              The rule itself, on a single number, couldn&apos;t be simpler:{" "}
              <span className="font-mono">max(0, x)</span> — keep positives, zero out
              negatives. Drag it:
            </p>
            <div className="flex items-end justify-center gap-8">
              <SignedBar value={x} label="input x" />
              <span className="mb-7 text-2xl text-stone-300">→</span>
              <SignedBar value={reluY} label="output max(0, x)" />
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
            <p className="mt-3 text-xs text-stone-400">
              Every one of the 512 wide-layer numbers gets this same treatment.
            </p>
          </div>

          <Continue onClick={onNext} label="What keeps it all stable" delay={0.2} />
        </motion.div>
      )}

      {/* per-phase advance buttons for the attention half */}
      {phase === "order" && <Continue onClick={onNext} label="Now: how it looks around" delay={0.2} />}
      {phase === "qkv" && <Continue onClick={onNext} label="See it on the whole line" delay={0.9} />}
      {phase === "attention" && <Continue onClick={onNext} label="Then each token thinks for itself" delay={0.4} />}
    </div>
  );
}
