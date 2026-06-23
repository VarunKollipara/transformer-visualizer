"use client";

import { useCallback, useState, type ComponentType } from "react";
import { AnimatePresence, motion } from "motion/react";
import OpeningScene from "./scenes/OpeningScene";
import WhyDimensionsScene from "./scenes/WhyDimensionsScene";
import PositionalScene from "./scenes/PositionalScene";
import QKVScene from "./scenes/QKVScene";
import AttentionScene from "./scenes/AttentionScene";
import MLPScene from "./scenes/MLPScene";
import NormResidualScene from "./scenes/NormResidualScene";
import BlocksScene from "./scenes/BlocksScene";
import SoftmaxScene from "./scenes/SoftmaxScene";
import TrainingScene from "./scenes/TrainingScene";
import GenerateScene from "./scenes/GenerateScene";
import type { SceneProps } from "./scenes/types";

// The whole experience is a linear sequence of fullscreen scenes. You advance by
// interacting; we never scroll the page — like an interactive movie.
const SCENES: { id: string; Component: ComponentType<SceneProps> }[] = [
  { id: "opening", Component: OpeningScene },
  { id: "why-dims", Component: WhyDimensionsScene },
  { id: "positional", Component: PositionalScene },
  { id: "qkv", Component: QKVScene },
  { id: "attention", Component: AttentionScene },
  { id: "mlp", Component: MLPScene },
  { id: "norm-residual", Component: NormResidualScene },
  { id: "blocks", Component: BlocksScene },
  { id: "softmax", Component: SoftmaxScene },
  { id: "training", Component: TrainingScene },
  { id: "generate", Component: GenerateScene },
];

// Each scene's "signature object" — a tiny bespoke visual of what that scene is
// about. They all share one layoutId, so motion MORPHS one scene's object into
// the next as you advance: letters → a vector → query/key/value → an attention
// grid → bars → a loss curve → a character. No transition is ever a plain fade.
function MiniBars({ vals, color }: { vals: number[]; color: string }) {
  return (
    <div className="flex h-4 items-end gap-[2px]">
      {vals.map((h, i) => (
        <span key={i} style={{ height: `${h * 100}%` }} className={`w-[3px] rounded-sm ${color}`} />
      ))}
    </div>
  );
}

function Signature({ id }: { id: string }) {
  switch (id) {
    case "opening":
      return (
        <div className="flex gap-0.5 font-mono text-[11px] text-stone-700">
          {["T", "o"].map((c) => (
            <span key={c} className="rounded border border-sky-200 bg-sky-50 px-1">{c}</span>
          ))}
        </div>
      );
    case "why-dims":
      return <MiniBars vals={[0.4, 0.8, 0.5, 0.9, 0.6, 0.7]} color="bg-violet-400" />;
    case "positional":
      return (
        <div className="flex items-center gap-1 font-mono text-[11px]">
          <span className="rounded bg-sky-100 px-1 text-sky-700">t</span>
          <span className="text-stone-300">+</span>
          <span className="rounded bg-amber-100 px-1 text-amber-700">pos</span>
        </div>
      );
    case "qkv":
      return (
        <div className="flex gap-1">
          <span className="h-2.5 w-2.5 rounded-full bg-teal-400" />
          <span className="h-2.5 w-2.5 rounded-full bg-sky-400" />
          <span className="h-2.5 w-2.5 rounded-full bg-violet-400" />
        </div>
      );
    case "attention":
      return (
        <div className="grid grid-cols-3 gap-[2px]">
          {[1, 0, 0, 1, 1, 0, 1, 1, 1].map((v, i) => (
            <span key={i} className={`h-2 w-2 rounded-[1px] ${v ? "bg-teal-400" : "bg-stone-100"}`} />
          ))}
        </div>
      );
    case "mlp":
      return <MiniBars vals={[0.5, 0.9, 0.4, 1, 0.6, 0.8]} color="bg-indigo-400" />;
    case "norm-residual":
      return <span className="font-mono text-xs text-emerald-600">x + ƒ(x)</span>;
    case "blocks":
      return (
        <div className="flex flex-col gap-[2px]">
          {[0, 1, 2].map((i) => (
            <span key={i} className="h-1.5 w-6 rounded-[1px] bg-teal-300" />
          ))}
        </div>
      );
    case "softmax":
      return (
        <div className="flex flex-col items-start gap-[2px]">
          {[1, 0.5, 0.3].map((w, i) => (
            <span key={i} style={{ width: `${w * 24}px` }} className="h-1.5 rounded-[1px] bg-indigo-400" />
          ))}
        </div>
      );
    case "training":
      return (
        <svg width="28" height="16" viewBox="0 0 28 16">
          <path d="M2 3 C10 3 12 12 26 13" fill="none" stroke="#f43f5e" strokeWidth="2" />
        </svg>
      );
    case "generate":
      return <span className="font-mono text-xs text-indigo-600">A▋</span>;
    default:
      return null;
  }
}

export default function Experience() {
  const [index, setIndex] = useState(0);
  const [dir, setDir] = useState(1);

  const next = useCallback(() => {
    setDir(1);
    setIndex((i) => Math.min(i + 1, SCENES.length - 1));
  }, []);
  const back = useCallback(() => {
    setDir(-1);
    setIndex((i) => Math.max(i - 1, 0));
  }, []);
  const restart = useCallback(() => {
    setDir(-1);
    setIndex(0);
  }, []);

  const { id, Component } = SCENES[index];

  return (
    <div className="relative flex h-[100svh] w-full flex-col overflow-hidden">
      {/* top bar: title + progress */}
      <header className="z-10 flex items-center justify-between px-6 py-4">
        <span className="text-sm font-semibold tracking-tight text-stone-500">
          How an AI actually works
        </span>
        <div className="flex items-center gap-1.5">
          {SCENES.map((s, i) => (
            <span
              key={s.id}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === index
                  ? "w-6 bg-indigo-600"
                  : i < index
                    ? "w-1.5 bg-indigo-300"
                    : "w-1.5 bg-stone-200"
              }`}
            />
          ))}
        </div>
      </header>

      {/* the scene's signature object — one shared layoutId, so it morphs from
          each scene's object into the next (never a plain fade) */}
      <div className="z-10 flex justify-center pb-1">
        <motion.div
          key={id}
          layoutId="signature"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            opacity: { duration: 0.45 },
            layout: { duration: 0.7, ease: [0.22, 1, 0.36, 1] },
          }}
          className="flex h-9 items-center justify-center rounded-xl border border-stone-200 bg-white px-3 shadow-sm"
        >
          <Signature id={id} />
        </motion.div>
      </div>

      {/* scene stage */}
      <main className="relative flex-1">
        <AnimatePresence mode="wait" custom={dir}>
          <motion.div
            key={id}
            custom={dir}
            initial={{ opacity: 0, y: dir * 40, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: dir * -40, scale: 0.98 }}
            transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
            className="absolute inset-0 flex items-center justify-center overflow-y-auto px-6 py-8"
          >
            <div className="w-full max-w-3xl">
              <Component onNext={next} onBack={back} restart={restart} />
            </div>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* back affordance */}
      {index > 0 && (
        <button
          onClick={back}
          className="absolute bottom-5 left-6 z-10 text-sm text-stone-400 transition hover:text-stone-700"
        >
          ← back
        </button>
      )}
    </div>
  );
}
