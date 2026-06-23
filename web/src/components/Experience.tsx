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

// A small element that is shared across every scene. Because it keeps one
// layoutId, motion morphs it from each scene into the next — so no transition is
// ever a plain fade; something always carries over.
const LABELS: Record<string, string> = {
  opening: "“To be, or not to be”",
  "why-dims": "each token → a vector",
  positional: "+ position",
  qkv: "query · key · value",
  attention: "attention",
  mlp: "the MLP",
  "norm-residual": "residual + LayerNorm",
  blocks: "stacked blocks",
  softmax: "→ probabilities",
  training: "training",
  generate: "writing the next character",
};

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

      {/* carried element — morphs from one scene to the next so transitions are
          never a plain fade */}
      <div className="z-10 flex justify-center pb-1">
        <motion.span
          key={id}
          layoutId="carry"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            opacity: { duration: 0.5 },
            layout: { duration: 0.6, ease: [0.22, 1, 0.36, 1] },
          }}
          className="rounded-full border border-stone-200 bg-white px-3 py-1 font-mono text-xs text-stone-500 shadow-sm"
        >
          {LABELS[id] ?? ""}
        </motion.span>
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
