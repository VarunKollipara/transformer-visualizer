"use client";

import { useCallback, useState, type ComponentType } from "react";
import { AnimatePresence, motion } from "motion/react";
import OpeningScene from "./scenes/OpeningScene";
import WhyDimensionsScene from "./scenes/WhyDimensionsScene";
import PositionQueryScene from "./scenes/PositionQueryScene";
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
  { id: "position-query", Component: PositionQueryScene },
  { id: "attention", Component: AttentionScene },
  { id: "mlp", Component: MLPScene },
  { id: "norm-residual", Component: NormResidualScene },
  { id: "blocks", Component: BlocksScene },
  { id: "softmax", Component: SoftmaxScene },
  { id: "training", Component: TrainingScene },
  { id: "generate", Component: GenerateScene },
];

// Each screen gets its OWN entrance/exit motion, so no two transitions feel the
// same. (Boundaries that share an object also morph it centrally, inside the
// scene components.)
type V = Record<string, number>;
const VARIANTS: Record<string, { initial: V; exit: V }> = {
  opening: { initial: { opacity: 0, scale: 0.94 }, exit: { opacity: 0, scale: 1.05 } },
  "why-dims": { initial: { opacity: 0, y: 48 }, exit: { opacity: 0, y: -48 } },
  "position-query": { initial: { opacity: 0, x: 72 }, exit: { opacity: 0, x: -72 } },
  attention: { initial: { opacity: 0, scale: 1.08 }, exit: { opacity: 0, scale: 0.94 } },
  mlp: { initial: { opacity: 0, x: -72 }, exit: { opacity: 0, x: 72 } },
  "norm-residual": { initial: { opacity: 0, y: 48 }, exit: { opacity: 0, y: -48 } },
  blocks: { initial: { opacity: 0, y: 64 }, exit: { opacity: 0, y: -32 } },
  softmax: { initial: { opacity: 0, x: 72 }, exit: { opacity: 0, x: -72 } },
  training: { initial: { opacity: 0, scale: 0.9 }, exit: { opacity: 0, scale: 1.05 } },
  generate: { initial: { opacity: 0, y: 52 }, exit: { opacity: 0, y: -52 } },
};
const DEFAULT_VARIANT = { initial: { opacity: 0, y: 40 }, exit: { opacity: 0, y: -40 } };

export default function Experience() {
  const [index, setIndex] = useState(0);

  const next = useCallback(() => {
    setIndex((i) => Math.min(i + 1, SCENES.length - 1));
  }, []);
  const back = useCallback(() => {
    setIndex((i) => Math.max(i - 1, 0));
  }, []);
  const restart = useCallback(() => {
    setIndex(0);
  }, []);

  const { id, Component } = SCENES[index];
  const variant = VARIANTS[id] ?? DEFAULT_VARIANT;

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

      {/* scene stage */}
      <main className="relative flex-1">
        <AnimatePresence mode="wait">
          <motion.div
            key={id}
            initial={variant.initial}
            animate={{ opacity: 1, x: 0, y: 0, scale: 1 }}
            exit={variant.exit}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
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
