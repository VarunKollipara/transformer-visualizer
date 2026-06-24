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
