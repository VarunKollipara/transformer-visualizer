"use client";

import { useCallback, useRef, useState, type ComponentType } from "react";
import { AnimatePresence, motion } from "motion/react";
import OpeningScene from "./scenes/OpeningScene";
import PositionQueryScene from "./scenes/PositionQueryScene";
import BlockScene from "./scenes/BlockScene";
import FinaleScene from "./scenes/FinaleScene";
import type { SceneProps } from "./scenes/types";

// The experience is a linear sequence of fullscreen scenes; most scenes contain
// several "slides" (phases) you click through. The Experience owns the current
// slide so the progress bar can address each one and clicking jumps there.
type SceneDef = {
  id: string;
  Component: ComponentType<SceneProps>;
  phases: { id: string; label: string }[];
};

const SCENES: SceneDef[] = [
  {
    id: "opening",
    Component: OpeningScene,
    phases: [
      { id: "invite", label: "The hook" },
      { id: "tokens", label: "Tokenizing" },
      { id: "vectors", label: "Embedding vectors" },
      { id: "dims", label: "Why 128 numbers" },
    ],
  },
  {
    id: "position-query",
    Component: PositionQueryScene,
    phases: [
      { id: "order", label: "Order matters" },
      { id: "qkv", label: "Query / Key / Value" },
      { id: "attention", label: "Attention" },
      { id: "think", label: "Each token thinks" },
      { id: "widen", label: "MLP: widen & shrink" },
      { id: "relu", label: "ReLU" },
    ],
  },
  {
    id: "block",
    Component: BlockScene,
    phases: [
      { id: "residual", label: "Residual connections" },
      { id: "layernorm", label: "LayerNorm" },
      { id: "stack", label: "Stacking blocks" },
      { id: "logits", label: "Logits" },
      { id: "softmax", label: "Softmax" },
    ],
  },
  {
    id: "finale",
    Component: FinaleScene,
    phases: [
      { id: "training", label: "Training" },
      { id: "generate", label: "Generation" },
    ],
  },
];

// Flatten into the linear slide list the progress bar and navigation use.
const SLIDES = SCENES.flatMap((s, sceneIndex) =>
  s.phases.map((p, phaseIndex) => ({
    sceneIndex,
    sceneId: s.id,
    phase: p.id,
    label: p.label,
    firstOfScene: phaseIndex === 0,
  })),
);

// Each SCENE gets its own entrance/exit motion (only plays when crossing a scene
// boundary; slide changes within a scene are handled inside the scene).
// One coherent transition for every scene change: a subtle fade + slight
// rise/recede. Keeping it uniform (instead of a different direction per scene)
// means the exit and the next entrance move along the same axis, so boundaries
// read as one smooth "advance" rather than a disjointed slide-then-jump. The
// within-scene morphs carry the visual interest.
const SCENE_VARIANT = {
  initial: { opacity: 0, scale: 0.98, y: 14 },
  exit: { opacity: 0, scale: 1.02, y: -14 },
};

export default function Experience() {
  const [slide, setSlide] = useState(0);

  // Once a visitor engages (first advance), start downloading + warming the
  // in-browser model in the background so it's ready by the attention slide.
  // Skipped for anyone who bounces on the first screen.
  const warmed = useRef(false);
  const prewarm = useCallback(() => {
    if (warmed.current) return;
    warmed.current = true;
    import("@/lib/onnx").then((m) => m.forward("\n").catch(() => {})).catch(() => {});
  }, []);

  const next = useCallback(() => {
    prewarm();
    setSlide((i) => Math.min(i + 1, SLIDES.length - 1));
  }, [prewarm]);
  const back = useCallback(() => setSlide((i) => Math.max(i - 1, 0)), []);
  const restart = useCallback(() => setSlide(0), []);

  const cur = SLIDES[slide];
  const scene = SCENES[cur.sceneIndex];

  return (
    <div className="relative flex h-[100svh] w-full flex-col overflow-hidden">
      {/* top bar: title + per-slide progress (click any dot to jump) */}
      <header className="z-10 flex items-center justify-center px-6 py-4 sm:justify-between">
        <span className="hidden text-sm font-semibold tracking-tight text-stone-500 sm:block">
          How an AI actually works
        </span>
        <div className="flex flex-wrap items-center justify-center">
          {SLIDES.map((s, i) => (
            <button
              key={i}
              onClick={() => setSlide(i)}
              aria-label={`Slide ${i + 1}: ${s.label}`}
              aria-current={i === slide ? "step" : undefined}
              title={`${i + 1}. ${s.label}`}
              className={`group flex items-center py-2 ${
                s.firstOfScene && i > 0 ? "pl-2.5 pr-1" : "px-1"
              }`}
            >
              <span
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i === slide
                    ? "w-5 bg-indigo-600"
                    : i < slide
                      ? "w-1.5 bg-indigo-300 group-hover:w-3 group-hover:bg-indigo-400"
                      : "w-1.5 bg-stone-200 group-hover:w-3 group-hover:bg-stone-300"
                }`}
              />
            </button>
          ))}
        </div>
      </header>

      {/* scene stage — AnimatePresence keyed by scene, so the entrance/exit only
          fires when crossing a scene boundary; within-scene slide changes are
          animated by the scene itself (morphs etc.) */}
      <main className="relative flex-1">
        <AnimatePresence mode="wait">
          <motion.div
            key={scene.id}
            initial={SCENE_VARIANT.initial}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={SCENE_VARIANT.exit}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="absolute inset-0 flex items-center justify-center overflow-y-auto px-6 py-8"
          >
            <div className="w-full max-w-3xl">
              <scene.Component phase={cur.phase} onNext={next} onBack={back} restart={restart} />
            </div>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* back affordance */}
      {slide > 0 && (
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
