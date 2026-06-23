"use client";

import { motion } from "motion/react";
import type { SceneProps } from "./types";

// Temporary end card while the remaining scenes (embeddings, attention,
// softmax, generation) are being built out.
export default function OutroScene({ onBack }: SceneProps) {
  return (
    <div className="text-center">
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-2xl font-semibold text-stone-900 sm:text-3xl"
      >
        More scenes coming next
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mx-auto mt-3 max-w-md text-stone-500"
      >
        From here the journey continues: turning these numbers into meaning
        (embeddings), letting tokens look at each other (attention), and finally
        running the model on your own prompt.
      </motion.p>
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        onClick={onBack}
        className="mt-8 text-sm text-stone-400 hover:text-stone-700"
      >
        ← replay tokenization
      </motion.button>
    </div>
  );
}
