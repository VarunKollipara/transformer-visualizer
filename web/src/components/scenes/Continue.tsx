"use client";

import { motion } from "motion/react";

// Shared "advance to the next scene" button, used at the end of each scene.
export default function Continue({
  onClick,
  label = "Continue",
  delay = 0,
}: {
  onClick: () => void;
  label?: string;
  delay?: number;
}) {
  return (
    <motion.button
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: "easeOut" }}
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className="mt-8 inline-flex items-center gap-2 rounded-full bg-indigo-600 px-6 py-2.5 font-semibold text-white shadow-sm hover:bg-indigo-500"
    >
      {label}
      <span aria-hidden>→</span>
    </motion.button>
  );
}
