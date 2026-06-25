"use client";

import { motion } from "motion/react";

export function SceneTitle({ children }: { children: React.ReactNode }) {
  return (
    <motion.h2
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: "easeOut" }}
      className="font-display text-2xl font-semibold text-stone-900 sm:text-3xl"
    >
      {children}
    </motion.h2>
  );
}

export function SceneText({
  children,
  delay = 0.2,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.p
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className={`mx-auto max-w-xl leading-relaxed text-stone-500 ${className}`}
    >
      {children}
    </motion.p>
  );
}
