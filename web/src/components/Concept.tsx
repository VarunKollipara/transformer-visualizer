"use client";

import { useState } from "react";
import { glossary } from "@/lib/glossary";

// Inline, clickable term. Click to reveal a plain-language definition popover.
// This is the "click into anything you don't understand" mechanism.
export default function Concept({
  id,
  children,
}: {
  id: keyof typeof glossary | string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const entry = glossary[id];

  return (
    <span className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="cursor-pointer font-medium text-emerald-400 underline decoration-dotted underline-offset-2 hover:text-emerald-300"
      >
        {children}
      </button>
      {open && entry && (
        <span
          className="absolute left-1/2 top-full z-20 mt-2 block w-72 -translate-x-1/2 rounded-lg border border-zinc-700 bg-zinc-900 p-3 text-left text-sm font-normal leading-relaxed text-zinc-200 shadow-xl"
          role="tooltip"
        >
          <span className="mb-1 block font-semibold text-emerald-400">
            {entry.title}
          </span>
          {entry.body}
        </span>
      )}
    </span>
  );
}
