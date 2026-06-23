"use client";

import { useState } from "react";

// Progressive disclosure: optional depth for readers who already get the basics.
// Collapsed by default so beginners aren't overwhelmed.
export default function GoDeeper({
  title = "Go deeper",
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-4 overflow-hidden rounded-xl border border-stone-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left text-sm font-semibold text-stone-700 hover:bg-stone-50"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          <span className="text-stone-400">{open ? "−" : "+"}</span>
          {title}
        </span>
        <span className="text-xs font-normal text-stone-400">
          {open ? "hide" : "for the curious"}
        </span>
      </button>
      {open && (
        <div className="border-t border-stone-200 px-4 py-4 text-[15px] leading-relaxed text-stone-600">
          {children}
        </div>
      )}
    </div>
  );
}
