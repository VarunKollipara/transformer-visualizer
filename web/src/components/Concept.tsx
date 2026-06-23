"use client";

import { useEffect, useRef, useState } from "react";
import { glossary } from "@/lib/glossary";

// Inline, clickable term. Click to reveal a plain-language definition popover;
// click outside or press Escape to dismiss.
export default function Concept({
  id,
  children,
}: {
  id: keyof typeof glossary | string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const entry = glossary[id];

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span className="relative inline-block" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`cursor-pointer rounded font-medium text-indigo-700 underline decoration-indigo-300 decoration-2 underline-offset-2 hover:bg-indigo-50 hover:text-indigo-800 ${
          open ? "bg-indigo-50" : ""
        }`}
      >
        {children}
      </button>
      {open && entry && (
        <span
          className="absolute left-1/2 top-full z-30 mt-2 block w-72 -translate-x-1/2 rounded-xl border border-stone-200 bg-white p-3.5 text-left text-sm font-normal leading-relaxed text-stone-600 shadow-lg shadow-stone-300/40"
          role="tooltip"
        >
          <span className="mb-1 block font-semibold text-indigo-700">
            {entry.title}
          </span>
          {entry.body}
        </span>
      )}
    </span>
  );
}
