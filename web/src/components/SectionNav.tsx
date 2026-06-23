"use client";

import { useEffect, useState } from "react";

const STEPS = [
  { id: "generation", n: 1, label: "It only does one thing" },
  { id: "corpus", n: 2, label: "What it learned from" },
  { id: "tokens", n: 3, label: "Text into numbers" },
  { id: "embeddings", n: 4, label: "Giving tokens meaning" },
  { id: "attention", n: 5, label: "Tokens look at each other" },
  { id: "training", n: 6, label: "Watch the loss fall" },
  { id: "choice", n: 7, label: "From scores to a choice" },
  { id: "pipeline", n: 8, label: "Putting it together" },
];

// Fixed scroll-spy rail (desktop only) so the reader always knows where they are
// in the journey and can jump around.
export default function SectionNav() {
  const [active, setActive] = useState("generation");

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id);
        });
      },
      { rootMargin: "-45% 0px -50% 0px" },
    );
    STEPS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  return (
    <nav className="fixed left-5 top-1/2 z-20 hidden -translate-y-1/2 lg:block">
      <ul className="space-y-1">
        {STEPS.map((s) => {
          const on = active === s.id;
          return (
            <li key={s.id}>
              <a href={`#${s.id}`} className="group flex items-center gap-2.5">
                <span
                  className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold transition ${
                    on
                      ? "bg-indigo-600 text-white"
                      : "bg-stone-200 text-stone-500 group-hover:bg-stone-300"
                  }`}
                >
                  {s.n}
                </span>
                <span
                  className={`max-w-0 overflow-hidden whitespace-nowrap text-xs transition-all duration-200 group-hover:max-w-[14rem] ${
                    on ? "max-w-[14rem] font-medium text-stone-700" : "text-stone-400"
                  }`}
                >
                  {s.label}
                </span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
