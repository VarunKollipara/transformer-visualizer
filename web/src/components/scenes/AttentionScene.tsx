"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { forward, type ForwardResponse } from "@/lib/api";
import { EXAMPLE_LINE } from "@/lib/example";
import Continue from "./Continue";
import type { SceneProps } from "./types";

const show = (ch: string) => (ch === " " ? "␣" : ch);

export default function AttentionScene({ onNext }: SceneProps) {
  const [data, setData] = useState<ForwardResponse | null>(null);
  const [head, setHead] = useState(0);
  const [query, setQuery] = useState<number | null>(null);

  useEffect(() => {
    forward(EXAMPLE_LINE, 5)
      .then((d) => {
        setData(d);
        setQuery(d.tokens.length - 1);
      })
      .catch(() => {});
  }, []);

  const row = useMemo(() => {
    if (!data || query === null) return null;
    return data.attention[0][head][query];
  }, [data, query, head]);

  const rowMax = useMemo(
    () => (row ? Math.max(...row.filter((_, j) => query !== null && j <= query)) : 1),
    [row, query],
  );

  return (
    <div className="text-center">
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="font-display text-2xl font-semibold text-stone-900 sm:text-3xl"
      >
        Now the tokens look at each other
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mx-auto mt-2 max-w-xl text-stone-500"
      >
        To predict what comes next, each character looks back at the earlier ones
        and pulls in what matters. <strong className="text-stone-700">Click any
        character</strong> to see where it looks. Brighter = more attention.
      </motion.p>

      {data && row && query !== null && (
        <>
          <p className="mt-6 text-sm text-stone-400">
            what{" "}
            <span className="rounded bg-teal-100 px-1.5 py-0.5 font-mono text-teal-800">
              {show(data.tokens[query])}
            </span>{" "}
            pays attention to:
          </p>

          <div className="mx-auto mt-3 flex max-w-2xl flex-wrap justify-center gap-1">
            {data.tokens.map((ch, j) => {
              const w = j <= query ? row[j] / (rowMax || 1) : 0;
              const isQuery = j === query;
              return (
                <button
                  key={j}
                  onClick={() => setQuery(j)}
                  className={`relative flex h-9 w-7 items-center justify-center overflow-hidden rounded-md border font-mono text-sm transition ${
                    isQuery
                      ? "border-teal-500 ring-2 ring-teal-300"
                      : "border-stone-200"
                  }`}
                  title={j <= query ? `${(row[j] * 100).toFixed(0)}%` : "in the future"}
                >
                  <motion.span
                    className="absolute inset-0 bg-teal-400"
                    initial={false}
                    animate={{ opacity: w }}
                    transition={{ duration: 0.35, ease: "easeOut" }}
                  />
                  <span className="relative text-stone-800">{show(ch)}</span>
                </button>
              );
            })}
          </div>

          {/* heads */}
          <div className="mt-5 flex items-center justify-center gap-2 text-sm text-stone-500">
            <span>head:</span>
            {Array.from({ length: data.num_heads }, (_, h) => (
              <button
                key={h}
                onClick={() => setHead(h)}
                className={`h-7 w-7 rounded-md border text-xs font-semibold transition ${
                  h === head
                    ? "border-teal-500 bg-teal-50 text-teal-700"
                    : "border-stone-200 text-stone-500 hover:bg-stone-50"
                }`}
              >
                {h}
              </button>
            ))}
            <span className="ml-1 text-xs text-stone-400">
              each head learns a different pattern
            </span>
          </div>

          <p className="mx-auto mt-5 max-w-lg text-sm text-stone-500">
            Notice everything to the right of the chosen character stays dark — a
            token can never look at the future, only the past.
          </p>
        </>
      )}

      <Continue onClick={onNext} label="Then each token thinks for itself" delay={0.4} />
    </div>
  );
}
