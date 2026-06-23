"use client";

import { useEffect, useState } from "react";
import { getEmbeddings, type EmbeddingPoint } from "@/lib/api";

const GROUP: Record<EmbeddingPoint["group"], { color: string; label: string }> = {
  upper: { color: "#4f46e5", label: "UPPER" },
  lower: { color: "#0ea5e9", label: "lower" },
  digit: { color: "#d97706", label: "digit" },
  punct: { color: "#e11d48", label: "punct" },
  space: { color: "#6b7280", label: "space" },
  newline: { color: "#9ca3af", label: "newline" },
};

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

const W = 600;
const H = 460;
const PAD = 30;
const sx = (x: number) => PAD + ((x + 1) / 2) * (W - 2 * PAD);
const sy = (y: number) => PAD + ((1 - (y + 1) / 2)) * (H - 2 * PAD);

export default function EmbeddingMap() {
  const [points, setPoints] = useState<EmbeddingPoint[] | null>(null);
  const [error, setError] = useState(false);
  const [hover, setHover] = useState<EmbeddingPoint | null>(null);

  useEffect(() => {
    getEmbeddings()
      .then((d) => setPoints(d.points))
      .catch(() => setError(true));
  }, []);

  if (error)
    return (
      <p className="text-sm text-amber-700">
        Backend not reachable — start the API server (see banner at top).
      </p>
    );

  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-stone-500">
        {Object.values(GROUP).map((g) => (
          <span key={g.label} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: g.color }}
            />
            {g.label}
          </span>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded-xl bg-stone-50"
        onMouseLeave={() => setHover(null)}
      >
        {/* faint axes through the origin */}
        <line x1={sx(-1)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#e7e5e4" />
        <line x1={sx(0)} y1={sy(-1)} x2={sx(0)} y2={sy(1)} stroke="#e7e5e4" />
        {points?.map((p) => {
          const isHover = hover?.id === p.id;
          return (
            <text
              key={p.id}
              x={sx(p.x)}
              y={sy(p.y)}
              fontSize={isHover ? 22 : 13}
              fontWeight={isHover ? 700 : 500}
              fill={GROUP[p.group].color}
              stroke="#faf8f5"
              strokeWidth={isHover ? 5 : 3}
              paintOrder="stroke"
              textAnchor="middle"
              dominantBaseline="central"
              className="cursor-pointer font-mono"
              onMouseEnter={() => setHover(p)}
            >
              {show(p.char)}
            </text>
          );
        })}
      </svg>

      <p className="mt-3 text-sm text-stone-500">
        {hover ? (
          <>
            Hovering{" "}
            <span className="font-mono font-semibold text-violet-700">
              “{show(hover.char)}”
            </span>{" "}
            — token ID {hover.id}. Each point is one character&apos;s learned
            vector, squashed from {/* n_embd */}128 dimensions down to 2.
          </>
        ) : (
          <>
            Each point is one character, placed by its learned embedding vector
            (128 numbers) projected to 2D. Characters the model uses similarly
            drift together — hover to explore.
          </>
        )}
      </p>
    </div>
  );
}
