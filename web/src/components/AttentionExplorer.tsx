"use client";

import { useEffect, useState } from "react";
import { forward, type ForwardResponse } from "@/lib/api";

const show = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

export default function AttentionExplorer() {
  const [text, setText] = useState("To be or not to be");
  const [data, setData] = useState<ForwardResponse | null>(null);
  const [error, setError] = useState(false);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      const clipped = text.slice(-64);
      if (!clipped) return;
      forward(clipped, 8)
        .then((d) => {
          setData(d);
          setError(false);
          setSelected(d.tokens.length - 1);
        })
        .catch(() => setError(true));
    }, 250);
    return () => clearTimeout(t);
  }, [text]);

  return (
    <div>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        maxLength={64}
        className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 font-mono text-zinc-100 outline-none focus:border-emerald-500"
        placeholder="Type text to see where each character looks…"
      />

      {error && (
        <p className="mt-3 text-sm text-amber-300">
          Backend not reachable — start the API server (see banner at top).
        </p>
      )}

      {data && (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-2 text-zinc-400">
              Layer
              <select
                value={layer}
                onChange={(e) => setLayer(+e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
              >
                {Array.from({ length: data.num_layers }, (_, i) => (
                  <option key={i} value={i}>
                    {i}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2 text-zinc-400">
              Head
              <select
                value={head}
                onChange={(e) => setHead(+e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
              >
                {Array.from({ length: data.num_heads }, (_, i) => (
                  <option key={i} value={i}>
                    {i}
                  </option>
                ))}
              </select>
            </label>
            <span className="text-zinc-500">
              rows = the querying token · columns = what it looks at
            </span>
          </div>

          <Heatmap
            tokens={data.tokens}
            matrix={data.attention[layer][head]}
            selected={selected}
            onSelect={setSelected}
          />

          {selected !== null && data.positions[selected] && (
            <Predictions
              token={data.tokens[selected]}
              topk={data.positions[selected].topk}
            />
          )}
        </>
      )}
    </div>
  );
}

function Heatmap({
  tokens,
  matrix,
  selected,
  onSelect,
}: {
  tokens: string[];
  matrix: number[][];
  selected: number | null;
  onSelect: (i: number) => void;
}) {
  const T = tokens.length;
  const cell = 26;
  return (
    <div className="mt-5 overflow-x-auto">
      <div
        className="grid w-max gap-px"
        style={{ gridTemplateColumns: `${cell}px repeat(${T}, ${cell}px)` }}
      >
        {/* corner */}
        <div style={{ width: cell, height: cell }} />
        {/* top labels (keys) */}
        {tokens.map((ch, j) => (
          <div
            key={`c${j}`}
            className="flex items-center justify-center font-mono text-xs text-zinc-500"
            style={{ width: cell, height: cell }}
          >
            {show(ch)}
          </div>
        ))}
        {/* rows */}
        {matrix.map((row, i) => (
          <Row
            key={`r${i}`}
            i={i}
            label={tokens[i]}
            row={row}
            cell={cell}
            selected={selected === i}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}

function Row({
  i,
  label,
  row,
  cell,
  selected,
  onSelect,
}: {
  i: number;
  label: string;
  row: number[];
  cell: number;
  selected: boolean;
  onSelect: (i: number) => void;
}) {
  return (
    <>
      <div
        className={`flex cursor-pointer items-center justify-center font-mono text-xs ${
          selected ? "text-emerald-400" : "text-zinc-500"
        }`}
        style={{ width: cell, height: cell }}
        onClick={() => onSelect(i)}
      >
        {show(label)}
      </div>
      {row.map((v, j) => (
        <div
          key={j}
          onClick={() => onSelect(i)}
          title={`${(v * 100).toFixed(0)}% attention`}
          className={`cursor-pointer rounded-[2px] ${
            selected ? "ring-1 ring-emerald-400/40" : ""
          }`}
          style={{
            width: cell,
            height: cell,
            backgroundColor: `rgba(16, 185, 129, ${v})`,
          }}
        />
      ))}
    </>
  );
}

function Predictions({ token, topk }: { token: string; topk: { char: string; prob: number }[] }) {
  const max = Math.max(...topk.map((t) => t.prob), 0.0001);
  return (
    <div className="mt-6 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <p className="mb-3 text-sm text-zinc-400">
        After{" "}
        <span className="font-mono text-emerald-400">
          “{show(token)}”
        </span>{" "}
        the model predicts the next character:
      </p>
      <div className="space-y-1.5">
        {topk.map((t, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="w-6 text-right font-mono text-sm text-zinc-300">
              {show(t.char)}
            </span>
            <div className="h-4 flex-1 overflow-hidden rounded bg-zinc-800">
              <div
                className="h-full rounded bg-emerald-500"
                style={{ width: `${(t.prob / max) * 100}%` }}
              />
            </div>
            <span className="w-12 text-right font-mono text-xs text-zinc-400">
              {(t.prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
