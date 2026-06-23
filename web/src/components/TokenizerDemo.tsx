"use client";

import { useEffect, useMemo, useState } from "react";
import { getInfo } from "@/lib/api";

// Live character tokenizer: type text, see each character mapped to its token ID.
// The vocab from /api/info is the sorted character list, so a char's ID is just
// its index in that list — we can tokenize entirely in the browser.
export default function TokenizerDemo() {
  const [vocab, setVocab] = useState<string[] | null>(null);
  const [text, setText] = useState("To be, or not to be");

  useEffect(() => {
    getInfo()
      .then((info) => setVocab(info.vocab))
      .catch(() => setVocab(null));
  }, []);

  const stoi = useMemo(() => {
    const m = new Map<string, number>();
    vocab?.forEach((ch, i) => m.set(ch, i));
    return m;
  }, [vocab]);

  const display = (ch: string) =>
    ch === " " ? "␣" : ch === "\n" ? "⏎" : ch;

  return (
    <div>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 font-mono text-zinc-100 outline-none focus:border-emerald-500"
        placeholder="Type some text…"
      />
      <div className="mt-4 flex flex-wrap gap-1.5">
        {[...text].map((ch, i) => {
          const id = stoi.get(ch);
          const known = id !== undefined;
          return (
            <span
              key={i}
              className={`flex flex-col items-center rounded-md border px-2 py-1 ${
                known
                  ? "border-zinc-700 bg-zinc-800"
                  : "border-red-500/50 bg-red-500/10"
              }`}
              title={known ? `'${ch}' → ID ${id}` : `'${ch}' not in vocabulary`}
            >
              <span className="font-mono text-base leading-none text-zinc-100">
                {display(ch)}
              </span>
              <span className="mt-1 font-mono text-[11px] leading-none text-emerald-400">
                {known ? id : "?"}
              </span>
            </span>
          );
        })}
      </div>
      <p className="mt-4 font-mono text-sm text-zinc-400">
        {[...text].map((ch) => stoi.get(ch) ?? "?").join(", ")}
      </p>
      <p className="mt-2 text-sm text-zinc-500">
        {[...text].length} characters → {[...text].length} token IDs. This exact
        array of numbers is what the model actually reads.
      </p>
    </div>
  );
}
