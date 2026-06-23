"use client";

import { useEffect, useMemo, useState } from "react";
import { getInfo } from "@/lib/api";

// Live character tokenizer: type text, see each character mapped to its token ID.
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

  const display = (ch: string) => (ch === " " ? "␣" : ch === "\n" ? "⏎" : ch);

  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="w-full rounded-lg border border-stone-300 bg-stone-50 px-4 py-2.5 font-mono text-stone-900 outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-200"
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
                  ? "border-sky-200 bg-sky-50"
                  : "border-rose-300 bg-rose-50"
              }`}
              title={known ? `'${ch}' → ID ${id}` : `'${ch}' not in vocabulary`}
            >
              <span className="font-mono text-base leading-none text-stone-800">
                {display(ch)}
              </span>
              <span className="mt-1 font-mono text-[11px] leading-none text-sky-600">
                {known ? id : "?"}
              </span>
            </span>
          );
        })}
      </div>
      <p className="mt-4 break-words rounded-lg bg-stone-50 px-3 py-2 font-mono text-sm text-stone-500">
        [{[...text].map((ch) => stoi.get(ch) ?? "?").join(", ")}]
      </p>
      <p className="mt-3 text-sm text-stone-500">
        {[...text].length} characters → {[...text].length} token IDs. This exact
        array of numbers is the only thing the model ever sees.
      </p>
    </div>
  );
}
