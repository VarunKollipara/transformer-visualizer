"use client";

// A clickable schematic of the whole forward pass. Each stage scrolls to its
// section, so this doubles as a map of the page.

type Stage = {
  label: string;
  sub: string;
  target?: string;
  cls: string;
};

const STAGES: Stage[] = [
  { label: "Text", sub: '"To be or not to b"', cls: "border-stone-300 bg-stone-50 text-stone-700" },
  { label: "Tokenize", sub: "characters → IDs", target: "tokens", cls: "border-sky-200 bg-sky-50 text-sky-800" },
  { label: "Embed (+ position)", sub: "IDs → vectors", target: "embeddings", cls: "border-violet-200 bg-violet-50 text-violet-800" },
  { label: "Transformer blocks ×3", sub: "attention + MLP mix context", target: "attention", cls: "border-teal-200 bg-teal-50 text-teal-800" },
  { label: "Logits → Softmax", sub: "scores → probabilities", target: "choice", cls: "border-indigo-200 bg-indigo-50 text-indigo-800" },
  { label: "Sample a character", sub: "pick one, append it", target: "choice", cls: "border-indigo-200 bg-indigo-50 text-indigo-800" },
];

export default function PipelineDiagram() {
  const go = (id?: string) => {
    if (id) document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-6 shadow-sm">
      <div className="flex flex-col items-stretch gap-0">
        {STAGES.map((s, i) => (
          <div key={i} className="flex flex-col items-center">
            <button
              onClick={() => go(s.target)}
              disabled={!s.target}
              className={`w-full max-w-md rounded-xl border px-4 py-2.5 text-left transition ${s.cls} ${
                s.target ? "cursor-pointer hover:brightness-95" : "cursor-default"
              }`}
            >
              <div className="font-semibold">{s.label}</div>
              <div className="font-mono text-xs opacity-70">{s.sub}</div>
            </button>
            {i < STAGES.length - 1 && (
              <span className="my-1 text-stone-300">↓</span>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-center justify-center gap-2 text-sm text-stone-500">
        <span className="text-base">⟲</span> then the new character is added to
        the text and the whole loop runs again
      </div>
    </div>
  );
}
