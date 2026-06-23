// Shows a slice of the real training corpus (tinyshakespeare) so the reader can
// see exactly what the model learned from — and nothing else.

const SNIPPET = `First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.`;

export default function CorpusSample() {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
      <pre className="overflow-x-auto whitespace-pre-wrap rounded-xl border border-stone-200 bg-stone-50 p-4 font-mono text-[13px] leading-relaxed text-stone-700">
        {SNIPPET}
      </pre>
      <div className="mt-4 grid grid-cols-3 gap-3 text-center">
        {[
          ["1,115,394", "characters of text"],
          ["65", "unique characters"],
          ["0", "facts about the world"],
        ].map(([n, label]) => (
          <div key={label} className="rounded-xl bg-amber-50 px-2 py-3">
            <div className="font-display text-2xl font-semibold text-amber-700">
              {n}
            </div>
            <div className="mt-1 text-xs text-stone-500">{label}</div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-sm text-stone-500">
        This is the model&apos;s entire universe — about a megabyte of
        Shakespeare. It never sees the internet, definitions, or grammar rules.
        Everything it &ldquo;knows&rdquo; is squeezed out of predicting the next
        character in <em>this</em> text, millions of times over.
      </p>
    </div>
  );
}
