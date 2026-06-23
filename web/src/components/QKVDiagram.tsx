// A small schematic of how one attention score is computed: a token makes a
// Query, compares it to each Key (dot product), softmaxes, and blends the Values.

export default function QKVDiagram() {
  const box = "rounded-lg px-3 py-2 text-center text-xs font-semibold";
  return (
    <div className="my-3 rounded-xl border border-stone-200 bg-stone-50 p-4">
      <div className="flex flex-col items-center gap-2 text-stone-600">
        <div className="font-mono text-sm text-stone-700">token vector</div>
        <div className="text-stone-300">↓ three learned projections ↓</div>
        <div className="grid w-full max-w-md grid-cols-3 gap-2">
          <div className={`${box} bg-teal-100 text-teal-800`}>
            Query
            <div className="font-normal opacity-70">what I want</div>
          </div>
          <div className={`${box} bg-sky-100 text-sky-800`}>
            Key
            <div className="font-normal opacity-70">what I offer</div>
          </div>
          <div className={`${box} bg-violet-100 text-violet-800`}>
            Value
            <div className="font-normal opacity-70">my info</div>
          </div>
        </div>
        <div className="mt-1 text-center font-mono text-xs text-stone-500">
          score(i, j) = Query<sub>i</sub> · Key<sub>j</sub> ÷ √d → softmax →
          weights → Σ weight · Value
        </div>
      </div>
    </div>
  );
}
