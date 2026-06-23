// Clickable example inputs so a reader always has somewhere to start.

export default function ExampleChips({
  items,
  onPick,
}: {
  items: string[];
  onPick: (s: string) => void;
}) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
      <span className="mr-1 text-xs text-stone-400">try:</span>
      {items.map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => onPick(s)}
          className="rounded-full border border-stone-200 bg-stone-50 px-2.5 py-1 font-mono text-xs text-stone-600 transition hover:border-stone-300 hover:bg-stone-100"
        >
          {s.replace(/\n/g, "⏎") || "⏎"}
        </button>
      ))}
    </div>
  );
}
