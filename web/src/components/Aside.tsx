// A plain-language analogy box — the on-ramp for readers with zero background.

export default function Aside({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-4 flex gap-3 rounded-xl border border-stone-200 bg-stone-50/70 p-4 text-[15px] leading-relaxed text-stone-600">
      <span className="select-none text-lg leading-none" aria-hidden>
        💡
      </span>
      <p>
        <span className="font-semibold text-stone-700">In plain words — </span>
        {children}
      </p>
    </div>
  );
}
