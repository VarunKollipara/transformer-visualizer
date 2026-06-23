// A numbered stage in the end-to-end journey. Each stage has its own accent hue
// so concepts stay visually color-coded across the page.

const ACCENTS: Record<string, { badge: string; rule: string }> = {
  indigo: { badge: "bg-indigo-100 text-indigo-700 ring-indigo-200", rule: "bg-indigo-200" },
  sky: { badge: "bg-sky-100 text-sky-700 ring-sky-200", rule: "bg-sky-200" },
  violet: { badge: "bg-violet-100 text-violet-700 ring-violet-200", rule: "bg-violet-200" },
  teal: { badge: "bg-teal-100 text-teal-700 ring-teal-200", rule: "bg-teal-200" },
  amber: { badge: "bg-amber-100 text-amber-700 ring-amber-200", rule: "bg-amber-200" },
  rose: { badge: "bg-rose-100 text-rose-700 ring-rose-200", rule: "bg-rose-200" },
};

export default function Section({
  step,
  title,
  accent = "indigo",
  subtitle,
  children,
}: {
  step: number;
  title: string;
  accent?: keyof typeof ACCENTS;
  subtitle?: React.ReactNode;
  children: React.ReactNode;
}) {
  const a = ACCENTS[accent] ?? ACCENTS.indigo;
  return (
    <section className="mx-auto w-full max-w-3xl scroll-mt-8 py-12">
      <div className={`mb-6 h-px w-full ${a.rule}`} />
      <div className="mb-6">
        <div className="mb-3 flex items-center gap-3">
          <span
            className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold ring-1 ${a.badge}`}
          >
            {step}
          </span>
          <h2 className="font-display text-[1.7rem] font-semibold leading-tight tracking-tight text-stone-900">
            {title}
          </h2>
        </div>
        {subtitle && (
          <p className="text-[15px] leading-relaxed text-stone-600">{subtitle}</p>
        )}
      </div>
      {children}
    </section>
  );
}
