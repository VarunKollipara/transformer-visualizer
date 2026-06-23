// A numbered stage in the end-to-end journey. Presentational wrapper.

export default function Section({
  step,
  title,
  subtitle,
  children,
}: {
  step: number;
  title: string;
  subtitle?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mx-auto w-full max-w-3xl scroll-mt-8 border-t border-zinc-800 py-14">
      <div className="mb-6">
        <div className="mb-2 flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-500/15 text-sm font-bold text-emerald-400 ring-1 ring-emerald-500/30">
            {step}
          </span>
          <h2 className="text-2xl font-bold tracking-tight text-zinc-50">
            {title}
          </h2>
        </div>
        {subtitle && (
          <p className="text-[15px] leading-relaxed text-zinc-400">{subtitle}</p>
        )}
      </div>
      {children}
    </section>
  );
}
