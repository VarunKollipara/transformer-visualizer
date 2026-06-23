// A connective bridge between two steps: an animated flowing line + a handoff
// message (and an optional artifact being passed along), so the journey reads as
// one continuous process rather than separate sections.

function Line({ height = 30 }: { height?: number }) {
  return (
    <svg
      width="3"
      height={height}
      viewBox={`0 0 3 ${height}`}
      className="block"
      aria-hidden
    >
      <line
        x1="1.5"
        y1="0"
        x2="1.5"
        y2={height}
        stroke="#d6d3d1"
        strokeWidth="2.5"
        strokeLinecap="round"
        className="flow-line"
      />
    </svg>
  );
}

export default function FlowConnector({
  children,
  artifact,
}: {
  children: React.ReactNode;
  artifact?: React.ReactNode;
}) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col items-center">
      <Line />
      <div className="flex max-w-lg flex-col items-center gap-2 rounded-2xl border border-stone-200 bg-white px-5 py-3 text-center shadow-sm">
        <p className="text-sm leading-relaxed text-stone-500">{children}</p>
        {artifact && <div className="text-stone-700">{artifact}</div>}
      </div>
      <Line />
    </div>
  );
}
