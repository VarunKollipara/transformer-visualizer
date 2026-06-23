"use client";

import { useEffect, useState } from "react";
import { API_URL, getInfo, type Config } from "@/lib/api";

// Banner: is the FastAPI backend reachable, and what is the live model?
export default function ApiStatus() {
  const [config, setConfig] = useState<Config | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getInfo()
      .then((info) => setConfig(info.config))
      .catch(() => setError(true));
  }, []);

  if (error) {
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
        Backend not reachable at <code className="font-mono">{API_URL}</code>.
        Start it with{" "}
        <code className="rounded bg-amber-100 px-1.5 py-0.5 font-mono">
          uv run uvicorn src.api:app --port 8000
        </code>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="rounded-xl border border-stone-200 bg-white px-4 py-3 text-sm text-stone-400">
        Connecting to the model…
      </div>
    );
  }

  const stats = [
    [`${config.vocab_size}`, "tokens"],
    [`${config.n_embd}`, "embedding dims"],
    [`${config.num_layers}×${config.num_heads}`, "layers × heads"],
    [`${config.block_size}`, "context"],
  ];

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-emerald-200 bg-emerald-50/60 px-4 py-3">
      <span className="flex items-center gap-2 text-sm font-semibold text-emerald-700">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
        </span>
        live model
      </span>
      {stats.map(([v, label]) => (
        <span key={label} className="text-sm text-stone-600">
          <span className="font-semibold text-stone-900">{v}</span> {label}
        </span>
      ))}
    </div>
  );
}
