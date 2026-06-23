"use client";

import { useEffect, useState } from "react";
import { API_URL, getInfo, type Config } from "@/lib/api";

// Small banner that checks whether the FastAPI backend is reachable, and shows
// the live model's configuration when it is.
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
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
        Backend not reachable at{" "}
        <code className="font-mono">{API_URL}</code>. Start it with{" "}
        <code className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-amber-100">
          uv run uvicorn src.api:app --port 8000
        </code>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-500">
        Connecting to the model…
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-sm text-zinc-300">
      <span className="font-semibold text-emerald-400">● live model</span>
      <span>{config.vocab_size} tokens</span>
      <span>{config.n_embd}-dim embeddings</span>
      <span>{config.num_layers} layers × {config.num_heads} heads</span>
      <span>context {config.block_size}</span>
    </div>
  );
}
