// Typed client for the FastAPI backend (src/api.py).
// Every response mirrors what the Python model actually computed — no mocks.

export type Config = {
  vocab_size: number;
  n_embd: number;
  num_heads: number;
  num_layers: number;
  block_size: number;
};

export type InfoResponse = {
  config: Config;
  vocab: string[]; // the sorted vocabulary; index == token ID
  vocab_size: number;
};

export type TopK = { char: string; prob: number };
export type Position = { char: string; topk: TopK[] };

export type ForwardResponse = {
  tokens: string[];
  positions: Position[];
  // attention[layer][head] is a (T x T) matrix of weights (rows sum to 1)
  attention: number[][][][];
  num_layers: number;
  num_heads: number;
};

export type Step = { char: string; topk: TopK[] };
export type GenerateResponse = {
  prompt: string;
  generated: string;
  steps: Step[];
};

// info / embeddings / training are identical every run, so they're baked to
// static JSON at build time (scripts/export_static.py -> web/public/data/*.json)
// and fetched as plain files. The live endpoints (forward / logits / generate)
// run the exported model client-side via ONNX (./onnx.ts), lazy-loaded on first
// use so onnxruntime-web isn't in the initial bundle. Net: no backend at all.
async function getStatic<T>(file: string): Promise<T> {
  const res = await fetch(`/data/${file}`);
  if (!res.ok) throw new Error(`GET /data/${file} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const getInfo = () => getStatic<InfoResponse>("info.json");

export const forward = (text: string, top_k = 10) =>
  import("./onnx").then((m) => m.forward(text, top_k));

export const generate = (
  prompt: string,
  max_new_tokens = 200,
  temperature = 1.0,
  top_k = 10,
) => import("./onnx").then((m) => m.generate(prompt, max_new_tokens, temperature, top_k));

export type EmbeddingPoint = {
  id: number;
  char: string;
  x: number;
  y: number;
  group: "upper" | "lower" | "digit" | "punct" | "space" | "newline";
};
export type EmbeddingsResponse = { points: EmbeddingPoint[] };

export const getEmbeddings = () =>
  getStatic<EmbeddingsResponse>("embeddings.json");

export type LossPoint = { step: number; train: number; val: number };
export type SamplePoint = { step: number; text: string };
export type TrainingResponse = {
  history: LossPoint[];
  samples: SamplePoint[];
  config: {
    n_embd: number;
    num_heads: number;
    num_layers: number;
    block_size: number;
    max_steps: number;
    params: number;
  };
};

export const getTraining = () => getStatic<TrainingResponse>("training.json");

export type Candidate = { char: string; logit: number };
export type LogitsResponse = { context: string; candidates: Candidate[] };

export const getLogits = (text: string, top_k = 8) =>
  import("./onnx").then((m) => m.next_logits(text, top_k));
