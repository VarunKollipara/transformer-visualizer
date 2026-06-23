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

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const getInfo = () => getJSON<InfoResponse>("/api/info");

export const forward = (text: string, top_k = 10) =>
  postJSON<ForwardResponse>("/api/forward", { text, top_k });

export const generate = (
  prompt: string,
  max_new_tokens = 200,
  temperature = 1.0,
  top_k = 10,
) =>
  postJSON<GenerateResponse>("/api/generate", {
    prompt,
    max_new_tokens,
    temperature,
    top_k,
  });

export type EmbeddingPoint = {
  id: number;
  char: string;
  x: number;
  y: number;
  group: "upper" | "lower" | "digit" | "punct" | "space" | "newline";
};
export type EmbeddingsResponse = { points: EmbeddingPoint[] };

export const getEmbeddings = () =>
  getJSON<EmbeddingsResponse>("/api/embeddings");

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

export const getTraining = () => getJSON<TrainingResponse>("/api/training");
