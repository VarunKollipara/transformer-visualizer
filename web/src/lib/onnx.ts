// Client-side inference for the three "live" endpoints (forward / logits /
// generate), so the deployed site needs no backend. Runs the exported char-GPT
// (public/model/gpt.onnx) in the browser via onnxruntime-web and reproduces the
// exact response shapes the FastAPI backend returned.

// CPU-wasm-only build (no WebGPU/JSEP) — a much smaller wasm binary to ship.
import * as ort from "onnxruntime-web/wasm";
import type {
  Candidate,
  ForwardResponse,
  GenerateResponse,
  InfoResponse,
  LogitsResponse,
  Position,
  Step,
  TopK,
} from "./api";

// Single-threaded WASM avoids the cross-origin-isolation (COOP/COEP) headers a
// static host won't set; the wasm binary is self-hosted from /ort/.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/ort/";

let sessionP: Promise<ort.InferenceSession> | null = null;
let metaP: Promise<{ vocab: string[]; blockSize: number }> | null = null;

function session() {
  if (!sessionP) {
    sessionP = ort.InferenceSession.create("/model/gpt.onnx", {
      executionProviders: ["wasm"],
    });
  }
  return sessionP;
}

function meta() {
  if (!metaP) {
    metaP = fetch("/data/info.json")
      .then((r) => r.json())
      .then((d: InfoResponse) => ({ vocab: d.vocab, blockSize: d.config.block_size }));
  }
  return metaP;
}

// ── one forward pass: token ids -> logits (T,V) + attention (L,H,T,T) ──
type RunOut = { logits: Float32Array; att: Float32Array; T: number; V: number; L: number; H: number };

async function run(ids: number[]): Promise<RunOut> {
  const sess = await session();
  const T = ids.length;
  const input = new ort.Tensor("int64", BigInt64Array.from(ids.map((i) => BigInt(i))), [1, T]);
  const out = await sess.run({ idx: input });
  const logits = out.logits.data as Float32Array; // (1, T, V)
  const att = out.attention.data as Float32Array; // (L, H, 1, T, T)
  const V = out.logits.dims[2] as number;
  const L = out.attention.dims[0] as number;
  const H = out.attention.dims[1] as number;
  return { logits, att, T, V, L, H };
}

// logits row for position t -> Float32Array length V
const rowAt = (o: RunOut, t: number) => o.logits.subarray(t * o.V, (t + 1) * o.V);

function softmax(logits: Float32Array | number[], temp = 1): Float32Array {
  const t = Math.max(temp, 1e-6);
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[i] / t);
  const e = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    e[i] = Math.exp(logits[i] / t - max);
    sum += e[i];
  }
  for (let i = 0; i < n; i++) e[i] /= sum;
  return e;
}

// indices of the top-k values in `arr`, descending
function topkIdx(arr: Float32Array | number[], k: number): number[] {
  return Array.from(arr.keys())
    .sort((a, b) => arr[b] - arr[a])
    .slice(0, k);
}

const topkChars = (probs: Float32Array, vocab: string[], k: number): TopK[] =>
  topkIdx(probs, k).map((i) => ({ char: vocab[i], prob: probs[i] }));

// ── public API (mirrors src/inference.py) ──

export async function forward(text: string, top_k = 10): Promise<ForwardResponse> {
  const { vocab, blockSize } = await meta();
  const src = text === "" ? "\n" : text;
  const ids = [...src].map((c) => vocab.indexOf(c)).filter((i) => i >= 0).slice(-blockSize);
  const o = await run(ids);

  const positions: Position[] = ids.map((id, t) => ({
    char: vocab[id],
    topk: topkChars(softmax(rowAt(o, t)), vocab, top_k),
  }));

  // attention[layer][head] -> (T x T); ONNX layout (L, H, 1, T, T)
  const { T } = o;
  const attention: number[][][][] = [];
  for (let l = 0; l < o.L; l++) {
    const heads: number[][][] = [];
    for (let h = 0; h < o.H; h++) {
      const base = (l * o.H + h) * T * T; // batch dim is 1
      const mat: number[][] = [];
      for (let i = 0; i < T; i++) {
        const rowStart = base + i * T;
        mat.push(Array.from(o.att.subarray(rowStart, rowStart + T)));
      }
      heads.push(mat);
    }
    attention.push(heads);
  }

  return {
    tokens: ids.map((i) => vocab[i]),
    positions,
    attention,
    num_layers: o.L,
    num_heads: o.H,
  };
}

export async function next_logits(text: string, top_k = 8): Promise<LogitsResponse> {
  const { vocab, blockSize } = await meta();
  const src = text === "" ? "\n" : text;
  const ids = [...src].map((c) => vocab.indexOf(c)).filter((i) => i >= 0).slice(-blockSize);
  const o = await run(ids);
  const last = rowAt(o, o.T - 1); // raw logits for the next char
  const candidates: Candidate[] = topkIdx(last, top_k).map((i) => ({
    char: vocab[i],
    logit: last[i],
  }));
  return { context: text, candidates };
}

// sample one index from a probability vector (multinomial)
function sample(probs: Float32Array): number {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1;
}

export async function generate(
  prompt: string,
  max_new_tokens = 200,
  temperature = 1.0,
  top_k = 10,
): Promise<GenerateResponse> {
  const { vocab, blockSize } = await meta();
  const src = prompt === "" ? "\n" : prompt;
  const ids = [...src].map((c) => vocab.indexOf(c)).filter((i) => i >= 0);
  const start = ids.length;
  const steps: Step[] = [];

  for (let n = 0; n < max_new_tokens; n++) {
    const o = await run(ids.slice(-blockSize));
    const last = rowAt(o, o.T - 1);
    const probs = softmax(last, temperature);
    const nextId = sample(probs);
    steps.push({ char: vocab[nextId], topk: topkChars(probs, vocab, top_k) });
    ids.push(nextId);
  }

  return {
    prompt: src,
    generated: ids.slice(start).map((i) => vocab[i]).join(""),
    steps,
  };
}
