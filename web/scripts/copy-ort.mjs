// Copy the onnxruntime-web CPU-wasm runtime into public/ort so it can be served
// as a static asset (self-hosted, version-matched to the installed package).
// Wired as predev/prebuild so it runs locally and on Vercel; public/ort is
// gitignored (regenerated from node_modules).
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "node_modules", "onnxruntime-web", "dist");
const dst = join(here, "..", "public", "ort");

mkdirSync(dst, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(join(src, f), join(dst, f));
}
console.log("copied onnxruntime-web wasm -> public/ort");
