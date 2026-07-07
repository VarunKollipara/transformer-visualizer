import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The app is 100% client-side (no server code, no API routes) — export it as
  // a plain folder of static files (web/out) so it deploys anywhere, without
  // depending on a host detecting Next.js.
  output: "export",
};

export default nextConfig;
