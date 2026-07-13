import type { NextConfig } from "next";
import path from "path";

const isStaticExport = process.env.NEXT_PUBLIC_STATIC_DEMO === "true";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  turbopack: {
    root: path.resolve(__dirname),
  },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "**" },
    ],
    // Static export can't run the Next.js image optimiser; every <Image>
    // must resolve to a plain URL. We're using regular <img> for the
    // demo fixture, but flip unoptimized on so any lingering <Image>
    // usage doesn't break the export.
    unoptimized: isStaticExport,
  },
  // Static export mode: produces an `out/` directory of plain HTML/CSS/JS
  // suitable for GitHub Pages. Enabled only when the build sets
  // NEXT_PUBLIC_STATIC_DEMO=true.
  ...(isStaticExport
    ? {
        output: "export" as const,
        // GitHub Pages serves from `/<repo-name>/` by default. Trailing
        // slashes make link resolution consistent between local `next start`
        // and the deployed site.
        basePath,
        trailingSlash: true,
      }
    : {}),
};

export default nextConfig;
