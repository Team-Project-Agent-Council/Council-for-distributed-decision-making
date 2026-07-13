export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * Static-demo mode. When `true`, the frontend replays a pre-recorded
 * Progressive Narrowing run from `public/demo-fixture/run.json` instead
 * of talking to a live backend. Used by the GitHub Pages build so the
 * demo is clickable without a running cluster.
 *
 * Set via `NEXT_PUBLIC_STATIC_DEMO=true` at build time.
 */
export const STATIC_DEMO_MODE = process.env.NEXT_PUBLIC_STATIC_DEMO === "true";

/**
 * Optional base path when the site is served from a sub-directory
 * (`https://user.github.io/geobench/` rather than a root domain).
 * Prepended to all `public/`-relative URLs at runtime.
 */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
