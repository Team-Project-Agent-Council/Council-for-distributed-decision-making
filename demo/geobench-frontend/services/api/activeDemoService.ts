import { STATIC_DEMO_MODE } from "@/lib/constants";
import { demoService as liveDemoService } from "./demoService";
import { staticDemoService } from "./staticDemoService";

/**
 * The active demo service — `demoService` (talks to the FastAPI backend)
 * or `staticDemoService` (replays a pre-recorded fixture). The choice is
 * made at build time via `NEXT_PUBLIC_STATIC_DEMO`.
 *
 * Consumers import this file rather than the concrete implementations so
 * they don't care which one is active.
 */
export const demoService = STATIC_DEMO_MODE ? staticDemoService : liveDemoService;

// Re-export the handler type so callers don't need to know which service
// they got.
export type { DemoSseHandlers } from "./demoService";
