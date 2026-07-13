import type {
  DemoSseEvent,
  DemoStartRunInput,
  DemoStartRunResponse,
  DemoRandomLocation,
} from "./types";
import type { DemoSseHandlers } from "./demoService";
import { BASE_PATH } from "@/lib/constants";

/**
 * Static-demo service. Replays a pre-recorded Progressive Narrowing run
 * from `public/demo-fixture/run.json` on the same schedule as the
 * original live run (timing preserved via each event's `t_ms`).
 *
 * Used by the GitHub Pages build (`NEXT_PUBLIC_STATIC_DEMO=true`) so the
 * demo is clickable without a running FastAPI backend + vLLM cluster.
 * The public API mirrors `DemoService` exactly so consumers (the Zustand
 * store, the orchestrator) don't know which one they got.
 */

interface RecordedEvent {
  t_ms: number;
  type: DemoSseEvent["type"];
  data: DemoSseEvent["data"];
}

interface FixtureFile {
  imageFile: string;
  groundTruth: { lat: number; lng: number; label: string } | null;
  events: RecordedEvent[];
}

// Playback speed. 1.0 = original wall-clock timing. Higher values compress
// the run so a viewer isn't waiting minutes for the final result.
const PLAYBACK_SPEED = 3;

async function loadFixture(): Promise<FixtureFile> {
  const url = `${BASE_PATH}/demo-fixture/run.json`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Static demo fixture missing at ${url} (status ${res.status}). ` +
        `Record one with geobench-backend/scripts/record_fixture.py.`,
    );
  }
  return (await res.json()) as FixtureFile;
}

class StaticDemoService {
  private fixture: FixtureFile | null = null;

  private async ensureFixture(): Promise<FixtureFile> {
    if (this.fixture) return this.fixture;
    this.fixture = await loadFixture();
    return this.fixture;
  }

  async startRun(_input: DemoStartRunInput): Promise<DemoStartRunResponse> {
    const fixture = await this.ensureFixture();
    return {
      runId: "static-demo",
      imageUrl: `${BASE_PATH}/demo-fixture/${fixture.imageFile}`,
      groundTruth: fixture.groundTruth ?? undefined,
    };
  }

  subscribeToRun(_runId: string, handlers: DemoSseHandlers): () => void {
    let cancelled = false;
    const timers: ReturnType<typeof setTimeout>[] = [];

    this.ensureFixture()
      .then((fixture) => {
        if (cancelled) return;
        // Schedule each event at t_ms/PLAYBACK_SPEED after subscription.
        for (const evt of fixture.events) {
          const delay = Math.max(0, Math.floor(evt.t_ms / PLAYBACK_SPEED));
          const timer = setTimeout(() => {
            if (cancelled) return;
            // The recorder captures whatever the backend sent, which for
            // upload-mode runs omits `groundTruth`. Splice the fixture's
            // top-level `groundTruth` into `run_started` at replay time so
            // the store sees it (the store's `run_started` handler would
            // otherwise clobber the value set by `startRun`).
            let data = evt.data as Record<string, unknown>;
            if (evt.type === "run_started" && fixture.groundTruth) {
              data = { ...data, groundTruth: fixture.groundTruth };
            }
            const sseEvent = { type: evt.type, data } as DemoSseEvent;
            handlers.onEvent(sseEvent);
            if (evt.type === "done") {
              handlers.onClose?.();
            }
          }, delay);
          timers.push(timer);
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
      });

    return () => {
      cancelled = true;
      for (const t of timers) clearTimeout(t);
    };
  }

  async getRandomLocation(): Promise<DemoRandomLocation> {
    const fixture = await this.ensureFixture();
    return {
      datasetId: "static-demo",
      imageUrl: `${BASE_PATH}/demo-fixture/${fixture.imageFile}`,
      lat: fixture.groundTruth?.lat ?? 0,
      lng: fixture.groundTruth?.lng ?? 0,
      label: fixture.groundTruth?.label ?? "?",
    };
  }
}

export const staticDemoService = new StaticDemoService();
