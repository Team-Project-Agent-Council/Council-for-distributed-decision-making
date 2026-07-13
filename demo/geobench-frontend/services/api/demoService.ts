import type {
  DemoSseEvent,
  DemoStartRunInput,
  DemoStartRunResponse,
  DemoRandomLocation,
} from "./types";
import { API_BASE_URL } from "@/lib/constants";

/**
 * Resolve an `imageUrl` returned by the backend to a fully-qualified URL.
 *
 * The backend serves dataset images from a relative path
 * (`/api/demo/dataset/image/<id>`). Relative URLs in `<img src>` resolve
 * against the frontend origin (e.g. `localhost:3000`) instead of the backend
 * origin (`localhost:8000`), which gives a 404 + broken-image icon. We
 * prefix `API_BASE_URL` for relative paths but leave any URL that already
 * has a scheme — `http(s):`, `data:`, `blob:` — unchanged.
 */
function resolveImageUrl(url: string): string {
  if (!url) return url;
  if (/^[a-z]+:/i.test(url)) return url; // http:, https:, data:, blob: …
  if (url.startsWith("//")) return `https:${url}`;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;
  return url;
}

export interface DemoSseHandlers {
  onEvent: (event: DemoSseEvent) => void;
  onError?: (err: Error) => void;
  onClose?: () => void;
}

/**
 * Talks to /api/demo/* on the FastAPI backend. The backend is the single
 * source of truth — there is no mock fallback. If the cluster + SSH tunnel
 * + backend are not all up, requests will fail and the UI will surface
 * the error.
 */
class DemoService {
  async startRun(input: DemoStartRunInput): Promise<DemoStartRunResponse> {
    const form = new FormData();
    if (input.file) {
      form.append("image", input.file);
    } else if (input.datasetId) {
      form.append("datasetId", input.datasetId);
    } else {
      throw new Error("startRun requires either a file or a datasetId");
    }

    const res = await fetch(`${API_BASE_URL}/api/demo/run`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      throw new Error(`Demo start failed: ${res.status} ${await res.text()}`);
    }
    const data = (await res.json()) as DemoStartRunResponse;
    return { ...data, imageUrl: resolveImageUrl(data.imageUrl) };
  }

  subscribeToRun(runId: string, handlers: DemoSseHandlers): () => void {
    const url = `${API_BASE_URL}/api/demo/runs/${encodeURIComponent(runId)}/events`;
    const source = new EventSource(url);

    const eventTypes: ReadonlyArray<DemoSseEvent["type"]> = [
      "run_started",
      "phase1_started",
      "agent_assessment",
      "region_consensus_result",
      "region_hypotheses_generated",
      "region_evaluation",
      "region_evaluation_complete",
      "region_decision",
      "country_assessment",
      "country_hypotheses_generated",
      "country_evaluation",
      "country_evaluation_complete",
      "final_started",
      "final_result",
      "error",
      "done",
    ];

    const listeners: Array<{ type: string; fn: (e: MessageEvent) => void }> = [];

    for (const type of eventTypes) {
      const fn = (e: MessageEvent) => {
        try {
          const parsed = JSON.parse(e.data) as DemoSseEvent;
          // run_started carries an imageUrl that may be a relative backend
          // path (e.g. /api/demo/dataset/image/<id>). Resolve it before the
          // store sees it so the <img> renders correctly.
          const normalised: DemoSseEvent =
            parsed.type === "run_started"
              ? {
                  ...parsed,
                  data: {
                    ...parsed.data,
                    imageUrl: resolveImageUrl(parsed.data.imageUrl),
                  },
                }
              : parsed;
          handlers.onEvent(normalised);
          if (normalised.type === "done") {
            source.close();
            handlers.onClose?.();
          }
        } catch (err) {
          handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
        }
      };
      source.addEventListener(type, fn as EventListener);
      listeners.push({ type, fn });
    }

    source.onerror = () => {
      handlers.onError?.(new Error("SSE connection lost"));
    };

    return () => {
      for (const { type, fn } of listeners) {
        source.removeEventListener(type, fn as EventListener);
      }
      source.close();
    };
  }

  async getRandomLocation(): Promise<DemoRandomLocation> {
    const res = await fetch(`${API_BASE_URL}/api/demo/dataset/random`);
    if (!res.ok) throw new Error(`Random location failed: ${res.status}`);
    const data = (await res.json()) as DemoRandomLocation;
    return { ...data, imageUrl: resolveImageUrl(data.imageUrl) };
  }
}

export const demoService = new DemoService();
