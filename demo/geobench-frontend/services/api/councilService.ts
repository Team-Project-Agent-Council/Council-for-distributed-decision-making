import type { CouncilInfo } from "./types";
import { API_BASE_URL, BASE_PATH, STATIC_DEMO_MODE } from "@/lib/constants";

export interface ICouncilService {
  getCouncilInfo(): Promise<CouncilInfo>;
}

/**
 * Council metadata service.
 *
 * In the regular build the config is fetched from the FastAPI backend at
 * `/api/council/agents`. In the static (GitHub Pages) build there is no
 * backend, so we serve a snapshot of `council_config.json` from
 * `public/demo-fixture/council.json` instead. The snapshot is a
 * byte-for-byte copy of `geobench-backend/data/council_config.json` —
 * kept in sync by hand for now (both files are ~5 KB and rarely change).
 */
class CouncilService implements ICouncilService {
  async getCouncilInfo(): Promise<CouncilInfo> {
    const url = STATIC_DEMO_MODE
      ? `${BASE_PATH}/demo-fixture/council.json`
      : `${API_BASE_URL}/api/council/agents`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Council metadata request failed: ${res.status}`);
    return res.json();
  }
}

export const councilService: ICouncilService = new CouncilService();
