# GeoBench Frontend: Next.js 16

The Next.js UI for the Progressive Narrowing visualisation. A thin shell
around the council metadata page and the live demo streamed from the
FastAPI backend.

For the wider project context (Mannheim team project on multi-LLM
councils, cluster setup, SSH tunnelling), see the
[top-level README](../README.md).

---

## Stack

- Next.js 16.2.1 (App Router, Turbopack), React 19, TypeScript strict
- Tailwind CSS v4 (CSS-first `@theme` config in `globals.css`), shadcn/ui primitives
- Zustand 5.0, one store only (`stores/demoStore.ts`)
- Leaflet, imported dynamically (no `react-leaflet`, no SSR). No interactive Street View. NMPZ-style static images.
- Framer Motion for subtle transitions (200 to 300 ms)

## Running

```bash
npm install
cp .env.example .env.local
npm run dev
```

Opens on <http://localhost:3000>:

- `/`, landing page with pipeline overview
- `/council`, council info (5 agents + 6 collaboration steps)
- `/test-council`, the Progressive Narrowing demo

The demo hits the FastAPI backend on `NEXT_PUBLIC_API_BASE_URL` (default
`http://localhost:8000`). Without the backend and a live vLLM tunnel the
demo raises network errors. There is no mock fallback. The `/council`
metadata page is fully static and works standalone.

### Build and type-check

```bash
npm run build
npm run start
npx tsc --noEmit
```

## Environment variables

See [`.env.example`](./.env.example):

```env
# Backend URL.
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## File map

| Path | Purpose |
|------|---------|
| `app/layout.tsx` | Root layout, navigation only, no providers |
| `app/page.tsx` | Landing page, pipeline overview + CTA |
| `app/council/page.tsx` | Council info and agent cards (data via `councilService`) |
| `app/test-council/page.tsx` | The Progressive Narrowing demo, primary feature |
| `components/council/` | Council page UI (`AgentCard`, `AgentGrid`, `CollaborationFlow`, …) |
| `components/test-council/` | Demo UI: `DemoMap`, `DemoAgentPanel`, `HypothesisMatrix`, `ProgressTimeline`, `TestCouncilOrchestrator` |
| `components/layout/Navigation.tsx` | Top nav: Home, Run the Council, The Council |
| `components/ui/` | shadcn/ui primitives, no business logic |
| `stores/demoStore.ts` | The only Zustand store, drives the demo SSE stream |
| `services/api/types.ts` | All TypeScript types including the demo SSE event union |
| `services/api/demoService.ts` | Talks to `/api/demo/*` on the FastAPI backend (single source of truth) |
| `services/api/staticDemoService.ts` | Replays a pre-recorded fixture for the GitHub Pages build |
| `services/api/activeDemoService.ts` | Barrel, selects `demoService` or `staticDemoService` from `NEXT_PUBLIC_STATIC_DEMO` |
| `services/api/demoAgents.ts` | `DEMO_AGENT_PROFILES` + `DEMO_AGENT_IDS`, static metadata |
| `services/api/councilService.ts` | Council info, fetches from `/api/council/agents` |
| `lib/constants.ts` | `API_BASE_URL`, `STATIC_DEMO_MODE`, `BASE_PATH` |

## Architecture patterns

### Service layer

Both the demo service and the council service hit the FastAPI backend
directly; there is no mock fallback. When adding a new API method:

1. Add the type to `services/api/types.ts`
2. Implement it on the service class
3. Components import the singleton and remain agnostic to the underlying
   implementation

### Zustand store

`stores/demoStore.ts` is the only store. It consumes SSE events from
`demoService.subscribeToRun()` and translates them into substate slots:

```ts
phase, agentAssessments, agentReady,
regionConsensus, regionHypotheses, regionEvaluations, regionEvaluationSummary,
regionDecision,
countryAssessments, countryHypotheses, countryEvaluations,
countryEvaluationSummary,
finalResult, error
```

Components select narrow slices and do not reach into the SSE shape.

### Component organisation

- Pages in `app/{route}/page.tsx`, thin, compose feature components
- Feature components in `components/{feature}/`, contain logic and layout
- UI primitives in `components/ui/`, shadcn/ui, no business logic

## Styling rules

1. Tailwind v4 utility classes only (CSS-first `@theme` config in `globals.css`)
2. GeoGuessr lime (`#c6ef38`) as the primary accent, used for CTAs, active nav, and consensus glow
3. Agent colours stay in sync between `demoAgents.ts` and `geobench-backend/data/council_config.json`
4. Dark mode via Tailwind `dark:` prefix
5. Animations via Framer Motion, subtle (200 to 300 ms)

## Map integration (`DemoMap`)

- Leaflet is imported dynamically inside a `useEffect` (never at module
  top level) so SSR does not attempt to reach `window`.
- `worldCopyJump: false`, `noWrap: true`, and `inertia: false` keep the
  SVG overlay pane locked to the tile pane. Without these, panning
  mid-stream desynchronises polygons from the countries beneath them.
- Auto-fit uses `fitBounds` / `setView` without animation during live
  phases; only the final reveal animates.
- Layer state is regenerated on remount via a `mapReady` counter in the
  dependency arrays. This ensures the map redraws correctly after a
  Next.js route change that unmounts and later remounts the demo page.
