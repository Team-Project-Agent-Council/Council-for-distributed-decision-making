"use client";

import { useEffect, useRef, useState } from "react";
import type { DemoAgentId, DemoFinalResult, DemoGroundTruth } from "@/services/api/types";
import { DEMO_AGENT_PROFILES } from "@/services/api/demoAgents";

export interface DemoAgentMarker {
  agentId: DemoAgentId;
  lat: number;
  lng: number;
  country: string;
  confidence: string;
}

export type DemoMapPhase =
  | "idle"
  | "phase1"
  | "judging"
  | "hypotheses"
  | "final"
  | "done";

interface DemoMapProps {
  /** Per-country: which agents currently vote for it. Computed from phase-1
   *  candidates minus phase-3 contradictions. */
  agentVotesByCountry: Record<string, DemoAgentId[]>;
  /** Country → representative {lat,lng} for placing the per-agent dot cluster. */
  countryCentroids: Record<string, { lat: number; lng: number }>;
  /** Final consensus point. */
  consensusPin: DemoFinalResult | null;
  /** Ground truth (only when running on dataset). */
  groundTruth: DemoGroundTruth | null;
  /** Pipeline phase — drives which colour layer dominates the map. */
  phase: DemoMapPhase;
}

const COUNTRY_NAME_TO_CODE: Record<string, string> = {
  // Minimal lookup for the demo. The real backend can ship a richer mapping.
  Spain: "724",
  Portugal: "620",
  France: "250",
  Italy: "380",
  Germany: "276",
  Andorra: "020",
  Greece: "300",
  Switzerland: "756",
  Austria: "040",
  Belgium: "056",
  Netherlands: "528",
  "United Kingdom": "826",
  Ireland: "372",
  Poland: "616",
  Czechia: "203",
  "Czech Republic": "203",
  Hungary: "348",
  Romania: "642",
  Turkey: "792",
  Croatia: "191",
  Norway: "578",
  Sweden: "752",
  Finland: "246",
  Denmark: "208",
  Iceland: "352",
  Russia: "643",
  Japan: "392",
  China: "156",
  India: "356",
  Brazil: "076",
  Argentina: "032",
  "United States": "840",
  USA: "840",
  Canada: "124",
  Mexico: "484",
  Australia: "036",
  "New Zealand": "554",
  Kyrgyzstan: "417",
  Kazakhstan: "398",
};

export function DemoMap({
  agentVotesByCountry,
  countryCentroids,
  consensusPin,
  groundTruth,
  phase,
}: DemoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import("leaflet").Map | null>(null);
  const geoLayerRef = useRef<import("leaflet").GeoJSON | null>(null);
  const agentLayerRef = useRef<import("leaflet").LayerGroup | null>(null);
  const consensusMarkerRef = useRef<import("leaflet").Marker | null>(null);
  const groundTruthMarkerRef = useRef<import("leaflet").Marker | null>(null);
  const consensusLineRef = useRef<import("leaflet").Polyline | null>(null);
  // Bumped once the Leaflet map + GeoJSON layer have finished initialising.
  // Every layer-mutating useEffect below lists `mapReady` in its deps so it
  // re-runs on remount (e.g. after a Next.js route change unmounts and later
  // remounts the /test-council page). Without this, layers that depend on
  // store values which haven't changed since the last mount (like the
  // ground-truth pin) would silently skip their initial render.
  const [mapReady, setMapReady] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      const L = (await import("leaflet")).default;
      if (cancelled || !containerRef.current || mapRef.current) return;

      if (!document.getElementById("leaflet-css")) {
        const link = document.createElement("link");
        link.id = "leaflet-css";
        link.rel = "stylesheet";
        link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(link);
      }

      const map = L.map(containerRef.current, {
        center: [25, 5],
        zoom: 2,
        minZoom: 1,
        maxZoom: 7,
        zoomControl: true,
        attributionControl: false,
        scrollWheelZoom: true,
        // Lock panning to a single world copy. `worldCopyJump: true` would let
        // the user pan past ±180° but causes GeoJSON polygons + divIcon
        // markers to drift away from the underlying tiles when the view
        // snaps back to the primary world (the SVG overlay pane keeps its
        // pixel origin while the tile origin jumps). Bounded panning avoids
        // that entirely.
        worldCopyJump: false,
        maxBounds: [
          [-85, -180],
          [85, 180],
        ],
        // Soft viscosity (0.7) instead of 1.0 — strict bounds clamping fights
        // momentum-pan, which can briefly desynchronise the tile and overlay
        // panes during quick drags. 0.7 still keeps the user inside the
        // world bounds without that stutter.
        maxBoundsViscosity: 0.7,
        // Disable inertia so that a pan ends synchronously when the user
        // releases the mouse — guarantees layers settle on the same tick.
        inertia: false,
      });
      mapRef.current = map;

      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
        // Don't repeat the world east/west — the GeoJSON layer only has one
        // copy of each country, so wrapping tiles would create a mismatch.
        noWrap: true,
      }).addTo(map);

      const res = await fetch("https://unpkg.com/world-atlas@2/countries-110m.json");
      if (cancelled) return;
      const topo = await res.json();
      const { feature } = (await import("topojson-client")) as {
        feature: (
          topo: unknown,
          obj: unknown
        ) => GeoJSON.FeatureCollection;
      };
      const geojson = feature(topo, topo.objects.countries);

      const layer = L.geoJSON(geojson, {
        style: () => ({
          fillColor: "transparent",
          fillOpacity: 0,
          color: "rgba(255,255,255,0.08)",
          weight: 0.5,
        }),
      }).addTo(map);
      geoLayerRef.current = layer;

      agentLayerRef.current = L.layerGroup().addTo(map);

      // Signal to the downstream layer-mutating effects that the map is
      // ready. Bumping the counter (rather than a boolean) ensures the
      // dependency array registers a change even across the very rare
      // second-mount-with-cached-store-values case.
      setMapReady((n) => n + 1);
    }

    init();
    return () => {
      cancelled = true;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        geoLayerRef.current = null;
        agentLayerRef.current = null;
        consensusMarkerRef.current = null;
        groundTruthMarkerRef.current = null;
        consensusLineRef.current = null;
      }
    };
  }, []);

  // Country fill driver: only the consensus country ever gets a fill.
  //
  //  - phase1 / judging / hypotheses : NO country fill or outline. Every
  //                                    live phase is represented purely by
  //                                    per-country agent dot clusters over
  //                                    the centroids. Country polygons stay
  //                                    at the neutral baseStyle — filling
  //                                    them during voting doubled-encoded
  //                                    the same information as the dots and
  //                                    cluttered the map.
  //  - final / done                  : ONLY the consensus country glows in
  //                                    lime with a matching outline; every
  //                                    other country stays at baseStyle
  //                                    (including countries an agent voted
  //                                    for during narrowing — no lingering
  //                                    faint trace).
  useEffect(() => {
    const layer = geoLayerRef.current;
    if (!layer) return;

    // Every phase except the reveal renders zero fills. The dots layer
    // (see the next useEffect) carries all the per-agent information.
    if (phase !== "final" && phase !== "done") {
      layer.setStyle(() => baseStyle());
      return;
    }

    // Reveal: highlight only the consensus country. Everything else stays
    // at the neutral base style.
    const finalCode = consensusPin ? lookupCode(consensusPin.country) : null;
    layer.setStyle((feature) => {
      const code = normaliseCode(feature?.id);
      if (finalCode && code === finalCode) {
        return {
          fillColor: "#c6ef38",
          fillOpacity: 0.42,
          color: "#c6ef38",
          weight: 2,
        };
      }
      return baseStyle();
    });
  }, [consensusPin, phase, mapReady]);

  // Auto-fit map view as the pipeline narrows.
  //
  // We deliberately use `fitBounds`/`setView` without animation here. The
  // earlier version used `flyToBounds` with a 1.2s animation, but that broke
  // visually when:
  //   1. The user panned mid-animation (their drag fought the animated view
  //      transition and Leaflet's tile pane could end up at a different
  //      pixel origin than the SVG overlay pane).
  //   2. Streaming SSE events triggered a new `flyToBounds` while a previous
  //      one was still animating, leaving the internal map state mid-flight.
  // Synchronous view changes (`animate: false`) avoid both: tiles and SVG
  // re-render in lockstep on the same tick.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const points: [number, number][] = [];

    if (
      phase === "phase1" ||
      phase === "judging" ||
      phase === "hypotheses"
    ) {
      // Fit to whichever countries still have ≥1 supporting agent. As the
      // judge phase eliminates contradicted countries, the bounds tighten
      // automatically.
      for (const country of Object.keys(agentVotesByCountry)) {
        const c = countryCentroids[country];
        if (c) points.push([c.lat, c.lng]);
      }
    } else if ((phase === "final" || phase === "done") && consensusPin) {
      points.push([consensusPin.lat, consensusPin.lng]);
      if (groundTruth) points.push([groundTruth.lat, groundTruth.lng]);
    }

    // For the final reveal we still want a smooth animation; for live updates
    // during the pipeline we skip animation so simultaneous panning + state
    // updates don't desync the layers.
    const animateView = phase === "final" || phase === "done";

    if (points.length >= 2) {
      map.fitBounds(points, {
        padding: [60, 60],
        maxZoom: 6,
        animate: animateView,
        duration: animateView ? 1.2 : 0,
      });
    } else if (points.length === 1) {
      map.setView(points[0], 5, {
        animate: animateView,
        duration: animateView ? 1.2 : 0,
      });
    }
  }, [phase, agentVotesByCountry, countryCentroids, consensusPin, groundTruth, mapReady]);

  // Per-country agent dot clusters (one small coloured dot per voting agent,
  // arranged horizontally over the country centroid).
  useEffect(() => {
    const map = mapRef.current;
    const group = agentLayerRef.current;
    if (!map || !group) return;

    import("leaflet").then(({ default: L }) => {
      group.clearLayers();
      const colorById = Object.fromEntries(
        DEMO_AGENT_PROFILES.map((p) => [p.agentId, p.color])
      ) as Record<DemoAgentId, string>;
      const dim = phase === "final" || phase === "done";
      const opacity = dim ? 0.5 : 1;

      for (const [country, agents] of Object.entries(agentVotesByCountry)) {
        if (agents.length === 0) continue;
        const c = countryCentroids[country];
        if (!c) continue;

        const dotsHtml = agents
          .map(
            (a) => `<span style="
              width: 12px; height: 12px; border-radius: 50%;
              background: ${colorById[a] ?? "#888"};
              border: 1.5px solid rgba(255,255,255,0.9);
              box-shadow: 0 0 6px ${colorById[a] ?? "#888"}aa;
              display: inline-block;
            "></span>`
          )
          .join("");

        const totalWidth = agents.length * 14 + 6;
        const html = `<div style="
          display: flex; gap: 2px; padding: 3px 4px;
          background: rgba(0,0,0,0.55);
          border-radius: 999px; opacity: ${opacity};
          transition: opacity 0.4s;
          backdrop-filter: blur(2px);
        ">${dotsHtml}</div>`;

        const icon = L.divIcon({
          className: "",
          html,
          iconSize: [totalWidth, 22],
          iconAnchor: [totalWidth / 2, 11],
        });

        const tooltipText = `${country} — ${agents.join(", ")}`;
        L.marker([c.lat, c.lng], { icon })
          .addTo(group)
          .bindTooltip(tooltipText, {
            direction: "top",
            offset: [0, -8],
          });
      }
    });
  }, [agentVotesByCountry, countryCentroids, phase, mapReady]);

  // Consensus pin + ground-truth + line.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    import("leaflet").then(({ default: L }) => {
      if (consensusMarkerRef.current) {
        consensusMarkerRef.current.remove();
        consensusMarkerRef.current = null;
      }
      if (groundTruthMarkerRef.current) {
        groundTruthMarkerRef.current.remove();
        groundTruthMarkerRef.current = null;
      }
      if (consensusLineRef.current) {
        consensusLineRef.current.remove();
        consensusLineRef.current = null;
      }

      if (consensusPin) {
        const icon = L.divIcon({
          className: "",
          html: `<div style="
            width: 22px; height: 22px; border-radius: 50%;
            background: #c6ef38; border: 3px solid #fff;
            box-shadow: 0 0 16px rgba(198,239,56,0.9);
            animation: pulse 1.4s ease-in-out infinite;
          "></div>
          <style>
            @keyframes pulse {
              0%,100% { transform: scale(1); }
              50% { transform: scale(1.15); }
            }
          </style>`,
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        });
        consensusMarkerRef.current = L.marker([consensusPin.lat, consensusPin.lng], { icon })
          .addTo(map)
          .bindTooltip(`<b>Council guess: ${consensusPin.country}</b>`, {
            direction: "top",
            offset: [0, -10],
          });
      }

      if (groundTruth) {
        const icon = L.divIcon({
          className: "",
          html: `<div style="
            width: 18px; height: 18px; border-radius: 50%;
            background: #4ade80; border: 3px solid #fff;
            box-shadow: 0 0 10px rgba(74,222,128,0.7);
          "></div>`,
          iconSize: [18, 18],
          iconAnchor: [9, 9],
        });
        groundTruthMarkerRef.current = L.marker([groundTruth.lat, groundTruth.lng], { icon })
          .addTo(map)
          .bindTooltip(`<b>Actual: ${groundTruth.label}</b>`, {
            direction: "top",
            offset: [0, -10],
          });
      }

      if (consensusPin && groundTruth) {
        consensusLineRef.current = L.polyline(
          [
            [consensusPin.lat, consensusPin.lng],
            [groundTruth.lat, groundTruth.lng],
          ],
          { color: "#c6ef38", weight: 2, dashArray: "6 6", opacity: 0.8 }
        ).addTo(map);
      }
    });
  }, [consensusPin, groundTruth, mapReady]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 16,
        overflow: "hidden",
        border: "1px solid var(--border)",
        background: "#111",
      }}
    />
  );
}

function lookupCode(country: string | undefined): string | null {
  if (!country) return null;
  return COUNTRY_NAME_TO_CODE[country] ?? null;
}

function normaliseCode(id: unknown): string {
  // world-atlas country features have numeric ids ("724"). Some are not zero-padded.
  const raw = String(id ?? "");
  if (!raw) return "";
  return raw.padStart(3, "0");
}

function baseStyle(): import("leaflet").PathOptions {
  return {
    fillColor: "transparent",
    fillOpacity: 0,
    color: "rgba(255,255,255,0.08)",
    weight: 0.5,
  };
}
