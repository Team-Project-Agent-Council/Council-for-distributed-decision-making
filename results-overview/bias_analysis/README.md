# Cross-Approach Geographic Bias Analysis

Aggregates the predicted-vs-ground-truth geographic deviations across **all
approaches at once** to answer a question the per-approach evaluations cannot:
*does the council, as a whole, systematically pull its guesses in a particular
direction* (e.g. north, or toward Europe)?

Each approach's own eval package (`eval_pnt/geo.py` and friends) already reports
per-approach bias. This folder **pools** those deviations into a single combined
analysis plus per-approach breakdowns, and renders maps and a bearing rose over
the combined error field.

## What it measures

For every image it takes the ground-truth coordinate and the model's predicted
coordinate and derives:

- **Signed latitude error** (`pred − truth`): positive = prediction north of truth.
- **Signed longitude error** (wrapped to `[-180, 180]`): positive = east of truth.
- **Bearing** truth → prediction (0° = N, 90° = E).
- **Quadrant** (NE / NW / SE / SW).
- **Haversine distance** (km).
- **Country over/under-prediction** — per ground-truth country, the net of
  false-positives (predicted but wasn't the truth) minus false-negatives
  (was the truth but missed). Positive = over-predicted, negative =
  under-predicted.

Bias significance is a **one-sample t-test** against `H0: mean error = 0`,
separately for the north/south (latitude) and east/west (longitude) axes.

The **primary combined test** is run on **per-image mean errors**: each image's
signed error is averaged across the approaches that predicted it, giving one
independent observation per distinct image (n = 500). Because the pooled record
set repeats every image once per approach, a naive t-test over all 3500 rows
would treat those repeated measurements as independent and understate the
standard error; averaging per image first respects the repeated-measures
structure. The naive pooled test is still reported alongside, for reference.
Per-approach tests use that approach's 500 rows directly (no repetition, so no
clustering needed). The p-value uses `scipy.stats.t` when available, otherwise
an exact Student-t computation via the regularised incomplete beta function
(accurate at any df — not a normal approximation).

## Approaches included

The 6 councils (from `results-overview/<Approach>/council_run/`) plus the
single-model **baseline** (`vlm-baseline/results/`) — 7 runs of 500 images each.
`initial-approach` has no per-image `result.json` and is excluded. The
registry lives in [`sources.py`](sources.py); approaches whose directories are
missing are skipped with a warning.

Council `result.json` stores `coordinates` as a dict `{"lat","lng"}`; the
baseline stores it as a string `"lat,lng"`. The loader
([`loader.py`](loader.py)) handles both, and falls back to parsing the
`Coordinates:` line out of `country_result`. When a prediction has no valid
coordinate (or the `(0,0)` null-island sentinel), it is replaced with the
predicted country's centroid (`country_centroids.json`) so the record still
counts — matching the per-approach loaders.

## Running

```bash
# from the results-overview/ directory
python -m bias_analysis.run
# custom output dir
python -m bias_analysis.run --out /tmp/bias
```

Dependencies: `matplotlib`, `numpy`, `pycountry` (required); `scipy` and
`geopandas` are optional (the t-test falls back to a normal approximation and
the maps fall back to plain lat/lng / bar charts when they are absent). The
over/under-prediction choropleth additionally needs a Natural Earth admin-0
shapefile, found automatically at `../data/natural_earth/ne_110m_admin_0_countries.shp`
(under `results-overview/data/`, shared with the approaches) or via `$VLM_NATURAL_EARTH_PATH`. See
[`requirements.txt`](requirements.txt).

## Outputs (`output/`)

Every plot is written as **PNG + PDF + SVG** (same basename).

- `bias_metrics.json` — `combined` + `per_approach` metrics (n, accuracy,
  haversine/lat/lng stats, north/east bias tests, quadrant counts, and
  per-country TP/FP/FN confusion keyed by ISO alpha-3).
- `report.md` — headline verdict + per-approach table.
- `plots/bearing_rose_combined.*` — pooled bearing rose.
- `plots/bearing_rose_by_approach.*` — small-multiples, one polar per approach.
- `plots/error_distribution.*` — pooled lat / lng / haversine histograms.
- `plots/quadrant_bars.*` — NE/NW/SE/SW shares per approach + combined.
- `plots/error_map_combined.*` — pooled truth→prediction error vectors with
  the mean error vector highlighted.
- `plots/over_under_map_combined.*` — **over/under-prediction choropleth**
  pooled across all approaches. Per ground-truth country, net error volume
  `sign(FP − FN)·log(1+|FP − FN|)`: red = over-predicted (the pooled council
  names it more often than it should, net false-positive), blue =
  under-predicted (net miss / false-negative). Mirrors the approach-specific
  `world_map_error_bias` map.
- `plots/over_under_by_approach/<approach>.*` — the same over/under map for
  each approach individually.
