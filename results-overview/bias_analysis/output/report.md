# Cross-Approach Geographic Bias

Pooled over 7 approaches, 3500 predictions (3500 with coordinates).

## Overall verdict

Primary test: **one-sample t-test on per-image mean errors** (n = 500 distinct images, each averaged over 7 approaches). Averaging per image before testing avoids treating the same image's repeated predictions as independent, which would understate the standard error.

- North/South: **strong north bias (p=0.0002)** (t=3.77)
- East/West: **no significant bias (p=0.7251)** (t=-0.35)
- Mean per-image signed error: +1.80° lat, -0.46° lng

_Naive pooled test (all 3500 rows, treats repeated images as independent — reported for reference only): N/S strong north bias (p=0.0000), E/W no significant bias (p=0.4235)._

- Mean haversine: 1535 km (median 421 km)

## Per-approach

| Approach | n | acc | mean lat err | mean lng err | N-bias p | E-bias p |
|---|--:|--:|--:|--:|--:|--:|
| VLM Progressive Narrowing with Parallel Hypotheses | 500 | 64.0% | +1.41° | +0.30° | 0.007 | 0.848 |
| VLM PN + PH + Tournament | 500 | 64.8% | +1.45° | -1.39° | 0.003 | 0.360 |
| VLM Tournament Only | 500 | 67.8% | +1.34° | -0.88° | 0.007 | 0.541 |
| VLM Hub and Spoke | 500 | 64.6% | +1.95° | -0.21° | 0.001 | 0.894 |
| VLM Global Context Reguess | 500 | 65.0% | +2.11° | -0.13° | 0.000 | 0.931 |
| VLM Debate | 500 | 63.0% | +1.92° | -0.64° | 0.000 | 0.675 |
| Baseline (single VLM) | 500 | 65.0% | +2.42° | -0.29° | 0.000 | 0.858 |

Positive lat error = prediction north of truth; positive lng error = prediction east of truth.
