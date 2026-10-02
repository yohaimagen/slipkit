# Mw 7.6 two-track InSAR synthetic revision

## Definitive case

The reproducible case is:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round3-mw76-two-track/mw76-dtp9yawu
```

It uses the existing 300 km by 25 km vertical right-lateral fault with 594
triangular strike-slip parameters. The fault is assigned an east-west strike.

The true slip contains three separated elliptical Gaussian asperities centered
at along-strike/depth coordinates `(-92, 7)`, `(-5, 13)`, and `(82, 8)` km.
Their anchor slips are exactly 5, 4, and 3 m. Slip in the two intervening
valleys is approximately 0.5-0.7 m. The area-weighted mean slip is
1.4054567 m and the moment is exactly `3.162277660168379e20 N m`, or Mw 7.6.

## Observations

The case simulates two right-looking SAR images:

| Track | Heading | Incidence | Fault-local ground-to-satellite LOS |
|---|---:|---:|---|
| Ascending | 347 degrees | 34 degrees | `[-0.544861, 0.125791, 0.829038]` |
| Descending | 193 degrees | 34 degrees | `[0.544861, 0.125791, 0.829038]` |

The same adaptive quadtree selects 10,003 surface locations in each image.
This produces 20,006 scalar LOS observations. A shared site-level split gives
16,006 training observations and 4,000 held-out observations. Independent
Gaussian noise has a 1 cm standard deviation.

## MAP smoothing scan

All solutions use bounds from 0 to `max(true slip) + 2 = 7.0057567 m`.
The smoothing precision multiplies unweighted slip differences between
triangles that share an edge.

| Precision (1/m) | Holdout RMS (m) | Slip RMS error (m) | Recovered Mw | Edge roughness (m) |
|---:|---:|---:|---:|---:|
| 0 | 0.010144 | 0.8901 | 7.60033 | 1.3678 |
| 1 | 0.010118 | 0.1449 | 7.59983 | 0.3069 |
| 3 | **0.010107** | **0.1037** | 7.60227 | 0.2541 |
| 10 | 0.010185 | 0.1655 | 7.60868 | 0.2218 |
| 30 | 0.011553 | 0.2655 | 7.61960 | 0.1809 |
| 100 | 0.019562 | 0.3977 | 7.62437 | 0.1347 |

The true shared-edge RMS is 0.2705 m. The held-out data select a precision of
3 1/m from this predeclared scan. Its slip correlation with truth is 0.9960,
and it recovers the three anchor slips as 5.178, 3.670, and 2.953 m.

The independent bounded model still fits the LOS data but produces a rough,
nonphysical slip map. The second viewing geometry improves the data, while a
weak spatial prior remains necessary at this mesh resolution.

## Outputs

```text
scenario_and_recovery.png
smoothing_scan.json
recovery_diagnostics.json
map-s0 through map-s100/map_report.json
```

The current native AlTar CPU/CUDA path still samples the independent bounded
uniform prior. The shared-edge prior in this revision is implemented for the
MAP baseline only. A full posterior run should follow after the bounded smooth
prior is represented explicitly in the native target density.
