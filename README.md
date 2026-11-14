# FAA outage attribution forecaster

Entry for make.fun competition `polymarket-faa-attribution-01`.

`main.py` (Python 3.13, standard library only) reads one Polymarket case at a
time — every trading market of the FAA outage attribution family, each with
its Yes price at every minute of the 72 hours up to a checkpoint — and writes,
for each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price. It runs offline in
the sandbox: no network, no model calls, fully deterministic — two runs on
the same input produce byte-identical output.

## Method

- **Median = last price.** A prediction-market price is close to a martingale,
  so the price at the checkpoint is the honest median. A drift term (EWMA of
  recent 1-minute changes, any half-life / shrink / cap) was tested and
  rejected: every variant raised the practice pinball loss and the median
  error.
- **Spread = recent volatility.** Sigma is an exponentially weighted mean of
  squared 1-minute changes (half-life 20 minutes), clamped to
  [0.0008, 0.008]. Volatility clusters, so a recently quiet book gets a narrow
  spread and a recently jumpy one a wide spread; the cap stops the estimate
  staying huge for hours after a jump. A long-window blend and an extra jump
  allowance were both tested and rejected: they inflate the spread from the
  markets' wild early minutes and cost far more on quiet minutes than they
  save on jumps.
- **Quantiles:** `q_tau(h) = clamp(x + Z_tau * sigma * sqrt(h), 0, 1)` with
  `Z = [-1.28, -0.5, 0, 0.5, 1.28]` — narrower than the normal's, because
  these books move in small steps most minutes.

## Practice score (tuned, not guessed)

Scored locally against Polymarket's own recorded prices for the six practice
cases (NVIDIA "Next CEO" event), with the same pinball loss the contest uses:

| program | pinball loss | median error | 90% coverage |
| --- | --- | --- | --- |
| house baseline (starter) | 0.363589 | 0.767503 | 0.949642 |
| this entry | **0.280473** | 0.767503 | 0.936393 |

About 23% lower loss than the baseline. The loss surface is flat across the
tuning plateau (half-life 10–30, z95 1.15–1.40, cap 0.006–0.010 all score
0.275–0.29), so middle-of-plateau values were picked rather than the point
optimum — the final's FAA markets are news-driven and jumpier than the
practice event, and the wider end of the plateau keeps more tail cover.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads `input/` (one JSON case per market family) and writes one
`.out` JSON file per case into `out/`, each with the five quantiles for all
240 forecast minutes. Runtime is about 1 second for six cases.

## License

MIT — see `LICENSE`.
