#!/usr/bin/env python3
"""Forecast Polymarket Yes prices minute by minute, 4 hours ahead.

Entry for make.fun competition polymarket-faa-attribution-01.

Program: python3 /bot/main.py (Python 3.13, standard library only).
Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"forecasts": [{"id": ..., "q": [[5 quantiles] x 240]}]}.

Method (per market, from its own 72h of 1-minute prices only):
  - Persistence median: the price at the checkpoint (a prediction-market price
    is close to a martingale, so the last price is the honest median).
    A drift term was tested and REJECTED: every drift variant (EWMA of recent
    changes, any half-life/shrink/cap) raised the practice pinball loss and
    the median error, so the median stays exactly at the checkpoint price.
  - Spread: per-minute volatility sigma from an exponentially weighted mean
    of squared 1-minute changes (half-life 20 minutes over the full history).
    Volatility clusters, so a recently quiet book gets a narrow spread and a
    recently jumpy one a wide spread. The estimate is clamped to
    [0.0008, 0.008]: the cap matters — after a jump the raw estimate would
    stay huge for hours and the forecast would pay at every quiet minute.
    A long-window blend and an extra jump allowance were both tested and
    REJECTED: they inflate the spread from the markets' wild early minutes
    and cost far more on quiet minutes than they save on jumps.
  - Quantiles: q_tau(h) = clamp(x + Z_TAU * sigma * sqrt(h), 0, 1), with
    Z = [-1.28, -0.5, 0, 0.5, 1.28]. These are NARROWER than the normal's
    (-1.645, -0.674): on the practice cases the books move in small steps
    most minutes, so honest quantiles sit closer in. Tuned, not guessed:
    on the six practice cases (scored against Polymarket's own recorded
    prices) this scores a pooled pinball loss of 0.2805, median error
    0.7675 and 90% coverage 0.936, vs the house baseline's 0.3636 / 0.7675 /
    0.9496 — about 23% lower loss. The loss surface is flat across the
    plateau (HL 10-30, z95 1.15-1.40, cap 0.006-0.010 all score 0.275-0.29),
    so these middle-of-plateau values were picked rather than the point
    optimum, to avoid overfitting one quiet event: the final's FAA markets
    are news-driven and jumpier, and the wider end of the plateau (z95 1.28,
    cap 0.008) keeps more tail cover.
  - Deterministic: no randomness, no network, no model calls. Fast: O(n)
    per market, far inside the 8-second cap.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Heavy-tailed standard quantiles: NARROWER than the normal's (-1.645, -0.674).
# Tuned on the practice cases (see module docstring): the books move in small
# steps most minutes, so honest quantiles sit closer in than a normal's.
Z = [-1.28, -0.5, 0.0, 0.5, 1.28]

# --- tunable parameters (tuned on the practice cases, see module docstring) ---
SIG_HL = 20.0         # half-life (min) of the EWMA variance
SIG_WIN = 4320        # minutes of history used for the variance (full case)
SIG_PRIOR = 0.0025    # prior sigma (house baseline) — unused, weight 0 (tested)
SIG_PRIOR_W = 0.0     # weight of the prior variance in the blend (0 = off)
SIG_MIN = 0.0008      # sigma floor
SIG_MAX = 0.008       # sigma cap (tested: 0.006-0.010 all score alike; 0.008 keeps tail cover)
DRIFT_HL = 60.0       # (drift tested and rejected — see module docstring)
DRIFT_WIN = 360
DRIFT_SHRINK = 0.0    # 0 = no drift: median stays at the checkpoint price
DRIFT_CAP = 0.0003


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def ewma_stats(diffs, half_life, window):
    """EWMA mean and variance of the last `window` diffs.

    Returns (mean, var). Newest diff last; weight decays with `half_life`.
    """
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0
    take = diffs[-window:] if window and n > window else diffs
    lam = 0.5 ** (1.0 / half_life)
    w_sum = 0.0
    w_mean = 0.0
    w_sq = 0.0
    w = 1.0
    for d in reversed(take):
        w_sum += w
        w_mean += w * d
        w_sq += w * d * d
        w *= lam
    mean = w_mean / w_sum
    var = w_sq / w_sum
    return mean, var


def forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market."""
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Spread: EWMA variance blended with the prior.
    _, var = ewma_stats(diffs, SIG_HL, SIG_WIN)
    var = (1.0 - SIG_PRIOR_W) * var + SIG_PRIOR_W * SIG_PRIOR * SIG_PRIOR
    sigma = math.sqrt(var)
    sigma = min(SIG_MAX, max(SIG_MIN, sigma))

    # Drift: shrunk, capped EWMA of recent changes.
    drift, _ = ewma_stats(diffs, DRIFT_HL, DRIFT_WIN)
    drift = max(-DRIFT_CAP, min(DRIFT_CAP, DRIFT_SHRINK * drift))

    rows = []
    for h in range(1, horizon + 1):
        s = sigma * math.sqrt(h)
        centre = x + drift * h
        row = [price6(centre + z * s) for z in Z]
        # Enforce non-decreasing (clamping at 0/1 can only keep order, but
        # be explicit so the output is always valid).
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[:-3]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(
                "%s asks for quantiles %s, not %s"
                % (fname, case["quantiles"], QUANTILES)
            )
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
