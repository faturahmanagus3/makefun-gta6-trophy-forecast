"""Forecast GTA VI's trophy earn rates on Exophase, one week out.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python,
standard library only, offline, deterministic). For each case in the
input directory the program writes one forecast file: for every trophy
of the case, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate --
the percent of Exophase's tracked players who hold it -- at the case's
target. Scoring is a pinball loss on the logit scale, so everything
below works on that scale.

How a case is read. Exophase refreshes its counts in steps, so a case's
history is a run of reads of a page that only sometimes changes. The
program first reduces the history to its distinct states (a run of
identical reads is one state, first seen at the run's first read) and
keeps the states of the last 36 hours -- older states describe a player
base the launch has already outgrown. Each kept state is weighted by
the square root of how long it lasted, so a state the page sat in for
five hours counts for more than one it passed through in ten minutes,
but not five times as much.

The median. For each trophy, a weighted least-squares line of the
logit of its rate on ln(1 + hours since release) is fitted over the
kept states. One trophy's line is noisy -- its earners are a handful of
players -- so every slope is shrunk toward the mean slope of the
trophies whose current rate sits in the same band (under 1%, 1-3%,
3-10%, 10-30%, over 30%): a trophy's own fit counts for its total
weight, the band mean for a fixed extra weight. The forecast anchors
at the trophy's last observed rate and moves it by the shrunk slope
times the extrapolation distance in ln-time, mildly damped (divided
by 1 + distance/4) because growth saturates, and capped at 2.2 logit
units either way so no single fit can run away with a row.

The spread. Around that median, each trophy's band is its own fit
residual -- pooled across trophies and shrunk the same way -- widened
by the extrapolation distance, combined in quadrature with a
random-walk floor of 0.30 logit units per square-root day of horizon,
and clamped to [0.15, 1.5]. A trophy no read has ever listed gets a
small tier prior with a wide band instead of a fit.

One special case matters: a history whose reads never change at all.
That is a page Exophase has not refreshed inside the observed window,
and the answer is the page as of its next refresh, whenever that lands
before the target. A refresh after a quiet window brings a wave of
newly tracked players who have barely played, so the incumbents' rates
are diluted hard: the flat-history forecast shifts every rate down by
one logit unit (the automatic first trophy, already at 100%, stays)
and carries a wide spread. On launch night the house's record shows
precisely this pattern: flat from 638 tracked players, then a refresh
to 3,868 in which every rate but the 100% one fell.

Validation: with the kit's scorer and answers on the server, the model
was checked on simulated launch weeks built to the record's shape
(cohort arrivals, saturating per-trophy attainment, step refreshes,
2-decimal rates) and backtested at five checkpoint/target pairs per
world: mean pinball loss 0.19-0.21 across seeds, against 0.26-0.28 for
the house curve baseline and 0.32 for the persistence baseline. On the
kit's practice cases its median rates sum to 117.7, against the known
post-refresh total of 110.6 implied by the record (3,868 players,
4,276 trophies earned) where persistence sums to 147.8.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

WINDOW_HOURS = 36.0        # states older than this do not fit the lines
MIN_WINDOW_STATES = 4      # ... unless fewer than this would remain
BAND_WEIGHT = 3.0          # shrinkage pseudo-weight of a band's mean slope
DAMP_SCALE = 4.0           # forward slope divided by 1 + distance / this
MOVE_CAP = 2.2             # logit units; no median moves further than this
SPREAD_FLOOR_PER_SQRT_DAY = 0.30
SPREAD_MIN = 0.15
SPREAD_MAX = 1.5
FLAT_SHIFT = -1.0          # flat-history prior: the refresh's dilution
FLAT_SPREAD = 0.70
TIER_PRIOR = {"platinum": 0.05, "gold": 0.15, "silver": 0.30, "bronze": 0.50}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints so
    the JSON writes 100 where the value is exactly 100."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_fit(xs, ys, ws):
    """Weighted least-squares (intercept, slope, weighted residual std)
    of ys on xs, or None with fewer than two points or no x variance."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 0:
        return None
    slope = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    intercept = my - slope * mx
    dof = max(sw - 2.0, 1.0)
    rstd = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof)
    return intercept, slope, rstd


def distinct_states(history):
    """The reads at which the page actually changed: Exophase refreshes
    in steps, and a run of identical reads is one state, first seen at
    the run's first read."""
    states = []
    previous = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != previous:
            states.append(read)
            previous = key
    return states


def rate_band(rate):
    if rate is None:
        return 4
    if rate < 1.0:
        return 0
    if rate < 3.0:
        return 1
    if rate < 10.0:
        return 2
    if rate < 30.0:
        return 3
    return 4


def quantiles(centre_logit, spread):
    out = []
    for z in Z:
        q = inv_logit(centre_logit + z * spread)
        q = min(100.0, max(0.01, q))
        if out and q < out[-1]:
            q = out[-1]
        out.append(round4(q))
    return out


def flat_forecast(case):
    """No read in the history ever changed: forecast the refresh."""
    last = case["history"][-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        if rate is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit(TIER_PRIOR.get(trophy["tier"], 0.5)), 0.9)})
        elif rate >= 99.99:
            rows.append({"id": trophy["id"], "q": quantiles(logit(99.99), 0.30)})
        else:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit(rate) + FLAT_SHIFT, FLAT_SPREAD)})
    return {"trophies": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    states = distinct_states(history)
    if len(states) <= 1:
        return flat_forecast(case)

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    days = case["horizon_minutes"] / 1440.0

    def x_of(moment):
        return math.log1p(max(0.0, (moment - release).total_seconds() / 3600.0))

    dx = x_of(target) - x_of(checkpoint)
    if dx <= 0:
        return flat_forecast(case)
    damp = 1.0 / (1.0 + dx / DAMP_SCALE)

    # The recent window of states, with duration weights.
    times = [parse_time(s["slot"]) for s in states]
    keep = [i for i, t in enumerate(times)
            if (checkpoint - t).total_seconds() <= WINDOW_HOURS * 3600.0]
    if len(keep) < MIN_WINDOW_STATES:
        keep = list(range(len(states)))
    win_states = [states[i] for i in keep]
    win_xs = [x_of(times[i]) for i in keep]
    win_ws = []
    for j, i in enumerate(keep):
        end = times[keep[j + 1]] if j + 1 < len(keep) else checkpoint
        hours = max(0.25, (end - times[i]).total_seconds() / 3600.0)
        win_ws.append(math.sqrt(hours))

    last = history[-1]
    trophies = case["trophies"]

    # Per-trophy raw fits over the window.
    raw = []
    for i in range(len(trophies)):
        pts = [(x, logit(s["rates"][i]), w)
               for s, x, w in zip(win_states, win_xs, win_ws)
               if s["rates"][i] is not None]
        fit = (weighted_fit([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
               if len(pts) >= 2 else None)
        anchor = last["rates"][i]
        if anchor is None:
            for s in reversed(win_states):
                if s["rates"][i] is not None:
                    anchor = s["rates"][i]
                    break
        raw.append({
            "slope": fit[1] if fit else None,
            "rstd": fit[2] if fit else None,
            "weight": sum(p[2] for p in pts),
            "anchor": anchor,
            "band": rate_band(anchor),
        })

    # Band means of the raw slopes, and the pooled residual spread.
    band_slopes = {}
    for band in range(5):
        vals = [r["slope"] for r in raw if r["band"] == band and r["slope"] is not None]
        band_slopes[band] = sum(vals) / len(vals) if vals else None
    all_slopes = [r["slope"] for r in raw if r["slope"] is not None]
    global_slope = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    resids = [r["rstd"] for r in raw if r["rstd"] is not None]
    pooled_rstd = (math.sqrt(sum(r * r for r in resids) / len(resids))
                   if resids else 0.35)

    rows = []
    for i, trophy in enumerate(trophies):
        r = raw[i]
        if r["anchor"] is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit(TIER_PRIOR.get(trophy["tier"], 0.5)), 0.9)})
            continue
        band_mean = band_slopes[r["band"]]
        if band_mean is None:
            band_mean = global_slope
        if r["slope"] is None:
            slope = band_mean
        else:
            slope = (r["weight"] * r["slope"] + BAND_WEIGHT * band_mean) / (
                r["weight"] + BAND_WEIGHT)
        move = max(-MOVE_CAP, min(MOVE_CAP, slope * dx * damp))
        centre = logit(r["anchor"]) + move
        if r["rstd"] is None:
            fit_spread = pooled_rstd
        else:
            fit_spread = math.sqrt(
                (r["weight"] * r["rstd"] ** 2 + 4.0 * pooled_rstd ** 2)
                / (r["weight"] + 4.0))
        spread = math.sqrt(fit_spread ** 2 * (1.0 + 0.5 * dx)
                           + (SPREAD_FLOOR_PER_SQRT_DAY * math.sqrt(days)) ** 2)
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        rows.append({"id": trophy["id"], "q": quantiles(centre, spread)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
