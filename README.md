# makefun-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/faturahmanagus3/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/faturahmanagus3/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A quantile forecaster for GTA VI's trophy earn rates on Exophase. Given a case holding every read the house made of the game's trophy page up to a checkpoint, it forecasts each of the 55 trophies' earn rate one week after launch as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale. The program reduces the history to the page's distinct states (Exophase refreshes in steps), fits each trophy a duration-weighted least-squares line of logit(rate) on ln(1 + hours since release) over the last 36 hours of states, shrinks every slope toward the mean slope of trophies in the same rate band, anchors the forecast at the last observed rate, and damps and caps the move. The spread is each trophy's own fit residual, pooled and shrunk, widened by the extrapolation distance over a random-walk floor. A history that never changes is a page waiting for a refresh: the forecast then shifts every rate down a logit unit for the dilution a wave of newly tracked players brings, with a wide band. Standard library only, deterministic, offline. On simulated launch weeks built to the record's shape (cohort arrivals, saturating per-trophy attainment, step refreshes), backtested at five checkpoint/target pairs per world across three seeds, it scores a mean pinball loss of 0.19-0.21 against the house curve baseline's 0.26-0.28 and the persistence baseline's 0.32-0.35; on the kit's practice cases its median rates sum to 117.7 against the known post-refresh total of 110.6, where persistence sums to 147.8.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dontol&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
