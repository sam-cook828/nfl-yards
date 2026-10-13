# nfl-yards

Predict NFL players' weekly passing, rushing and receiving yards from nflverse player stats.

```
uv sync
uv run python -m nfl_yards.train   # downloads data, trains, writes models/report.json
```

Features are leak-free rolling averages (3-game, 8-game, career) plus opponent yards allowed.
Models are gradient-boosted trees evaluated against an 8-game rolling-average baseline (see `models/report.json`).

## Predict upcoming games

```
uv run python -m nfl_yards.predict                       # next unplayed week, top 15 per stat
uv run python -m nfl_yards.predict --team KC --stat passing
uv run python -m nfl_yards.predict --player mahomes --week 7
uv run python -m nfl_yards.predict --csv preds.csv       # all players
```

Re-downloads current stats and the nflverse schedule each run, so predictions use the latest completed games.
Run `train` first to create `models/*.joblib` (they are not committed).

## Threshold probabilities

`predict` also prints the chance of reaching yardage thresholds (receiving/rushing 25, 40, 50, 60, 70, 80, 90, 100;
passing 125, 150, ... 350). One classifier per yard type is trained on rows stacked across thresholds with a
monotonic constraint, so probabilities never increase as the threshold rises. Edit `THRESHOLDS` in
`nfl_yards/probs.py` to change the cutoffs, then `uv run python -m nfl_yards.train --probs-only`.
Test-set log loss and calibration are in `models/report.json`.
