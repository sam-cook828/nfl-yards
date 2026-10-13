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
