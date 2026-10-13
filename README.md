# nfl-yards

Predict NFL players' weekly passing, rushing and receiving yards from nflverse player stats.

```
uv sync
uv run python -m nfl_yards.train   # downloads data, trains, writes models/report.json
```

Features are leak-free rolling averages (3-game, 8-game, career) plus opponent yards allowed.
Models are gradient-boosted trees evaluated against an 8-game rolling-average baseline (see `models/report.json`).
