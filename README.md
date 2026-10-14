# nfl-yards

Predict NFL players' weekly **passing, rushing and receiving yards**, and the **chance of reaching yardage
thresholds** (e.g. 60+ receiving yards), from public nflverse data.

## Quick start

```
uv sync
uv run python -m nfl_yards.train     # downloads data, trains all models, writes models/report.json
uv run python -m nfl_yards.predict   # predictions for the next unplayed week
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Trained models (`models/*.joblib`) and downloaded
data (`data/`) are not committed, so run `train` first.

## Predicting upcoming games

```
uv run python -m nfl_yards.predict                          # next unplayed week, top 15 per stat
uv run python -m nfl_yards.predict --team KC --stat passing
uv run python -m nfl_yards.predict --player mahomes --week 7
uv run python -m nfl_yards.predict --csv preds.csv          # write every player to a file
```

Each run re-downloads the latest stats and schedule, so features reflect all completed games. Output per player:
the point prediction (`pred_yards`), their recent 8-game average (`recent_avg`), and the probability of reaching
each threshold:

| Yard type | Thresholds |
|---|---|
| Rushing, receiving | 25, 40, 50, 60, 70, 80, 90, 100 |
| Passing | 125, 150, 175, 200, 225, 250, 275, 300, 325, 350 |

To change thresholds, edit `THRESHOLDS` in `nfl_yards/probs.py` and run `uv run python -m nfl_yards.train --probs-only`.

## Data

All data comes from [nflverse](https://github.com/nflverse), downloaded as CSV and cached as parquet in `data/`.

| Source | Used for |
|---|---|
| `stats_player_week_{year}.csv` (`stats_player` release) | Per-player, per-game box score stats: yards, attempts, carries, targets, EPA, air yards, target share, WOPR, etc. Regular season only, 2016 to present. |
| `snap_counts_{year}.csv` (`snap_counts` release) | Offensive snap share per player-game. Joined to the stats via `players.csv`, which maps PFR ids to GSIS ids. |
| `games.csv` (`nfldata` repo) | The schedule. Used only by `predict` to find the next unplayed week and each team's opponent. |

Each row is one player in one game. Only players with meaningful recent usage are modeled: a prior 8-game
average of at least 10 pass attempts (passing), 3 carries (rushing) or 2 targets (receiving).

## How the models work

Every yard type (passing, rushing, receiving) gets its own pair of models.

### Features (`nfl_yards/features.py`)

All features use only games **before** the one being predicted, so there is no leakage from the game itself.

- **Player form:** rolling 3-game and 8-game averages and the career average of yards and of volume (attempts,
  carries or targets), plus yard-type-specific stats such as EPA, air yards, first downs, receptions and sacks.
- **Role and usage:** snap share, target share and carry share (the player's fraction of his team's carries),
  each as rolling averages.
- **Consistency:** the last game's yards, the 8-game standard deviation, and games played.
- **Opponent:** rolling 3- and 8-game average of that yard type allowed by the opposing defense.
- **Other:** position and week of the season.

### Yardage regressor (`nfl_yards/train.py`)

A scikit-learn `HistGradientBoostingRegressor` with absolute-error loss, which predicts the **median** outcome and
minimizes mean absolute error (MAE). Because yardage is right-skewed, the median sits below the average, which is
most noticeable for running backs and low-volume receivers.

### Threshold probabilities (`nfl_yards/probs.py`)

A `HistGradientBoostingClassifier` per yard type. Every player-game is repeated once per threshold, with a label of
whether the player reached it, and the threshold (and its gap to the player's 8-game average) is added as a feature.
Both threshold features have a decreasing monotonic constraint, so P(60+) can never exceed P(50+).

### Training and evaluation

Splits are by time: seasons before 2024 train, 2024 picks the number of boosting rounds, and the final models are
refit on both. Seasons after 2024 (2025 and the 2026 weeks played so far) are the held-out test set.

**Yardage regressors vs. a baseline that predicts the player's 8-game average (test MAE, in yards):**

| Yard type | Model | Baseline | Improvement |
|---|---|---|---|
| Passing | 64.5 | 70.2 | 8.0% |
| Rushing | 20.5 | 22.5 | 8.9% |
| Receiving | 20.2 | 21.8 | 7.5% |

**Threshold probabilities vs. the historical base rate for each threshold (log loss, lower is better):**

| Yard type | Model | Base rate |
|---|---|---|
| Passing | 0.456 | 0.518 |
| Rushing | 0.379 | 0.481 |
| Receiving | 0.366 | 0.442 |

The probabilities are well calibrated: across the test set, outcomes predicted at 30% happen about 25-35% of the
time, and similarly across other probability ranges. Full metrics, including the calibration tables, are in
`models/report.json`.

## Limitations

- Weekly yardage is noisy. The gain over a rolling average is modest, and individual predictions can miss widely.
- The model does not see injuries, depth charts, Vegas lines, weather or home/away, so a backup who is about to
  start, or a star who is out, will be mispredicted.
- `predict` uses the saved models and does not retrain. Retrain after new seasons accumulate
  (`uv run python -m nfl_yards.train --val-season <year>`).
- The yardage regressor and the probability models are trained separately, so they can disagree slightly.

## Layout

```
nfl_yards/data.py       download and cache nflverse data, merge snap counts
nfl_yards/features.py   leak-free rolling features and opponent strength
nfl_yards/train.py      train regressors and probability models, write models/report.json
nfl_yards/probs.py      threshold-probability model
nfl_yards/predict.py    predictions for upcoming games
```
