"""Predict yards for upcoming games.

Finds the next unplayed week in the nflverse schedule, adds a placeholder row per active player
against that week's opponent, and runs the trained models on features built from prior games only.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from . import probs
from .data import load_all
from .features import TARGETS, build
from .train import MODELS_DIR

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
ID_COLS = ["player_id", "player_name", "player_display_name", "position", "position_group", "season", "week",
           "season_type", "team", "opponent_team"]


def next_week(season: int | None, week: int | None) -> tuple[pd.DataFrame, int, int]:
    """Unplayed games for the requested (or next) week."""
    sched = pd.read_csv(SCHEDULE_URL)
    sched = sched[sched["game_type"] == "REG"]
    unplayed = sched[sched["home_score"].isna()]
    if season is None:
        season = int(unplayed["season"].min())
    unplayed = unplayed[unplayed["season"] == season]
    if unplayed.empty:
        raise SystemExit(f"no unplayed regular-season games found for {season}")
    if week is None:
        week = int(unplayed["week"].min())
    games = unplayed[unplayed["week"] == week]
    if games.empty:
        raise SystemExit(f"no unplayed games in {season} week {week}")
    return games, season, week


def placeholders(raw: pd.DataFrame, games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    opp = {**dict(zip(games["home_team"], games["away_team"])), **dict(zip(games["away_team"], games["home_team"]))}
    # active = appeared this season or last, so retired/cut players drop out
    latest = raw[raw["season"] >= season - 1].drop_duplicates("player_id", keep="last")
    latest = latest[latest["team"].isin(opp)]
    ph = latest[ID_COLS].copy()
    ph["season"], ph["week"], ph["season_type"] = season, week, "REG"
    ph["opponent_team"] = ph["team"].map(opp)
    already_played = raw[(raw["season"] == season) & (raw["week"] == week)]["player_id"]
    return ph[~ph["player_id"].isin(already_played)]


def predict(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    games, season, week = next_week(season, week)
    raw = load_all(refresh=True)
    full = pd.concat([raw, placeholders(raw, games, season, week)], ignore_index=True)

    out = []
    for target in TARGETS:
        bundle = joblib.load(MODELS_DIR / f"{target}.joblib")
        df = build(full, target)
        df = df[(df["season"] == season) & (df["week"] == week) & df[target].isna()].copy()
        X = df.reindex(columns=bundle["features"])
        X[[c for c in X if c.startswith("pos_")]] = X[[c for c in X if c.startswith("pos_")]].fillna(0)
        df["pred_yards"] = np.clip(bundle["model"].predict(X), 0, None).round(1)
        df["recent_avg"] = df["baseline"].round(1)
        df["stat"] = target.replace("_yards", "")
        pbundle = joblib.load(MODELS_DIR / f"{target}_probs.joblib")
        df = df.join(probs.predict(pbundle["model"], df, pbundle["features"], target))
        keep = ["player_display_name", "position", "team", "opponent_team", "stat", "pred_yards", "recent_avg"]
        out.append(df[keep + [c for c in df if c.startswith("p_")]])
    res = pd.concat(out, ignore_index=True)
    res.attrs["label"] = f"{season} week {week}"
    return res.rename(columns={"player_display_name": "player", "opponent_team": "opp"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--stat", choices=[t.replace("_yards", "") for t in TARGETS], help="only this yard type")
    ap.add_argument("--player", help="case-insensitive name substring filter")
    ap.add_argument("--team", help="team abbreviation filter, e.g. KC")
    ap.add_argument("--top", type=int, default=15, help="rows per stat (default 15)")
    ap.add_argument("--csv", help="write all predictions to this path")
    args = ap.parse_args()

    res = predict(args.season, args.week)
    if args.csv:
        res.sort_values(["stat", "pred_yards"], ascending=[True, False]).to_csv(args.csv, index=False)
    if args.stat:
        res = res[res["stat"] == args.stat]
    if args.player:
        res = res[res["player"].str.contains(args.player, case=False, regex=False)]
    if args.team:
        res = res[(res["team"] == args.team.upper()) | (res["opp"] == args.team.upper())]
    print(f"Predictions for {res.attrs.get('label', '')}")
    for stat, g in res.groupby("stat", sort=False):
        print(f"\n== {stat} yards ==")
        g = g.sort_values("pred_yards", ascending=False).head(args.top).drop(columns="stat").dropna(axis=1, how="all")
        pcols = [c for c in g if c.startswith("p_")]
        g[pcols] = g[pcols].apply(lambda col: (col * 100).round().astype(int).astype(str) + "%")
        g = g.rename(columns={c: c.replace("p_", "") + "+" for c in pcols})
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
