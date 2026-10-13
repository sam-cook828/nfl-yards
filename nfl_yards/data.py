"""Download weekly player stats from nflverse and cache them locally."""
from pathlib import Path

import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.csv"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_season(year: int, refresh: bool = False) -> pd.DataFrame | None:
    path = DATA_DIR / f"stats_player_week_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        df = pd.read_csv(URL.format(year=year), low_memory=False)
    except Exception as e:  # season not published yet
        print(f"skip {year}: {e}")
        return None
    df = df.drop(columns=[c for c in df.columns if c.endswith("_list") or c == "headshot_url"])
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(path)
    return df


def _cached(url: str, name: str, refresh: bool, **kw) -> pd.DataFrame | None:
    path = DATA_DIR / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        df = pd.read_csv(url, low_memory=False, **kw)
    except Exception as e:
        print(f"skip {name}: {e}")
        return None
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(path)
    return df


def load_snaps(start: int, end: int, refresh: bool = False) -> pd.DataFrame:
    """Offensive snap share per player-week, keyed by gsis id (the id used in the stats files)."""
    ids = _cached(PLAYERS_URL, "players", refresh, usecols=["gsis_id", "pfr_id"]).dropna()
    frames = [f for y in range(start, end + 1)
              if (f := _cached(SNAP_URL.format(year=y), f"snap_counts_{y}", refresh,
                               usecols=["season", "week", "game_type", "pfr_player_id", "offense_pct"])) is not None]
    snaps = pd.concat(frames, ignore_index=True)
    snaps = snaps[snaps["game_type"] == "REG"].merge(ids, left_on="pfr_player_id", right_on="pfr_id")
    return (snaps.groupby(["gsis_id", "season", "week"], as_index=False)["offense_pct"].max()
            .rename(columns={"gsis_id": "player_id"}))


def load_all(start: int = 2016, end: int = 2026, refresh: bool = False) -> pd.DataFrame:
    frames = [f for y in range(start, end + 1) if (f := load_season(y, refresh)) is not None]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"].sort_values(["season", "week"]).reset_index(drop=True)
    df = df.merge(load_snaps(start, end, refresh), on=["player_id", "season", "week"], how="left")
    df["carry_share"] = df["carries"] / df.groupby(["season", "week", "team"])["carries"].transform("sum")
    return df
