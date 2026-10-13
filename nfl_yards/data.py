"""Download weekly player stats from nflverse and cache them locally."""
from pathlib import Path

import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
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


def load_all(start: int = 2016, end: int = 2026, refresh: bool = False) -> pd.DataFrame:
    frames = [f for y in range(start, end + 1) if (f := load_season(y, refresh)) is not None]
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"].sort_values(["season", "week"]).reset_index(drop=True)
