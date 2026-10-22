"""Download and clean match results from football-data.co.uk.

One raw CSV per (league, season). Cleaning renames columns to snake_case, parses
dates, drops blank or incomplete rows, and gives every match a stable id.
"""

from __future__ import annotations

import ssl
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import certifi
import pandas as pd

from soccer_predict import config

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Raw column -> clean column. Only these columns are kept.
COLUMNS: dict[str, str] = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    # Bet365 closing odds. Used only for the bookmaker reference baseline, never as features.
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}

REQUIRED = [c for c in COLUMNS.values() if not c.startswith("odds_")]
STAT_COLUMNS = [c for c in REQUIRED if c not in {"date", "home_team", "away_team", "result"}]
SOT_COLUMNS = ["home_shots_on_target", "away_shots_on_target"]
RESULTS = ("H", "D", "A")

# football-data.co.uk changed its Premier League shots-on-target source in 2013/14:
# the on-target share of all shots drops from about 55% to 33% overnight while the
# other leagues stay flat. Seasons before the break are rescaled so the column
# means the same thing everywhere. league -> first season on the new definition.
SOT_BREAKS: dict[str, str] = {"E0": "1314"}


def raw_path(league: str, season: str, raw_dir: Path = config.RAW_DIR) -> Path:
    return raw_dir / f"{league}_{season}.csv"


def download(
    leagues: Iterable[str] = config.LEAGUES,
    seasons: Iterable[str] = config.SEASONS,
    raw_dir: Path = config.RAW_DIR,
    force: bool = False,
) -> list[Path]:
    """Fetch raw CSVs into raw_dir, skipping files that already exist."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    # certifi's CA bundle: python.org builds on macOS ship without system certificates.
    context = ssl.create_default_context(cafile=certifi.where())
    paths = []
    for season in seasons:
        for league in leagues:
            path = raw_path(league, season, raw_dir)
            if force or not path.exists():
                url = BASE_URL.format(season=season, league=league)
                with urllib.request.urlopen(url, timeout=30, context=context) as response:
                    path.write_bytes(response.read())
            paths.append(path)
    return paths


def load_raw(
    leagues: Iterable[str] = config.LEAGUES,
    seasons: Iterable[str] = config.SEASONS,
    raw_dir: Path = config.RAW_DIR,
) -> pd.DataFrame:
    """Concatenate raw CSVs and tag each row with its league and season."""
    frames = []
    for season in seasons:
        for league in leagues:
            path = raw_path(league, season, raw_dir)
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing. Run `soccer download` first.")
            frame = pd.read_csv(
                path,
                encoding="latin-1",
                usecols=lambda col: col in COLUMNS,
                dtype=str,
            )
            frame["league"] = league
            frame["season"] = season
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse dd/mm/yy and dd/mm/yyyy (both appear across seasons)."""
    long_form = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    short_form = pd.to_datetime(values, format="%d/%m/%y", errors="coerce")
    return long_form.fillna(short_form)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one tidy row per completed match, sorted by date."""
    df = raw.rename(columns=COLUMNS).copy()
    for col in COLUMNS.values():
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(subset=REQUIRED)
    df["date"] = parse_dates(df["date"])
    df = df.dropna(subset=["date"])
    df = df[df["result"].isin(RESULTS)]

    for col in STAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=STAT_COLUMNS)
    df[STAT_COLUMNS] = df[STAT_COLUMNS].astype(int)
    df = harmonize_shots_on_target(df)
    for col in ("odds_home", "odds_draw", "odds_away"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    df["split"] = df["season"].map(config.split_for_season)

    df = df.sort_values(["date", "league", "home_team"], kind="stable").reset_index(drop=True)
    df["match_id"] = (
        df["league"]
        + "_"
        + df["season"]
        + "_"
        + df["date"].dt.strftime("%Y%m%d")
        + "_"
        + df["home_team"].str.replace(" ", "")
    )
    if df["match_id"].duplicated().any():
        raise ValueError("match_id is not unique; check the raw files for duplicate rows")

    ordered = ["match_id", "league", "season", "split", *COLUMNS.values()]
    return df[ordered]


def on_target_share(df: pd.DataFrame) -> float:
    shots = df["home_shots"].sum() + df["away_shots"].sum()
    return float((df["home_shots_on_target"].sum() + df["away_shots_on_target"].sum()) / shots)


def harmonize_shots_on_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale shots on target before a known definition change (see SOT_BREAKS).

    The factor is the on-target share after the break divided by the share
    before it, measured on training seasons only so no test data is used.
    """
    df = df.copy()
    df[SOT_COLUMNS] = df[SOT_COLUMNS].astype(float)
    for league, first_new in SOT_BREAKS.items():
        in_league = df["league"] == league
        before = in_league & (df["season"] < first_new)
        after = in_league & (df["season"] >= first_new) & df["season"].isin(config.TRAIN_SEASONS)
        if not before.any() or not after.any():
            continue
        factor = on_target_share(df[after]) / on_target_share(df[before])
        df.loc[before, SOT_COLUMNS] = df.loc[before, SOT_COLUMNS] * factor
    return df
