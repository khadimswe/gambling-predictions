"""Synthetic fixtures so tests never need the network or the real dataset."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from soccer_predict import config

STRENGTH = {"Alpha": 3, "Bravo": 2, "Charlie": 1, "Delta": 0}


def make_matches(
    seasons: tuple[str, ...] = ("0809", "0910"),
    teams: dict[str, int] = STRENGTH,
    league: str = "E0",
    rounds: int = 3,
) -> pd.DataFrame:
    """Round-robin seasons where the stronger side wins, except close games in odd rounds draw."""
    rows = []
    date = pd.Timestamp("2008-08-02")
    for season in seasons:
        for round_no in range(rounds):
            for home, away in itertools.permutations(teams, 2):
                gap = teams[home] - teams[away]
                if round_no % 2 == 1 and abs(gap) == 1:
                    gap = 0
                home_goals, away_goals = max(gap, 0) + 1, max(-gap, 0) + 1
                result = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"
                rows.append(
                    {
                        "league": league,
                        "season": season,
                        "split": config.split_for_season(season),
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "result": result,
                        "home_shots": 10 + 2 * teams[home],
                        "away_shots": 10 + 2 * teams[away],
                        "home_shots_on_target": 4 + teams[home],
                        "away_shots_on_target": 4 + teams[away],
                        "home_corners": 5 + teams[home],
                        "away_corners": 5 + teams[away],
                        "home_fouls": 12,
                        "away_fouls": 12,
                        "home_yellows": 2,
                        "away_yellows": 2,
                        "home_reds": 0,
                        "away_reds": 0,
                        "odds_home": 2.0,
                        "odds_draw": 3.4,
                        "odds_away": 3.8,
                    }
                )
                date += pd.Timedelta(days=1)
        date += pd.Timedelta(days=60)
    df = pd.DataFrame(rows)
    df["match_id"] = [f"{league}_{i:05d}" for i in range(len(df))]
    return df


@pytest.fixture
def matches() -> pd.DataFrame:
    return make_matches()


@pytest.fixture
def snapshots() -> pd.DataFrame:
    """Three well-separated groups of team snapshots in two leagues."""
    rng = np.random.default_rng(0)
    rows = []
    for league, offset in (("E0", 0.0), ("D1", 0.4)):
        for level, ppg in (("strong", 2.2), ("middle", 1.4), ("weak", 0.7)):
            for _ in range(40):
                gf = ppg + offset + rng.normal(0, 0.05)
                ga = 2.6 - ppg + rng.normal(0, 0.05)
                rows.append(
                    {
                        "league": league,
                        "level": level,
                        "form_ppg": ppg + rng.normal(0, 0.05),
                        "form_goals_for": gf,
                        "form_goals_against": ga,
                        "form_shots_for": 8 + 3 * ppg + rng.normal(0, 0.2),
                        "form_shots_against": 18 - 3 * ppg + rng.normal(0, 0.2),
                        "form_sot_for": 2 + 1.5 * ppg + rng.normal(0, 0.1),
                        "form_sot_against": 6 - 1.5 * ppg + rng.normal(0, 0.1),
                        "form_corners_for": 3 + ppg + rng.normal(0, 0.1),
                        "form_corners_against": 7 - ppg + rng.normal(0, 0.1),
                    }
                )
    return pd.DataFrame(rows)
