"""Leakage-safe rolling "form" features.

Every match is turned into two team-match rows (one per side). For each row we
average the team's previous ROLLING_WINDOW matches, using only matches played
strictly before this one. That pre-match snapshot is what the tier model clusters
and what the classifier sees. Stats from the match being predicted never leak in.
"""

from __future__ import annotations

import pandas as pd

from soccer_predict import config

# form column -> (match stat, whose number: the team's own or its opponent's)
FORM_STATS: dict[str, tuple[str, str]] = {
    "form_goals_for": ("goals", "own"),
    "form_goals_against": ("goals", "opp"),
    "form_shots_for": ("shots", "own"),
    "form_shots_against": ("shots", "opp"),
    "form_sot_for": ("shots_on_target", "own"),
    "form_sot_against": ("shots_on_target", "opp"),
    "form_corners_for": ("corners", "own"),
    "form_corners_against": ("corners", "opp"),
    "form_fouls": ("fouls", "own"),
    "form_yellows": ("yellows", "own"),
}
FORM_COLUMNS = [*FORM_STATS, "form_ppg"]

POINTS = {"W": 3, "D": 1, "L": 0}


def to_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match, one from each team's point of view."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame(
            {
                "match_id": matches["match_id"],
                "league": matches["league"],
                "season": matches["season"],
                "split": matches["split"],
                "date": matches["date"],
                "team": matches[f"{side}_team"],
                "opponent": matches[f"{other}_team"],
                "is_home": int(side == "home"),
            }
        )
        for form_col, (stat, whose) in FORM_STATS.items():
            source = side if whose == "own" else other
            part[form_col.removeprefix("form_")] = matches[f"{source}_{stat}"]
        won = matches[f"{side}_goals"] > matches[f"{other}_goals"]
        drew = matches[f"{side}_goals"] == matches[f"{other}_goals"]
        part["outcome"] = "L"
        part.loc[drew, "outcome"] = "D"
        part.loc[won, "outcome"] = "W"
        part["points"] = part["outcome"].map(POINTS)
        sides.append(part)
    team_matches = pd.concat(sides, ignore_index=True)
    return team_matches.sort_values(["league", "team", "date"], kind="stable").reset_index(
        drop=True
    )


def add_spells(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Label each unbroken run of top-flight seasons per team.

    A team relegated in 2009 and promoted back in 2012 should not carry its 2009
    form into 2012, so a missing season starts a new spell.
    """
    df = team_matches.copy()
    season_index = {code: i for i, code in enumerate(config.SEASONS)}
    df["season_idx"] = df["season"].map(season_index)
    gap = df.groupby(["league", "team"])["season_idx"].diff().fillna(0)
    df["spell"] = (gap > 1).astype(int).groupby([df["league"], df["team"]]).cumsum()
    return df.drop(columns="season_idx")


def add_rolling_form(
    team_matches: pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    min_periods: int = config.MIN_PRIOR_MATCHES,
) -> pd.DataFrame:
    """Add form_* columns: means over the previous `window` matches, excluding this one."""
    df = add_spells(team_matches)
    keys = [df["league"], df["team"], df["spell"]]
    df["n_prior"] = df.groupby(keys).cumcount()

    raw = {col: col.removeprefix("form_") for col in FORM_STATS}
    raw["form_ppg"] = "points"
    for form_col, raw_col in raw.items():
        df[form_col] = df.groupby(keys)[raw_col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
        )
    return df


def build_match_features(matches: pd.DataFrame, team_form: pd.DataFrame) -> pd.DataFrame:
    """Attach each side's pre-match form to the match row.

    Matches where either team has fewer than MIN_PRIOR_MATCHES previous games in
    the current spell are dropped (their form is undefined).
    """
    snapshot_cols = ["match_id", "team", "n_prior", *FORM_COLUMNS]
    snap = team_form[snapshot_cols]
    out = matches[
        ["match_id", "league", "season", "split", "date", "home_team", "away_team", "result"]
    ].copy()
    for side in ("home", "away"):
        columns = {c: f"{side}_{c}" for c in ["n_prior", *FORM_COLUMNS]}
        renamed = snap.rename(columns=columns | {"team": f"{side}_team"})
        out = out.merge(renamed, on=["match_id", f"{side}_team"], how="left", validate="1:1")
    form_cols = [f"{side}_{c}" for side in ("home", "away") for c in FORM_COLUMNS]
    out = out.dropna(subset=form_cols).reset_index(drop=True)
    out["target"] = out["result"].map({"H": 0, "D": 1, "A": 2})
    return out
