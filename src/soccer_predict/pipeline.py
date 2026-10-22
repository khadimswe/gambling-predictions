"""End-to-end build: raw CSVs -> matches.csv, team_form.csv, features.csv."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from soccer_predict import config, data, features, tiers


@dataclass
class BuildOutput:
    matches: pd.DataFrame
    team_form: pd.DataFrame
    features: pd.DataFrame
    tier_model: tiers.TierModel


def training_snapshots(team_form: pd.DataFrame) -> pd.DataFrame:
    """Team-match form snapshots from training seasons with a defined form."""
    usable = team_form[team_form["split"] == "train"]
    return usable.dropna(subset=tiers.TIER_FEATURES)


def build(n_tiers: int = config.N_TIERS, save: bool = True) -> BuildOutput:
    matches = data.clean(data.load_raw())
    team_form = features.add_rolling_form(features.to_team_matches(matches))
    match_features = features.build_match_features(matches, team_form)

    tier_model = tiers.fit_tier_model(training_snapshots(team_form), n_tiers=n_tiers)
    form_rows = team_form.dropna(subset=tiers.TIER_FEATURES).index
    team_form.loc[form_rows, "tier"] = tier_model.predict(team_form.loc[form_rows])
    match_features = tiers.add_tiers(match_features, tier_model)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        matches.to_csv(config.MATCHES_CSV, index=False)
        team_form.to_csv(config.TEAM_FORM_CSV, index=False)
        match_features.to_csv(config.FEATURES_CSV, index=False)
    return BuildOutput(matches, team_form, match_features, tier_model)


def load_features() -> pd.DataFrame:
    if not config.FEATURES_CSV.exists():
        raise FileNotFoundError(f"{config.FEATURES_CSV} is missing. Run `soccer build` first.")
    return pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
