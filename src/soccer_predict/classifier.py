"""Match outcome classifier (owner: Ryann).

Spec: docs/specs/ryann-classifier.md

Input is data/processed/features.csv (one row per match, built by `soccer build`).
Output is a model that predicts target 0 (home win), 1 (draw) or 2 (away win)
with probabilities. Train on split == "train", tune on "val", report "test" once.

Everything below the TODO markers is yours to change. Keep the function
signatures so `soccer train` and the tests keep working.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from soccer_predict.features import FORM_COLUMNS

# Pre-match features only. Never add home_goals, home_shots, etc. from the match
# itself: those are only known after the final whistle.
FORM_FEATURES = [f"{side}_{col}" for side in ("home", "away") for col in FORM_COLUMNS]
TIER_FEATURES = ["home_tier", "away_tier", "tier_diff"]


def feature_columns(use_tiers: bool = True) -> list[str]:
    """Columns the classifier is allowed to see. use_tiers=False is the ablation."""
    return FORM_FEATURES + (TIER_FEATURES if use_tiers else [])


def make_model(kind: str = "logreg", **params: Any) -> Any:
    """Return an unfitted scikit-learn estimator (ideally a Pipeline).

    "logreg": scale the features, then fit multinomial Logistic Regression.
              Scaling matters here because shots (~13) and points per game (~1.4)
              are on very different scales, and Logistic Regression is sensitive
              to that.
    "rf":     Random Forest. Trees split on raw values, so no scaling needed.
    """
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, **params)),
            ]
        )

    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        from soccer_predict import config

        rf_params = {
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": config.RANDOM_SEED,
        }
        rf_params.update(params)  # let a caller override any default
        return RandomForestClassifier(**rf_params)

    raise NotImplementedError(
        f"make_model({kind!r}) is not built yet; see docs/specs/ryann-classifier.md step 2"
    )


def train(train_df: pd.DataFrame, kind: str = "logreg", use_tiers: bool = True) -> Any:
    """Fit a model on the training split and return it."""
    model = make_model(kind)
    model.fit(train_df[feature_columns(use_tiers)], train_df["target"])
    return model


def predict(model: Any, df: pd.DataFrame, use_tiers: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted targets, class probabilities with columns H, D, A)."""
    X = df[feature_columns(use_tiers)]
    return np.asarray(model.predict(X)), np.asarray(model.predict_proba(X))
