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

    TODO(Ryann): spec step 2. Suggested kinds:
      "logreg": StandardScaler + LogisticRegression (multinomial is the default)
      "rf":     RandomForestClassifier
    """
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
