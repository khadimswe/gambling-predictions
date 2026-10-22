"""Shared evaluation: metrics and reference baselines.

Both halves of the project report numbers through these functions so results
are comparable. Labels are always ordered H, D, A (target 0, 1, 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

LABELS = ["H", "D", "A"]


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Accuracy, macro-F1, confusion matrix, and (if probabilities given) log loss.

    y_true and y_pred are integer targets (0=H, 1=D, 2=A). proba has shape (n, 3).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result: dict[str, Any] = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    if proba is not None:
        result["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1, 2]))
    return result


def predict_home_win(matches: pd.DataFrame) -> np.ndarray:
    """Baseline from the proposal: always predict a home win."""
    return np.zeros(len(matches), dtype=int)


def class_prior_proba(train: pd.DataFrame, matches: pd.DataFrame) -> np.ndarray:
    """Probabilistic version of the baseline: training-set H/D/A frequencies for every match."""
    prior = np.bincount(train["target"], minlength=3) / len(train)
    return np.tile(prior, (len(matches), 1))


def bookmaker_proba(matches: pd.DataFrame) -> np.ndarray:
    """Bet365 implied probabilities, normalized to remove the bookmaker margin.

    This is a reference ceiling, not a model we are allowed to train on.
    """
    inverse = 1.0 / matches[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    return np.asarray(inverse / inverse.sum(axis=1, keepdims=True))


def predict_tier_rule(matches: pd.DataFrame) -> np.ndarray:
    """Tier-only heuristic: home win unless the away side is in a stronger tier."""
    return np.where(matches["tier_diff"] < 0, 2, 0)
