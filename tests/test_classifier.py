"""Acceptance tests for the classifier (owner: Ryann).

The leakage test runs now. The training test skips until make_model is built,
then starts checking it automatically.
"""

import numpy as np
import pytest

from soccer_predict import classifier, features, tiers
from tests.conftest import make_matches


@pytest.fixture
def feature_table():
    matches = make_matches(seasons=("0809", "0910", "1011"), rounds=4)
    form = features.add_rolling_form(features.to_team_matches(matches))
    table = features.build_match_features(matches, form)
    snapshots = form[form["split"] == "train"].dropna(subset=tiers.TIER_FEATURES)
    model = tiers.fit_tier_model(snapshots, n_tiers=3, seed=0)
    return tiers.add_tiers(table, model)


def test_features_are_pre_match_only():
    match_stats = {"home_goals", "away_goals", "home_shots", "away_shots", "result", "target"}
    assert not match_stats & set(classifier.feature_columns(use_tiers=True))


def test_ablation_drops_only_tier_columns():
    with_tiers = set(classifier.feature_columns(use_tiers=True))
    without = set(classifier.feature_columns(use_tiers=False))
    assert with_tiers - without == {"home_tier", "away_tier", "tier_diff"}


@pytest.mark.parametrize("kind", ["logreg", "rf"])
def test_train_and_predict_probabilities(feature_table, kind):
    try:
        classifier.make_model(kind)
    except NotImplementedError:
        pytest.skip("make_model not implemented yet (Ryann, spec step 2)")
    train = feature_table[feature_table["split"] == "train"]
    model = classifier.train(train, kind=kind)
    pred, proba = classifier.predict(model, feature_table)
    assert proba.shape == (len(feature_table), 3)
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(np.unique(pred)) <= {0, 1, 2}
