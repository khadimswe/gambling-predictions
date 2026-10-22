import numpy as np
import pandas as pd
import pytest

from soccer_predict import evaluate


def test_evaluate_reports_accuracy_and_confusion_matrix():
    result = evaluate.evaluate(np.array([0, 1, 2, 0]), np.array([0, 0, 2, 0]))
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[2, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert "log_loss" not in result


def test_home_win_baseline_and_prior():
    train = pd.DataFrame({"target": [0, 0, 1, 2]})
    matches = pd.DataFrame({"target": [0, 2]})
    assert evaluate.predict_home_win(matches).tolist() == [0, 0]
    assert evaluate.class_prior_proba(train, matches)[0] == pytest.approx([0.5, 0.25, 0.25])


def test_bookmaker_probabilities_remove_margin_and_pick_favourite():
    odds = pd.DataFrame({"odds_home": [1.5, 5.0], "odds_draw": [4.0, 3.6], "odds_away": [6.0, 1.7]})
    proba = evaluate.bookmaker_proba(odds)
    assert np.allclose(proba.sum(axis=1), 1)
    assert proba.argmax(axis=1).tolist() == [0, 2]


def test_tier_rule_only_backs_the_away_side_when_it_is_stronger():
    matches = pd.DataFrame({"tier_diff": [2, 0, -1]})
    assert evaluate.predict_tier_rule(matches).tolist() == [0, 0, 2]
