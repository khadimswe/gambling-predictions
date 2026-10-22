import numpy as np
import pandas as pd
import pytest

from soccer_predict import tiers, tiers_report


def test_tiers_are_ordered_strongest_first(snapshots):
    model = tiers.fit_tier_model(snapshots, n_tiers=3, seed=0)
    snapshots = snapshots.assign(tier=model.predict(snapshots))
    by_level = snapshots.groupby("level")["tier"].agg(lambda s: s.mode()[0])
    assert by_level.to_dict() == {"strong": 1, "middle": 2, "weak": 3}


def test_predict_is_nearest_centroid(snapshots):
    model = tiers.fit_tier_model(snapshots, n_tiers=3, seed=0)
    clusters = model.kmeans.predict(model.scaler.transform(snapshots))
    assert np.array_equal(model.predict(snapshots), model.tier_of_cluster[clusters])


def test_scaler_standardizes_within_each_league(snapshots):
    scaler = tiers.LeagueScaler.fit(snapshots, tiers.TIER_FEATURES)
    z = pd.DataFrame(scaler.transform(snapshots), columns=tiers.TIER_FEATURES)
    per_league = z.groupby(snapshots["league"].to_numpy()).mean()
    assert np.allclose(per_league.to_numpy(), 0, atol=1e-9)


def test_add_tiers_sets_diff_positive_when_home_is_stronger(snapshots):
    model = tiers.fit_tier_model(snapshots, n_tiers=3, seed=0)
    strong = snapshots[snapshots["level"] == "strong"].iloc[0]
    weak = snapshots[snapshots["level"] == "weak"].iloc[0]
    row = {"league": strong["league"], "home_team": "S", "away_team": "W"}
    for col in tiers.TIER_FEATURES:
        row[f"home_{col}"] = strong[col]
        row[f"away_{col}"] = weak[col]
    out = tiers.add_tiers(pd.DataFrame([row]), model)
    assert out.loc[0, "home_tier"] == 1
    assert out.loc[0, "away_tier"] == 3
    assert out.loc[0, "tier_diff"] == 2


def test_tier_pair_table_is_a_distribution():
    train = pd.DataFrame({"home_tier": [1, 1, 2], "away_tier": [2, 2, 1], "target": [0, 1, 2]})
    table = tiers.tier_pair_table(train, n_tiers=2, alpha=1.0)
    assert table.shape == (2, 2, 3)
    assert np.allclose(table.sum(axis=2), 1)
    # 1v2: counts H=1, D=1, A=0 plus alpha=1 each -> (2, 2, 1) / 5
    assert table[0, 1] == pytest.approx([0.4, 0.4, 0.2])


def test_lloyd_trace_converges_on_two_obvious_groups():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    steps = tiers_report.lloyd_trace(points, init=points[[0, 1]])
    assert steps[-1]["assignment"] in ([0, 0, 1, 1], [1, 1, 0, 0])
    assert steps[-1]["inertia"] <= steps[0]["inertia"]
