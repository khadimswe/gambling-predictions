import pandas as pd
import pytest

from soccer_predict import features
from tests.conftest import STRENGTH, make_matches


def test_two_team_rows_per_match_with_points(matches):
    team_matches = features.to_team_matches(matches)
    assert len(team_matches) == 2 * len(matches)
    first = team_matches[team_matches["match_id"] == matches["match_id"].iloc[0]]
    assert sorted(first["is_home"]) == [0, 1]
    # Every match hands out 3 points (win) or 2 (draw) in total.
    totals = team_matches.groupby("match_id")["points"].sum()
    assert set(totals) <= {2, 3}


def test_form_is_mean_of_previous_matches_only(matches):
    form = features.add_rolling_form(features.to_team_matches(matches), window=3, min_periods=2)
    alpha = form[form["team"] == "Alpha"].reset_index(drop=True)
    assert pd.isna(alpha.loc[1, "form_goals_for"])  # only one prior match
    expected = alpha.loc[0:2, "goals_for"].mean()  # matches 0, 1, 2 precede match 3
    assert alpha.loc[3, "form_goals_for"] == pytest.approx(expected)


def test_changing_a_match_does_not_change_its_own_features(matches):
    before = features.add_rolling_form(features.to_team_matches(matches))
    tampered = matches.copy()
    target = tampered.index[30]
    tampered.loc[target, ["home_goals", "home_shots"]] = [9, 40]
    after = features.add_rolling_form(features.to_team_matches(tampered))

    match_id = matches.loc[target, "match_id"]
    cols = features.FORM_COLUMNS
    same_match_before = before[before["match_id"] == match_id][cols]
    same_match_after = after[after["match_id"] == match_id][cols]
    pd.testing.assert_frame_equal(same_match_before, same_match_after)
    # ...but later matches do see it.
    assert not before[cols].equals(after[cols])


def test_missing_season_starts_a_new_spell():
    with_gap = pd.concat(
        [
            make_matches(seasons=("0809",)),
            make_matches(seasons=("1011",), teams={**STRENGTH}).assign(
                date=lambda d: d["date"] + pd.Timedelta(days=800)
            ),
        ],
        ignore_index=True,
    )
    with_gap["match_id"] = [f"m{i}" for i in range(len(with_gap))]
    form = features.add_rolling_form(features.to_team_matches(with_gap))
    alpha = form[form["team"] == "Alpha"]
    assert alpha.groupby("season")["spell"].first().tolist() == [0, 1]
    assert alpha[alpha["season"] == "1011"]["n_prior"].min() == 0


def test_match_features_drop_rows_without_enough_history(matches):
    form = features.add_rolling_form(features.to_team_matches(matches))
    table = features.build_match_features(matches, form)
    assert len(table) < len(matches)
    assert table[[f"home_{c}" for c in features.FORM_COLUMNS]].notna().all().all()
    assert set(table["target"]) <= {0, 1, 2}
