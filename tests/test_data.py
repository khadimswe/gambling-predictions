import pandas as pd

from soccer_predict import data


def raw_frame() -> pd.DataFrame:
    base = {
        "FTHG": "2", "FTAG": "1", "FTR": "H", "HS": "12", "AS": "8", "HST": "5", "AST": "3",
        "HC": "6", "AC": "2", "HF": "10", "AF": "11", "HY": "1", "AY": "2", "HR": "0", "AR": "0",
        "B365H": "1.8", "B365D": "3.5", "B365A": "4.5",
    }  # fmt: skip
    rows = [
        {"Date": "16/08/08", "HomeTeam": "Arsenal", "AwayTeam": "West Brom", **base},
        {"Date": "17/08/2008", "HomeTeam": "Chelsea ", "AwayTeam": "Portsmouth", **base},
        {k: None for k in ["Date", "HomeTeam", "AwayTeam", *base]},  # trailing blank row
    ]
    raw = pd.DataFrame(rows)
    raw["league"] = "E0"
    raw["season"] = "0809"
    return raw


def test_parse_dates_handles_two_and_four_digit_years():
    parsed = data.parse_dates(pd.Series(["16/08/08", "16/08/2015"]))
    assert list(parsed.dt.year) == [2008, 2015]
    assert list(parsed.dt.month) == [8, 8]


def test_clean_drops_blank_rows_and_renames_columns():
    clean = data.clean(raw_frame())
    assert len(clean) == 2
    assert {"home_team", "away_goals", "home_shots_on_target", "split"} <= set(clean.columns)
    assert clean["home_team"].tolist() == ["Arsenal", "Chelsea"]  # whitespace stripped
    assert clean["match_id"].is_unique
    assert (clean["split"] == "train").all()


def test_harmonize_rescales_only_pre_break_seasons():
    def season(league: str, code: str, sot: float) -> dict:
        return {
            "league": league,
            "season": code,
            "home_shots": 10,
            "away_shots": 10,
            "home_shots_on_target": sot,
            "away_shots_on_target": sot,
        }

    df = pd.DataFrame([season("E0", "1213", 5), season("E0", "1314", 2.5), season("D1", "1213", 5)])
    out = data.harmonize_shots_on_target(df)
    assert out.loc[0, "home_shots_on_target"] == 2.5  # E0 before the break: halved
    assert out.loc[1, "home_shots_on_target"] == 2.5  # E0 after the break: untouched
    assert out.loc[2, "home_shots_on_target"] == 5  # other leagues: untouched
