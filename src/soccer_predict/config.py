"""Project-wide settings: paths, leagues, seasons, splits, and model defaults.

Change things here instead of hard-coding them in modules, so both halves of the
project (tiers and classifier) always agree on the data they see.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

MATCHES_CSV = PROCESSED_DIR / "matches.csv"
TEAM_FORM_CSV = PROCESSED_DIR / "team_form.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

# football-data.co.uk division codes: the five largest leagues in the Kaggle
# European Soccer Database named in the proposal.
LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Season codes as used in football-data.co.uk URLs ("0809" = 2008/09).
SEASONS: list[str] = ["0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516"]

# Chronological split. Never shuffle matches across seasons: that leaks the future.
TRAIN_SEASONS: list[str] = ["0809", "0910", "1011", "1112", "1213", "1314"]
VAL_SEASONS: list[str] = ["1415"]
TEST_SEASONS: list[str] = ["1516"]

# Rolling "form" window: how many previous matches summarize a team's current level.
ROLLING_WINDOW = 10
MIN_PRIOR_MATCHES = 5

# Team tiers (k-means). K was picked with `soccer tiers-report`; see the midterm report.
N_TIERS = 3
K_SWEEP: list[int] = [2, 3, 4, 5, 6, 7, 8]

RANDOM_SEED = 3642


def season_label(code: str) -> str:
    """Turn a season code like "1516" into "2015/16"."""
    return f"20{code[:2]}/{code[2:]}"


def split_for_season(code: str) -> str:
    """Return "train", "val", or "test" for a season code."""
    if code in TRAIN_SEASONS:
        return "train"
    if code in VAL_SEASONS:
        return "val"
    if code in TEST_SEASONS:
        return "test"
    raise ValueError(f"Season {code!r} is not assigned to a split in config.py")
