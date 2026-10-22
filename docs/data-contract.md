# Data contract: `data/processed/features.csv`

This file is the hand-off between the two halves of the project. Khadim's pipeline
writes it (`soccer build`); Ryann's classifier reads it. If a column changes,
update this page in the same pull request.

One row per match where both teams have at least 5 previous matches in their
current top-flight spell. 13,883 rows: train 10,365, val 1,758, test 1,760.

## Identifiers (never use as features)

| Column | Type | Meaning |
|---|---|---|
| `match_id` | str | `{league}_{season}_{yyyymmdd}_{HomeTeam}`, unique |
| `league` | str | `E0` Premier League, `SP1` La Liga, `D1` Bundesliga, `I1` Serie A, `F1` Ligue 1 |
| `season` | str | `0809` = 2008/09 ... `1516` = 2015/16 |
| `split` | str | `train` (0809 to 1314), `val` (1415), `test` (1516) |
| `date` | date | kickoff date |
| `home_team`, `away_team` | str | football-data.co.uk team names |

## Target

| Column | Type | Meaning |
|---|---|---|
| `result` | str | `H`, `D`, or `A` |
| `target` | int | `0` = H, `1` = D, `2` = A (use this for training) |

## Pre-match form features (safe to use)

For each side (`home_` and `away_` prefix), the mean over that team's **previous
10 matches** (home and away games both count), computed strictly before kickoff:

| Column suffix | Meaning (per match) |
|---|---|
| `form_ppg` | points per game (3 win, 1 draw, 0 loss) |
| `form_goals_for`, `form_goals_against` | goals scored / conceded |
| `form_shots_for`, `form_shots_against` | shots taken / allowed |
| `form_sot_for`, `form_sot_against` | shots on target taken / allowed (Premier League pre-2013/14 rescaled) |
| `form_corners_for`, `form_corners_against` | corners won / conceded |
| `form_fouls` | fouls committed |
| `form_yellows` | yellow cards received |
| `n_prior` | how many previous matches the team has in this spell (5 or more here) |

## Tier features (safe to use)

| Column | Type | Meaning |
|---|---|---|
| `home_tier`, `away_tier` | int | k-means tier of each side's form snapshot: 1 = strongest, 3 = weakest |
| `tier_diff` | int | `away_tier - home_tier`. Positive means the home side is in a stronger tier |

The tier model is fit on training seasons only. Validation and test matches are
assigned to the frozen training centroids, so these columns do not leak.

## Not in this file

The match's own stats (goals, shots, cards) and the bookmaker odds live in
`data/processed/matches.csv`. They are known only after the match (or come from
the betting market) and must not be classifier inputs. Odds are used only as a
reference ceiling in `evaluate.bookmaker_proba`.
