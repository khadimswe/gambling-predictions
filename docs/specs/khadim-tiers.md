# Spec: Data Pipeline and Team Tiers (Khadim)

**Owner:** Khadim Drame (kdrame1) · **Partner:** Ryann Ridgeway (rridgew4) · **Course:** AI 3642

## Your job in one paragraph

Own the data mining half: turn raw match results into leakage-safe rolling form
features, cluster each team's pre-match form with k-means into ordered
performance tiers, and hand Ryann a clean feature table with `home_tier`,
`away_tier`, and `tier_diff`. You also own the shared plumbing (data, evaluation,
CI) and the overall report.

## Status

| Piece | State | Where |
|---|---|---|
| Download + clean 5 leagues x 8 seasons (14,604 matches) | done | `data.py`, `soccer download` |
| Premier League shots-on-target fix (definition change in 2013/14) | done | `data.harmonize_shots_on_target` |
| Rolling form, previous 10 matches, spell reset for promoted teams | done | `features.py` |
| Per-league standardization + k-means + strength ordering | done | `tiers.py` |
| K and window chosen on validation (K = 3, window = 10) | done | `reports/results/k_sweep.csv`, `window_sensitivity.csv` |
| Baselines: home win, tier rule, tier-pair table, bookmaker | done | `reports/results/baselines.json` |
| Worked example + Lloyd's toy trace for the report | done | `worked_example.json`, `lloyd_toy.json` |
| Figures | done | `reports/figures/*.png` |
| CI: ruff, mypy, pytest, em dash check, pre-commit hygiene, gitleaks | done | `.github/workflows/ci.yml` |
| Midterm report: your sections | drafted | `reports/midterm/midterm-report.md` |
| Ryann's sections of the report | waiting on Ryann | `[RYANN: ...]` markers |

Reproduce everything with `make data features tiers` (about a minute after the download).

## Before the midterm is submitted

1. **Push the repo and add Ryann.** GitHub > Settings > Collaborators > add her account.
   Then turn on branch protection for `main`: require a pull request and require the
   CI checks (Lint and format, Type check, Tests, No em dashes, File hygiene, Secret scan).
2. **Send Ryann her spec** (`docs/specs/ryann-classifier.md` or `build/pdf/ryann-classifier.pdf`).
3. **Read the report draft** end to end and fix anything that does not sound like you.
   Every number in your sections comes from `reports/results/`; if you change code,
   rerun `make tiers` and update the numbers.
4. **Review Ryann's PR** with `make_model` and her results; merge when CI is green.
5. **Merge her report sections**, run `make pdf`, check the PDF, and **both of you submit it**.

## Design decisions to be able to defend

| Decision | Why | Evidence |
|---|---|---|
| football-data.co.uk instead of the Kaggle database | free direct download, the 5 biggest Kaggle leagues over the same seasons, clean per-match shots and corners; Kaggle needs an account and stores match events as XML | proposal named it as the fallback |
| Drop possession | not in football-data.co.uk | shots, shots on target, corners stand in for territory |
| Rescale Premier League shots on target before 2013/14 by 0.597 | on-target share drops from 55.6% to 33.2% in one summer while other leagues stay flat; factor uses training seasons only | `data.SOT_BREAKS`, report Section 2 |
| Window of 10 matches, at least 5 | window 5 is clearly worse on validation; 10, 15, 20 are within noise | `window_sensitivity.csv` |
| Reset form when a team misses a season | a promoted team's form from years ago is not its current level | `features.add_spells` |
| Standardize within each league, training data only | leagues differ in scoring; no test statistics leak in | `tiers.LeagueScaler` |
| K = 3 | best tier-rule accuracy on validation, second-best log loss (1.0070 vs. 1.0059 at K = 5), clean monotone profiles, 99.6% assignment agreement across seeds | `k_sweep.csv`, `summary.json` |
| Tier 1 = highest mean points per game | k-means labels are arbitrary; ordering makes the tier an ordinal feature | `tiers.fit_tier_model` |

## Next steps (midterm to final)

1. **The Leicester problem.** In 2015/16 Leicester won the league but were tier 2 in 22
   of their 38 matches, because 6 of the 9 clustering features describe shot and corner
   volume, and Leicester allowed many shots and corners. Try: weighting `form_ppg` and goal features
   more, PCA before k-means, or adding goal efficiency (goals per shot on target).
   Measure with the validation tier-pair log loss.
2. **Soft tiers.** Replace hard assignments with a Gaussian Mixture Model and give the
   classifier P(tier = k) for each side. Compare with k-means on the same metric.
3. **Playing-style clusters.** The proposal also mentions style. Cluster on fouls, cards,
   shot share and corner share (not strength) and test whether style matchups add
   anything on top of tiers.
4. **Cold start for promoted teams.** 721 matches are dropped because a team has fewer
   than 5 prior matches. Default promoted teams to tier 3 (or the relegated teams' average)
   instead of dropping them.
5. **Elo comparison.** Ren and Susnjak (2022) used Elo-style ratings. Add an Elo feature
   as a baseline to show whether the tiers capture the same information.
6. **Stretch: demo.** Pick two teams and a date, show tiers and Ryann's predicted
   probabilities (notebook first, small GUI if time allows).
