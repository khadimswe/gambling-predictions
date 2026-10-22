# Predicting Soccer Match Outcomes (AI 3642)

Predict home win / draw / away win for European top-flight matches by combining
**data mining** (k-means team tiers) with **machine learning** (an outcome classifier).

| Member | NetID | Owns |
|---|---|---|
| Khadim Drame | kdrame1 | Data pipeline, rolling form features, k-means team tiers, CI |
| Ryann Ridgeway | rridgew4 | Outcome classifier (Logistic Regression, Random Forest), ablations |

Proposal: [`docs/proposal/AI-Project-Proposal.pdf`](docs/proposal/AI-Project-Proposal.pdf)

## Where to start

- **Ryann:** read [`docs/specs/ryann-classifier.md`](docs/specs/ryann-classifier.md), then run the quickstart below.
- **Khadim:** [`docs/specs/khadim-tiers.md`](docs/specs/khadim-tiers.md) tracks what is done and what is next.
- **Midterm report draft:** [`reports/midterm/midterm-report.md`](reports/midterm/midterm-report.md)
- **Column-by-column description of `features.csv`:** [`docs/data-contract.md`](docs/data-contract.md)

## Quickstart

Requires Python 3.11+ and git. Takes about 3 minutes, most of it downloading.

```bash
git clone https://github.com/khadimswe/gambling-predictions.git
cd gambling-predictions
make setup        # .venv + dependencies + git hooks
make data         # 40 CSVs from football-data.co.uk into data/raw/
make features     # data/processed/features.csv with tiers
make tiers        # rerun the tier experiments (figures + numbers in reports/)
make check        # everything CI checks
```

No `make` (Windows)? Run the commands from the [Makefile](Makefile) directly, using
`.venv\Scripts\` instead of `.venv/bin/`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Pipeline

```
football-data.co.uk CSVs (5 leagues x 8 seasons, 2008/09 to 2015/16)
  -> data.py          clean, harmonize Premier League shots on target
  -> features.py      rolling form over each team's previous 10 matches (no leakage)
  -> tiers.py         k-means on form snapshots -> tier 1 (strong) .. 3 (weak)   [Khadim]
  -> features.csv     one row per match: home/away form + home_tier, away_tier, tier_diff
  -> classifier.py    Logistic Regression / Random Forest -> P(H), P(D), P(A)   [Ryann]
  -> evaluate.py      accuracy, macro-F1, log loss, confusion matrix vs. baselines
```

Seasons 2008/09 to 2013/14 train, 2014/15 validates, 2015/16 is the held-out test.

## Results so far (test season 2015/16, 1,760 matches)

| Model | Accuracy | Log loss |
|---|---|---|
| Always predict home win (proposal baseline) | 44.9% | 1.070 |
| Tier rule (home unless away tier is stronger) | 49.0% | n/a |
| Tier-pair lookup P(result \| home tier, away tier) | 49.4% | 1.025 |
| Bookmaker odds (reference ceiling, not a model) | 52.0% | 0.981 |
| Ryann's classifier | coming soon | |

Full numbers: [`reports/results/`](reports/results). Figures: [`reports/figures/`](reports/figures).

## Checks (CI runs these on every push and pull request)

| Check | Tool | Run locally |
|---|---|---|
| Lint and formatting | ruff | `make lint` (`make format` fixes) |
| Types | mypy | `make typecheck` |
| Tests + coverage | pytest | `make test` |
| No em dashes or invisible characters | `scripts/check_text.py` | `make text` |
| File hygiene (whitespace, YAML, large files, private keys, notebook outputs) | pre-commit | `pre-commit run --all-files` |
| Secret scan over full history | gitleaks | `make secrets` |

The same hooks run on every `git commit` once `make setup` has installed them.

## Layout

```
src/soccer_predict/   config, data, features, tiers, tiers_report, classifier, evaluate, cli
tests/                synthetic-data tests (no network needed)
scripts/              check_text.py (em dash check), build_pdf.py (Markdown -> PDF)
docs/                 proposal, plan, data contract, per-person specs
reports/              midterm report, generated results and figures
data/                 raw + processed CSVs (git-ignored; `make data` recreates them)
notebooks/            exploration only; outputs are stripped on commit
```
