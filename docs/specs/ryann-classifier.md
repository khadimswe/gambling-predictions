# Spec: Outcome Classifier (Ryann)

**Owner:** Ryann Ridgeway (rridgew4) · **Partner:** Khadim Drame (kdrame1) · **Course:** AI 3642

## Your job in one paragraph

Build the machine learning half of the project: a classifier that reads one row
per match from `data/processed/features.csv` and outputs probabilities for home
win (H), draw (D), and away win (A). Start with Logistic Regression and Random
Forest, as the proposal says. Then answer the key question: **do Khadim's k-means
tier features make the classifier better?** The pipeline, data, tiers, baselines,
evaluation code, and CLI are already built. You plug your model into one function.

## What is already done for you

| Thing | Where | Notes |
|---|---|---|
| Clean data for 14,604 matches, 5 leagues, 2008/09 to 2015/16 | `data/processed/matches.csv` | `make data && make features` |
| Feature table with 25 usable columns | `data/processed/features.csv` | column guide: [`docs/data-contract.md`](../data-contract.md) |
| Train / val / test split by season | `split` column | train 10,365 · val 1,758 · test 1,760 |
| Tier features (home_tier, away_tier, tier_diff) | same file | 1 = strongest, 3 = weakest |
| Metrics: accuracy, macro-F1, log loss, confusion matrix | `src/soccer_predict/evaluate.py` | use these so numbers match Khadim's |
| Training and evaluation harness | `soccer train` | writes `reports/results/classifier_*.json` |
| Your module, with signatures and TODOs | `src/soccer_predict/classifier.py` | you fill in `make_model` |
| Acceptance tests | `tests/test_classifier.py` | skip until `make_model` exists, then run automatically |

## Numbers to beat

Validation season 2014/15 (1,758 matches). Use these while you tune.

| Reference | Accuracy | Macro-F1 | Log loss |
|---|---|---|---|
| Always home win (proposal baseline) | 45.2% | 0.208 | 1.068 |
| Tier rule (Khadim) | 51.7% | 0.381 | n/a |
| Tier-pair lookup table (Khadim) | 50.2% | 0.343 | 1.007 |
| Bookmaker odds (ceiling, not a model) | 53.1% | 0.390 | 0.976 |

A realistic goal: beat the tier lookup on log loss (below 1.007) and get close to
52% accuracy. Matching the bookmaker would be a great result.

## Setup (30 minutes)

```bash
git clone https://github.com/khadimswe/gambling-predictions.git
cd gambling-predictions
make setup && make data && make features
make test          # 25 pass, 2 skipped (the skipped ones are yours)
```

Windows without `make`: see [CONTRIBUTING.md](../../CONTRIBUTING.md). Work on a
branch named `ryann/<topic>` and open pull requests into `main`.

## Steps

### Step 1. Look at the data (1 hour)

Open a notebook in `notebooks/` (outputs are stripped on commit, that is expected).

```python
import pandas as pd

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
train = df[df.split == "train"]
train.result.value_counts(normalize=True)  # H 46.8%, D 25.5%, A 27.7%
train.groupby("tier_diff").result.value_counts(normalize=True).unstack()
```

Things worth noticing for the report: the class imbalance (draws are the hardest
class), and how strongly `tier_diff` already separates outcomes.

**Done when:** you can say in two sentences what the inputs and the target look like.

### Step 2. Implement `make_model` (2 hours)

In `src/soccer_predict/classifier.py`, replace the `NotImplementedError` so that:

- `make_model("logreg")` returns a scikit-learn `Pipeline` of `StandardScaler` then
  `LogisticRegression(max_iter=1000)`. Scaling matters for Logistic Regression
  because shots (about 13) and points per game (about 1.4) have different units.
  With three classes, scikit-learn fits a multinomial (softmax) model by default.
- `make_model("rf")` returns a `RandomForestClassifier` with `random_state=config.RANDOM_SEED`.
  Start with `n_estimators=500, min_samples_leaf=5, n_jobs=-1`.
- Pass `**params` through so you can tune from the command line or a notebook.

**Done when:** `make test` shows 27 passed and nothing skipped.

### Step 3. First results on validation (1 hour)

```bash
soccer train --model logreg
soccer train --model rf
```

Each run prints and saves accuracy, macro-F1, log loss, and the confusion matrix
for the validation season to `reports/results/classifier_<model>.json`.

**Done when:** both JSON files exist and you have compared them to the table above.

### Step 4. Do the tiers help? (1 hour, this is the headline experiment)

```bash
soccer train --model logreg --no-tiers
soccer train --model rf --no-tiers
```

`--no-tiers` drops `home_tier`, `away_tier`, and `tier_diff` and keeps the 22 raw
form columns. Put the four runs in one table: {LR, RF} x {with tiers, without}.
This is the experiment that ties your half to Khadim's, so it goes in the report
whatever the answer is. "Tiers do not help once raw form is present" is a valid
and interesting finding.

**Done when:** the 2 x 2 table is in the midterm report (Section 4.3).

### Step 5. Tune and handle draws (after midterm, 3 to 5 hours)

Tune only on validation; never look at test while choosing settings.

- Logistic Regression: `C` in {0.01, 0.1, 1, 10}.
- Random Forest: `max_depth` in {None, 8, 16}, `min_samples_leaf` in {1, 5, 20}, `max_features` in {"sqrt", 0.5}.
- Draws: both baselines predict zero draws. Try `class_weight="balanced"` and report
  what it does to accuracy versus macro-F1. Explain the trade-off.
- Try tiers as one-hot categories instead of integers (Logistic Regression treats
  integers as evenly spaced; tier 2 is closer to tier 3 than to tier 1 in the data).
- Optional: Gradient Boosting, the ensemble direction that Ren and Susnjak (2022)
  and Patel (2024) found strongest.

### Step 6. Final evaluation (once, at the end)

```bash
soccer train --model <best> --test
```

Report test numbers exactly once with the settings chosen on validation. Add the
confusion matrix and the top feature importances (Random Forest `feature_importances_`
or Logistic Regression coefficients).

## Your midterm report sections

The group writes one report and **both of us submit it**. The draft is
[`reports/midterm/midterm-report.md`](../../reports/midterm/midterm-report.md).
Khadim's sections are filled in with real numbers. Yours are marked
`[RYANN: ...]`. You own:

1. **Section 3.3, Model B: the outcome classifier.** Describe Logistic Regression
   (and Random Forest) precisely *for this problem*, not in general:
   - input vector x for match m: the 25 columns from `feature_columns()` (list them by group)
   - output y in {H, D, A} encoded 0, 1, 2
   - the model: P(y = k | x) = exp(w_k · x + b_k) / sum over j of exp(w_j · x + b_j), for k in {H, D, A}
   - what is learned (a weight vector w_k with 25 entries plus a bias b_k for each of the 3 classes, so 78 numbers),
     the loss (cross-entropy with L2 penalty, strength 1/C), and how it is optimized (lbfgs)
   - for Random Forest: number of trees, bootstrap samples, Gini impurity, `max_features`, and how the votes become probabilities
2. **Section 3.4, the concrete example, part B.** Use the same match as Khadim's part A
   (Man City vs Leicester, 6 Feb 2016, `match_id E0_1516_20160206_ManCity`). Show its
   25 input values, the three scores w_k · x + b_k from your trained Logistic Regression,
   the softmax probabilities, and the prediction. Leicester actually won 3-1, and the
   tiers alone said "home win" with 69%. Does your model do better? Why or why not?
   Snippet:
   ```python
   from soccer_predict import classifier, pipeline

   df = pipeline.load_features()
   model = classifier.train(df[df.split == "train"], kind="logreg")
   row = df[df.match_id == "E0_1516_20160206_ManCity"]
   X = row[classifier.feature_columns()]
   lr = model[-1]
   z = model[:-1].transform(X)
   print(lr.decision_function(z), model.predict_proba(X))
   ```
   (This match is in the test season. It is used only as an illustration, never for tuning.)
3. **Section 4.3, classifier results:** the 2 x 2 table from Step 4 plus one confusion
   matrix, with two or three sentences of interpretation.
4. **Section 5, next steps:** your plan from Step 5.

Write the numbers by copying them from the JSON files your runs produce.

## Rules

- Never feed the match's own stats (goals, shots, result) into the model. A test enforces this.
- Never shuffle across seasons. Train on `train`, tune on `val`, test once.
- No em dashes anywhere: CI fails on them. Use commas, colons, or parentheses.
- If you need a new column in `features.csv`, ask Khadim or open a PR that also updates
  `docs/data-contract.md`.

## Checklist

- [ ] Step 1: data explored
- [ ] Step 2: `make_model` implemented, `make test` all green
- [ ] Step 3: LR and RF validation results saved
- [ ] Step 4: tier ablation table done
- [ ] Midterm Sections 3.3, 3.4B, 4.3, 5 written
- [ ] Report PDF built (`make pdf`) and submitted by **both** of us
- [ ] After midterm: Step 5 tuning, Step 6 final test run
