# Working on this repo

## One-time setup

```bash
make setup     # creates .venv, installs the package + dev tools, installs git hooks
make data      # downloads the raw CSVs (about 1 minute)
make features  # builds data/processed/features.csv
```

Without `make`:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,report]"
pre-commit install
soccer download
soccer build
```

`gitleaks` for `make secrets` is a separate install (`brew install gitleaks` on macOS).
CI runs it anyway, so it is optional locally.

**`CERTIFICATE_VERIFY_FAILED` on macOS** (usually during `pre-commit install` or the
first commit): the python.org installer ships without root certificates. Fix it once
by double-clicking `/Applications/Python 3.12/Install Certificates.command` (match your
version), or for one shell: `export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"`.
`soccer download` already works around this on its own.

## Daily flow

1. `git switch main && git pull`
2. `git switch -c ryann/logreg` (prefix branches with your name)
3. Make changes. Commit early; the hooks auto-fix formatting and block em dashes and secrets.
4. `make check` before pushing.
5. `git push -u origin HEAD` and open a pull request. The other person reviews.
6. Merge when CI is green.

Direct commits to `main` are blocked by a local hook. If you really need one (for
example the very first commit), use `SKIP=no-commit-to-branch git commit ...`.

## Rules that protect the results

- **Never use a match's own stats to predict it.** Only `*_form_*`, `*_tier`, and
  `tier_diff` columns are pre-match. `classifier.feature_columns()` enforces this
  and a test checks it.
- **Split by season, never shuffle.** Train 2008/09 to 2013/14, tune on 2014/15,
  touch 2015/16 (`--test`) once at the end.
- **Numbers in the report come from a command.** If you change code, rerun
  `make tiers` or `soccer train` and paste the regenerated values.
- **No data or secrets in git.** `data/` is ignored; `kaggle.json` and `.env` are
  ignored; gitleaks scans every commit in CI.
- **No em dashes** in code, docs, or the report. Use a comma, colon, parentheses,
  or " - ". `make text` lists every one with file and line.

## If CI fails

| Job | Fix |
|---|---|
| Lint and format | `make format`, commit the result |
| Type check | read the mypy message; add or fix the type hint |
| Tests | `make test`, read the first failure |
| No em dashes | `make text` shows `file:line:col`; rewrite those sentences |
| File hygiene | `pre-commit run --all-files`, commit what it fixed |
| Secret scan | remove the secret, **rotate it**, and tell your partner (it is in history) |

## Building PDFs

`make pdf` renders the two specs and the midterm report into `build/pdf/` using
headless Chrome. Without Chrome, open the generated `.html` in any browser and
print to PDF.
