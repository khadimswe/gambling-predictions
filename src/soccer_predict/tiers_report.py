"""Midterm experiment for the tier model (owner: Khadim).

`soccer tiers-report` rebuilds the features and writes every number and figure
the midterm report cites into reports/results and reports/figures:

  k_sweep.csv               inertia, silhouette, validation scores for K = 2..8
  window_sensitivity.csv    same validation scores for different rolling windows
  tier_profiles.csv         mean raw form per tier and the centroids in z units
  outcome_by_tier_diff.csv  H/D/A rates by tier difference (val + test)
  baselines.json            reference models on val and test
  worked_example.json       one match pushed through every step of the algorithm
  lloyd_toy.json            a small k-means run on six real teams, step by step
  summary.json              dataset sizes and headline numbers
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from soccer_predict import config, evaluate, features, pipeline, plots, tiers

DEFAULT_EXAMPLE = "E0_1516_20160206_ManCity"
TOY_DATE_RANGE = ("2016-02-06", "2016-02-07")
TOY_TEAMS = ["Man City", "Leicester", "Tottenham", "Arsenal", "Aston Villa", "Norwich"]
TOY_FEATURES = ["form_ppg", "form_sot_for"]


def _write_json(name: str, payload: Any) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / name).write_text(json.dumps(payload, indent=2, default=float) + "\n")


def _metrics(y: pd.Series, pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, Any]:
    result = evaluate.evaluate(y, pred, proba)
    rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}
    return rounded


def _val_scores(team_form: pd.DataFrame, match_feats: pd.DataFrame, k: int) -> dict[str, Any]:
    model = tiers.fit_tier_model(pipeline.training_snapshots(team_form), n_tiers=k)
    tiered = tiers.add_tiers(match_feats, model)
    train, val = tiered[tiered["split"] == "train"], tiered[tiered["split"] == "val"]
    proba = tiers.tier_pair_proba(tiers.tier_pair_table(train, k), val)
    return {
        "val_pair_table_log_loss": evaluate.evaluate(val["target"], proba.argmax(1), proba)[
            "log_loss"
        ],
        "val_tier_rule_accuracy": evaluate.evaluate(val["target"], evaluate.predict_tier_rule(val))[
            "accuracy"
        ],
        "n_val": len(val),
    }


def k_sweep(team_form: pd.DataFrame, match_feats: pd.DataFrame) -> pd.DataFrame:
    sweep = tiers.sweep_k(pipeline.training_snapshots(team_form))
    extra = pd.DataFrame([_val_scores(team_form, match_feats, k) for k in sweep["k"]])
    return pd.concat([sweep, extra.drop(columns="n_val")], axis=1)


def window_sensitivity(matches: pd.DataFrame) -> pd.DataFrame:
    team_matches = features.to_team_matches(matches)
    rows = []
    for window, min_periods in [(5, 3), (10, 5), (15, 5), (20, 5)]:
        team_form = features.add_rolling_form(team_matches, window, min_periods)
        match_feats = features.build_match_features(matches, team_form)
        scores = _val_scores(team_form, match_feats, config.N_TIERS)
        rows.append({"window": window, "min_prior_matches": min_periods, **scores})
    return pd.DataFrame(rows)


def tier_profiles(snapshots: pd.DataFrame, model: tiers.TierModel) -> pd.DataFrame:
    tiered = snapshots.assign(tier=model.predict(snapshots))
    raw = tiered.groupby("tier")[tiers.TIER_FEATURES].mean()
    centroids = pd.DataFrame(
        model.centroids(), index=raw.index, columns=[f"z_{c}" for c in tiers.TIER_FEATURES]
    )
    sizes = tiered["tier"].value_counts().sort_index().rename("n_snapshots")
    return pd.concat([sizes, raw, centroids], axis=1).reset_index()


def outcome_by_tier_diff(matches: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(matches["tier_diff"], matches["result"])[["H", "D", "A"]]
    rates = counts.div(counts.sum(axis=1), axis=0).add_prefix("p_")
    return pd.concat([counts.sum(axis=1).rename("n"), rates], axis=1).reset_index()


def baselines(tiered: pd.DataFrame, model: tiers.TierModel) -> dict[str, Any]:
    train = tiered[tiered["split"] == "train"]
    table = tiers.tier_pair_table(train, model.n_tiers)
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        part = tiered[tiered["split"] == split]
        y = part["target"]
        pair = tiers.tier_pair_proba(table, part)
        with_odds = part.dropna(subset=["odds_home", "odds_draw", "odds_away"])
        book = evaluate.bookmaker_proba(with_odds)
        out[split] = {
            "always_home_win": _metrics(
                y, evaluate.predict_home_win(part), evaluate.class_prior_proba(train, part)
            ),
            "tier_rule": _metrics(y, evaluate.predict_tier_rule(part)),
            "tier_pair_table": _metrics(y, pair.argmax(axis=1), pair),
            "bookmaker_reference": _metrics(with_odds["target"], book.argmax(axis=1), book),
        }
    out["tier_pair_table_probabilities"] = {
        f"{h + 1}v{a + 1}": [round(p, 3) for p in table[h, a]]
        for h in range(model.n_tiers)
        for a in range(model.n_tiers)
    }
    return out


def worked_example(tiered: pd.DataFrame, model: tiers.TierModel, match_id: str) -> dict[str, Any]:
    row = tiered[tiered["match_id"] == match_id]
    if row.empty:
        raise ValueError(f"match_id {match_id!r} not found in the feature table")
    league = row["league"].iloc[0]
    train = tiered[tiered["split"] == "train"]
    table = tiers.tier_pair_table(train, model.n_tiers)

    sides = {}
    for side in ("home", "away"):
        snap = tiers.side_snapshots(row, side)
        z = model.scaler.transform(snap)[0]
        dist = model.distances(snap)[0]
        sides[side] = {
            "team": snap["team"].iloc[0],
            "form": {c: round(float(snap[c].iloc[0]), 3) for c in model.features},
            "league_mean": dict(
                zip(model.features, np.round(model.scaler.means[league], 3).tolist(), strict=True)
            ),
            "league_std": dict(
                zip(model.features, np.round(model.scaler.stds[league], 3).tolist(), strict=True)
            ),
            "z": dict(zip(model.features, np.round(z, 3).tolist(), strict=True)),
            "squared_distance_to_tier": {
                str(t + 1): round(float(d), 3) for t, d in enumerate(dist)
            },
            "tier": int(dist.argmin() + 1),
        }
    home_tier, away_tier = sides["home"]["tier"], sides["away"]["tier"]
    pair = table[home_tier - 1, away_tier - 1]
    return {
        "match_id": match_id,
        "date": str(row["date"].iloc[0].date()),
        "league": config.LEAGUES[league],
        "season": config.season_label(row["season"].iloc[0]),
        "home": sides["home"],
        "away": sides["away"],
        "tier_diff": away_tier - home_tier,
        "tier_rule_prediction": evaluate.LABELS[int(evaluate.predict_tier_rule(row)[0])],
        "tier_pair_probabilities": dict(
            zip(evaluate.LABELS, np.round(pair, 3).tolist(), strict=True)
        ),
        "actual_result": row["result"].iloc[0],
        "centroids_z": {
            str(t + 1): dict(zip(model.features, np.round(c, 3).tolist(), strict=True))
            for t, c in enumerate(model.centroids())
        },
    }


def lloyd_trace(points: np.ndarray, init: np.ndarray, max_iter: int = 10) -> list[dict[str, Any]]:
    """Plain Lloyd's algorithm, recording each assign/update step (for the toy example)."""
    centroids = init.astype(float).copy()
    steps = []
    for iteration in range(1, max_iter + 1):
        dist = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new = np.vstack([points[labels == j].mean(axis=0) for j in range(len(centroids))])
        inertia = float(dist[np.arange(len(points)), labels].sum())
        steps.append(
            {
                "iteration": iteration,
                "squared_distances": np.round(dist, 3).tolist(),
                "assignment": labels.tolist(),
                "inertia": round(inertia, 3),
                "new_centroids": np.round(new, 3).tolist(),
            }
        )
        if np.allclose(new, centroids):
            break
        centroids = new
    return steps


def lloyd_toy(team_form: pd.DataFrame, model: tiers.TierModel) -> dict[str, Any]:
    start, end = TOY_DATE_RANGE
    snaps = team_form[
        (team_form["league"] == "E0")
        & team_form["date"].between(start, end)
        & team_form["team"].isin(TOY_TEAMS)
    ].set_index("team")
    snaps = snaps.loc[TOY_TEAMS]
    z = model.scaler.transform(snaps.reset_index())
    idx = [model.features.index(c) for c in TOY_FEATURES]
    points = z[:, idx]
    init = points[[0, 4]]  # fixed start (Man City, Aston Villa) so the example is reproducible
    return {
        "date_range": TOY_DATE_RANGE,
        "features": TOY_FEATURES,
        "teams": TOY_TEAMS,
        "raw": snaps[TOY_FEATURES].round(3).to_numpy().tolist(),
        "z": np.round(points, 3).tolist(),
        "initial_centroids": np.round(init, 3).tolist(),
        "steps": lloyd_trace(points, init),
    }


def season_sanity(team_form: pd.DataFrame) -> dict[str, Any]:
    """Spearman correlation between a team's mean tier and its final points, per league-season."""
    held_out = team_form[team_form["split"].isin(["val", "test"])]
    per_team = held_out.groupby(["league", "season", "team"]).agg(
        mean_tier=("tier", "mean"), points=("points", "sum")
    )
    corrs = per_team.groupby(["league", "season"]).apply(
        lambda g: g["mean_tier"].corr(g["points"], method="spearman")
    )
    return {
        "per_league_season": {f"{lg}_{s}": round(float(v), 3) for (lg, s), v in corrs.items()},
        "mean": round(float(corrs.mean()), 3),
    }


def plot_k_selection(sweep: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("inertia", "Inertia (within-cluster SSE)", "lower = tighter clusters"),
        ("silhouette", "Silhouette (5,000-point sample)", "higher = better separated"),
        ("val_pair_table_log_loss", "Validation log loss", "tier-pair lookup, lower = better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), sharex=True)
    for ax, (col, title, ylabel) in zip(axes, panels, strict=True):
        ax.plot(sweep["k"], sweep[col], color=plots.SERIES[0], marker="o", markersize=5)
        chosen = sweep[sweep["k"] == config.N_TIERS]
        ax.plot(chosen["k"], chosen[col], "o", markersize=9, color=plots.SERIES[1], zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("K (number of tiers)")
        ax.set_xticks(sweep["k"])
        ax.grid(axis="x", visible=False)
    axes[1].annotate(
        f"chosen K = {config.N_TIERS}",
        xy=(config.N_TIERS, float(sweep.loc[sweep["k"] == config.N_TIERS, "silhouette"].iloc[0])),
        xytext=(12, 10),
        textcoords="offset points",
        color=plots.INK_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "k_selection.png")


def plot_tier_profiles(profiles: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    labels = {
        "form_ppg": "Points per game",
        "form_goals_for": "Goals for",
        "form_goals_against": "Goals against",
        "form_shots_for": "Shots for",
        "form_shots_against": "Shots against",
        "form_sot_for": "Shots on target for",
        "form_sot_against": "Shots on target against",
        "form_corners_for": "Corners for",
        "form_corners_against": "Corners against",
    }
    z = profiles[[f"z_{c}" for c in labels]].to_numpy().T
    raw = profiles[list(labels)].to_numpy().T
    # Color says "better or worse than the league average": conceding less is good,
    # so the sign is flipped for the *_against rows before coloring.
    better = np.array([-1.0 if c.endswith("_against") else 1.0 for c in labels])[:, None] * z
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    limit = float(np.abs(better).max())
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.imshow(better, cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(
                j,
                i,
                f"{raw[i, j]:.2f}\n(z {z[i, j]:+.2f})",
                ha="center",
                va="center",
                fontsize=7.5,
                color=plots.INK,
            )
    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(
        [
            f"Tier {t}\n{n:,} snapshots"
            for t, n in zip(profiles["tier"], profiles["n_snapshots"], strict=True)
        ]
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(labels.values()))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Tier profiles: mean per match (z vs. league average)", pad=16)
    ax.text(
        0,
        1.01,
        "blue = better than league average, red = worse",
        transform=ax.transAxes,
        color=plots.MUTED,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "tier_profiles.png")


def plot_outcome_by_tier_diff(table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for result in ("H", "D", "A"):
        widths = table[f"p_{result}"].to_numpy()
        ax.barh(
            y,
            widths,
            left=left,
            height=0.62,
            color=plots.OUTCOME_COLORS[result],
            edgecolor="white",
            linewidth=2,
            label=names[result],
        )
        for yi, (lo, w) in enumerate(zip(left, widths, strict=True)):
            if w >= 0.08:
                ink = "white" if result != "D" else plots.INK
                ax.text(lo + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=8, color=ink)
        left += widths
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{d:+d}  (n={n:,})" if d else f"0  (n={n:,})"
            for d, n in zip(table["tier_diff"], table["n"], strict=True)
        ]
    )
    ax.set_ylabel("tier_diff = away tier - home tier\n(positive: home side stronger)")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Outcome rates by tier difference, held-out seasons 2014/15 and 2015/16")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8)
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "outcome_by_tier_diff.png")


def run(example_match_id: str = DEFAULT_EXAMPLE) -> dict[str, Any]:
    plots.apply_style()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    built = pipeline.build(save=True)
    matches, team_form, model = built.matches, built.team_form, built.tier_model
    odds = matches.set_index("match_id")[["odds_home", "odds_draw", "odds_away"]]
    tiered = built.features.join(odds, on="match_id")
    untiered = built.features.drop(columns=["home_tier", "away_tier", "tier_diff"])

    sweep = k_sweep(team_form, untiered)
    sweep.round(4).to_csv(config.RESULTS_DIR / "k_sweep.csv", index=False)
    plot_k_selection(sweep)

    window_sensitivity(matches).round(4).to_csv(
        config.RESULTS_DIR / "window_sensitivity.csv", index=False
    )

    profiles = tier_profiles(pipeline.training_snapshots(team_form), model)
    profiles.round(3).to_csv(config.RESULTS_DIR / "tier_profiles.csv", index=False)
    plot_tier_profiles(profiles)

    held_out = tiered[tiered["split"].isin(["val", "test"])]
    by_diff = outcome_by_tier_diff(held_out)
    by_diff.round(4).to_csv(config.RESULTS_DIR / "outcome_by_tier_diff.csv", index=False)
    plot_outcome_by_tier_diff(by_diff)

    _write_json("baselines.json", baselines(tiered, model))
    _write_json("worked_example.json", worked_example(tiered, model, example_match_id))
    _write_json("lloyd_toy.json", lloyd_toy(team_form, model))

    summary = {
        "leagues": list(config.LEAGUES.values()),
        "seasons": [config.season_label(s) for s in config.SEASONS],
        "n_matches_clean": len(matches),
        "n_matches_with_features": len(tiered),
        "matches_by_split": tiered["split"].value_counts().to_dict(),
        "n_training_snapshots": len(pipeline.training_snapshots(team_form)),
        "result_rates_train": tiered[tiered["split"] == "train"]["result"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "rolling_window": config.ROLLING_WINDOW,
        "min_prior_matches": config.MIN_PRIOR_MATCHES,
        "n_tiers": config.N_TIERS,
        "tier_vs_final_points_spearman": season_sanity(team_form),
        "seed_agreement": seed_agreement(pipeline.training_snapshots(team_form)),
    }
    _write_json("summary.json", summary)
    return summary


def seed_agreement(snapshots: pd.DataFrame, seeds: tuple[int, ...] = (1, 2, 3)) -> float:
    """Share of training snapshots that keep the same tier when k-means is reseeded."""
    base = tiers.fit_tier_model(snapshots).predict(snapshots)
    rates = [
        float((tiers.fit_tier_model(snapshots, seed=s).predict(snapshots) == base).mean())
        for s in seeds
    ]
    return round(min(rates), 4)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=float))
