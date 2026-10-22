"""Team performance tiers with k-means (owner: Khadim).

Each team-match snapshot (a team's rolling form just before kickoff) is a point
in feature space. We standardize each feature within its league, run k-means,
and relabel clusters so tier 1 is the strongest (highest points per game) and
tier K the weakest. The fitted model is frozen after training, so validation and
test matches are assigned with centroids learned only from training seasons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from soccer_predict import config

TIER_FEATURES: list[str] = [
    "form_ppg",
    "form_goals_for",
    "form_goals_against",
    "form_shots_for",
    "form_shots_against",
    "form_sot_for",
    "form_sot_against",
    "form_corners_for",
    "form_corners_against",
]


@dataclass
class LeagueScaler:
    """z-score each feature using the mean and std of its own league (training data only).

    Leagues differ in scoring rates, so 1.5 goals per game means something
    different in the Bundesliga than in Serie A. Scaling per league puts every team
    on a "relative to its league" footing before clustering.
    """

    features: list[str]
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]

    @classmethod
    def fit(cls, snapshots: pd.DataFrame, features: list[str]) -> LeagueScaler:
        grouped = snapshots.groupby("league")[features]
        means = {league: frame.mean().to_numpy() for league, frame in grouped}
        stds = {league: frame.std(ddof=0).to_numpy() for league, frame in grouped}
        return cls(features=features, means=means, stds=stds)

    def transform(self, snapshots: pd.DataFrame) -> np.ndarray:
        values = snapshots[self.features].to_numpy(dtype=float)
        mean = np.vstack([self.means[league] for league in snapshots["league"]])
        std = np.vstack([self.stds[league] for league in snapshots["league"]])
        return (values - mean) / std


@dataclass
class TierModel:
    scaler: LeagueScaler
    kmeans: KMeans
    tier_of_cluster: np.ndarray  # tier_of_cluster[cluster_id] = tier, 1 = strongest

    @property
    def n_tiers(self) -> int:
        return int(self.kmeans.n_clusters)

    @property
    def features(self) -> list[str]:
        return self.scaler.features

    def centroids(self) -> np.ndarray:
        """Centroids in standardized units, row i = tier i + 1."""
        order = np.argsort(self.tier_of_cluster)
        return np.asarray(self.kmeans.cluster_centers_)[order]

    def distances(self, snapshots: pd.DataFrame) -> np.ndarray:
        """Squared Euclidean distance from each snapshot to each tier centroid."""
        z = self.scaler.transform(snapshots)
        diff = z[:, None, :] - self.centroids()[None, :, :]
        return np.asarray((diff**2).sum(axis=2))

    def predict(self, snapshots: pd.DataFrame) -> np.ndarray:
        """Tier (1..K) of each snapshot: the nearest centroid."""
        return self.distances(snapshots).argmin(axis=1) + 1


def fit_tier_model(
    snapshots: pd.DataFrame,
    n_tiers: int = config.N_TIERS,
    features: list[str] | None = None,
    seed: int = config.RANDOM_SEED,
) -> TierModel:
    """Fit k-means on training snapshots and order clusters by strength."""
    features = features or TIER_FEATURES
    snapshots = snapshots.dropna(subset=features)
    scaler = LeagueScaler.fit(snapshots, features)
    z = scaler.transform(snapshots)
    kmeans = KMeans(n_clusters=n_tiers, init="k-means++", n_init=10, random_state=seed).fit(z)

    ppg_by_cluster = (
        pd.Series(snapshots["form_ppg"].to_numpy()).groupby(kmeans.labels_).mean().sort_index()
    )
    strongest_first = np.argsort(-ppg_by_cluster.to_numpy())
    tier_of_cluster = np.empty(n_tiers, dtype=int)
    tier_of_cluster[strongest_first] = np.arange(1, n_tiers + 1)
    return TierModel(scaler=scaler, kmeans=kmeans, tier_of_cluster=tier_of_cluster)


def side_snapshots(features: pd.DataFrame, side: str) -> pd.DataFrame:
    """Pull one side's form columns out of a match-level feature table."""
    prefix = f"{side}_"
    cols = {c: c.removeprefix(prefix) for c in features.columns if c.startswith(prefix + "form_")}
    out = features[["league", *cols]].rename(columns=cols)
    out["team"] = features[f"{side}_team"]
    return out


def add_tiers(features: pd.DataFrame, model: TierModel) -> pd.DataFrame:
    """Add home_tier, away_tier and tier_diff (positive = home side is stronger)."""
    out = features.copy()
    for side in ("home", "away"):
        out[f"{side}_tier"] = model.predict(side_snapshots(features, side))
    out["tier_diff"] = out["away_tier"] - out["home_tier"]
    return out


def sweep_k(
    snapshots: pd.DataFrame,
    ks: list[int] = config.K_SWEEP,
    features: list[str] | None = None,
    seed: int = config.RANDOM_SEED,
    silhouette_sample: int = 5000,
) -> pd.DataFrame:
    """Inertia and silhouette for each K (silhouette on a random sample to keep it fast)."""
    features = features or TIER_FEATURES
    snapshots = snapshots.dropna(subset=features)
    z = LeagueScaler.fit(snapshots, features).transform(snapshots)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(z)
        sil = silhouette_score(
            z, km.labels_, sample_size=min(silhouette_sample, len(z)), random_state=seed
        )
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def tier_pair_table(train: pd.DataFrame, n_tiers: int, alpha: float = 1.0) -> np.ndarray:
    """P(result | home_tier, away_tier) from training counts with add-alpha smoothing.

    Shape (K, K, 3); last axis is H, D, A. Used to score how much outcome
    information the tiers carry on their own, before any classifier.
    """
    counts = np.full((n_tiers, n_tiers, 3), alpha)
    np.add.at(
        counts,
        (train["home_tier"].to_numpy() - 1, train["away_tier"].to_numpy() - 1, train["target"]),
        1,
    )
    return np.asarray(counts / counts.sum(axis=2, keepdims=True))


def tier_pair_proba(table: np.ndarray, matches: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        table[matches["home_tier"].to_numpy() - 1, matches["away_tier"].to_numpy() - 1]
    )
