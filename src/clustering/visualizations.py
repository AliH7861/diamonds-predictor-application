"""Static segmentation visualizations intended for README/documentation assets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .feature_engineering import CUSTOMER_PILLARS
from .models import SegmentProfile


def generate_readme_visualizations(
    enriched_diamonds: pd.DataFrame,
    scaled_features: np.ndarray,
    comparison: pd.DataFrame,
    profiles: Sequence[SegmentProfile],
    output_dir: Path,
    random_state: int = 42,
) -> None:
    """Generate static PNG assets for the project's README.

    The function intentionally does not display plots or modify application UI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cluster-size distribution.
    shares = (
        enriched_diamonds["customer_profile_name"]
        .value_counts(normalize=True)
        .mul(100.0)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(shares.index.astype(str), shares.values)
    plt.ylabel("Share of segmented diamonds (%)")
    plt.xlabel("Buyer-preference archetype")
    plt.title("Customer-oriented diamond segments")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "segment_sizes.png", dpi=180)
    plt.close()

    # Method comparison.
    best_per_method = (
        comparison.sort_values(
            ["ValidCustomerSegments", "CompositeScore"],
            ascending=[False, False],
        )
        .groupby("Method", as_index=False)
        .first()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        best_per_method["Method"],
        best_per_method["CompositeScore"],
    )
    plt.ylabel("Practical segmentation score")
    plt.xlabel("Clustering method")
    plt.title("Best segmentation result by method")
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=180)
    plt.close()

    # Profile heatmap.
    medians = enriched_diamonds.groupby("cluster_id")[list(CUSTOMER_PILLARS)].median()
    overall = enriched_diamonds[list(CUSTOMER_PILLARS)].median()
    iqr = (
        enriched_diamonds[list(CUSTOMER_PILLARS)].quantile(0.75)
        - enriched_diamonds[list(CUSTOMER_PILLARS)].quantile(0.25)
    ).replace(0.0, np.nan)
    profile_delta = (medians - overall) / iqr

    profile_name_by_cluster = {profile.cluster_id: profile.name for profile in profiles}

    plt.figure(figsize=(11, max(5, len(profile_delta) * 0.8)))
    image = plt.imshow(
        profile_delta.to_numpy(),
        aspect="auto",
    )
    plt.xticks(
        range(len(CUSTOMER_PILLARS)),
        [name.replace("Pillar_", "") for name in CUSTOMER_PILLARS],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(len(profile_delta)),
        [profile_name_by_cluster[int(cluster_id)] for cluster_id in profile_delta.index],
    )
    plt.colorbar(
        image,
        label="Difference from overall median (IQR units)",
    )
    plt.title("Buyer-preference profile heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "profile_heatmap.png", dpi=180)
    plt.close()

    # PCA projection.
    rng = np.random.default_rng(random_state)
    pca = PCA(n_components=2, random_state=random_state)
    projection = pca.fit_transform(scaled_features)

    n_plot = min(15_000, len(enriched_diamonds))
    indices = rng.choice(
        len(enriched_diamonds),
        size=n_plot,
        replace=False,
    )

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        projection[indices, 0],
        projection[indices, 1],
        c=enriched_diamonds.iloc[indices]["cluster_id"].to_numpy(),
        s=8,
        alpha=0.5,
    )
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("2D view of customer-oriented diamond segments")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_segments.png", dpi=180)
    plt.close()
