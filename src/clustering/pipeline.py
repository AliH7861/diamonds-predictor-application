"""End-to-end K-Means comparison, profiling, selection, and persistence."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans

from config import PROJECT_ROOT, RANDOM_STATE, RAW_DATA_PATH

from .evaluation import add_selection_score, evaluate_kmeans
from .preprocessing import prepare_clustering_data
from .profiling import build_cluster_profiles
from .visualizations import save_comparison_plots, save_selected_plots


CANDIDATE_K = (3, 5, 7, 10)


def run_buyer_segmentation(
    data_path: str | Path = RAW_DATA_PATH,
    output_root: str | Path = PROJECT_ROOT / "outputs" / "clustering",
    model_dir: str | Path = PROJECT_ROOT / "models" / "clustering",
    smoke: bool = False,
) -> dict:
    """Compare all candidate K values and save the statistically balanced winner."""
    output_root = Path(output_root)
    model_dir = Path(model_dir)
    if smoke:
        output_root /= "smoke_test"
        model_dir /= "smoke_test"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_clustering_data(data_path, sample_rows=2500 if smoke else None)
    frame, matrix = prepared["profiles"], prepared["matrix"]
    runs = {}
    metric_rows = []
    all_summaries = []
    all_numeric = []
    all_categorical = []

    print(f"Prepared {len(frame):,} purchase profiles with {matrix.shape[1]} encoded inputs.")
    for k in CANDIDATE_K:
        print(f"Training K-Means with K={k}...", flush=True)
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=5 if smoke else 20,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(matrix)
        metrics = evaluate_kmeans(model, matrix, labels, sample_size=1000 if smoke else 5000)
        profiles = build_cluster_profiles(frame, labels, k)
        runs[k] = {"model": model, "labels": labels, "profiles": profiles}
        metric_rows.append(metrics)
        all_summaries.append(profiles["summary"])
        all_numeric.append(profiles["numeric"])
        all_categorical.append(profiles["categorical"])
        print(
            f"  Silhouette={metrics['Silhouette']:.4f}  "
            f"Inertia={metrics['Inertia']:,.0f}  "
            f"Smallest cluster={metrics['Min_Cluster_Pct']:.2f}%",
            flush=True,
        )

    comparison = add_selection_score(pd.DataFrame(metric_rows))
    selected_k = int(comparison.loc[comparison["Selection_Score"].idxmax(), "K"])
    selected = runs[selected_k]
    summary = pd.concat(all_summaries, ignore_index=True)
    numeric = pd.concat(all_numeric, ignore_index=True)
    categorical = pd.concat(all_categorical, ignore_index=True)

    comparison.to_csv(tables_dir / "k_comparison.csv", index=False)
    summary.to_csv(tables_dir / "all_cluster_profiles.csv", index=False)
    numeric.to_csv(tables_dir / "numeric_cluster_statistics.csv", index=False)
    categorical.to_csv(tables_dir / "categorical_cluster_distributions.csv", index=False)
    selected["profiles"]["summary"].to_csv(
        tables_dir / "selected_buyer_segments.csv", index=False
    )
    selected["profiles"]["labeled"].to_csv(
        tables_dir / "purchase_profiles_with_clusters.csv", index=False
    )

    save_comparison_plots(comparison, figures_dir)
    save_selected_plots(matrix, selected["labels"], selected["profiles"]["summary"], figures_dir)
    joblib.dump(
        {
            "preprocessor": prepared["preprocessor"],
            "model": selected["model"],
            "selected_k": selected_k,
            "segment_profiles": selected["profiles"]["summary"],
        },
        model_dir / "buyer_segmentation.joblib",
    )

    print(f"Selected K={selected_k} using separation, centroid distance, balance, and simplicity.")
    print(selected["profiles"]["summary"].to_string(index=False))
    return {
        "comparison": comparison,
        "selected_k": selected_k,
        "selected_profiles": selected["profiles"]["summary"],
        "artifact": model_dir / "buyer_segmentation.joblib",
    }
