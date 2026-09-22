"""Integration checks for persisted, reusable customer segmentation."""

from src.clustering.config import SegmentationConfig
from src.clustering.pipeline import persist_segmentation_result, run_segmentation
from src.clustering.prediction import assign_purchase_segment, load_segmentation_model
from tests.helpers import make_diamonds


def test_segmentation_compares_methods_and_reloads_assignment(tmp_path):
    frame = make_diamonds(rows=240).drop(columns="Unnamed: 0")
    config = SegmentationConfig(
        k_values=(3, 5),
        silhouette_sample_size=150,
        agglomerative_sample_size=150,
        stability_seeds=(0, 1),
    )
    result = run_segmentation(
        frame,
        config=config,
        visualization_dir=tmp_path / "figures",
    )
    assert set(result.comparison["Method"]) == {"KMeans", "GMM", "Agglomerative"}
    assert result.selected_k in {3, 5}
    assert bool(
        result.comparison.set_index("Solution").loc[
            result.selected_solution, "ValidCustomerSegments"
        ]
    )
    assert {"cluster_id", "customer_profile_name"}.issubset(result.enriched_diamonds)
    assert len(result.rag_records) == len(result.profiles)

    artifact_path = persist_segmentation_result(
        result,
        output_root=tmp_path / "outputs",
        model_dir=tmp_path / "models",
    )
    artifact = load_segmentation_model(artifact_path)
    assert artifact["artifact_version"] == 3
    assigned = assign_purchase_segment(
        artifact, make_diamonds(rows=1).drop(columns="Unnamed: 0").iloc[0].to_dict()
    )
    assert assigned.loc[0, "cluster_id"] in range(result.selected_k)
    assert isinstance(assigned.loc[0, "customer_profile_name"], str)

    for filename in (
        "segment_sizes.png",
        "method_comparison.png",
        "profile_heatmap.png",
        "pca_segments.png",
    ):
        assert (tmp_path / "figures" / filename).exists()
