from src.clustering.pipeline import CANDIDATE_K, run_buyer_segmentation
from src.clustering.prediction import assign_purchase_segment, load_segmentation_model
from tests.helpers import make_diamonds


def test_clustering_compares_all_k_values_and_reloads_assignment(tmp_path):
    data_path = tmp_path / "diamonds.csv"
    make_diamonds(rows=120).to_csv(data_path, index=False)
    result = run_buyer_segmentation(
        data_path=data_path,
        output_root=tmp_path / "outputs",
        model_dir=tmp_path / "models",
        smoke=True,
    )
    assert tuple(result["comparison"]["K"]) == CANDIDATE_K
    assert result["selected_k"] in CANDIDATE_K
    artifact = load_segmentation_model(result["artifact"])
    assigned = assign_purchase_segment(artifact, make_diamonds(rows=1).iloc[0].to_dict())
    assert assigned.loc[0, "Cluster"] in range(result["selected_k"])
    assert isinstance(assigned.loc[0, "Buyer_Interpretation"], str)
