"""End-to-end customer-oriented segmentation and artifact persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PROJECT_ROOT, RAW_DATA_PATH
from src.datasets import load_diamond_frame

from .clustering import run_candidate_clustering
from .config import SegmentationConfig
from .evaluation import evaluate_candidates, select_best_solution
from .feature_engineering import (
    CUSTOMER_PILLARS,
    REQUIRED_COLUMNS,
    build_customer_pillars,
    build_segmentation_features,
    split_geometry_anomalies,
)
from .models import SegmentationResult
from .profiling import attach_profile_names, build_profiles
from .rag import build_rag_records
from .visualizations import generate_readme_visualizations

LOGGER = logging.getLogger(__name__)
CANDIDATE_K = (3, 5, 7, 10)


class SegmentationPipeline:
    """Run segmentation from an existing cleaned diamond DataFrame."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()

    def run(
        self,
        frame: pd.DataFrame,
        *,
        visualization_dir: Path | None = None,
    ) -> SegmentationResult:
        """Return authoritative profiles, assignments, metrics, and RAG records."""
        engineered = build_segmentation_features(frame, self.config)
        segmentable, anomalies = split_geometry_anomalies(engineered, self.config)
        segmentable = build_customer_pillars(segmentable)

        pillar_frame = segmentable[list(CUSTOMER_PILLARS)].replace([np.inf, -np.inf], np.nan)
        pillar_frame = pillar_frame.fillna(pillar_frame.median(numeric_only=True))
        if pillar_frame.empty or not np.isfinite(pillar_frame.to_numpy()).all():
            raise ValueError("Segmentation pillars must contain finite values.")

        scaler = StandardScaler()
        matrix = scaler.fit_transform(pillar_frame)
        solutions = run_candidate_clustering(matrix, self.config)
        comparison = evaluate_candidates(matrix, solutions, self.config)
        selected_name = select_best_solution(comparison)
        selected = solutions[selected_name]

        profiles, pillar_medians, distances = build_profiles(segmentable, matrix, selected.labels)
        enriched = attach_profile_names(segmentable, selected.labels, profiles)
        rag_records = build_rag_records(profiles)

        if visualization_dir is not None:
            generate_readme_visualizations(
                enriched,
                matrix,
                comparison,
                profiles,
                visualization_dir,
                self.config.random_state,
            )

        selected_row = comparison.loc[comparison["Solution"].eq(selected_name)].iloc[0]
        return SegmentationResult(
            selected_solution=selected_name,
            selected_method=selected.method,
            selected_k=selected.k,
            enriched_diamonds=enriched,
            profiles=profiles,
            comparison=comparison,
            profile_distances=distances,
            pillar_medians=pillar_medians,
            anomaly_rows=anomalies,
            rag_records=rag_records,
            metadata={
                "input_rows": int(len(frame)),
                "segmented_rows": int(len(enriched)),
                "anomaly_rows": int(len(anomalies)),
                "customer_pillars": list(CUSTOMER_PILLARS),
                "selected_metrics": selected_row.to_dict(),
                "scaler": scaler,
                "selected_model": selected.model,
                "reference_frame": segmentable[list(REQUIRED_COLUMNS)].copy(),
            },
        )


def run_segmentation(
    frame: pd.DataFrame,
    *,
    config: SegmentationConfig | None = None,
    visualization_dir: Path | None = None,
) -> SegmentationResult:
    """Run customer segmentation using a DataFrame owned by the host application."""
    return SegmentationPipeline(config).run(
        frame,
        visualization_dir=visualization_dir,
    )


def _json_default(value):
    """Convert NumPy and pandas values into JSON-compatible Python values."""
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def _json_safe(value):
    """Recursively replace non-finite metric values with JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def persist_segmentation_result(
    result: SegmentationResult,
    *,
    output_root: str | Path,
    model_dir: str | Path,
) -> Path:
    """Save the selected model, profile registry, tables, and enriched data."""
    output_root = Path(output_root)
    model_dir = Path(model_dir)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    profile_records = [asdict(profile) for profile in result.profiles]
    result.comparison.to_csv(tables_dir / "method_comparison.csv", index=False)
    result.enriched_diamonds.to_csv(tables_dir / "diamonds_with_customer_profiles.csv", index=False)
    result.profile_distances.to_csv(tables_dir / "profile_distances.csv", index=False)
    result.pillar_medians.to_csv(tables_dir / "profile_pillar_medians.csv")
    result.anomaly_rows.to_csv(tables_dir / "geometry_anomalies.csv", index=False)
    (model_dir / "profile_registry.json").write_text(
        json.dumps(profile_records, indent=2, default=_json_default), encoding="utf-8"
    )
    (model_dir / "rag_records.json").write_text(
        json.dumps(result.rag_records, indent=2, default=_json_default), encoding="utf-8"
    )

    artifact_path = model_dir / "buyer_segmentation.joblib"
    joblib.dump(
        {
            "artifact_version": 3,
            "method": result.selected_method,
            "selected_solution": result.selected_solution,
            "selected_k": result.selected_k,
            "model": result.metadata["selected_model"],
            "scaler": result.metadata["scaler"],
            "customer_pillars": list(CUSTOMER_PILLARS),
            "profiles": result.profiles,
            "reference_frame": result.metadata["reference_frame"],
        },
        artifact_path,
    )
    manifest = {
        "artifact_version": 3,
        "selected_solution": result.selected_solution,
        "method": result.selected_method,
        "selected_k": result.selected_k,
        "customer_pillars": list(CUSTOMER_PILLARS),
        "input_rows": result.metadata["input_rows"],
        "segmented_rows": result.metadata["segmented_rows"],
        "anomaly_rows": result.metadata["anomaly_rows"],
        "selected_metrics": result.metadata["selected_metrics"],
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    return artifact_path


def run_buyer_segmentation(
    data_path: str | Path = RAW_DATA_PATH,
    output_root: str | Path = PROJECT_ROOT / "outputs" / "clustering",
    model_dir: str | Path = PROJECT_ROOT / "models" / "clustering",
    smoke: bool = False,
) -> dict:
    """Load once, run the DataFrame API, and save reusable outputs."""
    frame = load_diamond_frame(data_path)
    if smoke and len(frame) > 2500:
        frame = frame.sample(2500, random_state=42).reset_index(drop=True)

    if smoke:
        output_root = Path(output_root) / "smoke_test"
        model_dir = Path(model_dir) / "smoke_test"
    config = SegmentationConfig(
        k_values=CANDIDATE_K if not smoke else (3, 5),
        silhouette_sample_size=1000 if smoke else 7000,
        agglomerative_sample_size=1000 if smoke else 8000,
        stability_seeds=(0, 1) if smoke else (0, 1, 2, 3, 4),
    )
    LOGGER.info("Running customer segmentation for %s diamonds", f"{len(frame):,}")
    result = run_segmentation(frame, config=config, visualization_dir=Path(output_root) / "figures")
    artifact = persist_segmentation_result(result, output_root=output_root, model_dir=model_dir)
    return {
        "comparison": result.comparison,
        "selected_solution": result.selected_solution,
        "selected_method": result.selected_method,
        "selected_k": result.selected_k,
        "selected_profiles": pd.DataFrame([asdict(item) for item in result.profiles]),
        "enriched_diamonds": result.enriched_diamonds,
        "rag_records": result.rag_records,
        "artifact": artifact,
    }
