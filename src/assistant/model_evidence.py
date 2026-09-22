"""Load and execute the project's real saved ML artifacts.

Validation and feature ordering remain in the original prediction modules.
This adapter adds uniform provenance and execution traces for the assistant.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd


class ModelEvidenceProvider:
    """Load production artifacts once and expose normalized inference evidence."""

    def __init__(self, project_root: str | Path):
        root = Path(project_root)
        self.classification_dir = root / "models" / "classification"
        self.regression_dir = root / "models" / "regression"
        self.clustering_dir = root / "models" / "clustering"
        self.clustering_path = self.clustering_dir / "buyer_segmentation.joblib"
        self.classifier = None
        self.regressor = None
        self.clusterer = None
        self.load_errors: list[str] = []
        self.metadata: dict[str, dict[str, Any]] = {}
        self._load_artifacts()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _load_artifacts(self) -> None:
        """Load each independent artifact while preserving useful load errors."""
        if (self.classification_dir / "metadata.json").is_file():
            try:
                from src.classification.prediction import load_best_model

                self.classifier = load_best_model(self.classification_dir)
                self.metadata["classification"] = self.classifier["metadata"]
            except Exception as error:  # pragma: no cover - runtime specific
                self.load_errors.append(f"Classification model unavailable: {error}")
        if (self.regression_dir / "pipeline.joblib").is_file():
            try:
                from src.regression.prediction import load_best_model

                self.regressor = load_best_model(self.regression_dir)
                self.metadata["regression"] = self._read_json(self.regression_dir / "metadata.json")
            except Exception as error:  # pragma: no cover - runtime specific
                self.load_errors.append(f"Regression model unavailable: {error}")
        if self.clustering_path.is_file():
            try:
                from src.clustering.prediction import load_segmentation_model

                self.clusterer = load_segmentation_model(self.clustering_path)
                self.metadata["clustering"] = self._read_json(self.clustering_dir / "manifest.json")
            except Exception as error:  # pragma: no cover - runtime specific
                self.load_errors.append(f"Clustering model unavailable: {error}")

    def status(self) -> dict[str, Any]:
        """Describe saved-model availability for the developer trace."""
        return {
            "classification_loaded": self.classifier is not None,
            "regression_loaded": self.regressor is not None,
            "clustering_loaded": self.clusterer is not None,
            "errors": list(self.load_errors),
        }

    def required_inputs(self, tool: str) -> list[str]:
        """Return the exact raw input schema for one saved tool."""
        if tool in {"classification", "regression"}:
            return list(self.metadata.get(tool, {}).get("required_inputs", []))
        if tool == "clustering":
            from src.clustering.feature_engineering import REQUIRED_COLUMNS

            return list(REQUIRED_COLUMNS)
        raise ValueError(f"Unknown model tool: {tool}")

    def _artifact_details(self, tool: str) -> dict[str, Any]:
        metadata = self.metadata.get(tool, {})
        if tool == "classification":
            return {
                "artifact_path": str(self.classification_dir / metadata.get("model_file", "")),
                "model_id": metadata.get("model_name"),
                "model_type": metadata.get("model_type"),
                "model_version": metadata.get("saved_at"),
                "preprocessor": str(self.classification_dir / "preprocessor.joblib"),
            }
        if tool == "regression":
            return {
                "artifact_path": str(self.regression_dir / metadata.get("model_file", "")),
                "model_id": f"{metadata.get('algorithm')} / {metadata.get('feature_set')}",
                "model_type": metadata.get("algorithm"),
                "model_version": metadata.get("provenance", {}).get("version", "metadata-current"),
                "preprocessor": str(self.regression_dir / "pipeline.joblib"),
            }
        return {
            "artifact_path": str(self.clustering_path),
            "model_id": metadata.get("selected_solution"),
            "model_type": metadata.get("method"),
            "model_version": metadata.get("artifact_version"),
            "preprocessor": "embedded scaler + clustering feature engineering",
        }

    @staticmethod
    def _profile_dict(profile: Any) -> dict[str, Any]:
        return asdict(profile) if is_dataclass(profile) else dict(profile)

    def execute(self, tool: str, raw_inputs: dict[str, Any]) -> tuple[dict, dict]:
        """Execute one real model and return normalized evidence plus a trace."""
        required = self.required_inputs(tool)
        missing = [name for name in required if name not in raw_inputs]
        trace = {
            "tool": tool,
            **self._artifact_details(tool),
            "expected_features": required,
            "received_features": sorted(raw_inputs),
            "missing_features": missing,
            "inference_success": False,
            "raw_output": None,
            "normalized_output": None,
            "latency_ms": 0.0,
        }
        if missing:
            trace["skip_reason"] = "missing required model inputs"
            return {}, trace

        inputs = {name: raw_inputs[name] for name in required}
        started = perf_counter()
        try:
            if tool == "classification":
                if self.classifier is None:
                    raise RuntimeError("classification artifact is unavailable")
                from src.classification.prediction import predict_diamonds

                raw = predict_diamonds(self.classifier, inputs)[0]
                output = {
                    "source": "classification_model",
                    "predicted_class": raw["clarity_family"],
                    "class_probabilities": raw.get("probabilities", {}),
                    **self._artifact_details(tool),
                    "required_features": required,
                    "input_validation": "passed",
                }
            elif tool == "regression":
                if self.regressor is None:
                    raise RuntimeError("regression artifact is unavailable")
                from src.regression.prediction import predict_prices

                raw = predict_prices(self.regressor, inputs)[0]
                output = {
                    "source": "regression_model",
                    "prediction": raw["predicted_price"],
                    **self._artifact_details(tool),
                    "required_features": required,
                    "input_validation": "passed",
                }
            elif tool == "clustering":
                if self.clusterer is None:
                    raise RuntimeError("clustering artifact is unavailable")
                from src.clustering.prediction import assign_purchase_segment

                assigned = assign_purchase_segment(self.clusterer, inputs).iloc[0]
                cluster_id = int(assigned["cluster_id"])
                profile = next(
                    item
                    for item in self.clusterer["profiles"]
                    if int(item.cluster_id) == cluster_id
                )
                raw = {
                    "cluster_id": cluster_id,
                    "customer_profile_name": assigned["customer_profile_name"],
                }
                profile_data = self._profile_dict(profile)
                output = {
                    "source": "clustering_model",
                    "cluster_id": cluster_id,
                    "cluster_profile": profile_data,
                    "interpretation_limit": profile_data.get("limitation"),
                    **self._artifact_details(tool),
                    "required_features": required,
                    "input_validation": "passed",
                }
            else:
                raise ValueError(f"Unknown model tool: {tool}")
            trace["inference_success"] = True
            trace["raw_output"] = raw
            trace["normalized_output"] = output
            return output, trace
        except Exception as error:
            trace["error"] = f"{type(error).__name__}: {error}"
            return {}, trace
        finally:
            trace["latency_ms"] = round((perf_counter() - started) * 1000, 3)

    def enrich(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Run every compatible artifact on retrieved rows; never fit a model."""
        result = matches.copy()
        if result.empty:
            return result
        if self.regressor is not None:
            required = self.required_inputs("regression")
            from src.regression.prediction import predict_prices

            prices = predict_prices(self.regressor, result[required].to_dict("records"))
            result["model_price"] = [row["predicted_price"] for row in prices]
            result["price_difference_pct"] = (
                100 * (result["price"] - result["model_price"]) / result["model_price"]
            )
        if self.classifier is not None:
            required = self.required_inputs("classification")
            from src.classification.prediction import predict_diamonds

            clarity = predict_diamonds(self.classifier, result[required].to_dict("records"))
            result["predicted_clarity_family"] = [row["clarity_family"] for row in clarity]
        if self.clusterer is not None:
            from src.clustering.prediction import assign_purchase_segment

            segments = assign_purchase_segment(self.clusterer, result)
            result["buyer_segment"] = segments["cluster_id"].to_numpy()
            result["buyer_interpretation"] = segments["customer_profile_name"].to_numpy()
        return result

    def predict(self, intent: str, raw_inputs: dict[str, Any]) -> dict[str, Any]:
        """Compatibility adapter for existing single-model assistant routes."""
        tool = {
            "price_prediction": "regression",
            "clarity_prediction": "classification",
            "cluster_prediction": "clustering",
        }.get(intent)
        if tool is None:
            return {}
        output, trace = self.execute(tool, raw_inputs)
        if not output:
            return {
                "error": trace.get("error") or trace.get("skip_reason"),
                "trace": trace,
            }
        if tool == "regression":
            return {
                "predicted_price": output["prediction"],
                "provenance": output,
                "trace": trace,
            }
        if tool == "classification":
            return {
                "clarity_family": output["predicted_class"],
                "probabilities": output["class_probabilities"],
                "provenance": output,
                "trace": trace,
            }
        return {**output, "trace": trace}
