"""Optional bridge from retrieved rows to the saved supervised-learning models."""

from pathlib import Path

import pandas as pd


class ModelEvidenceProvider:
    """Add saved-model estimates when artifacts exist; preserve catalog use without them."""

    def __init__(self, project_root: str | Path):
        root = Path(project_root)
        self.classification_dir = root / "models" / "classification"
        self.regression_dir = root / "models" / "regression"
        self.classifier = None
        self.regressor = None
        self.load_errors = []

        # Streamlit constructs this object once when the tab opens. Loading the
        # artifacts here avoids reading and deserializing them for every question.
        if (self.classification_dir / "metadata.json").is_file():
            try:
                from src.classification.prediction import load_best_model

                self.classifier = load_best_model(self.classification_dir)
            except Exception as error:
                self.load_errors.append(f"Classification model unavailable: {error}")
        if (self.regression_dir / "pipeline.joblib").is_file():
            try:
                from src.regression.prediction import load_best_model

                self.regressor = load_best_model(self.regression_dir)
            except Exception as error:
                self.load_errors.append(f"Regression model unavailable: {error}")

    def status(self) -> dict:
        """Describe optional saved-model availability for the visible evidence trace."""
        return {
            "classification_loaded": self.classifier is not None,
            "regression_loaded": self.regressor is not None,
            "errors": self.load_errors,
        }

    def enrich(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Run inference only on routed rows; this method never fits a model."""
        result = matches.copy()
        records = result[["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]].to_dict("records")
        if records and self.regressor is not None:
            from src.regression.prediction import predict_prices

            prices = predict_prices(self.regressor, records)
            result["model_price"] = [row["predicted_price"] for row in prices]
            result["price_difference_pct"] = 100 * (result["price"] - result["model_price"]) / result["model_price"]
        if records and self.classifier is not None:
            from src.classification.prediction import predict_diamonds

            bundle = self.classifier
            classifier_records = result[bundle["metadata"]["required_inputs"]].to_dict("records")
            clarity = predict_diamonds(bundle, classifier_records)
            result["predicted_clarity_family"] = [row["clarity_family"] for row in clarity]
        return result

    def predict(self, intent: str, raw_inputs: dict) -> dict:
        """Run the primary saved ANN requested by a prediction intent."""
        if intent == "price_prediction":
            if self.regressor is None:
                return {"error": "Primary regression ANN is unavailable."}
            from src.regression.prediction import predict_prices

            return predict_prices(self.regressor, raw_inputs)[0]
        if intent == "clarity_prediction":
            if self.classifier is None:
                return {"error": "Primary classification ANN is unavailable."}
            from src.classification.prediction import predict_diamonds

            return predict_diamonds(self.classifier, raw_inputs)[0]
        return {}
