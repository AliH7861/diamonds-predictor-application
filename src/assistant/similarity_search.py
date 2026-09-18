"""Find comparable diamonds with scaled numeric and encoded categorical features."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import DiamondQueryPlan


NUMERIC_FEATURES = ["price", "carat", "depth", "table", "x", "y", "z"]
CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class StructuredSimilaritySearch:
    """Fit one reusable feature space and rank future comparisons within it."""

    def __init__(self, diamonds: pd.DataFrame):
        self.diamonds = diamonds.copy()
        self.preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        self.matrix = self.preprocessor.fit_transform(self.diamonds[FEATURES]).astype(np.float32)

    def find(
        self,
        reference: pd.Series | dict,
        plan: DiamondQueryPlan,
        question: str,
        limit: int = 5,
    ) -> pd.DataFrame:
        """Rank candidates near a reference while honoring comparative constraints."""
        reference_frame = pd.DataFrame([dict(reference)])[FEATURES]
        candidates = self.diamonds.copy()
        lowered = question.casefold()
        reference_price = float(reference_frame.iloc[0]["price"])
        reference_carat = float(reference_frame.iloc[0]["carat"])

        if "cheaper" in lowered:
            candidates = candidates[candidates["price"] < reference_price]
        if "bigger" in lowered or "larger" in lowered:
            candidates = candidates[candidates["carat"] > reference_carat]
        if plan.min_price is not None:
            candidates = candidates[candidates["price"] >= plan.min_price]
        if plan.max_price is not None:
            candidates = candidates[candidates["price"] <= plan.max_price]
        if candidates.empty:
            return candidates

        candidate_matrix = self.preprocessor.transform(candidates[FEATURES]).astype(np.float32)
        reference_vector = self.preprocessor.transform(reference_frame).astype(np.float32)
        distances = pairwise_distances(candidate_matrix, reference_vector).reshape(-1)
        ranked = candidates.assign(
            similarity_score=1.0 / (1.0 + distances),
            _distance=distances,
        )
        same_row = (
            np.isclose(ranked["price"], reference_price)
            & np.isclose(ranked["carat"], reference_carat)
            & ranked["cut"].eq(str(reference_frame.iloc[0]["cut"]))
            & ranked["color"].eq(str(reference_frame.iloc[0]["color"]))
            & ranked["clarity"].eq(str(reference_frame.iloc[0]["clarity"]))
        )
        return ranked.loc[~same_row].sort_values("_distance").drop(columns="_distance").head(limit)
