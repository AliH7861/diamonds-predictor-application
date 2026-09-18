"""Exact, auditable filtering of the structured diamond dataset."""

from pathlib import Path

import pandas as pd

from .schemas import DiamondQueryPlan

REQUIRED_COLUMNS = {"carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"}


class DiamondCatalog:
    """Load and search real diamond rows without asking the language model to calculate."""

    def __init__(self, diamonds: pd.DataFrame):
        # Kaggle exports often include the previously saved DataFrame index as
        # ``Unnamed: 0``. It identifies a CSV row, not a diamond characteristic.
        index_columns = [
            column for column in diamonds.columns if str(column).casefold().startswith("unnamed:")
        ]
        clean = diamonds.drop(columns=index_columns, errors="ignore")
        missing = REQUIRED_COLUMNS - set(clean.columns)
        if missing:
            raise ValueError(f"Diamond dataset is missing columns: {sorted(missing)}")
        self.diamonds = clean.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> "DiamondCatalog":
        return cls(pd.read_csv(path))

    def search(self, plan: DiamondQueryPlan, limit: int | None = 5) -> pd.DataFrame:
        """Apply the plan as Pandas filters and rank useful matches deterministically."""
        if not plan.search_dataset:
            return self.diamonds.head(0).copy()

        matches = self.diamonds.copy()
        if plan.min_price is not None:
            matches = matches[matches["price"] >= plan.min_price]
        if plan.max_price is not None:
            matches = matches[matches["price"] <= plan.max_price]
        if plan.target_carat is not None:
            tolerance = max(float(plan.carat_tolerance), 0)
            matches = matches[
                matches["carat"].between(
                    plan.target_carat - tolerance, plan.target_carat + tolerance
                )
            ]
        for column in ("depth", "table", "x", "y", "z"):
            expected = getattr(plan, column)
            if expected is not None:
                tolerance = 1.0 if column in {"depth", "table"} else 0.15
                matches = matches[
                    matches[column].between(expected - tolerance, expected + tolerance)
                ]
        for column in ("cut", "color"):
            expected = getattr(plan, column)
            if expected:
                matches = matches[matches[column].str.casefold() == expected.casefold()]
        if plan.clarity:
            matches = matches[
                matches["clarity"].str.casefold().str.startswith(plan.clarity.casefold())
            ]

        sort_columns = []
        ascending = []
        if plan.target_price is not None:
            matches = matches.assign(_price_distance=(matches["price"] - plan.target_price).abs())
            sort_columns.append("_price_distance")
            ascending.append(True)
        if plan.target_carat is not None:
            matches = matches.assign(_carat_distance=(matches["carat"] - plan.target_carat).abs())
            sort_columns.append("_carat_distance")
            ascending.append(True)
        sort_columns.append("price")
        ascending.append(True)
        ranked = matches.sort_values(sort_columns, ascending=ascending).drop(
            columns=["_price_distance", "_carat_distance"], errors="ignore"
        )
        return ranked.head(limit).copy() if limit is not None else ranked.copy()
