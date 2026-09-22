"""Exact, auditable filtering of the structured diamond dataset."""

from pathlib import Path

import pandas as pd

from src.datasets import load_diamond_frame

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
        clean = diamonds.drop(columns=index_columns, errors="ignore").reset_index(drop=True)
        clean.insert(0, "_row_id", range(len(clean)))
        missing = REQUIRED_COLUMNS - set(clean.columns)
        if missing:
            raise ValueError(f"Diamond dataset is missing columns: {sorted(missing)}")
        self.diamonds = clean.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> "DiamondCatalog":
        return cls(load_diamond_frame(path))

    def _filter(self, plan: DiamondQueryPlan) -> pd.DataFrame:
        """Apply only hard user constraints and return every qualifying row."""
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

        return matches

    @staticmethod
    def _quality_score(frame: pd.DataFrame) -> pd.Series:
        """Provide a stable tie-breaker without pretending to be a valuation model."""
        cut = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
        clarity = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
        color = {grade: score for score, grade in enumerate(reversed("DEFGHIJ"))}
        return (
            frame["cut"].map(cut).fillna(0)
            + frame["clarity"].map(clarity).fillna(0)
            + frame["color"].map(color).fillna(0)
        )

    def _rank(self, matches: pd.DataFrame, plan: DiamondQueryPlan, strategy: str) -> pd.DataFrame:
        """Rank candidates according to the explicit follow-up or price semantics."""
        ranked = matches.assign(_quality_score=self._quality_score(matches))
        sort_columns: list[str] = []
        ascending = []
        directional = {"lower_end", "cheaper", "middle", "bigger"}
        if plan.target_price is not None and strategy not in directional:
            ranked = ranked.assign(_price_distance=(ranked["price"] - plan.target_price).abs())
            sort_columns.append("_price_distance")
            ascending.append(True)
        if plan.target_carat is not None and strategy != "bigger":
            ranked = ranked.assign(_carat_distance=(ranked["carat"] - plan.target_carat).abs())
            sort_columns.append("_carat_distance")
            ascending.append(True)
        if strategy in {"lower_end", "cheaper"}:
            sort_columns.extend(["price", "_quality_score"])
            ascending.extend([True, False])
        elif strategy == "middle" and len(ranked):
            low = float(plan.min_price if plan.min_price is not None else ranked["price"].min())
            high = float(plan.max_price if plan.max_price is not None else ranked["price"].max())
            ranked = ranked.assign(_middle_distance=(ranked["price"] - (low + high) / 2).abs())
            sort_columns.extend(["_middle_distance", "_quality_score"])
            ascending.extend([True, False])
        elif strategy == "bigger":
            sort_columns.extend(["carat", "_quality_score", "price"])
            ascending.extend([False, False, True])
        elif not sort_columns:
            sort_columns.extend(["price", "_quality_score"])
            ascending.extend([plan.max_price is None, False])
        else:
            sort_columns.append("_quality_score")
            ascending.append(False)
        return ranked.sort_values(sort_columns, ascending=ascending)

    @staticmethod
    def _diverse_price_selection(ranked: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Select strong rows across evenly spaced price bands."""
        if len(ranked) <= limit:
            return ranked
        low, high = float(ranked["price"].min()), float(ranked["price"].max())
        if low == high:
            return ranked.head(limit)
        targets = pd.Series([low + index * (high - low) / (limit - 1) for index in range(limit)])
        available = ranked.copy()
        selected = []
        for target in targets:
            distance = (available["price"] - target).abs()
            best_distance = distance.min()
            band = available[distance == best_distance]
            choice = band.sort_values("_quality_score", ascending=False).iloc[0]
            selected.append(choice)
            available = available[available["_row_id"] != choice["_row_id"]]
            if available.empty:
                break
        return pd.DataFrame(selected, columns=ranked.columns)

    @staticmethod
    def _standout_labels(frame: pd.DataFrame, plan: DiamondQueryPlan) -> pd.Series:
        """Explain each selected row using values present in the result set."""
        if frame.empty:
            return pd.Series(dtype="object")
        labels = pd.Series("Balanced option", index=frame.index, dtype="object")
        labels.loc[frame["price"].idxmin()] = "Lowest-price option"
        labels.loc[frame["carat"].idxmax()] = "Largest option"
        if plan.target_price is not None:
            closest = (frame["price"] - plan.target_price).abs().idxmin()
            labels.loc[closest] = "Closest to target price"
        return labels

    def search_with_trace(
        self,
        plan: DiamondQueryPlan,
        limit: int = 5,
        strategy: str = "balanced",
        exclude_ids: list[int] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Filter, rank, diversify, and return an auditable execution record."""
        matches = self._filter(plan)
        qualifying_count = len(matches)
        excluded = set(int(item) for item in (exclude_ids or []))
        if excluded:
            matches = matches[~matches["_row_id"].isin(excluded)]
        ranked = self._rank(matches, plan, strategy)
        if strategy in {"diverse_range", "different_diverse"}:
            selected = self._diverse_price_selection(ranked, limit)
        else:
            selected = ranked.head(limit)
        selected = selected.drop(
            columns=["_quality_score", "_price_distance", "_carat_distance", "_middle_distance"],
            errors="ignore",
        )
        selected = selected.copy()
        selected["why_it_stands_out"] = self._standout_labels(selected, plan)
        trace = {
            "hard_filters": {
                key: value
                for key, value in {
                    "min_price": plan.min_price,
                    "max_price": plan.max_price,
                    "target_price": plan.target_price,
                    "target_carat": plan.target_carat,
                    "carat_tolerance": plan.carat_tolerance if plan.target_carat else None,
                    "cut": plan.cut,
                    "color": plan.color,
                    "clarity": plan.clarity,
                }.items()
                if value is not None
            },
            "qualifying_rows": qualifying_count,
            "ranking_strategy": strategy,
            "excluded_result_ids": sorted(excluded),
            "selected_result_ids": selected["_row_id"].astype(int).tolist(),
        }
        return selected, trace

    def search(self, plan: DiamondQueryPlan, limit: int | None = 5) -> pd.DataFrame:
        """Compatibility wrapper around the simplified search executor."""
        matches = self._filter(plan)
        if limit is None:
            return self._rank(matches, plan, "balanced").drop(
                columns=["_quality_score", "_price_distance", "_carat_distance"],
                errors="ignore",
            )
        selected, _ = self.search_with_trace(plan, limit=limit)
        return selected
