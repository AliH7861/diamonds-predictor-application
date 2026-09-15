import pandas as pd

from src.assistant.routing import route_question
from src.assistant.clarification import parse_model_inputs
from src.assistant.schemas import DiamondQueryPlan
from src.assistant.similarity_search import StructuredSimilaritySearch


def _diamonds():
    return pd.DataFrame([
        {"price": 6000, "carat": 1.00, "cut": "Ideal", "color": "G", "clarity": "VS2", "depth": 61.5, "table": 57, "x": 6.4, "y": 6.4, "z": 3.95},
        {"price": 5600, "carat": 0.98, "cut": "Ideal", "color": "G", "clarity": "VS2", "depth": 61.7, "table": 57, "x": 6.3, "y": 6.3, "z": 3.90},
        {"price": 3000, "carat": 0.60, "cut": "Good", "color": "J", "clarity": "SI2", "depth": 64.0, "table": 61, "x": 5.2, "y": 5.1, "z": 3.30},
    ])


def test_router_uses_only_knowledge_for_definition():
    route = route_question("What does VS2 clarity mean?")
    assert route.intent == "general_knowledge"
    assert route.use_knowledge
    assert not route.use_dataset
    assert not route.use_models


def test_router_selects_structured_similarity_for_comparative_request():
    route = route_question("Find something similar but cheaper.")
    assert route.intent == "similarity_search"
    assert route.use_similarity
    assert route.use_dataset


def test_similarity_search_uses_encoded_scaled_features_and_cheaper_constraint():
    diamonds = _diamonds()
    search = StructuredSimilaritySearch(diamonds)
    result = search.find(
        diamonds.iloc[0], DiamondQueryPlan(search_dataset=True),
        "Find something similar but cheaper.", limit=2,
    )
    assert not result.empty
    assert (result["price"] < 6000).all()
    assert result.iloc[0]["clarity"] == "VS2"
    assert 0 < result.iloc[0]["similarity_score"] <= 1


def test_price_prediction_inputs_are_collected_from_natural_language():
    values, missing = parse_model_inputs(
        "Predict price for 1 carat Ideal cut, G color, VS2 clarity, "
        "depth 61.5, table 57, x 6.4, y 6.4, z 3.95.",
        "price_prediction",
    )
    assert not missing
    assert values["carat"] == 1.0
    assert values["clarity"] == "VS2"


def test_llm_plan_rejects_invalid_categories_and_normalizes_ranges():
    plan = DiamondQueryPlan.from_dict({
        "search_dataset": True,
        "min_price": 6000,
        "max_price": 5000,
        "cut": "Invented",
        "color": "Z",
        "clarity": "VS2",
    })
    assert (plan.min_price, plan.max_price) == (5000, 6000)
    assert plan.cut is None
    assert plan.color is None
    assert plan.clarity == "VS2"
