from src.assistant.routing import route_question
from src.assistant.clarification import parse_model_inputs
from src.assistant.schemas import DiamondQueryPlan


def test_router_uses_only_knowledge_for_definition():
    route = route_question("What does VS2 clarity mean?")
    assert route.intent == "diamond_knowledge"
    assert route.use_knowledge
    assert not route.use_dataset
    assert not route.use_models


def test_router_keeps_greetings_out_of_rag_even_with_saved_preferences():
    route = route_question(
        "Hey",
        "Saved search preferences: between 0.25 and 0.55 carats; clarity VS.",
    )
    assert route.intent == "small_talk"
    assert not route.use_knowledge
    assert not route.use_memory


def test_buyer_segmentation_question_routes_to_knowledge_not_purchase_search():
    route = route_question("How is buyer segmentation evaluated?")
    assert route.intent == "diamond_knowledge"
    assert not route.use_dataset


def test_router_selects_structured_similarity_for_comparative_request():
    route = route_question(
        "Find something similar but cheaper.",
        "Saved search preferences: {}\nassistant: previous rows",
    )
    assert route.intent == "search"
    assert route.use_dataset


def test_price_prediction_inputs_are_collected_from_natural_language():
    values, missing = parse_model_inputs(
        "Predict price for 1 carat Ideal cut, G color, VS2 clarity, "
        "depth 61.5, table 57, x 6.4, y 6.4, z 3.95.",
        "price_prediction",
    )
    assert not missing
    assert values["carat"] == 1.0
    assert values["clarity"] == "VS2"


def test_price_prediction_accepts_buyer_facing_clarity_family():
    values, missing = parse_model_inputs(
        "Predict price for 1 carat Ideal cut, G color, VS clarity, "
        "depth 61.5, table 57, x 6.4, y 6.4, z 3.95.",
        "price_prediction",
    )
    assert not missing
    assert values["clarity"] == "VS2"


def test_llm_plan_rejects_invalid_categories_and_normalizes_ranges():
    plan = DiamondQueryPlan.from_dict(
        {
            "search_dataset": True,
            "min_price": 6000,
            "max_price": 5000,
            "cut": "Invented",
            "color": "Z",
            "clarity": "VS2",
        }
    )
    assert (plan.min_price, plan.max_price) == (5000, 6000)
    assert plan.cut is None
    assert plan.color is None
    assert plan.clarity == "VS"
