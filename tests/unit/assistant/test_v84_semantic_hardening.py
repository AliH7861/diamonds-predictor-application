"""Minimal pairs for the simplified intent and price semantics."""

from src.assistant.routing import route_question
from src.assistant.search_planning import build_search_plan


def test_search_and_analysis_minimal_pair():
    assert route_question("Show VS diamonds under $5000").intent == "search"
    assert route_question("What is the maximum price of VS diamonds?").intent == "dataset_analysis"


def test_explanation_and_calculated_relation_minimal_pair():
    assert route_question("Explain why carat affects price").intent == "diamond_knowledge"
    assert route_question("Calculate how carat correlates with price").intent == "dataset_analysis"


def test_price_meaning_is_explicit():
    under = build_search_plan("Show diamonds under $2500")
    ranged = build_search_plan("Show diamonds between $1500 and $2500")
    around = build_search_plan("Show diamonds around $2500")

    assert under.plan.max_price == 2500 and under.plan.target_price is None
    assert (ranged.plan.min_price, ranged.plan.max_price) == (1500, 2500)
    assert around.plan.target_price == 2500 and around.plan.max_price is None


def test_current_correction_preserves_unmentioned_state():
    state = {"max_price": 5000.0, "target_carat": 1.0, "cut": "Ideal"}
    parsed = build_search_plan("Actually 1.2 carats", state)

    assert parsed.state["target_carat"] == 1.2
    assert parsed.state["max_price"] == 5000.0
    assert parsed.state["cut"] == "Ideal"
