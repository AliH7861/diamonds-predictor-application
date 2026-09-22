"""Field parsing and compact-state tests for the simplified search planner."""

from src.assistant.search_planning import build_search_plan


def test_search_plan_keeps_budget_and_carat_range():
    parsed = build_search_plan("Find a diamond below $3,000 between 0.35 and 0.45 carats.")

    assert parsed.plan.max_price == 3000
    assert parsed.plan.target_carat == 0.40
    assert parsed.plan.carat_tolerance == 0.05
    assert not parsed.dropped_fields


def test_target_price_is_distinct_from_maximum_budget():
    around = build_search_plan("Find something around $2500")
    under = build_search_plan("Find something under $2500")

    assert around.plan.target_price == 2500
    assert around.plan.max_price is None
    assert under.plan.max_price == 2500
    assert under.plan.target_price is None


def test_follow_up_updates_one_field_and_preserves_the_rest():
    state = {"min_price": 1500, "max_price": 2500, "cut": "Ideal"}
    parsed = build_search_plan("Make it closer to $2200", state)

    assert parsed.plan.target_price == 2200
    assert parsed.plan.min_price == 1500
    assert parsed.plan.max_price == 2500
    assert parsed.plan.cut == "Ideal"


def test_invalid_value_is_explained_and_valid_budget_is_kept():
    parsed = build_search_plan("Find a VI clarity diamond under $2000")

    assert parsed.clarification
    assert parsed.state["max_price"] == 2000
    assert "clarity" in parsed.dropped_fields
