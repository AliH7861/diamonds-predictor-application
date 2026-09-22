from src.assistant.legacy.concepts import map_concepts
from src.assistant.legacy.execution_policy import execution_policy
from src.assistant.routing import route_question
from src.assistant.search_planning import build_search_plan


def test_concept_map_exposes_evidence():
    concepts = map_concepts("I want something large under $5000 with clean clarity.")
    names = {item.concept for item in concepts}
    assert {"BUDGET", "CARAT_SIZE", "CLARITY", "SEARCH_FILTER"} <= names
    assert all(item.evidence and item.confidence >= 0.9 for item in concepts)


def test_analysis_actions_are_deterministic():
    assert route_question("Give me the median price by clarity.").level_3 == "TREND_GROUP"
    assert route_question("What is the average price?").level_3 == "STATS"
    assert (
        route_question("Calculate the correlation between carat and price.").level_3
        == "TREND_RELATION"
    )


def test_state_correction_changes_only_supported_field():
    state = {"max_price": 6000.0, "target_carat": 1.0, "clarity": "VS", "cut": "Ideal"}
    parsed = build_search_plan("Actually make it 1.2 carats.", state)

    assert parsed.plan.target_carat == 1.2
    assert parsed.state["max_price"] == 6000.0
    assert parsed.state["clarity"] == "VS"
    assert parsed.state["cut"] == "Ideal"


def test_explicit_removal_does_not_clear_other_state():
    state = {"max_price": 5000.0, "clarity": "VS", "color": "G"}
    parsed = build_search_plan("Remove color.", state)

    assert "color" not in parsed.state
    assert parsed.state["max_price"] == 5000.0
    assert parsed.state["clarity"] == "VS"


def test_clear_tasks_have_zero_token_policy():
    for prompt in (
        "Find diamonds under $2000.",
        "What is the median price?",
        "Predict price for this diamond.",
    ):
        policy = execution_policy(route_question(prompt))
        assert (policy.llm_calls, policy.embedding_calls, policy.generation_calls) == (0, 0, 0)
