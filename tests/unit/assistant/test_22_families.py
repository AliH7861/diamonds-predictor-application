"""Routing contract for the simplified seven-intent assistant."""

import pytest

from src.assistant.legacy.execution_policy import execution_policy
from src.assistant.routing import route_question


CASES = [
    ("Find diamonds under $2,000.", "search", "data", False),
    ("How many diamonds are under $2,000?", "dataset_analysis", "data", False),
    ("Compare these.", "compare", "data", False),
    ("What does VS clarity mean?", "diamond_knowledge", "knowledge", True),
    ("Predict price for this diamond.", "model_prediction", "model", False),
    ("Hey", "small_talk", "conversation", False),
    ("What is the weather today?", "out_of_scope", "unsupported", False),
]


@pytest.mark.parametrize("prompt,intent,family,uses_tokens", CASES)
def test_seven_intent_routes(prompt, intent, family, uses_tokens):
    context = "Saved search preferences: {}\nassistant: five rows shown"
    route = route_question(prompt, context)
    policy = execution_policy(route)

    assert route.intent == intent
    assert route.family == family
    assert policy.use_embeddings is uses_tokens
    assert policy.use_generation is uses_tokens


def test_invalid_clarity_stays_on_search_path_without_dataset_execution():
    route = route_question("Find a VI clarity diamond under $2000")

    assert route.intent == "search"
    assert route.level_3 == "INVALID_CONSTRAINT"
    assert not route.use_dataset


def test_customer_profiles_require_explicit_profile_language():
    explicit = route_question("What customer profiles did clustering find?")
    ordinary = route_question("Find diamonds under $3000")

    assert explicit.intent == "diamond_knowledge"
    assert explicit.level_3 == "CUSTOMER_PROFILES"
    assert ordinary.intent == "search"
