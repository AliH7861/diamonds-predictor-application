from src.assistant.clarification import build_buying_plan
from src.assistant.clarity import clarity_family


def test_detailed_clarity_grades_collapse_to_project_families():
    assert clarity_family("I1") == "I"
    assert clarity_family("SI2") == "SI"
    assert clarity_family("VS1") == "VS"
    assert clarity_family("VVS2") == "VVS"
    assert clarity_family("IF") == "IF"


def test_buying_conversation_combines_natural_follow_ups_without_looping():
    first = (
        "I have a budget for 4000$ and want a balance between clarity and price. "
        "I want a medium size diamond."
    )
    initial = build_buying_plan(first, "No earlier conversation.")

    assert initial is not None
    assert not initial.needs_clarification
    assert initial.max_price == 4000
    assert initial.target_carat == 0.75
    assert initial.clarity == "VS"

    conversation = (
        f"user: {first}\n"
        "assistant: Before I search, what is your maximum budget, the carat size you are "
        "targeting, and the clarity grade or range you would accept?\n"
        "user: 4500\n"
        "assistant: Before I search, what is your maximum budget, the carat size you are "
        "targeting, and the clarity grade or range you would accept?\n"
        "user: 2500"
    )
    final = build_buying_plan(
        "2500, 0.35 carrot, and like high clarity",
        conversation,
    )

    assert final is not None
    assert not final.needs_clarification
    assert final.max_price == 2500
    assert final.target_carat == 0.35
    assert final.clarity == "VVS"


def test_budget_range_and_no_clarity_preference_complete_the_plan():
    first = "So I want a diamond for below 3000, between 0.35 and 0.45 carrot."
    initial = build_buying_plan(first, "No earlier conversation.")

    assert initial is not None
    assert initial.max_price == 3000
    assert initial.target_carat == 0.40
    assert initial.carat_tolerance == 0.05
    assert not initial.needs_clarification

    conversation = (
        f"user: {first}\n"
        f"assistant: {initial.clarifying_question}"
    )
    final = build_buying_plan("I don't care about clarity grade.", conversation)

    assert final is not None
    assert final.max_price == 3000
    assert final.target_carat == 0.40
    assert final.clarity is None
    assert not final.needs_clarification


def test_budget_alone_is_enough_to_return_recommendations():
    plan = build_buying_plan(
        "Can you give me a great diamond around the price $2000?",
        "No earlier conversation.",
    )

    assert plan is not None
    assert plan.min_price == 1800
    assert plan.max_price == 2200
    assert plan.target_price == 2000
    assert not plan.needs_clarification


def test_common_typos_and_k_budget_are_normalized():
    plan = build_buying_plan(
        "whats a good diamnd for like 4k maybe 1 carret, ideel cut and VS clrty",
        "No earlier conversation.",
    )

    assert plan is not None
    assert plan.max_price == 4000
    assert plan.target_carat == 1.0
    assert plan.cut == "Ideal"
    assert plan.clarity == "VS"
    assert not plan.needs_clarification
