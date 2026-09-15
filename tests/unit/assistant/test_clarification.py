from src.assistant.clarification import build_buying_plan


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
