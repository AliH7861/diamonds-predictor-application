"""Acceptance coverage for the simplified dataset-first assistant."""

from __future__ import annotations

from src.assistant import DiamondAssistant, DiamondCatalog
from tests.helpers import make_diamonds


class GroundedLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, system, prompt, on_token=None):
        self.prompts.append((system, prompt))
        answer = "VS has minor inclusions; VVS has fewer visible inclusions under magnification."
        if on_token:
            on_token(answer)
        return answer


class KnowledgeStore:
    def search_knowledge_details(self, queries, limit):
        return [
            {
                "query": queries[0],
                "source": "diamond_basics.md",
                "section": "Clarity",
                "similarity": 0.95,
                "document": "VS diamonds have minor inclusions. VVS diamonds have very slight inclusions.",
            }
        ]


def assistant():
    return DiamondAssistant(
        GroundedLLM(),
        DiamondCatalog(make_diamonds(rows=1200)),
        KnowledgeStore(),
        top_diamonds=5,
    )


def follow_up(bot, question, previous):
    conversation = [
        {"role": "user", "content": "Earlier search"},
        {"role": "assistant", "content": previous["answer"], "result": previous},
    ]
    return bot.ask(question, conversation=conversation, state=previous["conversation_state"])


def test_search_followups_diversity_and_five_rows():
    bot = assistant()
    first = bot.ask("Find me a diamond around $2500.")
    assert first["route"]["intent"] == "search"
    assert first["conversation_state"]["target_price"] == 2500
    assert len(first["matches"]) == 5
    assert first["diagnostics"]["tools_executed"] == ["dataframe"]

    closer = follow_up(bot, "Make it closer to $2200.", first)
    assert closer["conversation_state"]["target_price"] == 2200
    assert len(closer["matches"]) == 5

    different = follow_up(bot, "Give me five different ones.", closer)
    assert len(different["matches"]) == 5
    assert set(different["matches"]["_row_id"]).isdisjoint(closer["matches"]["_row_id"])


def test_price_range_is_diverse_and_followup_rankings_are_grounded():
    bot = assistant()
    ranged = bot.ask("Show me five Ideal diamonds between $1500 and $2500.")
    assert len(ranged["matches"]) == 5
    assert ranged["matches"]["cut"].eq("Ideal").all()
    assert ranged["matches"]["price"].between(1500, 2500).all()
    assert ranged["matches"]["price"].max() - ranged["matches"]["price"].min() > 500
    assert ranged["diagnostics"]["dataset_ranking_strategy"] == "diverse_range"

    alternate = follow_up(bot, "Give me five different ones.", ranged)
    assert alternate["matches"]["price"].max() - alternate["matches"]["price"].min() > 500
    assert set(alternate["matches"]["_row_id"]).isdisjoint(ranged["matches"]["_row_id"])

    lower = follow_up(bot, "Give me options near the lower end.", ranged)
    middle = follow_up(bot, "Now show me the middle of the range.", ranged)
    assert lower["matches"]["price"].mean() < middle["matches"]["price"].mean()

    bigger = follow_up(bot, "Make them bigger.", ranged)
    cheaper = follow_up(bot, "Show me cheaper alternatives.", ranged)
    assert bigger["matches"]["carat"].mean() >= ranged["matches"]["carat"].mean()
    assert cheaper["matches"]["price"].mean() <= ranged["matches"]["price"].mean()


def test_comparisons_use_only_displayed_rows():
    bot = assistant()
    search = bot.ask("Show me five diamonds around $2500.")
    tradeoffs = follow_up(bot, "What are the trade-offs here?", search)
    assert tradeoffs["route"]["intent"] == "compare"
    assert "customer" not in tradeoffs["answer"].casefold()
    assert len(tradeoffs["matches"]) == 5

    compared = follow_up(bot, "Compare the first and third.", search)
    expected = {
        int(search["matches"].iloc[0]["_row_id"]),
        int(search["matches"].iloc[2]["_row_id"]),
    }
    assert set(compared["matches"]["_row_id"].astype(int)) == expected

    explained = follow_up(bot, "Why is the third one more expensive?", search)
    assert len(explained["matches"]) == 1
    assert int(explained["matches"].iloc[0]["_row_id"]) == int(search["matches"].iloc[2]["_row_id"])


def test_analysis_knowledge_invalid_and_out_of_scope_paths():
    bot = assistant()
    largest = bot.ask("What is the largest carat diamond?")
    common = bot.ask("What are the most common carat weights?")
    vs = bot.ask("What does VS clarity mean?")
    difference = bot.ask("What is the difference between VS and VVS?")
    invalid = bot.ask("Find me a VI clarity diamond.")
    math = bot.ask("What is 2+2?")
    weather = bot.ask("What is the weather today?")

    assert largest["route"]["intent"] == "dataset_analysis"
    assert common["route"]["intent"] == "dataset_analysis"
    assert vs["route"]["intent"] == "diamond_knowledge"
    assert difference["route"]["intent"] == "diamond_knowledge"
    assert vs["diagnostics"]["tools_executed"] == ["rag"]
    assert invalid["status"] == "needs_clarification"
    assert "isn't a clarity grade" in invalid["answer"]
    assert invalid["matches"].empty
    assert math["route"]["intent"] == "out_of_scope"
    assert weather["route"]["intent"] == "out_of_scope"


def test_dataset_ranges_overview_and_category_questions_use_the_right_source():
    bot = assistant()

    prices = bot.ask("What is the range of prices in the dataset?")
    carats = bot.ask("Tell me the range of carats we have.")
    misspelled_carats = bot.ask("Tell me the range of carots we have.")
    overview = bot.ask("Tell me about the dataset features and their ranges.")
    feature_meanings = bot.ask("Tell me about the diamond features and what each means.")
    cuts = bot.ask("How many different types of cuts are there?")

    assert prices["route"]["intent"] == "dataset_analysis"
    assert prices["diagnostics"]["tools_executed"] == ["pandas"]
    assert "$" in prices["answer"] and "median" in prices["answer"]
    assert carats["route"]["intent"] == "dataset_analysis"
    assert "carats" in carats["answer"]
    assert misspelled_carats["route"]["intent"] == "dataset_analysis"
    assert "carats" in misspelled_carats["answer"]
    assert overview["route"]["intent"] == "dataset_analysis"
    assert "modeling fields" in overview["answer"]
    assert feature_meanings["route"]["intent"] == "dataset_analysis"
    assert "millimeter dimensions" in feature_meanings["answer"]
    assert cuts["route"]["intent"] == "diamond_knowledge"
    assert "five cut grades" in cuts["answer"]
    assert cuts["diagnostics"]["embedding_calls"] == 0


def test_short_price_followup_requests_the_missing_number():
    bot = assistant()
    previous = bot.ask("Find me a diamond.")
    result = follow_up(bot, "PRICE ONLY", previous)

    assert result["route"]["intent"] == "search"
    assert result["status"] == "needs_clarification"
    assert "maximum budget or target price" in result["answer"]
