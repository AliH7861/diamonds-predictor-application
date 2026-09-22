import unittest

from src.assistant import DiamondAssistant, DiamondCatalog
from tests.helpers import make_diamonds


class FakeLLM:
    def __init__(self):
        self.complete_calls = 0

    def complete(self, system, user, on_token=None):
        self.complete_calls += 1
        self.final_context = user
        answer = "Grounded diamond explanation."
        if on_token:
            on_token(answer)
        return answer


class FakeStores:
    def __init__(self):
        self.queries = []

    def search_knowledge(self, queries, limit):
        self.queries.extend(queries)
        return [f"Evidence for {query}" for query in queries]


class FakeEnricher:
    def predict(self, intent, raw_inputs):
        return {"predicted_price": 6020.0, "input_carat": raw_inputs["carat"]}


class AssistantPipelineTests(unittest.TestCase):
    def test_search_uses_dataframe_only_and_keeps_state_local(self):
        llm = FakeLLM()
        stores = FakeStores()
        assistant = DiamondAssistant(
            llm, DiamondCatalog(make_diamonds(rows=80)), stores, top_diamonds=3
        )

        result = assistant.ask("Find options around $10,000.")

        self.assertEqual(len(result["matches"]), 3)
        self.assertEqual(result["route"]["intent"], "search")
        self.assertEqual(result["diagnostics"]["tools_executed"], ["dataframe"])
        self.assertEqual(llm.complete_calls, 0)
        self.assertEqual(stores.queries, [])

    def test_new_conversation_does_not_reuse_another_chats_preferences(self):
        assistant = DiamondAssistant(FakeLLM(), DiamondCatalog(make_diamonds()), FakeStores())
        first = assistant.ask("Find an Ideal diamond under $3000.", state={})
        second = assistant.ask("What types of cuts are there?", state={})

        self.assertEqual(first["conversation_state"]["cut"], "Ideal")
        self.assertEqual(second["conversation_state"], {})
        self.assertEqual(second["retrieved_memory"], [])

    def test_invalid_clarity_keeps_budget_but_returns_no_rows(self):
        assistant = DiamondAssistant(FakeLLM(), DiamondCatalog(make_diamonds()), FakeStores())
        result = assistant.ask("$2000, lower than VI, yes", state={})

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["conversation_state"]["max_price"], 2000)
        self.assertTrue(result["matches"].empty)
        self.assertIn("isn't a clarity grade", result["answer"])

    def test_blank_question_is_rejected_before_any_component_runs(self):
        assistant = DiamondAssistant(FakeLLM(), DiamondCatalog(make_diamonds()), FakeStores())
        with self.assertRaisesRegex(ValueError, "contain text"):
            assistant.ask("  ")

    def test_greeting_skips_all_evidence_tools(self):
        llm = FakeLLM()
        assistant = DiamondAssistant(
            llm, DiamondCatalog(make_diamonds()), FakeStores(), FakeEnricher()
        )
        result = assistant.ask("Hey", state={"target_carat": 0.4, "clarity": "VS"})

        self.assertEqual(result["route"]["intent"], "small_talk")
        self.assertIn("Hi", result["answer"])
        self.assertEqual(llm.complete_calls, 0)
        self.assertTrue(result["matches"].empty)

    def test_explicit_price_prediction_uses_saved_model(self):
        assistant = DiamondAssistant(
            FakeLLM(), DiamondCatalog(make_diamonds(rows=80)), FakeStores(), FakeEnricher()
        )
        result = assistant.ask(
            "Predict price for 1 carat Ideal cut, G color, VS2 clarity, "
            "depth 61.5, table 57, x 6.4, y 6.4, z 3.95."
        )

        self.assertEqual(result["route"]["intent"], "model_prediction")
        self.assertEqual(result["evidence"]["model_evidence"]["predicted_price"], 6020.0)
        self.assertTrue(result["matches"].empty)

    def test_count_question_uses_pandas_without_generation(self):
        llm = FakeLLM()
        diamonds = make_diamonds(rows=80)
        assistant = DiamondAssistant(llm, DiamondCatalog(diamonds), FakeStores())
        result = assistant.ask("How many VS diamonds are under $10,000?")
        expected = len(
            diamonds[(diamonds["price"] <= 10000) & diamonds["clarity"].str.startswith("VS")]
        )

        self.assertEqual(result["route"]["intent"], "dataset_analysis")
        self.assertEqual(result["evidence"]["dataset_statistics"]["value"], expected)
        self.assertEqual(llm.complete_calls, 0)

    def test_compact_state_completes_follow_up_without_full_history(self):
        assistant = DiamondAssistant(
            FakeLLM(), DiamondCatalog(make_diamonds(rows=80)), FakeStores()
        )
        first = assistant.ask("I want a diamond.")
        self.assertEqual(first["status"], "needs_clarification")
        second = assistant.ask(
            "Below 3000, between 0.35 and 0.45 carat, and VS clarity.",
            state=first["conversation_state"],
        )

        self.assertEqual(second["status"], "answered")
        self.assertLessEqual(second["plan"]["max_price"], 3000)
        self.assertAlmostEqual(second["plan"]["target_carat"], 0.4)

    def test_missing_dataset_field_is_rejected_without_generation(self):
        llm = FakeLLM()
        assistant = DiamondAssistant(llm, DiamondCatalog(make_diamonds(rows=20)), FakeStores())
        result = assistant.ask("What country was this diamond mined in?")

        self.assertEqual(result["route"]["intent"], "out_of_scope")
        self.assertIn("not present", result["answer"])
        self.assertEqual(llm.complete_calls, 0)

    def test_impossible_constraints_do_not_hallucinate_matches(self):
        assistant = DiamondAssistant(
            FakeLLM(), DiamondCatalog(make_diamonds(rows=20)), FakeStores()
        )
        result = assistant.ask("I want a 20 carat Ideal cut IF clarity diamond under $100.")

        self.assertTrue(result["matches"].empty)
        self.assertIn("No dataset rows", result["answer"])

    def test_simple_search_skips_rag_and_generation(self):
        llm = FakeLLM()
        stores = FakeStores()
        assistant = DiamondAssistant(llm, DiamondCatalog(make_diamonds(rows=80)), stores)
        result = assistant.ask(
            "Find a diamond under 3000 between 0.35 and 0.45 carat; I don't care about clarity."
        )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["route"]["intent"], "search")
        self.assertEqual(llm.complete_calls, 0)
        self.assertEqual(stores.queries, [])


if __name__ == "__main__":
    unittest.main()
