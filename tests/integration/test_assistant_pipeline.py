import unittest

from src.assistant import DiamondAssistant, DiamondCatalog
from tests.helpers import make_diamonds


class FakeLLM:
    def __init__(self):
        self.structured_calls = 0

    def structured(self, system, user):
        self.structured_calls += 1
        if self.structured_calls == 1:
            return {"search_dataset": True, "max_price": 10000, "knowledge_queries": ["cut sparkle"]}
        if self.structured_calls == 2:
            return {"needs_more_context": True, "extra_queries": ["clarity value"]}
        return {"should_save": True, "memory_text": "User prioritizes sparkle."}

    def complete(self, system, user):
        self.final_context = user
        return "Choose the strongest cut within the budget."


class FakeStores:
    def __init__(self):
        self.saved = []
        self.queries = []

    def search_memory(self, question, limit):
        return ["User likes Ideal cut."]

    def search_knowledge(self, queries, limit):
        self.queries.extend(queries)
        return [f"Evidence for {query}" for query in queries]

    def add_memory(self, text):
        self.saved.append(text)


class FakeEnricher:
    def enrich(self, matches):
        return matches.assign(model_price=matches["price"] * 0.98)

    def predict(self, intent, raw_inputs):
        return {"predicted_price": 6020.0, "input_carat": raw_inputs["carat"]}


class AssistantPipelineTests(unittest.TestCase):
    def test_complete_orchestration_keeps_evidence_and_memory_separate(self):
        llm = FakeLLM()
        stores = FakeStores()
        assistant = DiamondAssistant(
            llm, DiamondCatalog(make_diamonds(rows=80)), stores, FakeEnricher(), top_diamonds=3
        )
        result = assistant.ask("I prioritize sparkle. Show options under $10,000.")
        self.assertEqual(len(result["matches"]), 3)
        self.assertIn("model_price", result["matches"])
        self.assertEqual(stores.queries, ["cut sparkle", "clarity value"])
        self.assertEqual(stores.saved, ["User prioritizes sparkle."])
        self.assertTrue(result["needed_second_retrieval"])
        self.assertIn("model_price", llm.final_context)

    def test_blank_question_is_rejected_before_any_component_runs(self):
        assistant = DiamondAssistant(FakeLLM(), DiamondCatalog(make_diamonds()), FakeStores())
        with self.assertRaisesRegex(ValueError, "contain text"):
            assistant.ask("  ")

    def test_natural_language_price_request_routes_to_saved_ann_evidence(self):
        assistant = DiamondAssistant(
            FakeLLM(), DiamondCatalog(make_diamonds(rows=80)), FakeStores(), FakeEnricher()
        )
        result = assistant.ask(
            "Predict price for 1 carat Ideal cut, G color, VS2 clarity, "
            "depth 61.5, table 57, x 6.4, y 6.4, z 3.95."
        )
        self.assertEqual(result["route"]["intent"], "price_prediction")
        self.assertEqual(result["evidence"]["model_evidence"]["predicted_price"], 6020.0)
        self.assertTrue(result["matches"].empty)


if __name__ == "__main__":
    unittest.main()
