import unittest

from src.assistant.dataset_search import DiamondCatalog
from src.assistant.schemas import DiamondQueryPlan
from src.assistant.vector_store import chunk_text
from tests.helpers import make_diamonds


class AssistantDatasetTests(unittest.TestCase):
    def setUp(self):
        self.catalog = DiamondCatalog(make_diamonds(rows=140))

    def test_exact_filters_are_applied_by_pandas(self):
        plan = DiamondQueryPlan(
            search_dataset=True, max_price=5000, target_carat=1.0,
            carat_tolerance=0.25, cut="Ideal", clarity="VS", depth=61.5,
        )
        result = self.catalog.search(plan, limit=20)
        self.assertTrue((result["price"] <= 5000).all())
        self.assertTrue(result["carat"].between(0.75, 1.25).all())
        self.assertTrue(result["cut"].eq("Ideal").all())
        self.assertTrue(result["clarity"].str.startswith("VS").all())
        self.assertTrue(result["depth"].between(60.5, 62.5).all())

    def test_no_search_returns_an_empty_frame_with_schema(self):
        result = self.catalog.search(DiamondQueryPlan(search_dataset=False))
        self.assertTrue(result.empty)
        self.assertIn("price", result.columns)

    def test_knowledge_chunking_is_deterministic_and_overlapping(self):
        text = "First paragraph.\n\n" + "A" * 30 + "\n\n" + "B" * 30
        first = chunk_text(text, size=45, overlap=8)
        self.assertEqual(first, chunk_text(text, size=45, overlap=8))
        self.assertGreater(len(first), 1)
        self.assertIn("A", first[1])


if __name__ == "__main__":
    unittest.main()
