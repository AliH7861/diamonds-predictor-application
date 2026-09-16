"""Test the complete RAG path in deterministic CI mode or against local Ollama."""

import argparse
from pathlib import Path
import sys
import tempfile

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.dataset_search import DiamondCatalog  # noqa: E402
from src.assistant.runtime import create_assistant  # noqa: E402
from src.assistant.service import DiamondAssistant  # noqa: E402
from src.assistant.vector_store import ChromaStores  # noqa: E402

TEST_CASES = {
    "What factors affect diamond price?": ("price", "carat"),
    "What does VS clarity mean?": ("clarity", "inclusion", "vs"),
    "Why might a model classify a diamond as VS?": ("clarity", "inclusion", "vs"),
    "What features does the price model use?": ("carat", "cut", "color", "price"),
    "How is buyer segmentation evaluated?": ("silhouette", "stability", "cluster"),
}


class DeterministicEmbedder:
    """Stable local text vectors used by CI without downloading an embedding model."""

    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=384, alternate_sign=False, norm="l2", ngram_range=(1, 2)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()


class DeterministicLLM:
    """Predictable planner and answer writer used only by the CI execution mode."""

    def __init__(self):
        self.last_complete_user = ""

    def structured(self, system: str, user: str) -> dict:
        if "search plan" in system:
            question = user.split("Question:", 1)[-1].strip()
            buying_conversation = "$6,000" in user or "$6000" in user
            has_size = "1 carat" in user.casefold()
            if buying_conversation and not has_size:
                return {
                    "search_dataset": True,
                    "max_price": 6000,
                    "priorities": ["clarity"],
                    "needs_clarification": True,
                    "clarifying_question": "What carat size and cut grade would you prefer?",
                    "knowledge_queries": [],
                }
            if buying_conversation and has_size:
                return {
                    "search_dataset": True,
                    "max_price": 6000,
                    "target_carat": 1.0,
                    "carat_tolerance": 0.15,
                    "cut": "Ideal",
                    "priorities": ["clarity"],
                    "needs_clarification": False,
                    "knowledge_queries": [
                        "clarity value trade-off VS diamonds",
                        "cut sparkle Ideal diamonds",
                    ],
                }
            return {
                "search_dataset": False,
                "priorities": [],
                "needs_clarification": False,
                "clarifying_question": None,
                "knowledge_queries": [question],
            }
        if "more_context" in system:
            return {"needs_more_context": False, "extra_queries": []}
        if "clarity matters" in user.casefold():
            return {"should_save": True, "memory_text": "Clarity matters most to the user."}
        return {"should_save": False, "memory_text": None}

    def complete(self, system: str, user: str) -> str:
        self.last_complete_user = user
        return "RAG answer generated from the retrieved diamond evidence. " + user


def _small_catalog() -> DiamondCatalog:
    """Create a valid catalog so the CI test remains independent of private data."""
    return DiamondCatalog(pd.DataFrame([
        {
            "carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
            "depth": 61.5, "table": 57.0, "price": 5800, "x": 6.45, "y": 6.43, "z": 3.96,
        },
        {
            "carat": 0.92, "cut": "Ideal", "color": "F", "clarity": "VVS2",
            "depth": 61.8, "table": 56.0, "price": 5900, "x": 6.25, "y": 6.22, "z": 3.85,
        },
        {
            "carat": 1.1, "cut": "Premium", "color": "H", "clarity": "SI1",
            "depth": 62.0, "table": 58.0, "price": 5600, "x": 6.60, "y": 6.56, "z": 4.08,
        },
    ]))


def _check_questions(assistant: DiamondAssistant, label: str) -> None:
    """Require retrieval relevance and a nonempty generated answer for each question."""
    for question, expected_terms in TEST_CASES.items():
        result = assistant.ask(question)
        retrieved = " ".join(result["knowledge"]).casefold()
        if not result["knowledge"]:
            raise AssertionError(f"{label}: no knowledge retrieved for: {question}")
        if not any(term in retrieved for term in expected_terms):
            raise AssertionError(f"{label}: retrieved knowledge was irrelevant for: {question}")
        if len(result["answer"].strip()) < 20:
            raise AssertionError(f"{label}: answer was empty or too short for: {question}")
        print(f"PASS  {question}")


def _check_buying_conversation(assistant: DiamondAssistant, llm: DeterministicLLM) -> None:
    """Verify clarification, memory, vector evidence, filtering, and grounding together."""
    first_question = "I want a diamond around $6,000 and clarity matters most."
    first = assistant.ask(first_question)
    if first["status"] != "needs_clarification" or first["knowledge"]:
        raise AssertionError("CI: incomplete buying request was embedded before clarification.")
    if "clarity" not in first["answer"].casefold() or first["saved_memory"]:
        raise AssertionError("CI: clarification should happen before embedding or memory storage.")
    print("PASS  incomplete buying request clarifies before any embedding")

    conversation = [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": first["answer"]},
    ]
    second_question = "About 1 carat."
    second = assistant.ask(
        second_question,
        conversation=conversation[-2:],
        state=first["conversation_state"],
    )
    if second["status"] != "needs_clarification" or second["knowledge"]:
        raise AssertionError("CI: clarity priority should require a concrete clarity grade.")
    print("PASS  assistant continues clarification until the priority is concrete")

    conversation.extend([
        {"role": "user", "content": second_question},
        {"role": "assistant", "content": second["answer"]},
    ])
    third = assistant.ask(
        "VS2 clarity or better, with an Ideal cut. Explain the trade-offs.",
        conversation=conversation[-2:],
        state=second["conversation_state"],
    )
    if third["status"] != "answered" or third["matches"].empty:
        raise AssertionError("CI: completed preferences did not produce dataset matches.")
    if (third["matches"]["price"] > 6600).any() or (third["matches"]["cut"] != "Ideal").any():
        raise AssertionError("CI: structured dataset filters were not applied.")
    if not third["knowledge_details"]:
        raise AssertionError("CI: vector retrieval did not return scored chunks.")
    if not all(
        item["source"] and item["similarity"] is not None
        for item in third["knowledge_details"]
    ):
        raise AssertionError("CI: retrieval provenance or similarity scores are missing.")
    if "knowledge" not in llm.last_complete_user.casefold() and ".md" not in llm.last_complete_user:
        raise AssertionError("CI: the final generator did not receive retrieved RAG context.")
    if not third["saved_memory"]:
        raise AssertionError("CI: completed conversation did not save the explicit preference.")
    print("PASS  completed request returns filtered rows and scored RAG context")

    recall = assistant.ask("What did I say matters most?")
    if not any("clarity" in item.casefold() for item in recall["retrieved_memory"]):
        raise AssertionError("CI: saved preference was not retrieved on a later turn.")
    print("PASS  later question retrieves the saved preference memory")


def run_ci_test() -> None:
    """Exercise embeddings, real Chroma, retrieval, orchestration, and answer output."""
    embedder = DeterministicEmbedder()
    probe = embedder.embed(["diamond clarity"])[0]
    if len(probe) != 384 or not any(probe):
        raise AssertionError("Embedding generation failed.")

    with tempfile.TemporaryDirectory() as directory:
        stores = ChromaStores(Path(directory) / "chroma", embedder)
        try:
            indexed = stores.index_knowledge(PROJECT_ROOT / "knowledge")
            if indexed < 1 or stores.knowledge.count() < 1:
                raise AssertionError("Chroma knowledge indexing failed.")
            llm = DeterministicLLM()
            assistant = DiamondAssistant(llm, _small_catalog(), stores)
            _check_questions(assistant, "CI")
            _check_buying_conversation(assistant, llm)
        finally:
            stores.close()
    print("\nRAG CI STATUS: SUCCESS")


def run_live_test() -> None:
    """Exercise real Ollama embeddings, persistent Chroma, retrieval, and Qwen."""
    try:
        assistant = create_assistant(include_models=False)
        for question, expected_terms in TEST_CASES.items():
            details = assistant.retriever.retrieve(question, [])["details"]
            retrieved = " ".join(item["document"] for item in details).casefold()
            if not any(term in retrieved for term in expected_terms):
                raise AssertionError(f"LIVE: retrieved knowledge was irrelevant for: {question}")
            print(f"RETRIEVAL PASS  {question}")

        # One real full answer proves the complete planner-to-generation route;
        # the checks above cover retrieval quality for the remaining questions.
        question = "What does VS clarity mean?"
        result = assistant.ask(question)
        if len(result["answer"].strip()) < 20 or not result["knowledge"]:
            raise AssertionError("LIVE: full chatbot answer was empty or ungrounded.")
        print(f"ANSWER PASS     {question}")
    except Exception as error:
        raise SystemExit(
            "Live RAG test failed. Run 'ollama serve', pull qwen3.5:0.8b and "
            f"nomic-embed-text, and install requirements-assistant.txt.\nReason: {error}"
        ) from error
    print("\nLIVE RAG STATUS: SUCCESS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci", action="store_true",
        help="Use deterministic generation and embeddings while exercising real Chroma",
    )
    args = parser.parse_args()
    run_ci_test() if args.ci else run_live_test()


if __name__ == "__main__":
    main()
