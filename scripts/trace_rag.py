"""Print an auditable multi-turn RAG trace using the real local assistant."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.runtime import create_assistant  # noqa: E402


DEFAULT_QUESTIONS = [
    "I want a diamond around $6,000 and clarity matters most.",
    "About 1 carat.",
    "VS2 clarity or better, with an Ideal cut.",
    "Find something similar but cheaper.",
]


def print_result(question: str, result: dict) -> None:
    """Print pipeline decisions and evidence without private model chain-of-thought."""
    print("\n" + "=" * 72)
    print(f"QUESTION: {question}")
    print(f"STATUS: {result['status']}")
    print("\nROUTED INTENT AND EVIDENCE SOURCES")
    print(json.dumps(result.get("route", {}), indent=2))
    print("\nQUERY PLAN")
    print(json.dumps(result["plan"], indent=2))
    print("\nEMBEDDING QUERIES")
    print(json.dumps(result["initial_queries"] + result["extra_queries"], indent=2))
    print("\nSIMILAR VECTOR CHUNKS")
    if not result["knowledge_details"]:
        print("No vector search yet; the assistant needs clarification first.")
    for number, item in enumerate(result["knowledge_details"], start=1):
        print(
            f"[{number}] source={item['source']} similarity={item['similarity']} "
            f"query={item['query']}"
        )
        print(item["document"])
    print("\nDATASET MATCHES")
    print("None" if result["matches"].empty else result["matches"].to_string(index=False))
    print("\nSTRUCTURED SIMILARITY MATCHES")
    similar = result.get("similar_matches")
    print("None" if similar is None or similar.empty else similar.to_string(index=False))
    print("\nMEMORY RETRIEVED")
    print(json.dumps(result["retrieved_memory"], indent=2))
    print("\nPIPELINE TRACE")
    print(json.dumps(result["trace"], indent=2))
    print("\nCOMPACT EVIDENCE SENT TO QWEN")
    print(json.dumps(result.get("evidence", {}), indent=2))
    print("\nASSISTANT RESPONSE")
    print(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Conversation turn to run. Repeat this option for multiple turns.",
    )
    args = parser.parse_args()
    questions = args.questions or DEFAULT_QUESTIONS
    assistant = create_assistant()
    conversation: list[dict] = []
    try:
        for question in questions:
            result = assistant.ask(question, conversation=conversation)
            print_result(question, result)
            conversation.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ])
    finally:
        if hasattr(assistant.stores, "close"):
            assistant.stores.close()


if __name__ == "__main__":
    main()
