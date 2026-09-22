"""Print the eight production checkpoints for one live assistant question.

Start the application first with ``powershell -File scripts/start_dev.ps1``.
This script calls the same HTTP endpoint as the React frontend, so its output
explains the behavior the user actually sees rather than a parallel test path.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:  # noqa: S310 - local dev URL
        return json.loads(response.read().decode("utf-8"))


def _print_checkpoint(number: int, title: str, value: Any) -> None:
    print(f"\n[{number}/8] {title}")
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Natural-language question to trace")
    parser.add_argument(
        "--state-json",
        default="{}",
        help="Optional state from an earlier turn, as a JSON object",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8770/chat")
    args = parser.parse_args()

    state = json.loads(args.state_json)
    if not isinstance(state, dict):
        raise SystemExit("--state-json must contain a JSON object")

    try:
        result = _post(
            args.api_url,
            {"question": args.question, "conversation": [], "state": state},
        )
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"The assistant returned HTTP {error.code}. Response: {detail}") from error
    except URLError as error:
        raise SystemExit(
            "The assistant backend is unavailable. Run "
            "`powershell -File scripts/start_dev.ps1` first. "
            f"Details: {error}"
        ) from error

    details = result.get("diagnostics", {})
    _print_checkpoint(1, "QUESTION", details.get("question"))
    _print_checkpoint(2, "ROUTE", details.get("route", {}))
    _print_checkpoint(
        3,
        "RAW PLAN",
        details.get("raw_criteria", {}),
    )
    _print_checkpoint(
        4,
        "VALIDATED PLAN AND STATE",
        {
            "accepted_criteria": details.get("validated_criteria", {}),
            "dropped_candidates": details.get("dropped_candidates", {}),
            "problems": details.get("validation_problems", []),
            "state_transaction": details.get("state_changes", {}),
            "resulting_state": result.get("conversation_state", {}),
        },
    )
    _print_checkpoint(
        5,
        "DATASET EXECUTION",
        {
            "filters": details.get("dataset_filters", {}),
            "qualifying_rows": details.get("dataset_qualifying_rows"),
            "ranking_strategy": details.get("dataset_ranking_strategy"),
            "selected_result_ids": details.get("selected_result_ids", []),
            "first_matches": details.get("dataset_preview", []),
        },
    )
    _print_checkpoint(
        6,
        "RAG",
        {
            "queries": details.get("rag_queries", []),
            "chunks": details.get("rag_chunks", []),
        },
    )
    _print_checkpoint(
        7,
        "GENERATION EVIDENCE",
        {
            "evidence": details.get("evidence_payload", {}),
            "prompt": details.get("generation_prompt"),
        },
    )
    _print_checkpoint(8, "FINAL ANSWER", result.get("answer"))


if __name__ == "__main__":
    main()
