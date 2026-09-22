"""Decide when a routed request is allowed to spend embedding or LLM tokens."""

from dataclasses import dataclass

from ..schemas import EvidenceRoute


@dataclass(frozen=True)
class ExecutionPolicy:
    """Token policy attached to one validated route."""

    use_embeddings: bool
    use_generation: bool
    path: str
    llm_calls: int
    embedding_calls: int
    generation_calls: int
    reason: str


def execution_policy(route: EvidenceRoute) -> ExecutionPolicy:
    """Use tokens only when a natural-language explanation needs retrieved knowledge."""
    if route.family in {"conversation", "unsupported"}:
        return ExecutionPolicy(
            False, False, "deterministic", 0, 0, 0, "deterministic conversational response"
        )
    if route.family == "model":
        return ExecutionPolicy(
            False, False, "deterministic", 0, 0, 0, "saved-model result can be formatted directly"
        )
    if route.family == "data" and not route.use_knowledge:
        return ExecutionPolicy(
            False, False, "deterministic", 0, 0, 0, "Pandas result can be formatted directly"
        )
    return ExecutionPolicy(
        bool(route.use_knowledge),
        bool(route.use_knowledge),
        "rag" if route.use_knowledge else "deterministic",
        1 if route.use_knowledge else 0,
        1 if route.use_knowledge else 0,
        1 if route.use_knowledge else 0,
        "grounded explanation requires retrieved knowledge",
    )
