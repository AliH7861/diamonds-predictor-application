"""V8.3-inspired control plane for safe, auditable assistant execution."""

from dataclasses import asdict

from .request_types import build_typed_request
from .request_validation import RequestEnvelope, validate_plan
from .execution_policy import execution_policy
from .concepts import concepts_as_dicts
from ..schemas import DiamondQueryPlan, EvidenceRoute
from .state_management import apply_state_transaction
from .text_preprocessing import canonicalize_message


def prepare_request(
    question: str,
    route: EvidenceRoute,
    plan: DiamondQueryPlan | None,
    state: dict | None = None,
) -> tuple[DiamondQueryPlan | None, dict]:
    """Canonicalize, evidence-check, freeze, and type one assistant request."""
    message = canonicalize_message(question)
    envelope: RequestEnvelope | None = None
    validated = plan
    if plan is not None:
        validated, envelope = validate_plan(message, plan, state)
    task = build_typed_request(route, validated)
    policy = execution_policy(route)
    audit = {
        "pipeline_version": "V8.4",
        "normalized_message": message.canonical,
        "concepts": concepts_as_dicts(message.canonical),
        "route": asdict(route),
        "typed_request": asdict(task),
        "request_type": type(task).__name__,
        "token_policy": asdict(policy),
        "envelope": envelope.public_dict()
        if envelope
        else {
            "raw_text": message.raw,
            "canonical_text": message.canonical,
            "candidate_criteria": {},
            "accepted_criteria": {},
            "field_evidence": {},
            "dropped_candidates": {},
            "problems": [],
        },
    }
    if envelope is not None:
        resulting_state, state_audit = apply_state_transaction(state, envelope)
    else:
        resulting_state = dict(state or {})
        state_audit = {
            "previous_validated_state": dict(state or {}),
            "candidate_delta": [],
            "accepted_delta": [],
            "rejected_delta": [],
            "resulting_state": resulting_state,
            "committed": True,
        }
    if validated is not None:
        resulting_state["pending_buying"] = bool(validated.needs_clarification)
    audit["state"] = state_audit
    audit["resulting_state"] = resulting_state
    audit["clarification_needed"] = bool(validated and validated.needs_clarification)
    audit["stage_audit"] = [
        {"stage": "normalize", "status": "PASS"},
        {
            "stage": "field_evidence",
            "status": "PASS" if not (envelope and envelope.dropped) else "DROP",
        },
        {
            "stage": "state_transaction",
            "status": "PASS" if state_audit["committed"] else "ROLLBACK",
        },
        {"stage": "route_freeze", "status": "PASS"},
    ]
    return validated, audit
