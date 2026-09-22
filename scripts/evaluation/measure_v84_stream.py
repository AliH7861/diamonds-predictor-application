"""Measure one real streaming API request."""

import json
from pathlib import Path
from time import perf_counter

import requests


started = perf_counter()
first_token_ms = None
result = None
token_events = 0
response = requests.post(
    "http://127.0.0.1:8770/chat/stream",
    json={"question": "What does VS clarity mean?", "conversation": [], "state": {}},
    stream=True,
    timeout=90,
)
response.raise_for_status()
for line in response.iter_lines():
    event = json.loads(line)
    elapsed = (perf_counter() - started) * 1000
    if event["type"] == "token":
        first_token_ms = first_token_ms or elapsed
        token_events += 1
    elif event["type"] == "result":
        result = event["value"]
diagnostics = result.get("diagnostics", {})
summary = {
    "http_status": response.status_code,
    "first_token_ms": first_token_ms,
    "total_ms": (perf_counter() - started) * 1000,
    "token_events": token_events,
    "request_id": result.get("request_id"),
    "api_handler_ms": diagnostics.get("api_handler_ms"),
    "service_total_ms": diagnostics.get("total_ms"),
}
target = Path("outputs/evaluation/v84_test4/streaming_measurement.json")
target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
