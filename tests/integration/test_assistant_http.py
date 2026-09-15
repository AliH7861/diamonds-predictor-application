"""Verify the Streamlit-compatible client can call the separated local backend."""

import threading

import pandas as pd

from src.assistant.api import create_server
from src.assistant.client import AssistantAPIClient


class FakeAssistant:
    def ask(self, question, conversation=None):
        return {
            "status": "answered",
            "answer": f"Grounded answer for: {question}",
            "matches": pd.DataFrame([{"carat": 1.0, "price": 5900}]),
            "similar_matches": pd.DataFrame(),
            "route": {"intent": "recommendation"},
            "plan": {"max_price": 6000},
            "initial_queries": ["diamond quality"],
            "extra_queries": [],
            "knowledge": ["Cut affects light performance."],
            "knowledge_details": [],
            "retrieved_memory": [],
            "saved_memory": None,
            "evidence": {},
            "trace": [{"stage": "generation", "result": "completed"}],
        }


def test_http_frontend_backend_round_trip():
    server = create_server(FakeAssistant(), "127.0.0.1", 0, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = AssistantAPIClient(
            f"http://127.0.0.1:{server.server_port}", token="test-token", timeout=5
        )
        result = client.ask("Find a diamond under $6,000.")
        assert result["status"] == "answered"
        assert result["matches"].iloc[0]["price"] == 5900
        assert result["similar_matches"].empty
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
