"""Verify the Streamlit-compatible client can call the separated local backend."""

import threading
from urllib.request import Request, urlopen

import pandas as pd

from src.assistant.api import create_server
from src.assistant.client import AssistantAPIClient


class FakeAssistant:
    def ask(self, question, conversation=None, on_token=None, state=None):
        if on_token:
            on_token("Grounded ")
            on_token("answer")
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
            "conversation_state": state or {},
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

        chunks = []
        streamed = client.ask("Stream a recommendation.", on_token=chunks.append)
        assert "".join(chunks) == "Grounded answer"
        assert streamed["status"] == "answered"
        assert streamed["matches"].iloc[0]["price"] == 5900

        preflight = Request(
            f"http://127.0.0.1:{server.server_port}/chat/stream",
            method="OPTIONS",
            headers={"Origin": "http://localhost:5173"},
        )
        with urlopen(preflight, timeout=5) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
            assert "Authorization" in response.headers["Access-Control-Allow-Headers"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
