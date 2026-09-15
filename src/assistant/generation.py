"""Minimal local Ollama client for structured chat and embedding generation."""

import json
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen


class OllamaClient:
    """Call a local Ollama server; no cloud credentials or network service is required."""

    def __init__(self, chat_model="qwen3.5:0.8b", embedding_model="nomic-embed-text", base_url="http://127.0.0.1:11434"):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            # CPU-only generation can take longer on the first request while the
            # model is loaded. Keep the connection open long enough for that run.
            with urlopen(request, timeout=300) as response:
                return json.load(response)
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                "Ollama took too long to answer. The local model may still be loading; "
                "retry once it is warm."
            ) from error
        except URLError as error:
            raise RuntimeError("Ollama is unavailable. Start it with 'ollama serve'.") from error
        except OSError as error:
            raise RuntimeError("Ollama is unavailable. Start it with 'ollama serve'.") from error

    def warmup(self) -> None:
        """Load the chat model into memory before the first user question."""
        self._post("/api/generate", {
            "model": self.chat_model,
            "prompt": "",
            "stream": False,
            "keep_alive": "30m",
            "options": {"num_predict": 1},
        })

    def complete(self, system: str, user: str, on_token=None) -> str:
        """Generate an answer, optionally forwarding each streamed text chunk."""
        payload = {
            "model": self.chat_model,
            "stream": on_token is not None,
            "think": False,
            "keep_alive": "30m",
            # Streaming keeps longer answers readable while they are generated.
            "options": {"temperature": 0, "num_predict": 400},
            "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user}
            ],
        }
        if on_token is None:
            result = self._post("/api/chat", payload)
            return result["message"]["content"].strip()

        request = Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        chunks = []
        try:
            with urlopen(request, timeout=300) as response:
                for line in response:
                    event = json.loads(line)
                    text = event.get("message", {}).get("content", "")
                    if text:
                        chunks.append(text)
                        on_token(text)
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                "Ollama took too long to answer. The local model may still be loading; "
                "retry once it is warm."
            ) from error
        except (URLError, OSError) as error:
            raise RuntimeError("Ollama is unavailable. Start it with 'ollama serve'.") from error
        return "".join(chunks).strip()

    def structured(self, system: str, user: str) -> dict:
        if "search plan" in system:
            properties = {
                "search_dataset": {"type": "boolean"},
                "min_price": {"type": ["number", "null"]},
                "max_price": {"type": ["number", "null"]},
                "target_carat": {"type": ["number", "null"]},
                "carat_tolerance": {"type": "number"},
                "depth": {"type": ["number", "null"]},
                "table": {"type": ["number", "null"]},
                "x": {"type": ["number", "null"]},
                "y": {"type": ["number", "null"]},
                "z": {"type": ["number", "null"]},
                "cut": {"type": ["string", "null"]},
                "color": {"type": ["string", "null"]},
                "clarity": {"type": ["string", "null"]},
                "priorities": {"type": "array", "items": {"type": "string"}},
                "needs_clarification": {"type": "boolean"},
                "clarifying_question": {"type": ["string", "null"]},
                "knowledge_queries": {"type": "array", "items": {"type": "string"}},
            }
        elif "more_context" in system:
            properties = {
                "needs_more_context": {"type": "boolean"},
                "extra_queries": {"type": "array", "items": {"type": "string"}},
            }
        else:
            properties = {
                "should_save": {"type": "boolean"},
                "memory_text": {"type": ["string", "null"]},
            }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        result = self._post("/api/chat", {
            "model": self.chat_model,
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": "30m",
            # The plan schema has several nullable fields. Give the local model enough
            # output room to close the JSON object even on slower CPU-only machines.
            "options": {"temperature": 0, "num_predict": 400},
            "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user}
            ],
        })
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("Ollama did not return valid structured JSON.") from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": texts, "keep_alive": "30m"},
        )
        return result["embeddings"]
