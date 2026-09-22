"""Local HTTP backend for the Diamond Decision Assistant."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter
from typing import Any, cast

from .runtime import create_assistant
from .transport import (
    decode_conversation,
    encode_result,
    extract_state_from_conversation,
    normalize_state,
)


MAX_REQUEST_BYTES = 1_000_000
LOGGER = logging.getLogger(__name__)


class AssistantHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying the initialized assistant and browser settings."""

    assistant: Any
    api_token: str | None
    allowed_origins: set[str]
    session_states: dict[str, dict]


class AssistantRequestHandler(BaseHTTPRequestHandler):
    """Serve health information and natural-language chat requests."""

    server_version = "DiamondAssistant/1.1"

    @property
    def assistant_server(self) -> AssistantHTTPServer:
        return cast(AssistantHTTPServer, self.server)

    def _cors_origin(self) -> str | None:
        supplied = self.headers.get("Origin")
        allowed = self.assistant_server.allowed_origins
        if supplied and ("*" in allowed or supplied in allowed):
            return "*" if "*" in allowed else supplied
        return None

    def _write_cors_headers(self) -> None:
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _write_json(self, status: int, value: dict) -> None:
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._write_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _write_stream_event(self, event_type: str, value: Any) -> None:
        payload = json.dumps({"type": event_type, "value": value}, allow_nan=False)
        self.wfile.write((payload + "\n").encode("utf-8"))
        self.wfile.flush()

    def _authorized(self) -> bool:
        expected = self.assistant_server.api_token
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    @staticmethod
    def _request_state(request: dict, conversation: list[dict]) -> dict:
        """Accept common frontend state names and recover message-attached state."""
        candidate = request.get("state")
        if candidate is None:
            candidate = request.get("conversation_state")
        if candidate is None:
            candidate = request.get("conversationState")
        state = normalize_state(candidate)
        return state or extract_state_from_conversation(conversation)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._write_json(404, {"error": "Not found"})
            return
        models = getattr(self.assistant_server.assistant, "enricher", None)
        self._write_json(
            200,
            {
                "status": "ok",
                "service": "diamond-assistant",
                "resources": models.status() if models and hasattr(models, "status") else {},
            },
        )

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._write_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        request_started = perf_counter()
        stream_started = False
        if self.path not in {"/chat", "/chat/stream"}:
            self._write_json(404, {"error": "Not found"})
            return
        if not self._authorized():
            self._write_json(401, {"error": "Unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body size is invalid.")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object.")

            question = request.get("question", request.get("message", ""))
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must contain text.")

            conversation = decode_conversation(request.get("conversation", []))
            state = self._request_state(request, conversation)

            # Optional session fallback. Explicit request state always wins.
            session_id = str(
                request.get("session_id") or request.get("conversation_id") or ""
            ).strip()
            if session_id and not state:
                state = dict(self.assistant_server.session_states.get(session_id, {}))

            if self.path == "/chat":
                result = self.assistant_server.assistant.ask(
                    question.strip(), conversation=conversation, state=state
                )
                result.setdefault("diagnostics", {})["api_handler_ms"] = round(
                    (perf_counter() - request_started) * 1000, 3
                )
                if session_id:
                    self.assistant_server.session_states[session_id] = dict(
                        result.get("conversation_state") or {}
                    )
                    result["session_id"] = session_id
                self._write_json(200, encode_result(result))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._write_cors_headers()
            self.end_headers()
            stream_started = True

            result = self.assistant_server.assistant.ask(
                question.strip(),
                conversation=conversation,
                on_token=lambda token: self._write_stream_event("token", token),
                state=state,
            )
            result.setdefault("diagnostics", {})["api_handler_ms"] = round(
                (perf_counter() - request_started) * 1000, 3
            )
            if session_id:
                self.assistant_server.session_states[session_id] = dict(
                    result.get("conversation_state") or {}
                )
                result["session_id"] = session_id
            self._write_stream_event("result", encode_result(result))

        except (json.JSONDecodeError, TypeError, ValueError) as error:
            if stream_started:
                try:
                    self._write_stream_event("error", str(error))
                except OSError:
                    pass
            else:
                self._write_json(400, {"error": str(error)})
        except Exception as error:
            LOGGER.exception("Assistant request failed")
            if stream_started:
                try:
                    self._write_stream_event("error", str(error))
                except OSError:
                    pass
            else:
                self._write_json(500, {"error": str(error)})

    def log_message(self, format_string: str, *args: object) -> None:
        LOGGER.info("HTTP: %s", format_string % args)


def create_server(
    assistant,
    host: str,
    port: int,
    token: str | None = None,
    allowed_origins: set[str] | None = None,
):
    server = AssistantHTTPServer((host, port), AssistantRequestHandler)
    server.assistant = assistant
    server.api_token = token
    server.allowed_origins = allowed_origins or {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    }
    server.session_states = {}
    return server


def main() -> None:
    logging.basicConfig(
        level=os.getenv("DIAMOND_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    token = os.getenv("DIAMOND_ASSISTANT_API_TOKEN") or None
    allowed_origins = {
        item.strip()
        for item in os.getenv(
            "DIAMOND_FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    }

    LOGGER.info("Loading dataset, vector index, Ollama clients, and saved models")
    server = create_server(create_assistant(), args.host, args.port, token, allowed_origins)
    LOGGER.info("Assistant backend ready at http://%s:%s", args.host, server.server_port)
    LOGGER.info("Health: GET /health | Chat: POST /chat | Stream: POST /chat/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Assistant backend stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
