"""Local HTTP backend for the Streamlit Diamond Decision Assistant."""

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .runtime import create_assistant
from .transport import decode_conversation, encode_result


MAX_REQUEST_BYTES = 1_000_000


class AssistantRequestHandler(BaseHTTPRequestHandler):
    """Serve health information and natural-language chat requests."""

    server_version = "DiamondAssistant/1.0"

    def _write_json(self, status: int, value: dict) -> None:
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_stream_event(self, event_type: str, value) -> None:
        """Write one newline-delimited JSON event and make it visible immediately."""
        payload = json.dumps({"type": event_type, "value": value}, allow_nan=False)
        self.wfile.write((payload + "\n").encode("utf-8"))
        self.wfile.flush()

    def _authorized(self) -> bool:
        expected = getattr(self.server, "api_token", None)
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._write_json(404, {"error": "Not found"})
            return
        self._write_json(200, {"status": "ok", "service": "diamond-assistant"})

    def do_POST(self) -> None:  # noqa: N802
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
            question = request.get("question", "")
            conversation = decode_conversation(request.get("conversation", []))
            if self.path == "/chat":
                result = self.server.assistant.ask(question, conversation=conversation)
                self._write_json(200, encode_result(result))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            stream_started = True
            result = self.server.assistant.ask(
                question,
                conversation=conversation,
                on_token=lambda token: self._write_stream_event("token", token),
            )
            self._write_stream_event("result", encode_result(result))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._write_json(400, {"error": str(error)})
        except Exception as error:
            if stream_started:
                try:
                    self._write_stream_event("error", str(error))
                except OSError:
                    pass
            else:
                self._write_json(500, {"error": str(error)})

    def log_message(self, format_string: str, *args) -> None:
        """Keep concise standard HTTP logs in the terminal."""
        print(f"Assistant API: {format_string % args}")


def create_server(assistant, host: str, port: int, token: str | None = None):
    """Create a testable threaded HTTP server around an assistant instance."""
    server = ThreadingHTTPServer((host, port), AssistantRequestHandler)
    server.assistant = assistant
    server.api_token = token
    return server


def main() -> None:
    """Load local models/RAG once and run the backend until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    token = os.getenv("DIAMOND_ASSISTANT_API_TOKEN") or None
    print("Loading dataset, vector index, Ollama clients, and saved models...")
    server = create_server(create_assistant(), args.host, args.port, token)
    print(f"Assistant backend ready at http://{args.host}:{server.server_port}")
    print("Health: GET /health | Chat: POST /chat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAssistant backend stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
