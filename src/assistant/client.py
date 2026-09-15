"""Small HTTP client used when Streamlit and the local assistant backend are separate."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .transport import decode_result, encode_conversation


class AssistantAPIClient:
    """Expose the same ``ask`` method as the in-process assistant service."""

    def __init__(self, base_url: str, token: str | None = None, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def ask(self, question: str, conversation: list[dict] | None = None) -> dict:
        """Send a natural-language turn to the configured local backend."""
        body = json.dumps({
            "question": question,
            "conversation": encode_conversation(conversation),
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(f"{self.base_url}/chat", data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return decode_result(json.loads(response.read().decode("utf-8")))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Assistant backend returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach the assistant backend at {self.base_url}. "
                "Start it locally or update DIAMOND_ASSISTANT_API_URL."
            ) from error


def create_frontend_assistant():
    """Use HTTP separation when configured, otherwise preserve one-command local mode."""
    api_url = os.getenv("DIAMOND_ASSISTANT_API_URL", "").strip()
    if api_url:
        return AssistantAPIClient(
            api_url,
            token=os.getenv("DIAMOND_ASSISTANT_API_TOKEN") or None,
        )
    # Import the heavier local runtime only when Streamlit is running the
    # backend in-process. HTTP frontend mode needs only this small client.
    from .runtime import create_assistant

    return create_assistant()
