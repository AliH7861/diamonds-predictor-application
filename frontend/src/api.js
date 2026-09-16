const API_BASE = (import.meta.env.VITE_ASSISTANT_API_URL || "/assistant-api").replace(/\/$/, "");
const API_TOKEN = import.meta.env.VITE_ASSISTANT_API_TOKEN || "";

function headers() {
  const value = { "Content-Type": "application/json" };
  if (API_TOKEN) value.Authorization = `Bearer ${API_TOKEN}`;
  return value;
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`, { headers: headers() });
  if (!response.ok) throw new Error("The local assistant backend is unavailable.");
  return response.json();
}

export async function streamChat({ question, conversation, state, onToken }) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ question, conversation, state }),
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(detail || `Assistant request failed with HTTP ${response.status}.`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "token") onToken(event.value);
      if (event.type === "result") result = event.value;
      if (event.type === "error") throw new Error(event.value);
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "result") result = event.value;
    if (event.type === "error") throw new Error(event.value);
  }
  if (!result) throw new Error("The assistant stream ended without a result.");
  return result;
}
