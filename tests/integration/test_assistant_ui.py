import unittest

import pandas as pd

from src.assistant.ui import render_app


class Block:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeStreamlit:
    def __init__(self, question=None, developer_mode=False):
        self.question = question
        self.developer_mode = developer_mode
        self.session_state = {}
        self.events = []

    def cache_resource(self, function):
        return function

    def chat_input(self, _label):
        return self.question

    def chat_message(self, role):
        self.events.append(("chat", role))
        return Block()

    def toggle(self, _label, value=False):
        return self.developer_mode

    def spinner(self, label):
        self.events.append(("spinner", label))
        return Block()

    def expander(self, label, **_kwargs):
        self.events.append(("expander", label))
        return Block()

    def columns(self, widths, **_kwargs):
        return tuple(Block() for _ in widths)

    def button(self, label, **_kwargs):
        self.events.append(("button", label))
        return False

    def empty(self):
        return self

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append((name, args[0] if args else kwargs))
        return record


class FakeAssistant:
    def ask(self, question, conversation=None, on_token=None, state=None):
        if on_token:
            on_token("A tested ")
            on_token("answer.")
        return {
            "status": "answered", "answer": "A tested answer.",
            "matches": pd.DataFrame([{"price": 5000}]),
            "similar_matches": pd.DataFrame(), "route": {"intent": "recommendation"},
            "plan": {"max_price": 6000}, "initial_queries": ["cut"],
            "extra_queries": [], "knowledge": ["Cut affects sparkle."],
            "knowledge_details": [{
                "source": "diamond_basics.md", "similarity": 0.8,
                "query": "cut", "document": "Cut affects sparkle.",
            }],
            "retrieved_memory": [], "saved_memory": None, "evidence": {}, "trace": [],
            "conversation_state": {"max_price": 6000},
        }


class AssistantUITests(unittest.TestCase):
    def test_normal_mode_hides_debug_panel(self):
        st = FakeStreamlit("Find a diamond", developer_mode=False)
        render_app(st, FakeAssistant)
        active_id = st.session_state["active_chat_id"]
        messages = st.session_state["chat_sessions"][active_id]["messages"]
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(
            st.session_state["chat_sessions"][active_id]["state"]["max_price"],
            6000,
        )
        self.assertIn(("markdown", "A tested answer."), st.events)
        self.assertFalse(any(event[0] == "dataframe" for event in st.events))
        self.assertIn(("subheader", "Closest dataset matches"), st.events)
        self.assertFalse(any(event[0] == "expander" for event in st.events))

    def test_developer_mode_renders_same_result_with_trace(self):
        st = FakeStreamlit("Find a diamond", developer_mode=True)
        render_app(st, FakeAssistant)
        self.assertIn(("expander", "Developer evidence and execution trace"), st.events)

    def test_setup_failure_is_explained_in_the_page(self):
        st = FakeStreamlit()

        def fail():
            raise RuntimeError("Ollama offline")

        render_app(st, fail)
        self.assertTrue(any(event[0] == "error" and "Ollama offline" in event[1] for event in st.events))


if __name__ == "__main__":
    unittest.main()
