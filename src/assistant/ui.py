"""Streamlit presentation kept separate from retrieval and model logic."""

import pandas as pd

from .ui_content import APP_DESCRIPTION, APP_TITLE, INPUT_PLACEHOLDER, STARTER_QUESTIONS


PAGE_STYLES = """
<style>
    .stApp { background: #f5f3ee; color: #1f2933; }
    .block-container { max-width: 1120px; padding-top: 3.2rem; padding-bottom: 5rem; }
    .assistant-kicker { color: #8b5e34; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .5rem; }
    .assistant-title { color: #17212b; font-family: Georgia, serif; font-size: 3rem;
        line-height: 1.05; margin: 0 0 .8rem 0; }
    .assistant-description { color: #52606d; font-size: 1.05rem; max-width: 760px;
        line-height: 1.7; margin-bottom: 1.5rem; }
    .assistant-rule { border: 0; border-top: 1px solid #d9d3c7; margin: 1.5rem 0 2rem; }
    [data-testid="stChatMessage"] { background: #ffffff; border: 1px solid #e3ded4;
        border-radius: 12px; padding: .55rem .8rem; margin-bottom: .75rem; box-shadow: none; }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    [data-testid="stChatInput"] { border-color: #b9aa94; }
    div[data-testid="stExpander"] { background: #fff; border: 1px solid #d9d3c7;
        border-radius: 10px; }
    .starter { color: #6b6258; font-size: .9rem; padding: .35rem 0; }
</style>
"""


def _render_trace(st, result: dict, expanded: bool = True) -> None:
    """Show every internal stage only when developer mode is enabled."""
    with st.expander("Developer evidence and execution trace", expanded=expanded):
        st.markdown("#### 1. Routed intent and sources")
        st.json(result.get("route", {}))

        st.markdown("#### 2. Internal structured plan")
        st.json(result["plan"])

        st.markdown("#### 3. Exact dataset matches")
        if result["matches"].empty:
            st.write("No exact dataset rows were requested or matched.")
        else:
            st.dataframe(result["matches"], width="stretch", hide_index=True)

        st.markdown("#### 4. Structured similarity matches")
        similar = result.get("similar_matches", pd.DataFrame())
        if similar.empty:
            st.write("Structured similarity was not needed for this question.")
        else:
            st.dataframe(similar, width="stretch", hide_index=True)

        st.markdown("#### 5. RAG embedding searches")
        st.write(result["initial_queries"] + result["extra_queries"] or "No embedding search yet.")

        st.markdown("#### 6. Similar knowledge chunks")
        details = result.get("knowledge_details", [])
        if details:
            score_rows = pd.DataFrame([
                {
                    "Source": item["source"],
                    "Similarity": item["similarity"],
                    "Matched query": item["query"],
                }
                for item in details
            ])
            st.dataframe(score_rows, width="stretch", hide_index=True)
            for number, item in enumerate(details, start=1):
                st.markdown(f"**Context {number} — {item['source']}**")
                st.write(item["document"])
        else:
            st.write("Retrieval waits until the assistant has enough buying information.")

        st.markdown("#### 7. Preference memory")
        st.write(result["retrieved_memory"] or "No earlier preferences retrieved.")
        if result["saved_memory"]:
            st.caption(f"Saved this preference: {result['saved_memory']}")

        st.markdown("#### 8. Compact evidence sent to Qwen")
        st.json(result.get("evidence", {}))

        st.markdown("#### 9. Pipeline stages and ANN status")
        st.json(result.get("trace", []))


def render_app(st, assistant_factory) -> None:
    """Render the chat page with an injected factory so CI can test it locally."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(PAGE_STYLES, unsafe_allow_html=True)
    st.markdown('<div class="assistant-kicker">Dataset and model research assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="assistant-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="assistant-description">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="assistant-rule">', unsafe_allow_html=True)
    developer_mode = bool(st.toggle("Developer mode", value=False))

    @st.cache_resource
    def load_assistant():
        return assistant_factory()

    try:
        assistant = load_assistant()
    except Exception as error:
        st.error(f"Assistant setup failed: {error}")
        st.info(
            "For local mode, check the dataset and Ollama. For split mode, check "
            "DIAMOND_ASSISTANT_API_URL, the API token, and the local backend."
        )
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.markdown("**Try one of these:**")
        for starter in STARTER_QUESTIONS:
            st.markdown(f'<div class="starter">{starter}</div>', unsafe_allow_html=True)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("result"):
                recommendations = message["result"].get("similar_matches")
                if recommendations is None or recommendations.empty:
                    recommendations = message["result"]["matches"]
                if not recommendations.empty:
                    st.dataframe(
                        recommendations, width="stretch", hide_index=True
                    )
                if developer_mode:
                    _render_trace(st, message["result"], expanded=False)

    question = st.chat_input(INPUT_PLACEHOLDER)
    if not question:
        return
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching diamonds and relevant knowledge..."):
            try:
                result = assistant.ask(
                    question,
                    conversation=st.session_state["messages"][:-1],
                )
            except Exception as error:
                st.error(f"The assistant could not answer: {error}")
                return
        st.markdown(result["answer"])
        recommendations = result.get("similar_matches")
        if recommendations is None or recommendations.empty:
            recommendations = result["matches"]
        if not recommendations.empty:
            st.subheader("Recommended dataset examples")
            st.dataframe(recommendations, width="stretch", hide_index=True)
        if developer_mode:
            _render_trace(st, result)
    st.session_state["messages"].append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )
