"""Streamlit workspace for the local diamond research assistant."""

from uuid import uuid4

import pandas as pd

from .ui_content import APP_DESCRIPTION, APP_TITLE, INPUT_PLACEHOLDER


PAGE_STYLES = """
<style>
    :root {
        --forest: #123b2a;
        --leaf: #247a52;
        --mint: #e6f2eb;
        --line: #d6e3da;
        --ink: #193028;
        --muted: #63756d;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .stApp { background: #eef4f0; color: var(--ink); }
    .block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 4rem; }
    .topbar {
        align-items: center; background: #fff; border: 1px solid var(--line);
        border-radius: 14px; display: flex; justify-content: space-between;
        margin-bottom: 1rem; min-height: 58px; padding: 0 1.15rem;
    }
    .brand { align-items: center; color: var(--forest); display: flex; font-weight: 750; gap: .7rem; }
    .brand-mark { background: var(--leaf); border-radius: 3px; display: inline-block;
        height: 18px; transform: rotate(45deg); width: 18px; }
    .top-links { display: flex; gap: 1.6rem; color: var(--muted); font-size: .88rem; }
    .top-links span:first-child { color: var(--forest); font-weight: 700; }
    .system-pill { background: var(--mint); border: 1px solid #c5decf; border-radius: 999px;
        color: var(--forest); font-size: .78rem; font-weight: 700; padding: .4rem .75rem; }
    [data-testid="stHorizontalBlock"] { align-items: stretch; gap: .85rem; }
    [data-testid="column"] { background: #fff; border: 1px solid var(--line); border-radius: 14px;
        min-height: 690px; padding: 1rem 1rem 1.25rem; }
    .rail-kicker { color: var(--leaf); font-size: .72rem; font-weight: 800;
        letter-spacing: .1em; text-transform: uppercase; }
    .rail-title { color: var(--forest); font-size: 1.05rem; font-weight: 750; margin: .3rem 0 .1rem; }
    .rail-copy { color: var(--muted); font-size: .78rem; margin-bottom: .85rem; }
    .history-label { color: var(--muted); font-size: .72rem; font-weight: 700;
        letter-spacing: .06em; margin: 1.25rem 0 .4rem; text-transform: uppercase; }
    .workspace-kicker { color: var(--leaf); font-size: .75rem; font-weight: 800;
        letter-spacing: .11em; text-transform: uppercase; }
    .assistant-title { color: var(--forest); font-family: Georgia, serif; font-size: 2.25rem;
        line-height: 1.08; margin: .35rem 0 .65rem; }
    .assistant-description { color: var(--muted); font-size: .98rem; line-height: 1.65;
        max-width: 720px; }
    .welcome-panel { align-items: center; background: linear-gradient(145deg, #f7fbf8, #e8f3ec);
        border: 1px solid #d2e5d8; border-radius: 14px; display: flex;
        flex-direction: column; justify-content: center; margin: 1.3rem 0 1rem;
        min-height: 430px; padding: 2rem; text-align: center; }
    .welcome-title { color: var(--forest); font-family: Georgia, serif; font-size: 1.45rem;
        margin-bottom: .4rem; }
    .welcome-copy { color: var(--muted); font-size: .9rem; line-height: 1.55; max-width: 560px; }
    .stButton > button { background: #f8fbf9; border: 1px solid var(--line); border-radius: 9px;
        color: var(--forest); min-height: 2.55rem; text-align: left; }
    .stButton > button:hover { border-color: var(--leaf); color: var(--leaf); }
    [data-testid="stChatMessage"] { background: #fbfdfb; border: 1px solid var(--line);
        border-radius: 12px; margin-bottom: .7rem; padding: .55rem .8rem; }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    [data-testid="stChatInput"] { border-color: #a9c9b5; }
    div[data-testid="stExpander"] { background: #fbfdfb; border: 1px solid var(--line); }
    @media (max-width: 760px) {
        .top-links { display: none; }
        [data-testid="column"] { min-height: auto; }
        .assistant-title { font-size: 1.8rem; }
    }
</style>
"""


def _title_from_messages(messages: list[dict]) -> str:
    """Create a compact history label from the first user message."""
    first = next((item["content"] for item in messages if item.get("role") == "user"), None)
    if not first:
        return "New diamond search"
    return first.strip()[:38] + ("..." if len(first.strip()) > 38 else "")


def _initialize_sessions(st) -> None:
    """Create conversation storage and migrate the earlier single-chat state."""
    if "chat_sessions" in st.session_state:
        return
    existing = list(st.session_state.get("messages", []))
    session_id = "chat-initial"
    st.session_state["chat_sessions"] = {
        session_id: {"title": _title_from_messages(existing), "messages": existing}
    }
    st.session_state["active_chat_id"] = session_id


def _create_session(st) -> None:
    """Add and activate an empty conversation workspace."""
    session_id = f"chat-{uuid4().hex[:8]}"
    st.session_state["chat_sessions"][session_id] = {
        "title": "New diamond search",
        "messages": [],
    }
    st.session_state["active_chat_id"] = session_id


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
            rows = [{"Source": item["source"], "Similarity": item["similarity"], "Matched query": item["query"]} for item in details]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
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


def _describe_matches(frame: pd.DataFrame) -> list[str]:
    """Turn ranked dataset rows into concise descriptions for ordinary users."""
    descriptions = []
    for number, (_, row) in enumerate(frame.head(5).iterrows(), start=1):
        price = f"${float(row['price']):,.0f}" if pd.notna(row.get("price")) else "Price unavailable"
        traits = []
        if pd.notna(row.get("carat")):
            traits.append(f"{float(row['carat']):.2f} carats")
        if pd.notna(row.get("cut")):
            traits.append(f"{row['cut']} cut")
        if pd.notna(row.get("color")):
            traits.append(f"{row['color']} color")
        if pd.notna(row.get("clarity")):
            traits.append(f"{row['clarity']} clarity")
        sentence = f"{number}. **{price}** — " + (", ".join(traits) or "matching dataset diamond") + "."
        details = []
        if pd.notna(row.get("similarity_score")):
            details.append(f"{float(row['similarity_score']) * 100:.0f}% similarity to the reference")
        if pd.notna(row.get("model_price")):
            details.append(f"ANN price estimate ${float(row['model_price']):,.0f}")
        if details:
            sentence += " " + "; ".join(details).capitalize() + "."
        descriptions.append(sentence)
    return descriptions


def _render_message_result(st, result: dict, developer_mode: bool, heading: bool = False) -> None:
    """Describe ranked matches in prose and expose raw tables only to developers."""
    recommendations = result.get("similar_matches")
    if recommendations is None or recommendations.empty:
        recommendations = result["matches"]
    if not recommendations.empty:
        if heading:
            st.subheader("Closest dataset matches")
        st.markdown("\n\n".join(_describe_matches(recommendations)))
    if developer_mode:
        _render_trace(st, result, expanded=heading)


def render_app(st, assistant_factory) -> None:
    """Render a multi-conversation research workspace with an injected backend."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(PAGE_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="topbar">
            <div class="brand"><span class="brand-mark"></span><span>Diamond Lab</span></div>
            <div class="top-links"><span>Assistant</span><span>Models</span><span>Experiments</span><span>Research</span></div>
            <div class="system-pill">Local workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _initialize_sessions(st)

    @st.cache_resource
    def load_assistant():
        return assistant_factory()

    try:
        assistant = load_assistant()
    except Exception as error:
        st.error(f"Assistant setup failed: {error}")
        st.info("Check the dataset, Ollama, or the configured local assistant API.")
        return

    history_column, workspace_column = st.columns([0.27, 0.73], gap="small")
    with history_column:
        st.markdown('<div class="rail-kicker">Research workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="rail-title">Conversation jobs</div>', unsafe_allow_html=True)
        st.markdown('<div class="rail-copy">Open an earlier analysis or begin a new one.</div>', unsafe_allow_html=True)
        if st.button("New conversation", use_container_width=True, type="primary"):
            _create_session(st)
            st.rerun()
        sessions = st.session_state["chat_sessions"]
        st.markdown(f'<div class="history-label">History ({len(sessions):02d})</div>', unsafe_allow_html=True)
        for session_id, session in reversed(list(sessions.items())):
            label = session["title"]
            if session_id == st.session_state["active_chat_id"]:
                label = f"Current · {label}"
            if st.button(label, key=f"open-{session_id}", use_container_width=True):
                st.session_state["active_chat_id"] = session_id
                st.rerun()
        st.markdown('<div class="history-label">Tools</div>', unsafe_allow_html=True)
        developer_mode = bool(st.toggle("Developer evidence", value=False))
        st.caption("Inspect routes, filters, embeddings, retrieved context, and model evidence.")

    active_id = st.session_state["active_chat_id"]
    active_session = st.session_state["chat_sessions"][active_id]
    messages = active_session["messages"]

    with workspace_column:
        st.markdown('<div class="workspace-kicker">Dataset and model research assistant</div>', unsafe_allow_html=True)
        st.markdown(f'<h1 class="assistant-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown(f'<div class="assistant-description">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)
        if not messages:
            st.markdown(
                '<div class="welcome-panel"><div class="welcome-title">Make a confident diamond decision</div>'
                '<div class="welcome-copy">Describe your budget, preferred size, and quality priorities. '
                'The assistant searches real dataset examples and grounds its explanation in retrieved diamond knowledge.</div></div>',
                unsafe_allow_html=True,
            )
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("result"):
                    _render_message_result(st, message["result"], developer_mode)

        question = st.chat_input(INPUT_PLACEHOLDER)
        if not question:
            return
        messages.append({"role": "user", "content": question})
        active_session["title"] = _title_from_messages(messages)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching dataset rows, model evidence, and diamond knowledge..."):
                try:
                    response = st.empty()
                    streamed_text = []

                    def show_token(token: str) -> None:
                        streamed_text.append(token)
                        response.markdown("".join(streamed_text) + " ▌")

                    result = assistant.ask(
                        question,
                        conversation=messages[:-1],
                        on_token=show_token,
                    )
                except Exception as error:
                    st.error(f"The assistant could not answer: {error}")
                    return
            response.markdown(result["answer"])
            _render_message_result(st, result, developer_mode, heading=True)
        messages.append({"role": "assistant", "content": result["answer"], "result": result})
