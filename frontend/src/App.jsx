import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamChat } from "./api";

const STORAGE_KEY = "diamond-react-conversations-v1";

function Icon({ name, size = 22 }) {
  const paths = {
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    plus: <><path d="M12 5v14"/><path d="M5 12h14"/></>,
    trash: <><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="m19 6-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/></>,
    chat: <><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z"/></>,
    sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"/></>,
    moon: <path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/>,
    diamond: <><path d="m3 8 4-5h10l4 5-9 13Z"/><path d="m3 8 9 5 9-5M7 3l5 10 5-10"/></>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

function DiamondMark({ large = false }) {
  return (
    <div className={large ? "diamond-mark diamond-mark--hero" : "diamond-mark"}>
      <svg viewBox="0 0 180 130" role="img" aria-label="Faceted diamond">
        <defs>
          <linearGradient id={large ? "gem-large" : "gem-small"} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#dff8ff"/><stop offset=".45" stopColor="#55c9ff"/><stop offset="1" stopColor="#1578d4"/>
          </linearGradient>
        </defs>
        <path d="M20 42 48 14h84l28 28-70 78Z" fill={`url(#${large ? "gem-large" : "gem-small"})`} opacity=".92"/>
        <g fill="none" stroke="#e9fbff" strokeWidth="2" opacity=".9">
          <path d="m20 42 70 78 70-78M20 42h140M48 14l42 106 42-106M20 42l28-28 42 28 42-28 28 28M48 14l-8 28 50 78 50-78-8-28M40 42h100"/>
        </g>
      </svg>
    </div>
  );
}

function newSession() {
  return {
    id: crypto.randomUUID(),
    title: "New diamond search",
    subtitle: "Start a new analysis",
    date: new Intl.DateTimeFormat("en", { day: "2-digit", month: "2-digit" }).format(new Date()),
    messages: [],
    state: {},
  };
}

function loadSessions() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(saved) && saved.length) return saved;
  } catch {
    // Invalid local state should never prevent the application from loading.
  }
  return [newSession()];
}

function family(value) {
  const grade = String(value || "").toUpperCase();
  if (grade === "IF") return "IF";
  if (grade.startsWith("VVS")) return "VVS";
  if (grade.startsWith("VS")) return "VS";
  if (grade.startsWith("SI")) return "SI";
  if (grade.startsWith("I")) return "I";
  return grade;
}

function MatchCards({ result }) {
  const rows = result?.similar_matches?.length ? result.similar_matches : result?.matches;
  if (!rows?.length) return null;
  return (
    <div className="match-grid">
      {rows.slice(0, 3).map((row, index) => (
        <article className="match-card" key={`${row.price}-${row.carat}-${index}`}>
          <span className="match-card__number">0{index + 1}</span>
          <strong>${Number(row.price).toLocaleString()}</strong>
          <p>{Number(row.carat).toFixed(2)} carat · {row.cut} cut</p>
          <p>{row.color} color · {family(row.clarity)} clarity</p>
        </article>
      ))}
    </div>
  );
}

export default function App() {
  const [sessions, setSessions] = useState(() => loadSessions());
  const [activeId, setActiveId] = useState(() => sessions[0].id);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("diamond-theme") || "dark");
  const endRef = useRef(null);

  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) || sessions[0],
    [sessions, activeId],
  );

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("diamond-theme", theme);
  }, [theme]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages]);

  function updateActive(updater) {
    setSessions((items) => items.map((session) => (
      session.id === activeId ? updater(session) : session
    )));
  }

  function createConversation() {
    const session = newSession();
    setSessions((items) => [session, ...items]);
    setActiveId(session.id);
  }

  function deleteConversation(event, id) {
    event.stopPropagation();
    const remaining = sessions.filter((session) => session.id !== id);
    const next = remaining.length ? remaining : [newSession()];
    setSessions(next);
    if (id === activeId) setActiveId(next[0].id);
  }

  async function submit(event) {
    event?.preventDefault();
    const question = input.trim();
    if (!question || sending) return;
    const prior = active.messages.slice(-2).map((message) => ({
      role: message.role,
      content: message.content,
      ...(message.result ? {
        result: {
          matches: message.result.matches || [],
          similar_matches: message.result.similar_matches || [],
        },
      } : {}),
    }));
    const userMessage = { id: crypto.randomUUID(), role: "user", content: question };
    const assistantId = crypto.randomUUID();
    updateActive((session) => ({
      ...session,
      title: session.messages.length ? session.title : question.slice(0, 34),
      subtitle: session.messages.length ? session.subtitle : "Dataset and model research",
      messages: [...session.messages, userMessage, { id: assistantId, role: "assistant", content: "" }],
    }));
    setInput("");
    setSending(true);
    try {
      let streamed = "";
      const result = await streamChat({
        question,
        conversation: prior,
        state: active.state || {},
        onToken: (token) => {
          streamed += token;
          updateActive((session) => ({
            ...session,
            messages: session.messages.map((message) => (
              message.id === assistantId ? { ...message, content: streamed } : message
            )),
          }));
        },
      });
      updateActive((session) => ({
        ...session,
        state: result.conversation_state || session.state,
        messages: session.messages.map((message) => (
          message.id === assistantId
            ? { ...message, content: result.answer, result }
            : message
        )),
      }));
    } catch (error) {
      updateActive((session) => ({
        ...session,
        messages: session.messages.map((message) => (
          message.id === assistantId
            ? { ...message, content: `I couldn't reach the assistant backend. ${error.message}` }
            : message
        )),
      }));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="app-shell">
        <header className="app-header">
          <DiamondMark/>
          <div>
            <h1>Diamonds Predictor Application</h1>
            <p>Dataset and model research assistant</p>
          </div>
          <button className="theme-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Switch color theme">
            <Icon name={theme === "dark" ? "sun" : "moon"} size={18}/>
            <span>{theme === "dark" ? "Light" : "Dark"}</span>
          </button>
        </header>

        <div className="workspace">
          <aside className="history-panel">
            <div className="history-heading">
              <div><h2>Chat History <span>({String(sessions.length).padStart(2, "0")})</span></h2><p>Your saved research threads</p></div>
              <button className="icon-button" onClick={createConversation} aria-label="New conversation"><Icon name="plus" size={20}/></button>
            </div>
            <div className="history-list">
              {sessions.map((session) => (
                <button className={`history-item ${session.id === activeId ? "is-active" : ""}`} key={session.id} onClick={() => setActiveId(session.id)}>
                  <span className="history-item__icon"><Icon name="chat" size={18}/></span>
                  <span className="history-item__copy"><strong>{session.title}</strong><small>{session.subtitle}</small></span>
                  <span className="history-item__meta">{session.date}</span>
                  <span className="history-item__delete" onClick={(event) => deleteConversation(event, session.id)} role="button" tabIndex="0" aria-label="Delete conversation"><Icon name="trash" size={16}/></span>
                </button>
              ))}
            </div>
            <div className="sidebar-note"><span>Local workspace</span><p>Your conversations stay in this browser.</p></div>
          </aside>

          <main className={`conversation ${active.messages.length ? "conversation--active" : ""}`}>
            <div className="ambient-lines" aria-hidden="true"><i/><i/><i/></div>
            {!active.messages.length ? (
              <section className="hero">
                <div className="hero-glow"/>
                <DiamondMark large/>
                <p className="eyebrow">Intelligent diamond research</p>
                <h2>Accurate Insights.<br/>Brighter Decisions.</h2>
                <p className="hero-copy">Compare real diamonds, understand quality trade-offs, and make a confident decision with grounded model evidence.</p>
              </section>
            ) : (
              <div className="message-list">
                <div className="conversation-label"><span>Active research</span><h2>{active.title}</h2></div>
                {active.messages.map((message) => (
                  <article className={`message message--${message.role}`} key={message.id}>
                    <div className="message__label">{message.role === "user" ? "You" : "Diamond adviser"}</div>
                    <div className="message__body">
                      {message.content ? <ReactMarkdown>{message.content}</ReactMarkdown> : <span className="typing"><i/><i/><i/></span>}
                      <MatchCards result={message.result}/>
                    </div>
                  </article>
                ))}
                <div ref={endRef}/>
              </div>
            )}

            <form className="composer" onSubmit={submit}>
              <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about a diamond, price, quality, or model result..." aria-label="Message"/>
              <button className="send-button" type="submit" disabled={!input.trim() || sending} aria-label="Send message"><Icon name="send" size={20}/></button>
            </form>
          </main>
        </div>
      </div>
    </div>
  );
}
