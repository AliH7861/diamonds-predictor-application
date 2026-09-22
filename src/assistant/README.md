# Assistant Architecture and Debug Map

The top level now contains only the live React/backend pipeline: 20 implementation
modules plus `__init__.py`. Historical V8.4 helpers live under `legacy/`, and the
optional Streamlit adapter lives under `streamlit_app/`.

## Live request path

```text
React App
  -> frontend/src/api.js
  -> api.py
  -> runtime.py
  -> service.py
       -> routing.py
       -> search_planning.py -> dataset_search.py
       -> comparison.py
       -> data_analysis.py
       -> domain_answers.py
       -> retrieval.py -> vector_store.py -> generation.py
       -> clarification.py -> tool_executor.py -> model_evidence.py
  -> transport.py
  -> React App
```

`service.py` is the orchestrator. Start there when a response uses the wrong tool.
Start in `routing.py` when the intent is wrong, and then move to the executor named
in the route.

## File map

| File | Status | Responsibility | Connected to |
|---|---|---|---|
| `__init__.py` | Active | Lazy public exports for `DiamondAssistant` and `DiamondCatalog`. | Imports `schemas.py`; loads `service.py` or `dataset_search.py` on demand. |
| `api.py` | Active | `/health`, `/chat`, and `/chat/stream` HTTP endpoints. | Creates the app through `runtime.py`; serializes through `transport.py`. |
| `clarification.py` | Partly active | Normalizes text and parses saved-model inputs. It also retains the older `build_buying_plan()` implementation for compatibility tests. | Active calls come from `routing.py`, `search_planning.py`, and `service.py`. |
| `clarity.py` | Active | Converts detailed clarity grades to I, SI, VS, VVS, or IF. | Used by planning, schemas, comparisons, prompts, and both UIs. |
| `comparison.py` | Active | Recovers the rows displayed in the current chat and compares requested positions. | Called by `service.py`; uses `clarity.py`. |
| `config.py` | Active | Dataset path, Ollama models, RAG limits, and other runtime settings. | Read by `runtime.py`. |
| `data_analysis.py` | Active | Pandas counts, ranges, overview, distributions, correlations, and grouped statistics. | Called by `service._analyze()`. |
| `dataset_search.py` | Active | Loads rows, applies exact filters, ranks candidates, diversifies five results, and returns audit details. | Receives `DiamondQueryPlan` from `search_planning.py`; called by `service.py`. |
| `domain_answers.py` | Active | Short authoritative answers for cut, color, clarity, and clustering vocabulary. | Called by `service._knowledge()` before RAG. |
| `generation.py` | Active | Ollama chat and embedding HTTP client, streaming, and warmup. | Created by `runtime.py`; used by `retrieval.py`, `vector_store.py`, and `service.py`. |
| `model_evidence.py` | Active | Loads classification, regression, and clustering artifacts and executes predictions. | Created by `runtime.py`; invoked through `tool_executor.py` or `service.py`. |
| `prompt_builder.py` | Active | Builds compact, labeled evidence and the knowledge-generation prompt. | Called by `service.py`; uses `schemas.py` and `clarity.py`. |
| `retrieval.py` | Active | Creates up to two focused knowledge queries and filters irrelevant clustering chunks. | Called by `service._knowledge()`; searches `vector_store.py`. |
| `routing.py` | Active | Assigns exactly one of seven intents and declares its evidence source. | Called first by `service._ask()`. |
| `runtime.py` | Active | Builds the dataset, Ollama client, Chroma stores, saved models, and assistant once. | Called by `api.py` and diagnostic scripts. |
| `schemas.py` | Active | Shared route, query-plan, and compatibility dataclasses. | Imported throughout the backend. |
| `search_planning.py` | Active | Extracts search constraints, validates supported values, applies current-chat state, and selects ranking behavior. | Called by `service._search()`; produces a plan for `dataset_search.py`. |
| `service.py` | Active | Main orchestrator and eight-stage diagnostic trace. Dispatches the seven supported intents. | Connects every active evidence component. |
| `tool_executor.py` | Active | Validates and executes multi-model requests and formats their combined result. | Called by `service._predict()`; calls `model_evidence.py`. |
| `transport.py` | Active | Converts DataFrames and nested Python values to/from HTTP-safe JSON. | Used by `api.py` and `client.py`. |
| `vector_store.py` | Active | Chunks Markdown, indexes Chroma, embeds questions, and returns scored knowledge chunks. | Created by `runtime.py`; queried through `retrieval.py`. |

### Organized supporting folders

| Folder | Files | Purpose |
|---|---|---|
| `legacy/` | `concepts.py`, `control_plane.py`, `execution_policy.py`, `request_types.py`, `request_validation.py`, `state_management.py`, `text_preprocessing.py` | Historical V8.4 evaluation pipeline. None are called by the live service. |
| `streamlit_app/` | `client.py`, `ui.py` | Optional Streamlit interface retained for `streamlit run app.py`. UI text now lives directly in `ui.py`. |

## Function-to-function flow

| Stage | Function | Receives | Calls / produces | Debug when |
|---|---|---|---|---|
| 1 | `AssistantRequestHandler._handle_chat()` | JSON question, conversation, state | `DiamondAssistant.ask()` | HTTP body, streaming, or CORS fails. |
| 2 | `DiamondAssistant.ask()` | Current request | `_ask()` plus eight-stage diagnostics | Trace fields or final packaging are wrong. |
| 3 | `route_question()` | Question and short chat context | One of seven intents | The assistant chooses search, RAG, model, or analysis incorrectly. |
| 4A | `build_search_plan()` | Search text and this chat's state | `SearchInterpretation` and `DiamondQueryPlan` | Budget, carat, category, follow-up, or clarification is wrong. |
| 5A | `DiamondCatalog.search_with_trace()` | Validated plan | Five filtered/ranked rows and execution trace | Prices, filters, ordering, diversity, or repeated IDs are wrong. |
| 4B | `displayed_rows()` / `compare_rows()` | Conversation results and comparison text | Exact displayed rows and grounded comparison | “First,” “third,” or trade-off comparisons use the wrong row. |
| 4C | `execute_analysis()` | Full DataFrame and analysis action | Pandas statistic plus evidence dictionary | Counts, ranges, averages, or distributions are wrong. |
| 4D | `answer_domain_catalog()` | Fixed category topic | Direct authoritative answer | Cut, color, clarity, or segment names are wrong. |
| 5D | `KnowledgeRetriever.retrieve()` | Knowledge question | Focused queries and retrieved chunks | RAG retrieves the wrong document. |
| 6D | `build_evidence_payload()` / `build_generation_prompt()` | Route, plan, rows, and chunks | Clearly separated evidence and prompt | Good evidence exists but the prompt mixes sources. |
| 7D | `OllamaClient.complete()` | System instruction and prompt | Streamed natural-language answer | Generation is slow, unavailable, or ignores good evidence. |
| 4E | `parse_model_inputs()` | Explicit prediction request | Model fields and missing fields | Model input extraction asks for the wrong information. |
| 5E | `execute_model_plan()` | Requested model families and inputs | Validated prediction package | The wrong saved model runs or a model is skipped. |
| 6E | `ModelEvidenceProvider.predict()` | Model name and raw diamond values | Classification, regression, or cluster result | Loading or inference fails. |
| 8 | `encode_result()` | Python result with DataFrames | JSON-safe result for React | Backend answer is right but frontend receives malformed data. |
| 9 | `streamChat()` in `frontend/src/api.js` | HTTP stream | Token updates and final result | Tokens or the final event do not appear. |
| 10 | `App` in `frontend/src/App.jsx` | Final result | Chat bubble, table, and developer trace | Backend is right but the visible UI is wrong. |

## Seven live intents

| Intent | Executor | Evidence source |
|---|---|---|
| `search` | `service._search()` | Real DataFrame rows |
| `compare` | `service._compare()` | Rows already displayed in this chat |
| `dataset_analysis` | `service._analyze()` | Pandas calculations |
| `diamond_knowledge` | `service._knowledge()` | Fixed vocabulary first, then RAG |
| `model_prediction` | `service._predict()` | Saved model artifacts |
| `small_talk` | `service._small_talk()` | Direct response |
| `out_of_scope` | `service._out_of_scope()` | Direct boundary response |

## Fast debugging commands

```powershell
# Print the eight production checkpoints for one question
python scripts\trace_assistant_question.py 'Find me a diamond around $2500'

# Exercise RAG and print retrieved chunks
python scripts\trace_rag.py

# Run the simplified assistant acceptance tests
.\.venv\Scripts\python.exe -m pytest -q tests\e2e\test_simplified_assistant.py

# Run the full project suite
.\.venv\Scripts\python.exe -m pytest -q
```

## Cleanup completed

- Removed the unused semantic `memory.py` module.
- Removed the unused nearest-neighbor `similarity_search.py` experiment and its test.
- Merged `ui_content.py` into the Streamlit UI.
- Moved seven historical V8.4 modules into `legacy/`.
- Moved the two optional Streamlit modules into `streamlit_app/`.
