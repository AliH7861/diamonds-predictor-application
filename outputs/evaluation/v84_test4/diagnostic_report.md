# V8.4 Test 4 Diagnostic Report

## Acceptance result

- Final representative integration matrix: **11/11 passed**.
- Initial matrix before the conjunction-routing fix: **7/11 passed**.
- Full repository verification: **94 passed**, 52 existing Keras/NumPy warnings.
- Ruff: passed.
- React production build: passed.

## Model reliability

| Model | Held-out result |
|---|---:|
| Clarity default ordinal XGBoost | Accuracy 56.49%; Macro F1 47.67% |
| Clarity ANN | Accuracy 51.13%; Macro F1 27.34% |
| Clarity Random Forest | Accuracy 56.31%; Macro F1 45.68% |
| Price primary ANN | MAE $271.15; R2 0.9798 |
| Price benchmark winner | MAE $245.44; R2 0.9834 |
| KMeans clustering | Execution/profile mapping verified; supervised accuracy is not applicable |

## Measured latency

| Use case | P50 ms | P95 ms | Status |
|---|---:|---:|---|
| Pandas only | 28.3 | 48.4 | Working well |
| Regression | 245.7 | 327.9 | Working well |
| Classification | 133.0 | 137.8 | Working well |
| Clustering | 366.4 | 531.3 | Working well |
| RAG only | 8,737.8 | 8,737.8 | Working, slow |
| Pandas + regression | 165.6 | 236.4 | Working well |
| Classification + regression | 148.9 | 182.5 | Working well |
| Regression + clustering | 394.5 | 448.9 | Working well |
| Pandas + RAG | 8,972.3 | 8,972.3 | Working, slow |
| All three models | 428.5 | 451.1 | Working well |
| All models + explanation | 25,539.9 | 25,539.9 | Working, very slow |

Cold startup was **4,580.9 ms** in the final run. One real streamed HTTP RAG request measured **779.7 ms to first token**, **10,046.7 ms total**, and 280 token events.

## Frontend acceptance

Verified through the actual React -> Vite proxy -> HTTP streaming API -> assistant path:

- message submission and response rendering;
- dataset cards;
- stateful follow-up preserving budget and cut while changing carat;
- conversation persistence after reload;
- request IDs;
- optional developer panel;
- displayed route, concepts, criteria, state delta, tool plan, token counts, cache state and total latency;
- deterministic request showed zero LLM and embedding calls.

Responsive mobile/tablet behavior was **UNTESTED** in this run. Explicit live progress-stage messages are **not implemented**; the UI shows a typing indicator until events arrive. Backend errors are converted into a readable assistant message.

## Diagnostics and limitations

Structured request logging now records a request ID and the diagnostic package without logging the full user prompt. Expensive resources are constructed once by `create_assistant()` and reused by the server. The displayed cache values reflect this architecture; per-resource hit/miss counters are not yet instrumented.

Total latency, API handler latency, time to first token and model inference latency are measured. Separate normalization, concept, state, validation, L1, L2, L3, embedding, retrieval and generation timers remain **UNTESTED/UNMEASURED** and are explicitly listed in each diagnostic response.

## Root cause fixed

Conjoined model requests were previously collapsed into one model or RAG route. General request-clause decomposition now recognizes clarity, price and segment targets independently. Multi-model explanation requests execute classification, regression and clustering first, then retrieve supporting RAG context and generate only from that evidence.

## Next fixes

1. Add precise timers around every control-plane and RAG stage.
2. Emit real backend progress events and render them in React.
3. Add cache hit/miss counters rather than static reuse status.
4. Add automated responsive viewport and browser error-state tests.
5. Reduce RAG and multi-tool explanation latency.
6. Add factual scoring for generated explanations.
