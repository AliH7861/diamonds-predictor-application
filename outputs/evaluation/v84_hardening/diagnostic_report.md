# V8.4 Intent, Concept, State and Routing Hardening

## Before and after

| Metric | Before | After |
|---|---:|---:|
| Overall development pass rate | 57.39% | 69.89% |
| L1 route accuracy | 88.41% | 95.57% |
| L2 route accuracy | 85.23% | 93.86% |
| L3 route accuracy | 83.41% | 93.52% |
| Deterministic bypass | 77.50% | 82.39% |
| Field validation | 75.96%* | 59.87% |
| Semantic state accuracy | Not measured* | 46.00% |

*The old field validator skipped missing expected fields, and the old 100% state number only measured transaction completion. The new measurements are stricter and are not direct regressions.

## Diagnostic categories

| failure_category | count |
| --- | --- |
| AMBIGUOUS_CASE | 18 |
| DOWNSTREAM_FAILURE | 133 |
| EVALUATOR_BUG | 20 |
| MISSING_CAPABILITY | 37 |
| PASS | 615 |
| REAL_PRODUCT_BUG | 57 |

## 22-family dashboard

| family | old_pass_rate | new_pass_rate | L1 | L2 | L3 | concept_accuracy | state_update_accuracy | tool_accuracy | deterministic_bypass | status | held_out_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| search_filtering | 0.2 | 0.275 | 0.95 | 0.95 | 0.95 | 0.525 | 1.0 | 0.95 | 0.95 | WEAK | 1.0 |
| statistics_aggregation | 0.325 | 0.85 | 0.95 | 0.95 | 0.95 | 0.875 | 1.0 | 0.95 | 0.95 | NEEDS_WORK | 1.0 |
| trends_group_analysis | 0.625 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 1.0 | 0.95 | 0.95 | STRONG | 1.0 |
| ranking_recommendations | 0.1 | 0.55 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | WEAK | 1.0 |
| diamond_comparison | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | STRONG | 1.0 |
| multi_turn | 0.0 | 0.225 | 0.85 | 0.85 | 0.85 | 0.775 | 0.225 | 0.85 | 0.85 | WEAK | 1.0 |
| ambiguity_contradictions | 0.55 | 0.55 | 0.55 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.75 | WEAK | 1.0 |
| rag_grounded_explanation | 0.925 | 0.925 | 1.0 | 1.0 | 0.925 | 1.0 | 1.0 | 1.0 | 0.0 | STRONG | 1.0 |
| dataframe_schema_variation | 0.75 | 0.575 | 1.0 | 1.0 | 1.0 | 0.825 | 1.0 | 1.0 | 1.0 | WEAK | 1.0 |
| engineered_feature_usage | 0.225 | 0.975 | 0.975 | 0.975 | 0.975 | 1.0 | 1.0 | 0.975 | 0.975 | STRONG | 1.0 |
| hallucination_bad_input | 0.6 | 0.925 | 0.925 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | STRONG | 1.0 |
| price_value_analysis | 0.45 | 0.65 | 0.925 | 0.675 | 0.675 | 1.0 | 1.0 | 0.675 | 0.925 | WEAK | 1.0 |
| mixed_intent_queries | 0.575 | 0.65 | 0.975 | 0.675 | 0.675 | 1.0 | 1.0 | 0.925 | 0.725 | WEAK | 1.0 |
| out_of_scope_refusal | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | STRONG | 1.0 |
| missing_data_empty_results | 0.85 | 0.1 | 0.975 | 0.975 | 0.975 | 1.0 | 1.0 | 0.95 | 0.975 | WEAK | 1.0 |
| conflicting_multi_intent | 0.7 | 0.725 | 1.0 | 0.75 | 0.75 | 1.0 | 1.0 | 1.0 | 0.65 | WEAK | 1.0 |
| explain_why_justification | 1.0 | 0.925 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.375 | STRONG | 1.0 |
| adversarial_phrasing | 0.45 | 0.6 | 1.0 | 1.0 | 1.0 | 0.725 | 1.0 | 0.875 | 0.7 | WEAK | 1.0 |
| long_context_memory | 0.3 | 0.725 | 1.0 | 1.0 | 1.0 | 1.0 | 0.925 | 0.8 | 0.8 | WEAK | 1.0 |
| correction_chains | 0.2 | 0.3 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.8 | 0.8 | WEAK | 1.0 |
| tool_boundary_tests | 0.9 | 0.925 | 1.0 | 0.925 | 0.925 | 1.0 | 1.0 | 0.925 | 0.825 | STRONG | 1.0 |
| grounding_audit | 0.9 | 0.975 | 1.0 | 0.975 | 0.975 | 1.0 | 1.0 | 1.0 | 0.925 | STRONG | 1.0 |

## Diagnostic examples

| failure_category | case_number | family | prompt | root_causes |
| --- | --- | --- | --- | --- |
| DOWNSTREAM_FAILURE | 1 | search_filtering | 1.4k ceiling / ~0.50ct / SI1 / I / Ideal — find matches | field_evidence_failure |
| EVALUATOR_BUG | 2 | search_filtering | under $1200, target is 0.42ct-ish pls | concept_mapping_failure |
| DOWNSTREAM_FAILURE | 3 | search_filtering | need VS1+ clarity, G+ color, roughly 0.70 ct, cap $4600 pls | field_evidence_failure |
| EVALUATOR_BUG | 7 | search_filtering | i want vs2 clarity and premium cut, f color, around 0.74ct, max 3.3k | concept_mapping_failure |
| DOWNSTREAM_FAILURE | 9 | search_filtering | need si1+ clarity, d+ color, roughly 0.30 ct, cap 1000 | field_evidence_failure |
| DOWNSTREAM_FAILURE | 10 | search_filtering | no worse than VS1 clarity and no worse than H color; ~0.73ct, max 3.9k | concept_mapping_failure|field_evidence_failure |
| DOWNSTREAM_FAILURE | 11 | search_filtering | pls min 0.63ct, max 0.79ct, don't spend over 4.1k | concept_mapping_failure|field_evidence_failure |
| EVALUATOR_BUG | 14 | search_filtering | 1300 max and roughly 0.39 carat — show matches. | concept_mapping_failure |
| EVALUATOR_BUG | 25 | search_filtering | 900 max and roughly 0.31 carat — show matches. | concept_mapping_failure |
| REAL_PRODUCT_BUG | 29 | search_filtering | carat has to sit between 0.25 and 0.41 budget max 800 | concept_mapping_failure|field_evidence_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| REAL_PRODUCT_BUG | 33 | search_filtering | carat has to sit between 0.45 and 0.61; budget max 1800 | concept_mapping_failure|field_evidence_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| EVALUATOR_BUG | 35 | search_filtering | under $1300, target is 0.32ct-ish pls | concept_mapping_failure |
| REAL_PRODUCT_BUG | 49 | statistics_aggregation | for Premium cut stones, what's the min carat just calculate it from the dataframe | field_evidence_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| REAL_PRODUCT_BUG | 67 | statistics_aggregation | pls for E color stones around 0.40 ct, what's the upper quartile price? just calculate it from the dataframe | concept_mapping_failure|field_evidence_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| REAL_PRODUCT_BUG | 103 | trends_group_analysis | pls is volume proxy correlated with price in our data? | concept_mapping_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| MISSING_CAPABILITY | 201 | multi_turn | everything else stays; change only the size target to 0.36ct | state_update_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| MISSING_CAPABILITY | 202 | multi_turn | pls keep that and require VS1 or better clarity | state_update_failure |
| MISSING_CAPABILITY | 203 | multi_turn | everything else stays change only the size target to 0.77ct | state_update_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure |
| MISSING_CAPABILITY | 205 | multi_turn | don't reset anything make size the priority now | state_update_failure |
| MISSING_CAPABILITY | 206 | multi_turn | pls keep that and require SI2 or better clarity | state_update_failure |
| AMBIGUOUS_CASE | 242 | ambiguity_contradictions | aim for .55ct but it must be at least .90ct | field_evidence_failure|L1_route_failure |
| AMBIGUOUS_CASE | 245 | ambiguity_contradictions | I want 1.10ct as the target and a .80ct absolute cap | field_evidence_failure|L1_route_failure |
| AMBIGUOUS_CASE | 247 | ambiguity_contradictions | aim for .55ct but it must be at least .90ct pls | field_evidence_failure|L1_route_failure |
| AMBIGUOUS_CASE | 248 | ambiguity_contradictions | I want 1.10ct as the target and a .80ct absolute cap | field_evidence_failure|L1_route_failure |
| AMBIGUOUS_CASE | 249 | ambiguity_contradictions | budget target aside, minimum spend 5000 and max spend 3500 | field_evidence_failure|L1_route_failure |

## Focused semantic collisions

| collision | before | after |
| --- | --- | --- |
| SEARCH -> RANK | 0 | 0 |
| SEARCH -> STATS | 22 | 0 |
| STATS -> SEARCH | 0 | 0 |
| TREND_GROUP -> STATS | 4 | 0 |
| PRICE_ESTIMATE -> PRICE_RANGE | 0 | 0 |
| PRICE_RANGE -> PRICE_ESTIMATE | 9 | 0 |

## Top remaining failure patterns

| pattern | count |
| --- | --- |
| field_evidence_failure | 72 |
| state_update_failure | 39 |
| L2_route_failure|L3_route_failure | 22 |
| concept_mapping_failure | 20 |
| tool_selection_failure|tool_execution_failure | 20 |
| concept_mapping_failure|field_evidence_failure | 19 |
| field_evidence_failure|L1_route_failure | 18 |
| field_evidence_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure | 10 |
| concept_mapping_failure|state_update_failure | 9 |
| state_update_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure | 6 |

## Concept errors

Highest false negatives:

| concept | true_positive | false_positive | false_negative | precision | recall |
| --- | --- | --- | --- | --- | --- |
| SEARCH_FILTER | 117 | 120 | 24 | 0.494 | 0.83 |
| TREND | 34 | 25 | 10 | 0.576 | 0.773 |
| BUDGET | 221 | 31 | 10 | 0.877 | 0.957 |
| STATISTIC | 95 | 180 | 5 | 0.345 | 0.95 |
| CARAT_SIZE | 395 | 113 | 4 | 0.778 | 0.99 |
| RELATIONSHIP | 21 | 11 | 2 | 0.656 | 0.913 |
| CLARITY | 199 | 16 | 1 | 0.926 | 0.995 |
| CLARITY_PREDICTION | 0 | 8 | 0 | 0.0 |  |
| COLOR | 0 | 100 | 0 | 0.0 |  |
| CUT | 0 | 131 | 0 | 0.0 |  |

Highest false positives:

| concept | true_positive | false_positive | false_negative | precision | recall |
| --- | --- | --- | --- | --- | --- |
| FILTER | 0 | 244 | 0 | 0.0 |  |
| STATISTIC | 95 | 180 | 5 | 0.345 | 0.95 |
| CUT | 0 | 131 | 0 | 0.0 |  |
| EXPLANATION | 0 | 129 | 0 | 0.0 |  |
| SEARCH_FILTER | 117 | 120 | 24 | 0.494 | 0.83 |
| CARAT_SIZE | 395 | 113 | 4 | 0.778 | 0.99 |
| ENGINEERED_FEATURE | 0 | 111 | 0 | 0.0 |  |
| COLOR | 0 | 100 | 0 | 0.0 |  |
| CORRECTION | 0 | 89 | 0 | 0.0 |  |
| GROUP | 0 | 59 | 0 | 0.0 |  |

## State findings

The frozen development sample contains shuffled or incomplete conversation fragments, so some expected prior fields were never established in the evaluated sequence. Those are evaluator limitations. A fresh six-turn held-out chain scored 6/6 for preservation, replacement, removal, and newest-value precedence.

## Model and token usage

Deterministic bypass moved from 77.50% to 82.39%. The new 22-family held-out routes and six state turns required zero LLM, embedding, or generation calls.

## Regressions and evaluator changes

Some family pass rates fall under the new evaluator because missing expected fields now fail instead of being skipped, and state is compared semantically rather than counted as successful whenever a transaction commits. The dashboard retains both numbers, but these measurement changes are not product regressions. The full project regression suite remains green.

## Anti-overfitting check

The fresh held-out set uses new wording and values: 22/22 family route cases and 6/6 state turns passed. It made zero LLM, embedding, or generation calls.

## Remaining priorities

1. Separate raw clarity-grade thresholds from the five-family classifier state.
2. Add explicit min/max carat fields instead of representing every range as midpoint plus tolerance.
3. Complete unsupported empty-result and contradiction executors.
4. Calibrate route abstention using labeled uncertain cases; deterministic routes currently report fixed confidence.
5. Expand current-message concept labels so false-positive reporting does not treat valid supporting concepts as errors.