# V8.4 Remaining-Failure Diagnostic Report

Current frozen result: **615/880 (69.89%)**, leaving **265 failures**. The official 1,200-case benchmark was not run or modified.

## Family health, worst to best

| Rank | Family | Tests | Passed | Failed | Pass rate | L1 | L2 | L3 | Concept | Field validation | Semantic state | Tool selection | Tool execution | Grounding | Deterministic | Held-out | Primary failure stage | Main reason | Severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | missing_data_empty_results | 40 | 4 | 36 | 10.00% | 97.50% | 97.50% | 97.50% | 100.00% | 0.00% | N/A | 95.00% | 85.00% | 87.50% | 97.50% | PASS | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | CRITICAL |
| 2 | multi_turn | 40 | 9 | 31 | 22.50% | 85.00% | 85.00% | 85.00% | 77.50% | 100.00% | 22.50% | 85.00% | 85.00% | 100.00% | 85.00% | PASS | state | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | CRITICAL |
| 3 | search_filtering | 40 | 11 | 29 | 27.50% | 95.00% | 95.00% | 95.00% | 52.50% | 40.00% | N/A | 95.00% | 95.00% | 100.00% | 95.00% | PASS | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | CRITICAL |
| 4 | correction_chains | 40 | 12 | 28 | 30.00% | 100.00% | 100.00% | 100.00% | 100.00% | N/A | 50.00% | 80.00% | 80.00% | 100.00% | 80.00% | PASS | state | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | CRITICAL |
| 5 | ranking_recommendations | 40 | 22 | 18 | 55.00% | 100.00% | 100.00% | 100.00% | 100.00% | 55.00% | N/A | 100.00% | 100.00% | 100.00% | 100.00% | PASS | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | WEAK |
| 6 | ambiguity_contradictions | 40 | 22 | 18 | 55.00% | 55.00% | 100.00% | 100.00% | 100.00% | 0.00% | N/A | 100.00% | 100.00% | 100.00% | 75.00% | PASS | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | WEAK |
| 7 | dataframe_schema_variation | 40 | 23 | 17 | 57.50% | 100.00% | 100.00% | 100.00% | 82.50% | 57.50% | N/A | 100.00% | 100.00% | 100.00% | 100.00% | PASS | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | WEAK |
| 8 | adversarial_phrasing | 40 | 24 | 16 | 60.00% | 100.00% | 100.00% | 100.00% | 72.50% | N/A | N/A | 87.50% | 95.00% | 100.00% | 70.00% | PASS | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | WEAK |
| 9 | price_value_analysis | 40 | 26 | 14 | 65.00% | 92.50% | 67.50% | 67.50% | 100.00% | 65.00% | N/A | 67.50% | 67.50% | 100.00% | 92.50% | PASS | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | WEAK |
| 10 | mixed_intent_queries | 40 | 26 | 14 | 65.00% | 97.50% | 67.50% | 67.50% | 100.00% | 100.00% | N/A | 92.50% | 92.50% | 97.50% | 72.50% | PASS | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | WEAK |
| 11 | conflicting_multi_intent | 40 | 29 | 11 | 72.50% | 100.00% | 75.00% | 75.00% | 100.00% | 100.00% | N/A | 100.00% | 97.50% | 97.50% | 65.00% | PASS | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | WEAK |
| 12 | long_context_memory | 40 | 29 | 11 | 72.50% | 100.00% | 100.00% | 100.00% | 100.00% | N/A | 92.50% | 80.00% | 80.00% | 100.00% | 80.00% | PASS | tool_selection | The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks. | WEAK |
| 13 | statistics_aggregation | 40 | 34 | 6 | 85.00% | 95.00% | 95.00% | 95.00% | 87.50% | 95.00% | N/A | 95.00% | 95.00% | 100.00% | 95.00% | PASS | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | NEEDS_WORK |
| 14 | rag_grounded_explanation | 40 | 37 | 3 | 92.50% | 100.00% | 100.00% | 92.50% | 100.00% | N/A | N/A | 100.00% | 100.00% | 100.00% | 0.00% | PASS | L3_routing | The correct broad operation is selected, but its specialized subtype is wrong. | NEEDS_WORK |
| 15 | hallucination_bad_input | 40 | 37 | 3 | 92.50% | 92.50% | 100.00% | 100.00% | 100.00% | N/A | N/A | 100.00% | 100.00% | 100.00% | 100.00% | PASS | L1_routing | The request falls outside the supported domain or clarification rules before its intended data operation is selected. | NEEDS_WORK |
| 16 | explain_why_justification | 40 | 37 | 3 | 92.50% | 100.00% | 100.00% | 100.00% | 100.00% | 80.00% | N/A | 100.00% | 100.00% | 100.00% | 37.50% | PASS | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | NEEDS_WORK |
| 17 | tool_boundary_tests | 40 | 37 | 3 | 92.50% | 100.00% | 92.50% | 92.50% | 100.00% | N/A | N/A | 92.50% | 92.50% | 100.00% | 82.50% | PASS | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | NEEDS_WORK |
| 18 | trends_group_analysis | 40 | 38 | 2 | 95.00% | 95.00% | 95.00% | 95.00% | 95.00% | N/A | N/A | 95.00% | 95.00% | 100.00% | 95.00% | PASS | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | STRONG |
| 19 | engineered_feature_usage | 40 | 39 | 1 | 97.50% | 97.50% | 97.50% | 97.50% | 100.00% | N/A | N/A | 97.50% | 97.50% | 100.00% | 97.50% | PASS | L1_routing | The request falls outside the supported domain or clarification rules before its intended data operation is selected. | STRONG |
| 20 | grounding_audit | 40 | 39 | 1 | 97.50% | 100.00% | 97.50% | 97.50% | 100.00% | 100.00% | N/A | 100.00% | 100.00% | 100.00% | 92.50% | PASS | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | STRONG |
| 21 | diamond_comparison | 40 | 40 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | 100.00% | 100.00% | 100.00% | 100.00% | PASS | none | No dominant remaining failure. | STRONG |
| 22 | out_of_scope_refusal | 40 | 40 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | 100.00% | 100.00% | 100.00% | 100.00% | PASS | none | No dominant remaining failure. | STRONG |

## Weak-family explanations

| Family | What works | What fails | Where it fails | Why it fails | Example failure | Recommended fix |
| --- | --- | --- | --- | --- | --- | --- |
| missing_data_empty_results | L1/L2/L3 = 97.50%/97.50%/97.50%; held-out PASS | 36 of 40 cases fail | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Find diamonds at least 5.0 ct under $100. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| multi_turn | L1/L2/L3 = 85.00%/85.00%/85.00%; held-out PASS | 31 of 40 cases fail | state | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | everything else stays; change only the size target to 0.36ct | Evaluate complete ordered conversations and add explicit threshold/removal state fields. |
| search_filtering | L1/L2/L3 = 95.00%/95.00%/95.00%; held-out PASS | 29 of 40 cases fail | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | 1.4k ceiling / ~0.50ct / SI1 / I / Ideal — find matches | Define current-message concept expectations separately from route-supporting concepts. |
| correction_chains | L1/L2/L3 = 100.00%/100.00%/100.00%; held-out PASS | 28 of 40 cases fail | state | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | Find around 0.60 ct under $3,000. | Evaluate complete ordered conversations and add explicit threshold/removal state fields. |
| ranking_recommendations | L1/L2/L3 = 100.00%/100.00%/100.00%; held-out PASS | 18 of 40 cases fail | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | under 700 and about 0.31 ct — sort the matches for me; best bang for buck pls | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| ambiguity_contradictions | L1/L2/L3 = 55.00%/100.00%/100.00%; held-out PASS | 18 of 40 cases fail | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | aim for .55ct but it must be at least .90ct | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| dataframe_schema_variation | L1/L2/L3 = 100.00%/100.00%/100.00%; held-out PASS | 17 of 40 cases fail | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | using whatever columns exist here, show stones about 1.54 carat below 14.7 grand | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| adversarial_phrasing | L1/L2/L3 = 100.00%/100.00%/100.00%; held-out PASS | 16 of 40 cases fail | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | medain prce by clrity pls, grp it dont give 1 overall num | Define current-message concept expectations separately from route-supporting concepts. |
| price_value_analysis | L1/L2/L3 = 92.50%/67.50%/67.50%; held-out PASS | 14 of 40 cases fail | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | estimate the typical dataset price for 0.70ct VVS2 Good | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| mixed_intent_queries | L1/L2/L3 = 97.50%/67.50%/67.50%; held-out PASS | 14 of 40 cases fail | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | What price range do SI1 diamonds around 0.50 ct have; and explain price per carat too. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| conflicting_multi_intent | L1/L2/L3 = 100.00%/75.00%/75.00%; held-out PASS | 11 of 40 cases fail | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Compare diamond 15570 with diamond 42875; then tell me the median price of diamonds in the whole dataset. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| long_context_memory | L1/L2/L3 = 100.00%/100.00%/100.00%; held-out PASS | 11 of 40 cases fail | tool_selection | The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks. | Find around 0.70 ct under $3,500. | Create explicit tool plans for supported mixed tasks and report unsupported combinations early. |
| statistics_aggregation | L1/L2/L3 = 95.00%/95.00%/95.00%; held-out PASS | 6 of 40 cases fail | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | for Premium cut stones, what's the min carat just calculate it from the dataframe | Define current-message concept expectations separately from route-supporting concepts. |
| rag_grounded_explanation | L1/L2/L3 = 100.00%/100.00%/92.50%; held-out PASS | 3 of 40 cases fail | L3_routing | The correct broad operation is selected, but its specialized subtype is wrong. | Why do we have carat squared as an engineered feature | Use operation evidence and negative evidence for the remaining specialized collisions. |
| hallucination_bad_input | L1/L2/L3 = 92.50%/100.00%/100.00%; held-out PASS | 3 of 40 cases fail | L1_routing | The request falls outside the supported domain or clarification rules before its intended data operation is selected. | show stones with negative 1.2ct size pls | Add calibrated abstention for genuinely unresolved operations without broad keyword changes. |
| explain_why_justification | L1/L2/L3 = 100.00%/100.00%/100.00%; held-out PASS | 3 of 40 cases fail | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Recommend the best options around 0.71 ct under $3,700; prioritize cleanliness. Why is the first option there? | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| tool_boundary_tests | L1/L2/L3 = 100.00%/92.50%/92.50%; held-out PASS | 3 of 40 cases fail | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Estimate the dataset price for a 1.0 ct VS2 diamond. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |

## Primary failure stages

| Failure stage | Cases | % of all failures | Most affected families | Example | Likely root cause |
| --- | --- | --- | --- | --- | --- |
| normalization | 0 | 0.00% | — | — | The message could not be normalized or processed. |
| concept_mapping | 53 | 20.00% | search_filtering, adversarial_phrasing, multi_turn | under $1200, target is 0.42ct-ish pls | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. |
| field_evidence | 104 | 39.25% | missing_data_empty_results, ranking_recommendations, ambiguity_contradictions | 1.4k ceiling / ~0.50ct / SI1 / I / Ideal — find matches | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. |
| state | 45 | 16.98% | multi_turn, correction_chains, long_context_memory | everything else stays; change only the size target to 0.36ct | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| clarification | 0 | 0.00% | — | — | Not a primary failure in the current trace. |
| L1_routing | 6 | 2.26% | hallucination_bad_input, engineered_feature_usage, mixed_intent_queries | pls gimme the max carat squared from the actual rows | The request falls outside the supported domain or clarification rules before its intended data operation is selected. |
| L2_routing | 26 | 9.81% | mixed_intent_queries, conflicting_multi_intent, tool_boundary_tests | What price range do SI1 diamonds around 0.50 ct have; and explain price per carat too. | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L3_routing | 3 | 1.13% | rag_grounded_explanation | Why do we have carat squared as an engineered feature | The correct broad operation is selected, but its specialized subtype is wrong. |
| tool_selection | 23 | 8.68% | long_context_memory, correction_chains, adversarial_phrasing | Classify clarity and compare with observed diamonds: 1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96. | The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks. |
| tool_execution | 5 | 1.89% | missing_data_empty_results, conflicting_multi_intent | Predict price for a 1 carat diamond. | The tool was selected but lacked validated inputs or a corresponding executor. |
| evidence_selection | 0 | 0.00% | — | — | Not a primary failure in the current trace. |
| grounding | 0 | 0.00% | — | — | The intended source was selected but did not return usable evidence. |
| answer_evaluator | 0 | 0.00% | — | — | Not a primary failure in the current trace. |
| missing_capability | 0 | 0.00% | — | — | Not a primary failure in the current trace. |

Primary-stage total: **265**, which reconciles to all **265** failed cases.

## Root causes

| Root cause | Failures affected | Families | Why it happens | General fix | Regression risk |
| --- | --- | --- | --- | --- | --- |
| numeric/category attachment | 104 | search, ranking, schema, price | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. | medium |
| concept ontology mismatch | 53 | search, schema, adversarial | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | Define current-message concept expectations separately from route-supporting concepts. | low |
| state sequence/representation | 45 | multi_turn, long_context, corrections | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | Evaluate complete ordered conversations and add explicit threshold/removal state fields. | medium |
| route precedence | 35 | price, mixed, search, statistics | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. | medium |
| missing or incomplete executor | 28 | missing-data, mixed, corrections | The tool was selected but lacked validated inputs or a corresponding executor. | Implement missing empty-result, contradiction, and mixed-task executors. | medium |
| evaluator expectation mismatch | 20 | search, state, concepts | The evaluator requires implicit concepts or state that the current message never established. | Correct labels and evaluate complete ordered conversations. | low |
| legitimate ambiguity | 18 | ambiguity_contradictions | More than one user interpretation is valid. | Clarify instead of forcing a route. | low |
| grounding/evidence absence | 7 | mixed, empty results | The intended source was selected but did not return usable evidence. | Add source fallback and evidence-presence assertions. | low |

## Remaining L2/L3 confusions

| Level | Expected route | Actual wrong route | Count | Example wording | Why confused |
| --- | --- | --- | --- | --- | --- |
| L2 | PRICE | MODEL_ANALYSIS | 13 | price estimate only: 0.35ct / f / vs1 / premium | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | TOOL_PLAN | ANALYZE | 11 | Median price by clarity and explain what clarity means. | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | TOOL_PLAN | PRICE | 10 | What price range do SI1 diamonds around 0.50 ct have; and explain price per carat too. | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | RETRIEVE | EXPLAIN | 8 | carat has to sit between 0.25 and 0.41 budget max 800 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | ANALYZE | EXPLAIN | 5 | for Premium cut stones, what's the min carat just calculate it from the dataframe | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | PRICE | EXPLAIN | 3 | about what would 1.23 carat, VS2, I color, Very Good cut cost based on this data | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | MODEL_ANALYSIS | TOOL_PLAN | 2 | Predict price and compare it with actual dataset values for 1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96. | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | MODEL_ANALYSIS | EXPLAIN | 1 | Which segment fits if price is unknown? | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L2 | TOOL_PLAN | EXPLAIN | 1 | Compare price by cut, then explain the strongest pattern. | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| L3 | PRICE_ESTIMATE | PRICE_PREDICTION | 13 | price estimate only: 0.35ct / f / vs1 / premium | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | MULTI_TASK | STATS | 10 | Compare diamond 15570 with diamond 42875; then tell me the median price of diamonds in the whole dataset. | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | MULTI_TASK | PRICE_RANGE | 10 | What price range do SI1 diamonds around 0.50 ct have; and explain price per carat too. | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | SEARCH | RAG_EXPLANATION | 8 | carat has to sit between 0.25 and 0.41 budget max 800 | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | STATS | RAG_EXPLANATION | 3 | for Premium cut stones, what's the min carat just calculate it from the dataframe | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | PRICE_ESTIMATE | RAG_EXPLANATION | 3 | about what would 1.23 carat, VS2, I color, Very Good cut cost based on this data | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | RAG_EXPLANATION | DOMAIN_EXPLANATION | 3 | Why do we have carat squared as an engineered feature | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | PRICE_PREDICTION | MULTI_TASK | 2 | Predict price and compare it with actual dataset values for 1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96. | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | TREND_RELATION | RAG_EXPLANATION | 2 | pls is volume proxy correlated with price in our data? | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | MULTI_TASK | DOMAIN_EXPLANATION | 1 | Compare price by cut, then explain the strongest pattern. | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | CLUSTER_PREDICTION | RAG_EXPLANATION | 1 | Which segment fits if price is unknown? | The correct broad operation is selected, but its specialized subtype is wrong. |
| L3 | MULTI_TASK | TREND_GROUP | 1 | Median price by clarity and explain what clarity means. | The correct broad operation is selected, but its specialized subtype is wrong. |

## Concept false positives and negatives

| Direction | Concept | Count | Most common wording | Why |
| --- | --- | --- | --- | --- |
| false positive | FILTER | 244 | under $1200, target is 0.42ct-ish pls | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | STATISTIC | 180 | i want vs2 clarity and premium cut, f color, around 0.74ct, max 3.3k | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | CUT | 131 | 1.4k ceiling / ~0.50ct / SI1 / I / Ideal — find matches | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | EXPLANATION | 129 | Using our dataset context, explain how clarity groups relate to price; I want explanation, not a table. | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | SEARCH_FILTER | 120 | pls need a grouped view: mean price per carat, grouped by color | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | CARAT_SIZE | 113 | Diamond 22021 vs 6815 compare them mainly for size. | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | ENGINEERED_FEATURE | 111 | pls for D color stones around 0.35 ct, what's the median price per carat? just calculate it from the dataframe | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | COLOR | 100 | need VS1+ clarity, G+ color, roughly 0.70 ct, cap $4600 pls | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | CORRECTION | 89 | under $1,600 and about 0.50 ct — sort the matches for me; keep it balanced overall pls | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | GROUP | 59 | compare carat across clarity groups using the median | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | COMPARE | 47 | compare carat across clarity groups using the median | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false positive | DIMENSIONS | 44 | i want the numeric relationship for depth and price, not a definition | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | SEARCH_FILTER | 24 | under $1200, target is 0.42ct-ish pls | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | TREND | 10 | medain prce by clrity pls, grp it dont give 1 overall num | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | BUDGET | 10 | i want vs2 clarity and premium cut, f color, around 0.74ct, max 3.3k | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | STATISTIC | 5 | for good cut stones, what's the lowest price? just calculate it from the dataframe | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | CARAT_SIZE | 4 | crt has to sit between 0.51 and 0.67; budget max $3,100 | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | RELATIONSHIP | 2 | pls is volume proxy correlated with price in our data? | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |
| false negative | CLARITY | 1 | is VS versus VVS actually a thing? | The detector and expected-label ontology use different explicitness rules or lack a phrase alias. |

## Semantic state summary

| Metric | Count |
| --- | --- |
| field preservation/change correctness | 301 |
| correct removal | 0 |
| incorrect change | 5 |
| lost/missing field | 88 |
| invented field | 17 |
| unsupported threshold capability | 106 |

## State failure cases

| Case | Family | Previous state | User message | Expected delta | Actual delta | Incorrectly changed fields | Lost fields | Stale fields | Root cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | multi_turn | {} | everything else stays; change only the size target to 0.36ct | {"carat_target": 0.36, "budget_max": 1100.0, "priority": "size", "clarity": "VS2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 201 | multi_turn | {} | pls keep that and require VS1 or better clarity | {"carat_target": 0.32, "budget_max": 1400.0, "priority": "value", "clarity": "VS1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 202 | multi_turn | {} | everything else stays change only the size target to 0.77ct | {"carat_target": 0.77, "budget_max": 2800.0, "priority": "size", "clarity": "SI2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 203 | multi_turn | {} | start me near 0.41ct under $1,200; value first | {"carat_target": 0.41, "budget_max": 1200.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 1200.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.41, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 204 | multi_turn | {} | don't reset anything make size the priority now | {"carat_target": 0.32, "budget_max": 1400.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 205 | multi_turn | {} | pls keep that and require SI2 or better clarity | {"carat_target": 0.56, "budget_max": 2300.0, "priority": "value", "clarity": "SI2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 206 | multi_turn | {} | start me near 0.31ct under $500; value first | {"carat_target": 0.31, "budget_max": 500.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 500.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.31, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 207 | multi_turn | {} | everything else stays; change only the size target to 0.37ct pls | {"carat_target": 0.37, "budget_max": 1800.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 208 | multi_turn | {} | don't reset anything; make size the priority now pls | {"carat_target": 0.41, "budget_max": 1900.0, "priority": "size", "clarity": "VVS2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 209 | multi_turn | {} | everything else stays change only the size target to 0.46ct | {"carat_target": 0.46, "budget_max": 1900.0, "priority": "size", "clarity": "VVS2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 210 | multi_turn | {} | start me near 0.72ct under $4,500; value first | {"carat_target": 0.72, "budget_max": 4500.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 4500.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.72, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 211 | multi_turn | {} | pls start me near 0.72ct under $2,200; value first | {"carat_target": 0.72, "budget_max": 2200.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 2200.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.72, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 212 | multi_turn | {} | pls keep that and require VVS2 or better clarity | {"carat_target": 0.41, "budget_max": 1900.0, "priority": "value", "clarity": "VVS2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 213 | multi_turn | {} | don't reset anything; make size the priority now | {"carat_target": 0.41, "budget_max": 1500.0, "priority": "size", "clarity": "SI1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 214 | multi_turn | {} | keep that and require VVS2 or better clarity | {"carat_target": 0.4, "budget_max": 1600.0, "priority": "value", "clarity": "VVS2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 215 | multi_turn | {} | don't reset anything make size the priority now | {"carat_target": 0.4, "budget_max": 1600.0, "priority": "size", "clarity": "VVS2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 216 | multi_turn | {} | start me near 0.40ct under $1,000 value first | {"carat_target": 0.4, "budget_max": 1000.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 1000.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.4, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 217 | multi_turn | {} | start me near 1.02ct under $5,300; value first | {"carat_target": 1.02, "budget_max": 5300.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 5300.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 1.02, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 220 | multi_turn | {} | everything else stays; change only the size target to 0.37ct | {"carat_target": 0.37, "budget_max": 1400.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 221 | multi_turn | {"max_price": 500.0, "target_carat": 0.31, "carat_tolerance": 0.15, "priorities": ["value"]} | keep that and require VS2 or better clarity pls | {"carat_target": 0.31, "budget_max": 1100.0, "priority": "value", "clarity": "VS2", "clarity_or_better": true} | [{"field": "clarity", "operation": "SET", "old_value": null, "new_value": "VS", "evidence": "explicit clarity family"}] | budget_max |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 222 | multi_turn | {} | start me near 0.32ct under $800; value first | {"carat_target": 0.32, "budget_max": 800.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 800.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.32, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 225 | multi_turn | {} | pls start me near 0.41ct under $900; value first | {"carat_target": 0.41, "budget_max": 900.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 900.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.41, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 226 | multi_turn | {} | pls don't reset anything; make size the priority now | {"carat_target": 0.41, "budget_max": 1800.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 227 | multi_turn | {} | don't reset anything make size the priority now | {"carat_target": 0.56, "budget_max": 2300.0, "priority": "size", "clarity": "SI2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 228 | multi_turn | {"max_price": 1200.0, "target_carat": 0.41, "carat_tolerance": 0.15, "priorities": ["value"]} | everything else stays change only the size target to 0.37ct | {"carat_target": 0.37, "budget_max": 1800.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true} | [{"field": "target_carat", "operation": "REPLACE", "old_value": 0.41, "new_value": 0.37, "evidence": "explicit carat expression"}] | budget_max, priority | clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 230 | multi_turn | {} | keep that and require VS1 or better clarity pls | {"carat_target": 0.41, "budget_max": 1800.0, "priority": "value", "clarity": "VS1", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 231 | multi_turn | {} | don't reset anything make size the priority now | {"carat_target": 1.07, "budget_max": 5000.0, "priority": "size", "clarity": "SI2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 232 | multi_turn | {"max_price": 900.0, "target_carat": 0.41, "carat_tolerance": 0.15, "priorities": ["value"]} | keep that and require si1 or better clarity | {"carat_target": 0.41, "budget_max": 1500.0, "priority": "value", "clarity": "SI1", "clarity_or_better": true} | [{"field": "clarity", "operation": "SET", "old_value": null, "new_value": "SI", "evidence": "explicit clarity family"}] | budget_max |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 234 | multi_turn | {} | keep that and require SI2 or better clarity | {"carat_target": 1.07, "budget_max": 5000.0, "priority": "value", "clarity": "SI2", "clarity_or_better": true} | [] |  | carat_target, budget_max, priority, clarity |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| 236 | multi_turn | {} | start me near 1.07ct under $4,400; value first pls | {"carat_target": 1.07, "budget_max": 4400.0, "priority": "value"} | [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 4400.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 1.07, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}] |  |  |  | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |

The frozen state sample contains shuffled or incomplete conversation fragments. The separate ordered held-out chain remains 6/6; evaluator-sequence failures must not be patched into production routing.

## Product versus evaluator

| failure_category | Count | % failures | Example | Action |
| --- | --- | --- | --- | --- |
| AMBIGUOUS_CASE | 18 | 6.79% | aim for .55ct but it must be at least .90ct | Clarify or abstain. |
| DOWNSTREAM_FAILURE | 133 | 50.19% | 1.4k ceiling / ~0.50ct / SI1 / I / Ideal — find matches | Fix field/state/tool execution after the correct route. |
| EVALUATOR_BUG | 20 | 7.55% | under $1200, target is 0.42ct-ish pls | Correct the label or test sequence; do not change production behavior. |
| MISSING_CAPABILITY | 37 | 13.96% | everything else stays; change only the size target to 0.36ct | Implement explicitly or report unsupported behavior. |
| REAL_PRODUCT_BUG | 57 | 21.51% | carat has to sit between 0.25 and 0.41 budget max 800 | Fix the general production rule and add held-out counterexamples. |

## Ten largest failure patterns

### 1. `field_evidence_failure` — 72 cases

**USER:** 1.4k ceiling / ~0.50ct / SI1 / I / Ideal — find matches

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|CLARITY|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|CLARITY|CUT|SEARCH_FILTER`; state `{"target_carat": 0.5, "carat_tolerance": 0.15, "cut": "Ideal", "color": "I", "clarity": "SI"}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

**USER:** need VS1+ clarity, G+ color, roughly 0.70 ct, cap $4600 pls

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|CLARITY|PRICE|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|CLARITY|COLOR|PRICE|SEARCH_FILTER`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

**USER:** need si1+ clarity, d+ color, roughly 0.30 ct, cap 1000

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|CLARITY|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|CLARITY|COLOR|SEARCH_FILTER`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

### 2. `state_update_failure` — 39 cases

**USER:** pls keep that and require VS1 or better clarity

**EXPECTED:** concepts `CLARITY`; state `{"applicable": true, "expected": {"carat_target": 0.32, "budget_max": 1400.0, "priority": "value", "clarity": "VS1", "clarity_or_better": true}, "actual": {}, "fields": {"carat_target": "INCORRECTLY_REMOVED_OR_MISSING", "budget_max": "INCORRECTLY_REMOVED_OR_MISSING", "priority": "INCORRECTLY_REMOVED_OR_MISSING", "clarity": "INCORRECTLY_REMOVED_OR_MISSING", "clarity_or_better": "MISSING_CAPABILITY"}, "accepted_delta": []}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CLARITY|CORRECTION|FILTER`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `state_update_failure`
**ROOT CAUSE:** The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.
**FIX:** Evaluate complete ordered conversations and add explicit threshold/removal state fields.

**USER:** don't reset anything make size the priority now

**EXPECTED:** concepts `nan`; state `{"applicable": true, "expected": {"carat_target": 0.32, "budget_max": 1400.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true}, "actual": {}, "fields": {"carat_target": "INCORRECTLY_REMOVED_OR_MISSING", "budget_max": "INCORRECTLY_REMOVED_OR_MISSING", "priority": "INCORRECTLY_REMOVED_OR_MISSING", "clarity": "INCORRECTLY_REMOVED_OR_MISSING", "clarity_or_better": "MISSING_CAPABILITY"}, "accepted_delta": []}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `state_update_failure`
**ROOT CAUSE:** The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.
**FIX:** Evaluate complete ordered conversations and add explicit threshold/removal state fields.

**USER:** pls keep that and require SI2 or better clarity

**EXPECTED:** concepts `CLARITY`; state `{"applicable": true, "expected": {"carat_target": 0.56, "budget_max": 2300.0, "priority": "value", "clarity": "SI2", "clarity_or_better": true}, "actual": {}, "fields": {"carat_target": "INCORRECTLY_REMOVED_OR_MISSING", "budget_max": "INCORRECTLY_REMOVED_OR_MISSING", "priority": "INCORRECTLY_REMOVED_OR_MISSING", "clarity": "INCORRECTLY_REMOVED_OR_MISSING", "clarity_or_better": "MISSING_CAPABILITY"}, "accepted_delta": []}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CLARITY|CORRECTION|FILTER`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `state_update_failure`
**ROOT CAUSE:** The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.
**FIX:** Evaluate complete ordered conversations and add explicit threshold/removal state fields.

### 3. `L2_route_failure|L3_route_failure` — 22 cases

**USER:** What price range do SI1 diamonds around 0.50 ct have; and explain price per carat too.

**EXPECTED:** concepts `CARAT_SIZE|CLARITY|PRICE`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/TOOL_PLAN/MULTI_TASK`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|ENGINEERED_FEATURE|EXPLANATION|PRICE|STATISTIC`; state `{"target_carat": 0.5, "carat_tolerance": 0.15, "clarity": "SI"}`; route `DATA_TASK/PRICE/PRICE_RANGE`; tools `dataframe`.

**FIRST WRONG STAGE:** `L2_route_failure`
**ROOT CAUSE:** Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan.
**FIX:** Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution.

**USER:** What price range do VS2 diamonds around 0.70 ct have; and explain price per carat too.

**EXPECTED:** concepts `CARAT_SIZE|CLARITY|PRICE`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/TOOL_PLAN/MULTI_TASK`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|ENGINEERED_FEATURE|EXPLANATION|PRICE|STATISTIC`; state `{"target_carat": 0.7, "carat_tolerance": 0.15, "clarity": "VS"}`; route `DATA_TASK/PRICE/PRICE_RANGE`; tools `dataframe`.

**FIRST WRONG STAGE:** `L2_route_failure`
**ROOT CAUSE:** Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan.
**FIX:** Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution.

**USER:** What price range do VS1 diamonds around 0.90 ct have; and explain price per carat too.

**EXPECTED:** concepts `CARAT_SIZE|CLARITY|PRICE`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/TOOL_PLAN/MULTI_TASK`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|ENGINEERED_FEATURE|EXPLANATION|PRICE|STATISTIC`; state `{"target_carat": 0.9, "carat_tolerance": 0.15, "clarity": "VS"}`; route `DATA_TASK/PRICE/PRICE_RANGE`; tools `dataframe`.

**FIRST WRONG STAGE:** `L2_route_failure`
**ROOT CAUSE:** Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan.
**FIX:** Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution.

### 4. `concept_mapping_failure` — 20 cases

**USER:** under $1200, target is 0.42ct-ish pls

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|PRICE|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|FILTER|PRICE`; state `{"max_price": 1200.0, "target_carat": 0.42, "carat_tolerance": 0.15}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

**USER:** i want vs2 clarity and premium cut, f color, around 0.74ct, max 3.3k

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|CLARITY|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|COLOR|CUT|SEARCH_FILTER|STATISTIC`; state `{"max_price": 3300.0, "target_carat": 0.74, "carat_tolerance": 0.15, "cut": "Premium", "color": "F", "clarity": "VS"}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

**USER:** 1300 max and roughly 0.39 carat — show matches.

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|SEARCH_FILTER|STATISTIC`; state `{"max_price": 1300.0, "target_carat": 0.39, "carat_tolerance": 0.15}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

### 5. `tool_selection_failure|tool_execution_failure` — 20 cases

**USER:** Classify clarity and compare with observed diamonds: 1.0 carat Ideal G VS2, depth 61.5, table 57, x 6.45, y 6.43, z 3.96.

**EXPECTED:** concepts `CARAT_SIZE|CLARITY`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/MODEL_ANALYSIS/CLARITY_PREDICTION`; tools `classification|dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|CLARITY_PREDICTION|COMPARE|CUT|DIMENSIONS|MODEL_PREDICTION`; state `{}`; route `DATA_TASK/MODEL_ANALYSIS/CLARITY_PREDICTION`; tools `classification`.

**FIRST WRONG STAGE:** `tool_selection_failure`
**ROOT CAUSE:** The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks.
**FIX:** Create explicit tool plans for supported mixed tasks and report unsupported combinations early.

**USER:** Find diamonds matching impossible constraints.

**EXPECTED:** concepts `nan`; state `{"applicable": false, "fields": {}}`; route `nan/nan/nan`; tools `dataframe`.

**ACTUAL:** concepts `SEARCH_FILTER`; state `{}`; route `OUT_OF_SCOPE/REFUSE/OUT_OF_SCOPE`; tools `nan`.

**FIRST WRONG STAGE:** `tool_selection_failure`
**ROOT CAUSE:** The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks.
**FIX:** Create explicit tool plans for supported mixed tasks and report unsupported combinations early.

**USER:** gimme the middle money by clarity

**EXPECTED:** concepts `CLARITY`; state `{"applicable": false, "fields": {}}`; route `nan/nan/nan`; tools `dataframe`.

**ACTUAL:** concepts `CLARITY|GROUP|PRICE|TREND`; state `{}`; route `KNOWLEDGE_TASK/EXPLAIN/RAG_EXPLANATION`; tools `nan`.

**FIRST WRONG STAGE:** `tool_selection_failure`
**ROOT CAUSE:** The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks.
**FIX:** Create explicit tool plans for supported mixed tasks and report unsupported combinations early.

### 6. `concept_mapping_failure|field_evidence_failure` — 19 cases

**USER:** no worse than VS1 clarity and no worse than H color; ~0.73ct, max 3.9k

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|CLARITY|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|COLOR|STATISTIC`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

**USER:** pls min 0.63ct, max 0.79ct, don't spend over 4.1k

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|STATISTIC`; state `{}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

**USER:** pls about 1.01ct under 5.3 grand; SI2 or better and H or better

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|CLARITY|SEARCH_FILTER`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|CLARITY|FILTER`; state `{"target_carat": 1.01, "carat_tolerance": 0.15, "color": "H", "clarity": "SI"}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

### 7. `field_evidence_failure|L1_route_failure` — 18 cases

**USER:** aim for .55ct but it must be at least .90ct

**EXPECTED:** concepts `CARAT_SIZE`; state `{"applicable": false, "fields": {}}`; route `CONVERSATION_CONTROL/nan/nan`; tools `nan`.

**ACTUAL:** concepts `CARAT_SIZE|FILTER`; state `{}`; route `KNOWLEDGE_TASK/EXPLAIN/RAG_EXPLANATION`; tools `nan`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

**USER:** I want 1.10ct as the target and a .80ct absolute cap

**EXPECTED:** concepts `CARAT_SIZE`; state `{"applicable": false, "fields": {}}`; route `CONVERSATION_CONTROL/nan/nan`; tools `nan`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE`; state `{"target_carat": 0.8, "carat_tolerance": 0.15}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `nan`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

**USER:** aim for .55ct but it must be at least .90ct pls

**EXPECTED:** concepts `CARAT_SIZE`; state `{"applicable": false, "fields": {}}`; route `CONVERSATION_CONTROL/nan/nan`; tools `nan`.

**ACTUAL:** concepts `CARAT_SIZE|FILTER`; state `{}`; route `KNOWLEDGE_TASK/EXPLAIN/RAG_EXPLANATION`; tools `nan`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

### 8. `field_evidence_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure` — 10 cases

**USER:** price estimate only: 0.35ct / f / vs1 / premium

**EXPECTED:** concepts `CARAT_SIZE|CLARITY|PRICE`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/PRICE/PRICE_ESTIMATE`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|CUT|MODEL_PREDICTION|PRICE|PRICE_PREDICTION`; state `{}`; route `DATA_TASK/MODEL_ANALYSIS/PRICE_PREDICTION`; tools `nan`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

**USER:** pls rough dataset estimate for a 0.50ct E VS2 Good?

**EXPECTED:** concepts `CARAT_SIZE|CLARITY`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/PRICE/PRICE_ESTIMATE`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY`; state `{}`; route `DATA_TASK/MODEL_ANALYSIS/PRICE_PREDICTION`; tools `nan`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

**USER:** price estimate only: 0.62ct / G / VS2 / Premium

**EXPECTED:** concepts `CARAT_SIZE|CLARITY|PRICE`; state `{"applicable": false, "fields": {}}`; route `DATA_TASK/PRICE/PRICE_ESTIMATE`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CLARITY|CUT|MODEL_PREDICTION|PRICE|PRICE_PREDICTION`; state `{}`; route `DATA_TASK/MODEL_ANALYSIS/PRICE_PREDICTION`; tools `nan`.

**FIRST WRONG STAGE:** `field_evidence_failure`
**ROOT CAUSE:** Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation.
**FIX:** Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels.

### 9. `concept_mapping_failure|state_update_failure` — 9 cases

**USER:** start me near 0.41ct under $1,200; value first

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|PRICE|SEARCH_FILTER`; state `{"applicable": true, "expected": {"carat_target": 0.41, "budget_max": 1200.0, "priority": "value"}, "actual": {"max_price": 1200.0, "target_carat": 0.41, "carat_tolerance": 0.15, "priorities": ["value"]}, "fields": {"carat_target": "CHANGED_CORRECTLY", "budget_max": "CHANGED_CORRECTLY", "priority": "CHANGED_CORRECTLY", "carat_tolerance": "INVENTED"}, "accepted_delta": [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 1200.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.41, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}]}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|FILTER|PRICE|VALUE`; state `{"max_price": 1200.0, "target_carat": 0.41, "carat_tolerance": 0.15, "priorities": ["value"]}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

**USER:** start me near 0.31ct under $500; value first

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|PRICE|SEARCH_FILTER`; state `{"applicable": true, "expected": {"carat_target": 0.31, "budget_max": 500.0, "priority": "value"}, "actual": {"max_price": 500.0, "target_carat": 0.31, "carat_tolerance": 0.15, "priorities": ["value"]}, "fields": {"carat_target": "CHANGED_CORRECTLY", "budget_max": "CHANGED_CORRECTLY", "priority": "CHANGED_CORRECTLY", "carat_tolerance": "INVENTED"}, "accepted_delta": [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 500.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.31, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}]}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|FILTER|PRICE|VALUE`; state `{"max_price": 500.0, "target_carat": 0.31, "carat_tolerance": 0.15, "priorities": ["value"]}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

**USER:** start me near 0.72ct under $4,500; value first

**EXPECTED:** concepts `BUDGET|CARAT_SIZE|PRICE|SEARCH_FILTER`; state `{"applicable": true, "expected": {"carat_target": 0.72, "budget_max": 4500.0, "priority": "value"}, "actual": {"max_price": 4500.0, "target_carat": 0.72, "carat_tolerance": 0.15, "priorities": ["value"]}, "fields": {"carat_target": "CHANGED_CORRECTLY", "budget_max": "CHANGED_CORRECTLY", "priority": "CHANGED_CORRECTLY", "carat_tolerance": "INVENTED"}, "accepted_delta": [{"field": "max_price", "operation": "SET", "old_value": null, "new_value": 4500.0, "evidence": "explicit budget or price expression"}, {"field": "target_carat", "operation": "SET", "old_value": null, "new_value": 0.72, "evidence": "explicit carat expression"}, {"field": "carat_tolerance", "operation": "SET", "old_value": null, "new_value": 0.15, "evidence": "tolerance derived from supported carat target"}, {"field": "priorities", "operation": "SET", "old_value": null, "new_value": ["value"], "evidence": "explicit preference words: value"}]}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `BUDGET|CARAT_SIZE|FILTER|PRICE|VALUE`; state `{"max_price": 4500.0, "target_carat": 0.72, "carat_tolerance": 0.15, "priorities": ["value"]}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**FIRST WRONG STAGE:** `concept_mapping_failure`
**ROOT CAUSE:** The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts.
**FIX:** Define current-message concept expectations separately from route-supporting concepts.

### 10. `state_update_failure|L1_route_failure|L2_route_failure|L3_route_failure|tool_selection_failure|tool_execution_failure` — 6 cases

**USER:** everything else stays; change only the size target to 0.36ct

**EXPECTED:** concepts `nan`; state `{"applicable": true, "expected": {"carat_target": 0.36, "budget_max": 1100.0, "priority": "size", "clarity": "VS2", "clarity_or_better": true}, "actual": {}, "fields": {"carat_target": "INCORRECTLY_REMOVED_OR_MISSING", "budget_max": "INCORRECTLY_REMOVED_OR_MISSING", "priority": "INCORRECTLY_REMOVED_OR_MISSING", "clarity": "INCORRECTLY_REMOVED_OR_MISSING", "clarity_or_better": "MISSING_CAPABILITY"}, "accepted_delta": []}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CORRECTION`; state `{}`; route `KNOWLEDGE_TASK/EXPLAIN/RAG_EXPLANATION`; tools `nan`.

**FIRST WRONG STAGE:** `state_update_failure`
**ROOT CAUSE:** The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.
**FIX:** Evaluate complete ordered conversations and add explicit threshold/removal state fields.

**USER:** everything else stays change only the size target to 0.77ct

**EXPECTED:** concepts `nan`; state `{"applicable": true, "expected": {"carat_target": 0.77, "budget_max": 2800.0, "priority": "size", "clarity": "SI2", "clarity_or_better": true}, "actual": {}, "fields": {"carat_target": "INCORRECTLY_REMOVED_OR_MISSING", "budget_max": "INCORRECTLY_REMOVED_OR_MISSING", "priority": "INCORRECTLY_REMOVED_OR_MISSING", "clarity": "INCORRECTLY_REMOVED_OR_MISSING", "clarity_or_better": "MISSING_CAPABILITY"}, "accepted_delta": []}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CORRECTION`; state `{}`; route `KNOWLEDGE_TASK/EXPLAIN/RAG_EXPLANATION`; tools `nan`.

**FIRST WRONG STAGE:** `state_update_failure`
**ROOT CAUSE:** The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.
**FIX:** Evaluate complete ordered conversations and add explicit threshold/removal state fields.

**USER:** everything else stays; change only the size target to 0.37ct pls

**EXPECTED:** concepts `nan`; state `{"applicable": true, "expected": {"carat_target": 0.37, "budget_max": 1800.0, "priority": "size", "clarity": "VS1", "clarity_or_better": true}, "actual": {}, "fields": {"carat_target": "INCORRECTLY_REMOVED_OR_MISSING", "budget_max": "INCORRECTLY_REMOVED_OR_MISSING", "priority": "INCORRECTLY_REMOVED_OR_MISSING", "clarity": "INCORRECTLY_REMOVED_OR_MISSING", "clarity_or_better": "MISSING_CAPABILITY"}, "accepted_delta": []}`; route `DATA_TASK/RETRIEVE/SEARCH`; tools `dataframe`.

**ACTUAL:** concepts `CARAT_SIZE|CORRECTION`; state `{}`; route `KNOWLEDGE_TASK/EXPLAIN/RAG_EXPLANATION`; tools `nan`.

**FIRST WRONG STAGE:** `state_update_failure`
**ROOT CAUSE:** The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state.
**FIX:** Evaluate complete ordered conversations and add explicit threshold/removal state fields.

## Prioritized next fixes

| Priority | Fix | Cases potentially recovered | Families improved | Difficulty | Regression risk | Recommended? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Add raw-grade thresholds and explicit carat min/max fields | 104 | search, ranking, schema, price | medium | medium | YES |
| 2 | Evaluate complete ordered state conversations | 45 | multi_turn, long_context, corrections | medium | low | YES |
| 3 | Separate dataset price estimates from model predictions | 26 | price, tool boundaries | medium | medium | YES |
| 4 | Compose supported mixed operations before tool selection | 23 | mixed, conflicting multi-intent | medium | medium | YES |
| 5 | Align current-message concept labels with detector semantics | 20 | search, schema, adversarial | low | low | YES |
| 6 | Implement explicit empty-result and contradiction executors | 18 | missing data, ambiguity | medium | medium | YES |
| 7 | Calibrate route uncertainty and abstention | 8 | search, analysis | high | medium | LATER |

## Verified runtime configuration

| Setting | Verified value |
| --- | --- |
| Frontend framework | React 19.1.1 with Vite 7.1.5 |
| Package manager | npm; package-lock.json is present |
| Frontend command | npm run dev -- --host 127.0.0.1 --port 5173 |
| Frontend port | 5173 |
| Backend command | .venv\Scripts\python.exe -m src.assistant.api --host 127.0.0.1 --port 8770 |
| Backend port | 8770 |
| Frontend API base | VITE_ASSISTANT_API_URL or /assistant-api |
| Development proxy | /assistant-api → http://127.0.0.1:8770 |
| CORS | localhost:5173 and 127.0.0.1:5173; configurable |
| Environment variables | VITE_ASSISTANT_API_URL, VITE_ASSISTANT_API_TOKEN, DIAMOND_ASSISTANT_API_TOKEN, DIAMOND_FRONTEND_ORIGINS, DIAMOND_LOG_LEVEL |
| One-command launcher | powershell -ExecutionPolicy Bypass -File scripts\start_dev.ps1 |

# Required final tables

## Table 1 — Family health

| Family | Pass rate | Failures | Main failure stage | Main root cause | Next fix |
| --- | --- | --- | --- | --- | --- |
| missing_data_empty_results | 10.00% | 36 | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| multi_turn | 22.50% | 31 | state | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | Evaluate complete ordered conversations and add explicit threshold/removal state fields. |
| search_filtering | 27.50% | 29 | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | Define current-message concept expectations separately from route-supporting concepts. |
| correction_chains | 30.00% | 28 | state | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. | Evaluate complete ordered conversations and add explicit threshold/removal state fields. |
| ranking_recommendations | 55.00% | 18 | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| ambiguity_contradictions | 55.00% | 18 | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| dataframe_schema_variation | 57.50% | 17 | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| adversarial_phrasing | 60.00% | 16 | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | Define current-message concept expectations separately from route-supporting concepts. |
| price_value_analysis | 65.00% | 14 | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| mixed_intent_queries | 65.00% | 14 | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| conflicting_multi_intent | 72.50% | 11 | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| long_context_memory | 72.50% | 11 | tool_selection | The route does not request every tool expected by the case, commonly on incomplete state fragments or unsupported mixed tasks. | Create explicit tool plans for supported mixed tasks and report unsupported combinations early. |
| statistics_aggregation | 85.00% | 6 | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | Define current-message concept expectations separately from route-supporting concepts. |
| rag_grounded_explanation | 92.50% | 3 | L3_routing | The correct broad operation is selected, but its specialized subtype is wrong. | Use operation evidence and negative evidence for the remaining specialized collisions. |
| hallucination_bad_input | 92.50% | 3 | L1_routing | The request falls outside the supported domain or clarification rules before its intended data operation is selected. | Add calibrated abstention for genuinely unresolved operations without broad keyword changes. |
| explain_why_justification | 92.50% | 3 | field_evidence | Natural-language values are parsed, but raw-grade thresholds, compact money syntax, ranges, or unsupported evaluator fields do not match the accepted plan representation. | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| tool_boundary_tests | 92.50% | 3 | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| trends_group_analysis | 95.00% | 2 | concept_mapping | The expected and detected concept ontologies still disagree on implicit search, filters, and supporting concepts. | Define current-message concept expectations separately from route-supporting concepts. |
| engineered_feature_usage | 97.50% | 1 | L1_routing | The request falls outside the supported domain or clarification rules before its intended data operation is selected. | Add calibrated abstention for genuinely unresolved operations without broad keyword changes. |
| grounding_audit | 97.50% | 1 | L2_routing | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| diamond_comparison | 100.00% | 0 | none | No dominant remaining failure. | None |
| out_of_scope_refusal | 100.00% | 0 | none | No dominant remaining failure. | None |

## Table 2 — Worst root causes

| Root cause | Cases | Families affected | Fix |
| --- | --- | --- | --- |
| numeric/category attachment | 104 | search, ranking, schema, price | Add explicit raw-grade thresholds and min/max carat fields, then attach compact values by nearby units and labels. |
| concept ontology mismatch | 53 | search, schema, adversarial | Define current-message concept expectations separately from route-supporting concepts. |
| state sequence/representation | 45 | multi_turn, long_context, corrections | Evaluate complete ordered conversations and add explicit threshold/removal state fields. |
| route precedence | 35 | price, mixed, search, statistics | Separate dataset estimate language from saved-model prediction and compose multi-task operations before execution. |
| missing or incomplete executor | 28 | missing-data, mixed, corrections | Implement missing empty-result, contradiction, and mixed-task executors. |
| evaluator expectation mismatch | 20 | search, state, concepts | Correct labels and evaluate complete ordered conversations. |
| legitimate ambiguity | 18 | ambiguity_contradictions | Clarify instead of forcing a route. |
| grounding/evidence absence | 7 | mixed, empty results | Add source fallback and evidence-presence assertions. |

## Table 3 — Routing confusions

| Expected | Actual | Count | Cause |
| --- | --- | --- | --- |
| PRICE | MODEL_ANALYSIS | 13 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| TOOL_PLAN | ANALYZE | 11 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| TOOL_PLAN | PRICE | 10 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| RETRIEVE | EXPLAIN | 8 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| ANALYZE | EXPLAIN | 5 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| PRICE | EXPLAIN | 3 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| MODEL_ANALYSIS | TOOL_PLAN | 2 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| MODEL_ANALYSIS | EXPLAIN | 1 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| TOOL_PLAN | EXPLAIN | 1 | Operation precedence still confuses dataset price summaries with model prediction, or one operation with a multi-task plan. |
| PRICE_ESTIMATE | PRICE_PREDICTION | 13 | The correct broad operation is selected, but its specialized subtype is wrong. |
| MULTI_TASK | STATS | 10 | The correct broad operation is selected, but its specialized subtype is wrong. |
| MULTI_TASK | PRICE_RANGE | 10 | The correct broad operation is selected, but its specialized subtype is wrong. |
| SEARCH | RAG_EXPLANATION | 8 | The correct broad operation is selected, but its specialized subtype is wrong. |
| STATS | RAG_EXPLANATION | 3 | The correct broad operation is selected, but its specialized subtype is wrong. |
| PRICE_ESTIMATE | RAG_EXPLANATION | 3 | The correct broad operation is selected, but its specialized subtype is wrong. |
| RAG_EXPLANATION | DOMAIN_EXPLANATION | 3 | The correct broad operation is selected, but its specialized subtype is wrong. |
| PRICE_PREDICTION | MULTI_TASK | 2 | The correct broad operation is selected, but its specialized subtype is wrong. |
| TREND_RELATION | RAG_EXPLANATION | 2 | The correct broad operation is selected, but its specialized subtype is wrong. |
| MULTI_TASK | DOMAIN_EXPLANATION | 1 | The correct broad operation is selected, but its specialized subtype is wrong. |
| CLUSTER_PREDICTION | RAG_EXPLANATION | 1 | The correct broad operation is selected, but its specialized subtype is wrong. |
| MULTI_TASK | TREND_GROUP | 1 | The correct broad operation is selected, but its specialized subtype is wrong. |

## Table 4 — State failures

| Type | Count | Example | Cause |
| --- | --- | --- | --- |
| Lost or missing field | 88 | Keep that and require VS1 or better clarity. | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| Unsupported threshold state | 106 | Require VS1 or better clarity. | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| Invented field | 17 | Everything else stays; change only size. | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| Incorrectly changed field | 5 | Make size the priority now. | The frozen sample contains incomplete/shuffled conversations, and raw-grade threshold flags are not represented in the five-family state. |
| Correct removal | 0 | No observed case in this frozen sample. | No count to debug. |

## Table 5 — Product versus evaluator

| Classification | Count |
| --- | --- |
| AMBIGUOUS_CASE | 18 |
| DOWNSTREAM_FAILURE | 133 |
| EVALUATOR_BUG | 20 |
| MISSING_CAPABILITY | 37 |
| REAL_PRODUCT_BUG | 57 |

## Table 6 — Frontend status

| Check | Result |
| --- | --- |
| Frontend starts | PASS — Vite served HTTP 200 |
| Backend starts | PASS — models, dataset, and vector resources loaded |
| React → API | PASS — Vite proxy health returned ok |
| Streaming | PASS — 277 token events; 14,065 ms first token |
| State persistence | PASS — 0.70 → 0.75 correction survived refresh |
| Developer panel | PASS — route, tools, state, cache, and latency visible |
| Request logging | PASS — compact timestamped request trace |
| Browser smoke test | PASS — search, follow-up, refresh, and ANN request |

## Table 7 — Local access

| Service | URL | Status |
| --- | --- | --- |
| Frontend | http://localhost:5173/ | RUNNING — HTTP 200 |
| Backend | http://localhost:8770 | RUNNING |
| Health | http://localhost:8770/health | OK; all three saved models loaded |
| Streaming API | http://localhost:8770/chat/stream | VERIFIED |

The React UI was opened and exercised through browser automation. The official 1,200-case benchmark remains untouched.
