# Recommendation Policy

## Purpose

This document explains how the assistant should search for and explain diamond recommendations.

Actual filtering and ranking behavior is implemented in search_planning.py and dataset_search.py. This document should explain that behavior, not silently override it.

# Evidence order

For a buying request:

1. Parse the user's stated constraints.
2. Filter real dataset rows.
3. Rank the closest matches.
4. Diversify results when appropriate.
5. Show observed row values.
6. Explain trade-offs in natural language.
7. Use model predictions only as supporting evidence.
8. Use RAG knowledge for explanations and definitions.

Real dataset rows are the primary evidence for search recommendations.

# Usable search constraints

One concrete constraint is enough to begin a search.

Examples include:

- maximum budget
- target or minimum carat
- cut preference
- color preference
- clarity preference
- explicit priority such as size or quality

Do not force the user to specify every diamond characteristic before showing results.

# Missing preferences

Ask a follow-up when:

- the request has no usable search constraint
- the user names a priority but leaves its acceptable range too unclear to execute
- the request is internally contradictory and cannot be interpreted safely

Do not ask unnecessary questions when an initial search can already be performed.

# Exact matches

If exact matches exist, show the strongest relevant matches according to the active ranking policy.

The assistant should explain why they match the user's request using actual row values.

# No exact matches

If no exact matches exist:

- do not invent matching diamonds
- show the closest available alternatives when the search policy supports relaxation
- clearly identify which constraint is being missed or relaxed
- explain the trade-off

For example, a result may be slightly over budget but closer in carat, or slightly below a requested clarity family while staying inside budget.

# Result diversity

Avoid returning five nearly identical rows when more useful alternatives are available.

A diverse result set can help the user understand trade-offs between:

- price
- carat
- cut
- color
- clarity

The exact diversity algorithm belongs in dataset_search.py.

# Meaning of common recommendation words

## Cheapest

The result with the lowest observed price among the considered rows.

## Largest

The result with the highest carat among the considered rows.

## Closest match

The result that best satisfies the active search/ranking plan.

## Best value

"Best value" is preference-dependent.

Do not treat cheapest as automatically equal to best value.

If the code defines a value score, explain that score. Otherwise explain value as a trade-off instead of pretending there is one universal winner.

## Highest quality

Avoid using this as an undefined overall score.

Instead, state the dimensions being compared, such as higher cut, cleaner clarity, or more colorless grade.

# Comparison behavior

When the user refers to "first," "second," "third," and similar positions, compare the rows already displayed in the conversation.

Do not silently run a new search and compare different rows unless the user asks for a new search.

# Communication

Recommendations should state the observed:

- price
- carat
- cut
- color
- five-family clarity

Explain compromises without inventing:

- certification
- certificate number
- origin
- mine
- customer identity
- shape
- fluorescence
- polish
- symmetry
- seller
- current market quote

# Use of model evidence

Model predictions are optional supporting evidence.

Do not let a model prediction override known values in an existing dataset row.

# Use of RAG

RAG should explain:

- what a feature means
- why a trade-off matters
- what a quality grade means
- what the model can and cannot conclude

RAG should not replace deterministic row filtering or live Pandas statistics.
