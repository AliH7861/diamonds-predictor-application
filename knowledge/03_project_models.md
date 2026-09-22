# Project Model Evidence

## Purpose

This document explains the modeling tasks used by the Diamond Decision Assistant.

Model outputs are evidence, not guaranteed truth. The deployed artifact loaded by the application is the source of truth for which model is currently active.

# Clarity classification

## Task

The classification workflow predicts one of five clarity families:

- I
- SI
- VS
- VVS
- IF

The original dataset contains more detailed subgrades, but the prediction target used by the buyer-facing assistant is grouped into these five families.

## Inputs

The clarity workflow uses structured diamond measurements and categorical characteristics.

Project experiments have used original and engineered features derived from:

- carat
- cut
- color
- depth
- table
- x
- y
- z
- geometric relationships
- ratio features

The exact deployed feature list must match the saved preprocessing/training artifact.

## Output meaning

The output is a model estimate of clarity family.

It does not replace examination by a trained grader and should not be described as a certification result.

## Ordinal nature

The five clarity families are ordered:

I < SI < VS < VVS < IF

Because the classes are ordered, a prediction one family away is less severe than a prediction several families away.

This is why the project evaluates both exact accuracy and ordinal metrics.

# Price regression

## Task

The regression workflow predicts historical dataset price from supplied diamond characteristics.

## Candidate algorithms

The project has compared:

- Artificial Neural Network (ANN)
- XGBoost
- Random Forest

The assistant should use the saved deployed/winning artifact rather than choosing a model dynamically from memory.

## Output meaning

The output is a predicted price based on the training distribution.

It is not a current retail quote, appraisal, or guaranteed selling price.

## When to use it

Use the saved price model when:

- the user explicitly asks for a price estimate
- the application intentionally includes model evidence in an explanation

Do not replace the real recorded dataset price with a prediction when the user is asking about an existing dataset row.

# Buyer segmentation

## Task

The clustering workflow groups diamond rows into broad purchase-profile patterns.

Each diamond row is treated as an anonymous purchase profile because the dataset contains no customer identifier.

## Inputs

The maintained workflow uses combinations of:

- price/value
- physical size
- proportions
- cut
- color
- five-family clarity

Numeric inputs are scaled. Skewed price and size measurements may use logarithmic transformations before distance-based clustering.

## Output meaning

A cluster indicates similarity in purchase characteristics.

It does not prove that a real person belongs to a psychological customer type.

Buyer-profile labels are interpretations of patterns, not verified identities.

# Model evidence rules

The assistant should:

- label predictions as estimates
- distinguish predicted values from observed dataset values
- avoid claiming professional grading or appraisal
- avoid inventing unavailable inputs
- report missing model inputs clearly
- use the exact preprocessing associated with the saved model
- avoid comparing model metrics from incompatible experiments as if they were directly equivalent

# Model questions the RAG should answer

Examples:

- What does the clarity model predict?
- What classes can it output?
- Why is clarity treated as ordered?
- What does the price model output mean?
- What algorithms were tested?
- What does clustering represent?
- Is a buyer segment a real customer identity?
- Why should model output be treated as supporting evidence?
