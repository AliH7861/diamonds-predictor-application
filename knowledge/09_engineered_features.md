# Engineered Features

## Purpose

Engineered features are calculated from the original dataset fields to give models additional information about geometry, proportions, and relationships.

These features are not original measurements from the dataset.

The exact feature-engineering code used to train a saved model is the final source of truth.

# Important rule

Do not guess a formula from a feature name.

This document only gives exact formulas where the project name itself clearly defines the operation or where the formula is already established in the project.

For any ambiguous feature such as DepthError or XYAsym, keep the exact implementation synchronized with the training code.

# Volume

## Formula

Volume = x * y * z

## Meaning

A bounding-box-style geometric size proxy based on the three measured dimensions.

## Why it helps

Two diamonds with similar carat can have different physical dimensions.

Volume gives a model an additional representation of geometric size.

## Limitation

This is not the exact physical volume of a faceted diamond.

It is a simple engineered proxy.

# FaceArea

## Formula used when defined as x by y

FaceArea = x * y

## Meaning

A simple top-facing rectangular area proxy.

## Why it helps

It helps represent how large the x/y footprint of a diamond is independently of z.

## Limitation

It is not the exact visible face-up area of a faceted diamond.

# MeanXY

## Meaning

A feature summarizing the x and y dimensions.

If the training code defines it as the arithmetic mean, the formula is:

MeanXY = (x + y) / 2

The saved feature-engineering code should be checked before treating this formula as authoritative.

# Carat / Volume

## Formula

CaratVolumeRatio = carat / Volume

## Meaning

Relates diamond weight to the approximate geometric volume proxy.

## Why it helps

Diamonds with similar geometric size can have different weights, and diamonds with similar weight can have different geometric proportions.

## Limitation

Because Volume is itself a proxy, this ratio should not be described as a laboratory density measurement.

# Z / Carat^(1/3)

## Formula

ZCaratScale = z / carat^(1/3)

## Meaning

Compares vertical dimension with a cube-root scaling of carat.

Cube-root scaling is useful when comparing a one-dimensional measurement with a quantity that behaves more like three-dimensional size.

# MeanXY / Carat^(1/3)

## Formula

MeanXYCaratScale = MeanXY / carat^(1/3)

## Meaning

Compares a horizontal size summary with cube-root-scaled carat.

It can help distinguish stones whose physical dimensions are unusual relative to their weight.

# XYAsym

## Meaning

An engineered measure of difference or asymmetry between x and y.

## Exact formula

The exact formula must be copied from the feature-engineering code before this RAG document treats it as authoritative.

Possible definitions should not be guessed.

# Carat * XYAsym

## Formula

CaratXYAsym = carat * XYAsym

## Meaning

An interaction feature combining stone weight with x/y asymmetry.

## Why it helps

The effect of asymmetry may differ for smaller and larger stones, so the interaction can give a model additional signal.

# FaceArea * Depth

## Formula

FaceAreaDepth = FaceArea * depth

## Meaning

Combines the x/y area proxy with the dataset depth percentage.

## Limitation

This is an engineered interaction, not a physical volume formula unless the training code explicitly defines and interprets it that way.

# DepthError

## Meaning

A project-specific engineered feature describing some deviation involving depth.

## Exact formula

The exact formula must be copied from the training/feature-engineering code.

Do not infer the formula from the name alone.

# Ratio features

Ratio features compare two measurements to represent shape or proportion rather than absolute size.

Examples can include:

- x / y
- y / x
- z / MeanXY
- carat / Volume

Only document ratios that are actually created by the training code.

# Interaction features

Interaction features combine multiple variables, for example:

- carat * XYAsym
- FaceArea * depth

They allow a model to capture relationships that may not be represented by either input alone.

# Transformed features

Some workflows use logarithms for skewed variables such as price or size.

A log transform compresses very large values and can make some statistical/modeling relationships easier to learn.

The exact transform must match the saved preprocessing pipeline.

# RAG safety rule

When the user asks what an engineered feature means:

1. give the exact formula if documented
2. explain the intuition
3. state that it is derived rather than original
4. avoid presenting a proxy as a direct gemological measurement
5. defer to training code when the formula is not documented
