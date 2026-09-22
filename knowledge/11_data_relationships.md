# Dataset Relationships and Observed Signals

## Purpose

This document explains observed relationships in the project without turning association into causation.

Exact correlations and grouped statistics should be recomputed from the live DataFrame when requested.

# Carat and price

Carat is a strong simple numeric price signal in this dataset.

Larger carat values are generally associated with higher prices.

The relationship is not a complete price formula because cut, color, clarity, proportions, and dimensions also vary.

The assistant should say "associated with" rather than claiming that one feature alone causes the full price difference.

# Dimensions and carat

x, y, and z generally increase as physical stone size increases.

However, stones with similar carat can have different proportions, so dimensions provide information beyond carat alone.

# Cut and price

Different cut categories have different price distributions, but raw average price by cut can be influenced by carat and other characteristics.

Therefore, a simple grouped average should not be interpreted as the isolated causal effect of cut.

# Color and price

Color grades can be associated with different price distributions.

Neighboring grades may have overlapping prices because carat, cut, clarity, and other factors vary at the same time.

# Clarity and price

Cleaner clarity grades can be associated with higher price, but the relationship overlaps heavily with carat and other quality variables.

A lower clarity grade does not automatically mean a lower price than every higher-clarity diamond.

# Depth and table

Depth and table describe proportions.

Their relationship with price can be nonlinear and confounded by the rest of the diamond's characteristics.

The assistant should avoid claiming a universal ideal range unless one is documented from a trusted source.

# Recorded feature signals for clarity experiments

A recorded project snapshot showed the following approximate relationships with the clarity target:

- z: eta-squared about 0.1517; Spearman about -0.3747
- x: eta-squared about 0.1490; Spearman about -0.3710
- y: eta-squared about 0.1448; Spearman about -0.3654
- carat: eta-squared about 0.1372; Spearman about -0.3741
- Volume: eta-squared about 0.1342; Spearman about -0.3697

Recorded categorical signal snapshots included approximately:

- cut: 0.1428
- color: 0.0796

These are experiment-specific feature signals and should not be described as universal gemological laws.

# Recorded mutual-information feature snapshot

A later ordinal experiment recorded high mutual-information values for engineered features including approximately:

- FaceArea * Depth: 0.1645
- Carat * XYAsym: 0.1200
- MeanXY / Carat^(1/3): 0.1127
- Carat / Volume: 0.1073
- Z / Carat^(1/3): 0.1058

Mutual information indicates statistical dependence with the target. It does not prove causation.

# Live analysis rules

When the user asks:

- What is the correlation between carat and price?
- Which feature correlates most with price?
- What is the mean price by cut?
- What clarity has the highest median price?
- How does price change by carat band?

calculate the answer from the live DataFrame.

RAG should explain how to interpret the result and its limitations.
