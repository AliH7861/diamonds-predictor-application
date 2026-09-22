# Dataset Profile and Observed Ranges

## Purpose

This is a static background profile of the local diamonds dataset.

When the loaded DataFrame differs because of cleaning or preprocessing, live Pandas calculations are authoritative.

# Dataset size and fields

The raw local diamonds dataset contains 53,940 rows.

Its ten original modeling fields are:

- price
- carat
- cut
- color
- clarity
- depth
- table
- x
- y
- z

These are historical dataset observations rather than live retail listings.

# Price range

Observed raw prices range from:

- minimum: $326
- lower quartile: $950
- median: $2,401
- upper quartile: $5,324.25
- maximum: $18,823

If the user asks for current values from the loaded DataFrame, calculate them with Pandas.

# Carat range

Observed raw carat values range from:

- minimum: 0.20 carat
- lower quartile: 0.40 carat
- median: 0.70 carat
- upper quartile: 1.04 carats
- maximum: 5.01 carats

# Depth range

Observed depth ranges from 43% to 79%.

The median is approximately 61.8%.

# Table range

Observed table ranges from 43% to 95%.

The median is approximately 57%.

# Dimension fields

x, y, and z are measured in millimeters.

Some raw rows contain zero dimensions, so geometry/modeling workflows remove invalid zero-size rows before calculating engineered geometric features.

# Cut categories

The dataset contains:

- Fair
- Good
- Very Good
- Premium
- Ideal

Recorded raw counts:

- Ideal: 21,551
- Premium: 13,791
- Very Good: 12,082
- Good: 4,906
- Fair: 1,610

# Color categories

The dataset spans:

D, E, F, G, H, I, J

The most common raw color is G with 11,292 rows.

# Source clarity categories

The raw source clarity grades are:

- I1
- SI2
- SI1
- VS2
- VS1
- VVS2
- VVS1
- IF

The most common source clarity grade is SI1 with 13,065 rows.

VS2 follows with 12,258 rows.

# Five-family clarity mapping

The project reports:

- I1 -> I
- SI1/SI2 -> SI
- VS1/VS2 -> VS
- VVS1/VVS2 -> VVS
- IF -> IF

# Feature meanings

## Price

Recorded historical dollar value.

## Carat

Diamond weight, not direct physical width.

## Cut

Dataset cut-quality category, not named shape.

## Color

D-to-J color grade.

## Clarity

Grade describing inclusions and blemishes.

## Depth

Proportion percentage.

## Table

Proportion percentage.

## x

Measured length in millimeters.

## y

Measured width in millimeters.

## z

Measured depth/height in millimeters.

# Mode and frequency questions

Questions such as:

- What price occurs most often?
- What is the mode price?
- What carat appears most often?
- What are the ten most common prices?
- What is the most common cut/color/clarity combination?

should be computed with Pandas from the live DataFrame.

Do not rely on a static Markdown value for these unless it is clearly labeled as a historical snapshot.

# Statistics source of truth

Questions asking for current:

- counts
- ranges
- means
- medians
- modes
- quartiles
- standard deviations
- distributions
- percentages
- correlations
- grouped statistics

should be calculated directly with Pandas.

RAG supplies this profile as background context only.
