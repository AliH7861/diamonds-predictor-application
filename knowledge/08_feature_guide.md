# Dataset Feature Guide

## Purpose

This document explains every original dataset feature in plain language.

It is intended for questions such as:

- What does depth mean?
- What does table mean?
- What is x?
- Is carat the same thing as size?
- Why do dimensions matter?

# Price

## Meaning

Price is the recorded historical dollar value associated with a diamond row.

## Why it matters

Price is the outcome used for budget searches, value comparisons, and price-regression modeling.

## Limitation

It is not a live retail quote and does not include a timestamp, seller, or market context.

# Carat

## Meaning

Carat measures diamond weight.

It is not a direct measure of length or visible face-up size.

## Why it matters

Carat is a strong numeric price signal in this dataset and is also one of the most common buyer constraints.

## Important distinction

Two diamonds with the same carat can have different x, y, and z dimensions.

That means equal-weight diamonds can have different physical proportions.

# Cut

## Meaning

Cut is the dataset's cut-quality category.

Available values:

- Fair
- Good
- Very Good
- Premium
- Ideal

## Why it matters

Cut is associated with proportion quality and visual performance.

## Limitation

Cut is not diamond shape.

# Color

## Meaning

Color is graded from D through J in this dataset.

D is at the more colorless end of the represented range, while J has more body color.

## Why it matters

Color is a quality characteristic that can influence appearance and price.

# Clarity

## Meaning

Clarity describes inclusions and blemishes observed during grading.

Raw grades:

- I1
- SI2
- SI1
- VS2
- VS1
- VVS2
- VVS1
- IF

Buyer-facing families:

- I
- SI
- VS
- VVS
- IF

## Why it matters

Clarity provides information about the cleanliness of the stone under grading conditions.

## Limitation

The dataset does not include inclusion maps, inclusion types, or images.

# Depth

## Meaning

Depth is a proportion percentage supplied by the dataset.

It describes a relationship involving the stone's vertical depth relative to its overall dimensions.

## Why it matters

Depth helps describe a diamond's proportions and can help distinguish stones with similar carat but different geometry.

## Limitation

The assistant should not invent an "ideal" depth range unless the project explicitly documents one from a trusted source.

# Table

## Meaning

Table is a proportion percentage supplied by the dataset.

It represents the table proportion used in the source dataset.

## Why it matters

Table helps describe diamond proportions and provides additional information beyond carat alone.

## Limitation

The assistant should not invent a universal ideal table range unless it has been added from a trusted reference.

# x

## Meaning

x records a physical dimension in millimeters, commonly treated as the diamond's length measurement in this dataset.

## Why it matters

x helps represent physical size and geometry.

# y

## Meaning

y records a physical dimension in millimeters, commonly treated as width.

## Why it matters

Together with x, it helps describe face-up dimensions and asymmetry.

# z

## Meaning

z records a physical dimension in millimeters, commonly treated as depth/height.

## Why it matters

z helps describe vertical geometry and contributes to approximate volume features.

# Why x, y, and z matter together

Carat measures weight.

x, y, and z describe physical dimensions.

Together they allow models to distinguish stones that have similar weight but different proportions.

# Units

- price: dollars in the source dataset
- carat: carats
- depth: percent
- table: percent
- x: millimeters
- y: millimeters
- z: millimeters

# Live numeric questions

Use Pandas for:

- minimum/maximum x, y, or z
- average depth
- mode table
- dimension distributions
- grouped averages
- correlations
