# Diamond Quality Grades

## Purpose

This document explains the meaning and ordering of the dataset's cut, color, and clarity categories. Exact category counts should be calculated from the live DataFrame when the user asks for current statistics.

# Cut

## What cut means

Cut is the dataset's cut-quality category. It relates to the quality of a diamond's proportions and visual performance.

Cut is not the same as shape.

The dataset does not provide named shapes such as round, oval, cushion, pear, emerald, or princess.

## Cut order

From the lowest to highest category represented in this dataset:

1. Fair
2. Good
3. Very Good
4. Premium
5. Ideal

### Fair

Fair is the lowest cut category represented in the dataset.

It can represent a trade-off where a buyer accepts lower cut quality in exchange for price, carat, color, clarity, or another preference.

### Good

Good is above Fair and below Very Good.

It represents a middle-lower cut category in the dataset's ordering.

### Very Good

Very Good is above Good and below Premium.

It represents a strong cut category while still leaving room for Premium and Ideal above it.

### Premium

Premium is one of the highest cut categories represented in the dataset.

It sits directly below Ideal in the dataset ordering.

### Ideal

Ideal is the highest cut category represented in the dataset.

It should be described as the highest dataset cut category, not as proof that a physical diamond has been independently inspected or certified by the assistant.

## Cut and price

Higher cut categories can be associated with different price distributions, but cut alone does not determine price.

Carat, color, clarity, proportions, and dimensions also matter.

# Clarity

## What clarity means

Clarity describes inclusions and blemishes observed during grading.

The raw dataset contains eight detailed grades:

- I1
- SI2
- SI1
- VS2
- VS1
- VVS2
- VVS1
- IF

The assistant reports five buyer-facing clarity families:

- I
- SI
- VS
- VVS
- IF

## Detailed clarity ordering

From lower to higher clarity in this dataset:

1. I1
2. SI2
3. SI1
4. VS2
5. VS1
6. VVS2
7. VVS1
8. IF

Within a family, a grade ending in 1 is higher than the corresponding grade ending in 2. For example, VS1 is above VS2 and VVS1 is above VVS2.

## Five-family mapping

- I1 -> I
- SI1 and SI2 -> SI
- VS1 and VS2 -> VS
- VVS1 and VVS2 -> VVS
- IF -> IF

### I — Included

I is the included family represented here by I1.

Inclusions are more significant than in the higher clarity families.

The assistant should avoid claiming whether a specific inclusion is visible to a particular person because the dataset does not contain inclusion maps, images, or inclusion-type fields.

### SI — Slightly Included

SI combines SI1 and SI2.

SI diamonds contain inclusions that are more noticeable under grading conditions than the VS, VVS, and IF families.

SI can be part of a value trade-off when a buyer prefers more carat or a lower price.

### VS — Very Slightly Included

VS combines VS1 and VS2.

It represents a cleaner family than SI and a lower family than VVS.

VS can provide a practical balance between clarity and price for some buyers, but whether it is the "best" choice depends on the user's priorities.

### VVS — Very Very Slightly Included

VVS combines VVS1 and VVS2.

It represents a very high clarity family with very small inclusions under grading conditions.

VVS is above VS and below IF in the five-family ordering.

### IF — Internally Flawless

IF is the highest clarity family represented in this project.

The assistant should describe it as the highest clarity category in this dataset, not as a substitute for professional inspection of a physical stone.

## Clarity and price

Higher clarity can be associated with higher prices, but clarity does not determine price by itself.

Carat, cut, color, and proportions may create larger differences between individual stones.

# Color

## What color means

The dataset contains color grades D through J.

The represented order is:

D, E, F, G, H, I, J

D is at the more colorless end of the represented range. J shows more body color within the represented range.

## Practical color bands

For simple explanations, the assistant may describe:

- D-F as the more colorless end of this dataset range
- G-H as the middle portion of the range
- I-J as showing more body color

These are broad explanatory bands, not replacement grading categories.

## Neighboring grades

Differences between neighboring color grades can be subtle.

Appearance can depend on lighting, setting, cut, and the physical stone. The dataset does not contain photographs or setting information, so the assistant should avoid claiming exactly how visible a color difference will be.

## Color and price

Color can contribute to price differences, but it should be considered together with carat, cut, clarity, and the rest of the measurements.

# Questions that should use Pandas

Use the live DataFrame for questions such as:

- Which cut appears most often?
- Which color appears most often?
- Which clarity grade appears most often?
- What percentage of diamonds are Ideal?
- How many VS diamonds exist?
- What is the average price for each clarity family?
- What is the median carat by cut category?
