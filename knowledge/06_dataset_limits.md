# Dataset Coverage and Limits

## Purpose

This document defines what information exists in the dataset and what the assistant must not invent.

# Available original fields

The dataset contains:

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

These fields support search, analysis, feature engineering, and saved-model inference.

# Engineered fields

The modeling workflows may derive additional features from original columns.

Examples include:

- geometric size proxies
- face-area proxies
- dimension ratios
- asymmetry measures
- carat-to-volume relationships
- transformed price/size variables

Engineered features are calculations, not original measurements supplied by the dataset.

# Unavailable fields

The dataset does not contain:

- customer ID
- customer demographics
- customer name
- certification laboratory
- certificate number
- country of origin
- mine of origin
- named diamond shape
- fluorescence
- symmetry
- polish
- inclusion type
- seller
- brand
- sales channel
- transaction date
- live market price
- setting
- photographs
- damage/condition inspection

The assistant must not invent these fields.

# Historical data limitation

Dataset prices are historical observations.

They should not be presented as guaranteed current retail prices.

Without transaction date or live-market data, the assistant cannot determine whether a row is currently available for sale.

# Customer limitation

A diamond row is not a known customer.

Clustering treats rows as anonymous purchase profiles only because customer-level data is absent.

The assistant must not infer demographics, identity, or psychological traits from a cluster.

# Shape limitation

Cut is not shape.

The dataset's cut categories are Fair, Good, Very Good, Premium, and Ideal.

The assistant cannot identify round, oval, princess, emerald, pear, cushion, or other named shapes because shape is not a dataset field.

# Physical verification limitation

The assistant cannot verify:

- whether a physical stone matches a row
- whether a stone is genuine
- whether an inclusion is visible to the naked eye
- whether a certificate is authentic
- whether a stone has damage
- exact light performance of a physical diamond

# Zero-dimension and cleaning limitation

Some raw rows contain zero values in x, y, or z.

Modeling workflows remove invalid zero-size rows before geometry-based feature engineering.

Therefore, the live cleaned modeling dataset can contain fewer rows than the raw dataset profile.

# Outliers

Modeling workflows may detect or remove unusual dimension/proportion rows.

The assistant should distinguish:

- raw dataset statistics
- cleaned-modeling statistics
- model-training statistics

Do not assume they have identical row counts.

# Safe conclusions

The project can:

- search real dataset rows
- compare displayed rows
- calculate statistics with Pandas
- summarize historical relationships
- estimate price using saved models
- estimate clarity family using saved models
- infer broad purchase-profile segments

The project cannot:

- verify a physical stone
- identify a real buyer
- prove origin
- provide a certificate
- guarantee a current sale price
- guarantee an appraisal
- provide live inventory unless a separate live source is added

# Source-of-truth rules

Use:

- Pandas for live dataset counts, ranges, averages, medians, modes, distributions, and correlations
- dataset_search.py for actual filtered rows
- saved model artifacts for predictions
- profile registry for authoritative buyer profile names
- RAG for definitions, methodology, limitations, and explanations
