# Buyer Profiles and Cluster Interpretation

## Purpose

This document explains how to talk about buyer-profile clusters safely.

Authoritative profile names, cluster IDs, and profile descriptions should come from the saved profile registry that is separately indexed into Chroma.

This Markdown file should not invent labels that are missing from the profile registry.

# What a buyer profile means

A buyer profile is an interpretation of a cluster of diamond rows with similar purchase characteristics.

It can summarize patterns involving:

- price
- carat
- cut
- color
- clarity
- proportions
- value-related features

It does not identify a real person.

# What a buyer profile does not mean

A cluster does not prove:

- age
- gender
- income
- location
- personality
- relationship status
- occupation
- ethnicity
- motivation
- psychological type

The dataset does not contain those fields.

# Profile language

Use language such as:

- "This cluster tends to contain..."
- "Rows in this profile are characterized by..."
- "The profile represents a purchase pattern..."
- "Compared with other clusters, this group tends to..."

Avoid language such as:

- "These customers are..."
- "This person is..."
- "People in this group definitely want..."
- "This cluster proves..."

# Profile statistics

When describing a saved profile, useful characteristics include:

- cluster size
- percentage of rows
- median price
- typical price range
- median carat
- typical carat range
- common cut categories
- common color grades
- common clarity families
- distinguishing engineered/value features

These values should come from the saved profile registry or live cluster analysis.

# Example profile structure

Use this structure when the registry provides the values:

## Cluster <ID> — <Profile Name>

### Typical characteristics

- cluster size:
- share of dataset:
- median price:
- price range:
- median carat:
- common cut:
- common color:
- common clarity:

### Interpretation

Explain the purchase pattern represented by the rows.

### Main trade-off

Explain whether the cluster tends to emphasize size, price, or quality characteristics.

### Limitations

State that the label is an interpretation of diamond-row patterns rather than a known customer identity.

# Comparison between profiles

When comparing profiles, focus on observable cluster statistics.

Examples:

- one cluster may contain larger but more expensive stones
- one may contain lower-priced stones with smaller carat
- one may emphasize higher quality grades

Only make these statements when supported by the actual profile records.

# Retrieval rule

Questions explicitly asking about:

- buyer profile
- customer profile
- cluster
- segmentation

may use the separately indexed segmentation collection.

Unrelated questions should not retrieve buyer-profile content.

# Source-of-truth rule

If this file conflicts with profile_registry.json, the saved profile registry is authoritative for:

- profile names
- cluster IDs
- profile-specific text
- profile statistics
