# Buyer Segmentation Method

## Purpose

This document explains how the project creates and evaluates broad purchase-profile clusters.

The clusters describe groups of similar diamond rows. They do not identify real people.

# What is clustered

Each cleaned diamond row represents an anonymous purchase profile.

The model uses combinations of:

- price and value
- physical size
- proportions
- cut
- color
- five-family clarity

Because the dataset contains no customer IDs, demographics, browsing history, or repeat purchases, a cluster cannot be interpreted as a verified customer identity.

# Preprocessing

Numeric inputs are scaled before distance-based clustering.

The maintained workflow uses robust scaling for numeric features.

Skewed measurements such as price and physical size can be log-transformed before K-Means distance is calculated.

Categorical diamond characteristics are converted into numeric model inputs as required by the training pipeline.

# Candidate K values

The maintained comparison evaluates:

- K = 3
- K = 5
- K = 7
- K = 10

These candidate solutions use the same general feature space so that cluster quality can be compared fairly.

# Evaluation metrics

## Silhouette score

Measures how well each point fits inside its assigned cluster compared with neighboring clusters.

Higher values generally indicate better separation.

## Inertia

Measures within-cluster squared distance.

Lower inertia means points are closer to their assigned cluster centers, but inertia always tends to decrease as K increases, so it should not be used alone.

## Calinski-Harabasz score

Measures separation between clusters relative to compactness within clusters.

Higher values generally indicate stronger separation.

## Davies-Bouldin score

Measures similarity between clusters based on within-cluster spread and separation.

Lower values generally indicate better-separated clusters.

## Stability

Checks whether similar cluster structure appears across repeated fits or perturbations.

Stable solutions are easier to interpret and trust.

## Cluster-size balance

Checks whether a candidate solution creates extremely tiny clusters.

A mathematically possible cluster can still be impractical if it contains only a negligible fraction of rows.

# Recorded K comparison snapshot

The project recorded the following comparison snapshot:

- K=3: silhouette about 0.265, inertia about 366,385.66, smallest cluster about 14.29%
- K=5: silhouette about 0.156, inertia about 322,697.56, smallest cluster about 10.90%
- K=7: silhouette about 0.140, inertia about 293,809.94, smallest cluster about 4.67%
- K=10: silhouette about 0.135, inertia about 262,776.22, smallest cluster approximately 0.004%

This snapshot should be treated as project experiment evidence. If the final clustering notebook or saved artifact changes, update this document.

# Interpretation limit

Clusters are broad purchase patterns.

They do not prove:

- a person's income
- a person's personality
- a person's demographic group
- a person's intent
- a fixed psychological buyer type

The assistant should describe cluster labels as interpretations of diamond purchase characteristics.

# Overlapping clusters

Clusters can overlap in real-world meaning even when K-Means assigns every row to exactly one cluster.

A row near a decision boundary may be similar to more than one cluster.

Therefore, cluster descriptions should be cautious rather than absolute.

# Buyer profile source

Authoritative buyer-profile labels and profile descriptions should come from the saved profile registry used by index_segmentation_records().

Do not invent cluster names in RAG when the profile registry has not defined them.

# Questions this document should answer

Examples:

- What does a cluster represent?
- Why are rows treated as purchase profiles?
- Why is K tested at several values?
- What is silhouette score?
- Why can a tiny cluster be a problem?
- Are these real customer identities?
- Why should profile labels be interpreted cautiously?
