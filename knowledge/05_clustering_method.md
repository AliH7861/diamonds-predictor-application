# Buyer Segmentation Method

## What is clustered

Each cleaned diamond row represents a purchase profile. The model uses price and value, physical size, proportions, cut, color, and five-family clarity. Numeric inputs are robustly scaled and skewed price and size measurements use logarithms before K-Means distance is calculated.

## Candidate solutions

The maintained workflow compares K equals 3, 5, 7, and 10 with the same features and random process. It checks silhouette score, inertia, Calinski-Harabasz separation, Davies-Bouldin compactness, stability across repeat fits, and cluster-size balance.

## Interpretation limit

Clusters are broad purchase patterns. They do not prove that real customers belong to fixed psychological groups, and overlapping clusters should be described cautiously.
