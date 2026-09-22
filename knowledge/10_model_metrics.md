# Model Metrics and Evaluation

## Purpose

This document explains how to interpret the project's model metrics.

Metrics from different experiments should not be mixed unless they come from the same split, target definition, and preprocessing pipeline.

# Classification metrics

## Accuracy

Accuracy is the percentage of predictions that exactly match the true class.

For five ordered clarity families, exact accuracy is strict because predicting a neighboring family still counts as incorrect.

## Balanced accuracy

Balanced accuracy gives each class more equal influence by averaging recall across classes.

It is useful when some clarity families are much more common than others.

## Macro precision

Calculates precision separately for each class and then averages them equally.

## Macro recall

Calculates recall separately for each class and then averages them equally.

## Macro F1

Calculates F1 for each class and then averages classes equally.

## Weighted F1

Weights each class by how many examples it contains.

## MCC

Matthews Correlation Coefficient summarizes classification agreement while accounting for the full confusion matrix.

It is useful when classes are imbalanced.

## Quadratic Weighted Kappa

Quadratic Weighted Kappa measures agreement for ordered classes.

Mistakes several clarity families away are penalized more strongly than neighboring mistakes.

## Ordinal MAE

Measures the average distance between the predicted and true ordered class index.

Lower is better.

For example, predicting VS instead of VVS is a smaller ordinal error than predicting I instead of VVS.

## Within-1

Percentage of predictions that are either exactly correct or within one neighboring clarity family.

This is useful for an ordered target.

## Severe error

Percentage of predictions more than one family away from the true class.

Lower is better.

# Recorded five-family classification snapshot

A recorded five-family experiment produced approximately:

- validation accuracy: 54.80%
- validation balanced accuracy: 48.28%
- test accuracy: 54.38%
- test balanced accuracy: 46.72%
- within-one-family: 91.09%
- severe error greater than one family: 8.91%

Treat this as an experiment snapshot, not automatically as the deployed model's current metrics.

# Recorded ordinal Random Forest snapshot

A later ordinal Random Forest test snapshot recorded approximately:

- accuracy: 57.12%
- balanced accuracy: 44.87%
- MCC: 0.3506
- Quadratic Weighted Kappa: 0.4845
- ordinal MAE: about 0.536

The final saved/deployed artifact should remain the source of truth if training is rerun.

# Regression metrics

## MAE

Mean Absolute Error is the average absolute difference between predicted and true price.

It is easy to interpret because it uses the same dollar units as price.

Lower is better.

## RMSE

Root Mean Squared Error penalizes larger mistakes more strongly than MAE.

It is also in price units.

Lower is better.

## R-squared

If reported, R-squared describes how much variance in the target is explained relative to a simple baseline.

It should not be used alone to judge whether dollar errors are practically acceptable.

# Price target scale

The project recorded a price standard deviation of approximately $3,989.44 in one modeling snapshot.

This provides context for regression errors but is not itself a model-performance metric.

# Clustering metrics

## Silhouette

Measures separation and compactness.

Higher is generally better.

## Inertia

Measures within-cluster squared distance.

Lower is better for a fixed K, but inertia naturally decreases as more clusters are added.

## Calinski-Harabasz

Higher generally indicates stronger separation relative to within-cluster spread.

## Davies-Bouldin

Lower generally indicates better-separated clusters.

## Stability

Checks whether the cluster solution is consistent across repeated fits or perturbations.

## Smallest cluster percentage

Helps detect solutions that create tiny, potentially impractical clusters.

# Recorded clustering snapshot

The project recorded approximately:

- K=3: silhouette 0.265, inertia 366,385.66, smallest cluster 14.29%
- K=5: silhouette 0.156, inertia 322,697.56, smallest cluster 10.90%
- K=7: silhouette 0.140, inertia 293,809.94, smallest cluster 4.67%
- K=10: silhouette 0.135, inertia 262,776.22, smallest cluster 0.004%

These metrics should be interpreted together rather than selecting K from inertia alone.

# Communication rules

The assistant should:

- identify which experiment a metric belongs to
- avoid calling a model "accurate" without saying which metric
- distinguish exact accuracy from within-one-family performance
- avoid comparing classification accuracy directly with regression error
- avoid treating clustering scores as prediction accuracy
- state that rerun training may change the numbers
