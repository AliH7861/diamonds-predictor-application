# Project Model Evidence

## Clarity classification

The classification workflow predicts one of five clarity families: I, SI, VS, VVS, or IF. Its saved ANN uses structured measurements and categorical characteristics. A prediction is a model estimate and does not replace examination by a trained grader.

## Price regression

The regression workflow predicts price. It compares three maintained representations and three algorithms: ANN, XGBoost, and Random Forest. The assistant uses the saved winning artifact only when a user explicitly requests a price estimate or when model evidence is useful for an explained recommendation.

## Buyer segmentation

The clustering workflow treats each diamond row as an anonymous purchase profile because the dataset has no customer identifiers. A segment describes diamonds with similar price, size, and quality characteristics. Its buyer label is an interpretation of the purchase pattern, not a known customer identity.
