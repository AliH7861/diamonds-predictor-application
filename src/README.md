# Source code

Reusable code lives here so notebooks contain explanations, experiment calls, tables, and plots instead of hidden implementation state.

- `classification/`: clarity-family classification. The target is clarity; price may be an input for experiments that explicitly include it.
- `regression/`: price regression. The target is price; clarity is an input.

Keep generated datasets, figures, reports, and fitted models out of `src/`.
