"""
Build, inspect, and export the datasets used by the diamond price-regression workflow.

This module saves cleaned data, base and human feature splits, experiment feature
frames, and processed model matrices together with descriptive statistics.

It also creates a manifest and README describing the generated regression datasets,
and provides notebook helpers for inspecting raw and encoded experiment data.
"""

from pathlib import Path
import json
import pandas as pd

def export_datasets(prepared, cleaned, output_dir):
    """Export regression datasets, statistics, manifest metadata, and documentation."""

    output_dir = Path(output_dir)

    # Start with the cleaned dataset, then add the shared base and human feature splits.
    frames = {"cleaned/dataset": cleaned}

    for split, df in prepared["raw"].items():
        frames[f"base/{split}"] = df
        frames[f"human/{split}"] = prepared["human"][split]

    # Add both original-unit feature frames and exact processed matrices for every experiment.
    for name, data in prepared["experiments"].items():
        for split, frame in data["frames"].items():

            raw = frame.copy()
            raw["price"] = prepared["targets"][split]
            frames[f"experiments/{name}/features/{split}"] = raw

            columns = data["preprocessor"].get_feature_names_out()
            matrix = pd.DataFrame(data["matrices"][split], index=frame.index, columns=columns)
            matrix["price"] = prepared["targets"][split]

            frames[f"experiments/{name}/processed/{split}"] = matrix

    manifest = {}

    # Save every dataset beside a statistics version describing the same frame.
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        stats = path.parent / "statistics" / path.name

        stats.parent.mkdir(parents=True, exist_ok=True)

        frame.to_csv(path, index_label="Source_Row")
        frame.describe(include="all").T.to_csv(stats)

        manifest[name] = {
            "rows": len(frame), "columns": list(frame.columns),
            "file": path.relative_to(output_dir).as_posix(),
            "statistics": stats.relative_to(output_dir).as_posix()
        }

    # Save a machine-readable list of every exported dataset and statistics file.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Explain the regression export structure and important metadata rules.
    readme = (
        "# Regression datasets\n\n"
        "Price is the target; clarity is an input. Source_Row is the original raw CSV row position, never a feature.\n"
        "All split files use the same row assignments. Encoders and human context fit training rows only.\n"
        "Historical cleaning preserves duplicates and computes volume limits before splitting.\n"
        "Feature CSVs contain selected original-unit features plus price; processed CSVs contain encoded inputs plus price.\n"
        "Rerunning the notebook replaces these generated files. manifest.json lists every dataset and statistics path.\n"
    )

    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    # Return a compact summary of the generated regression datasets.
    summary = [{"Dataset": name, "Rows": entry["rows"], "File": entry["file"]} for name, entry in manifest.items()]

    return pd.DataFrame(summary)

def show_dataframe(df, name):
    """Display one DataFrame with its shape, schema, missing values, and statistics."""

    from IPython.display import display

    print(f"{name}: {df.shape}")

    display(df)
    df.info()
    display(df.isna().sum().rename("Missing values").to_frame())
    display(df.describe(include="all").T)

def show_experiment_data(prepared, name):
    """Display the original and processed data for every split of one regression experiment."""

    data = prepared["experiments"][name]
    columns = data["preprocessor"].get_feature_names_out()

    for split, frame in data["frames"].items():

        # Show the original-unit feature frame first.
        show_dataframe(frame, f"{name}: {split} features")

        # Rebuild the encoded/scaled matrix as a readable DataFrame.
        matrix = pd.DataFrame(data["matrices"][split], columns=columns)

        show_dataframe(matrix, f"{name}: {split} encoded inputs")