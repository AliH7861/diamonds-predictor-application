"""Physical-only Y23 features for ordered diamond-clarity classification.

Price is intentionally excluded. Every maintained classifier receives the
same physical representation so algorithm comparisons remain fair.
"""

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["cut", "color"]
RAW_CLASSIFICATION_INPUTS = [
    "carat",
    "cut",
    "color",
    "depth",
    "table",
    "x",
    "y",
    "z",
]

NUMERIC_FEATURES_ALL = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "Y23_FaceArea",
    "Y23_Volume",
    "Y23_FaceDiagonal",
    "Y23_XYRatio",
    "Y23_XZRatio",
    "Y23_YZRatio",
    "Y23_XYAbsDiff",
    "Y23_XZAbsDiff",
    "Y23_YZAbsDiff",
    "Y23_XYNormalizedDiff",
    "Y23_XZNormalizedDiff",
    "Y23_YZNormalizedDiff",
    "Y23_XYSymmetryProxy",
    "Y23_MeanXY",
    "Y23_MeanXYZ",
    "Y23_MaxDimension",
    "Y23_MinDimension",
    "Y23_DimensionRange",
    "Y23_DimensionStd",
    "Y23_DimensionCV",
    "Y23_XPerCaratCubeRoot",
    "Y23_YPerCaratCubeRoot",
    "Y23_ZPerCaratCubeRoot",
    "Y23_MeanXYPerCaratCubeRoot",
    "Y23_MeanXYZPerCaratCubeRoot",
    "Y23_CaratPerVolume",
    "Y23_VolumePerCarat",
    "Y23_CaratPerFaceArea",
    "Y23_FaceAreaPerCarat",
    "Y23_CaratPerMeanDimension",
    "Y23_CalculatedDepthPct",
    "Y23_DepthError",
    "Y23_AbsDepthError",
    "Y23_DepthTableRatio",
    "Y23_TableDepthRatio",
    "Y23_DepthToFaceDiagonal",
    "Y23_DepthToFaceArea",
    "Y23_Compactness",
    "Y23_VolumeToMeanCube",
    "Y23_DepthTimesTable",
    "Y23_CaratTimesDepth",
    "Y23_CaratTimesTable",
    "Y23_CaratTimesXYAsymmetry",
    "Y23_VolumeTimesDepth",
    "Y23_FaceAreaTimesDepth",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the notebook's physical Y23 features without using price."""
    data = frame.copy()
    data["Y23_FaceArea"] = data["x"] * data["y"]
    data["Y23_Volume"] = data["x"] * data["y"] * data["z"]
    data["Y23_FaceDiagonal"] = np.sqrt(data["x"] ** 2 + data["y"] ** 2)

    data["Y23_XYRatio"] = data["x"] / data["y"]
    data["Y23_XZRatio"] = data["x"] / data["z"]
    data["Y23_YZRatio"] = data["y"] / data["z"]
    data["Y23_XYAbsDiff"] = np.abs(data["x"] - data["y"])
    data["Y23_XZAbsDiff"] = np.abs(data["x"] - data["z"])
    data["Y23_YZAbsDiff"] = np.abs(data["y"] - data["z"])

    xy_mean = (data["x"] + data["y"]) / 2.0
    xz_mean = (data["x"] + data["z"]) / 2.0
    yz_mean = (data["y"] + data["z"]) / 2.0
    data["Y23_XYNormalizedDiff"] = data["Y23_XYAbsDiff"] / xy_mean
    data["Y23_XZNormalizedDiff"] = data["Y23_XZAbsDiff"] / xz_mean
    data["Y23_YZNormalizedDiff"] = data["Y23_YZAbsDiff"] / yz_mean
    data["Y23_XYSymmetryProxy"] = 1.0 - data["Y23_XYNormalizedDiff"]

    dimensions = data[["x", "y", "z"]]
    data["Y23_MeanXY"] = xy_mean
    data["Y23_MeanXYZ"] = dimensions.mean(axis=1)
    data["Y23_MaxDimension"] = dimensions.max(axis=1)
    data["Y23_MinDimension"] = dimensions.min(axis=1)
    data["Y23_DimensionRange"] = data["Y23_MaxDimension"] - data["Y23_MinDimension"]
    data["Y23_DimensionStd"] = dimensions.std(axis=1, ddof=0)
    data["Y23_DimensionCV"] = data["Y23_DimensionStd"] / data["Y23_MeanXYZ"]

    carat_root = np.cbrt(data["carat"])
    data["Y23_XPerCaratCubeRoot"] = data["x"] / carat_root
    data["Y23_YPerCaratCubeRoot"] = data["y"] / carat_root
    data["Y23_ZPerCaratCubeRoot"] = data["z"] / carat_root
    data["Y23_MeanXYPerCaratCubeRoot"] = data["Y23_MeanXY"] / carat_root
    data["Y23_MeanXYZPerCaratCubeRoot"] = data["Y23_MeanXYZ"] / carat_root

    data["Y23_CaratPerVolume"] = data["carat"] / data["Y23_Volume"]
    data["Y23_VolumePerCarat"] = data["Y23_Volume"] / data["carat"]
    data["Y23_CaratPerFaceArea"] = data["carat"] / data["Y23_FaceArea"]
    data["Y23_FaceAreaPerCarat"] = data["Y23_FaceArea"] / data["carat"]
    data["Y23_CaratPerMeanDimension"] = data["carat"] / data["Y23_MeanXYZ"]

    data["Y23_CalculatedDepthPct"] = 2.0 * data["z"] / (data["x"] + data["y"]) * 100.0
    data["Y23_DepthError"] = data["depth"] - data["Y23_CalculatedDepthPct"]
    data["Y23_AbsDepthError"] = np.abs(data["Y23_DepthError"])
    data["Y23_DepthTableRatio"] = data["depth"] / data["table"]
    data["Y23_TableDepthRatio"] = data["table"] / data["depth"]
    data["Y23_DepthToFaceDiagonal"] = data["z"] / data["Y23_FaceDiagonal"]
    data["Y23_DepthToFaceArea"] = data["z"] / data["Y23_FaceArea"]
    data["Y23_Compactness"] = data["carat"] / data["Y23_MeanXYZ"] ** 3
    data["Y23_VolumeToMeanCube"] = data["Y23_Volume"] / data["Y23_MeanXYZ"] ** 3

    data["Y23_DepthTimesTable"] = data["depth"] * data["table"]
    data["Y23_CaratTimesDepth"] = data["carat"] * data["depth"]
    data["Y23_CaratTimesTable"] = data["carat"] * data["table"]
    data["Y23_CaratTimesXYAsymmetry"] = data["carat"] * data["Y23_XYNormalizedDiff"]
    data["Y23_VolumeTimesDepth"] = data["Y23_Volume"] * data["depth"]
    data["Y23_FaceAreaTimesDepth"] = data["Y23_FaceArea"] * data["depth"]
    return data.replace([np.inf, -np.inf], np.nan)
