"""Customer-oriented segmentation for diamond purchase profiles."""

from .config import SegmentationConfig
from .pipeline import SegmentationPipeline, run_buyer_segmentation, run_segmentation
from .prediction import assign_purchase_segment, load_segmentation_model

__all__ = [
    "SegmentationConfig",
    "SegmentationPipeline",
    "assign_purchase_segment",
    "load_segmentation_model",
    "run_buyer_segmentation",
    "run_segmentation",
]
