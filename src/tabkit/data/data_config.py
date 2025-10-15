"""
Configuration dataclasses for TableProcessor.

These provide type-safe, autocomplete-friendly configuration options.
You can still use plain dictionaries if you prefer - both work!
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading.

    Args:
        dataset_name: Name for your dataset (used for logging/identification)
        data_source: Where to load data from - "disk", "openml", "uci", or "automm"
        file_path: Path to data file (required when data_source="disk")
        file_type: File format - "csv" or "parquet" (required when data_source="disk")
        label_col: Name of the target/label column
        openml_task_id: OpenML task ID (for data_source="openml")
        openml_dataset_id: OpenML dataset ID (for data_source="openml")
        openml_split_idx: Which OpenML split to use (for data_source="openml")
        uci_dataset_id: UCI dataset identifier (for data_source="uci")
        automm_dataset_id: AutoMM dataset identifier (for data_source="automm")
        split_file_path: Path to predefined train/test split indices

    Example:
        >>> # Load from local CSV
        >>> config = DatasetConfig(
        ...     dataset_name="my_data",
        ...     data_source="disk",
        ...     file_path="data/train.csv",
        ...     file_type="csv",
        ...     label_col="target"
        ... )

        >>> # Load from OpenML
        >>> config = DatasetConfig(
        ...     dataset_name="adult",
        ...     data_source="openml",
        ...     openml_task_id=7592
        ... )
    """
    dataset_name: str = "default"
    data_source: Literal["disk", "openml", "uci", "automm"] | None = None
    openml_task_id: int | None = None
    openml_dataset_id: int | None = None
    openml_split_idx: int | None = None
    uci_dataset_id: str | None = None
    automm_dataset_id: str | None = None
    file_path: str | None = None
    file_type: Literal["csv", "parquet"] | None = None
    label_col: str | None = None
    split_file_path: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary (for backward compatibility)."""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class TableProcessorConfig:
    """
    Configuration for table preprocessing and train/val/test splitting.

    Args:
        pipeline: List of preprocessing transforms to apply. Each is a dict with:
            - "class": Transform name (e.g., "Impute", "Encode", "Scale")
            - "params": Dict of parameters for that transform

        task_kind: Type of prediction task - "classification" or "regression"

        --- Splitting Configuration ---
        Two modes available (ratio-based takes precedence if both are set):

        MODE 1: Ratio-Based Splitting (Simple)
            test_ratio: Fraction for test set (e.g., 0.2 = 20%)
            val_ratio: Fraction for validation set (e.g., 0.1 = 10%)
            Example: test_ratio=0.2, val_ratio=0.1 → 70/10/20 train/val/test split

        MODE 2: K-Fold Splitting (Robust)
            n_splits: Number of folds for train/test split
            split_idx: Which fold to use as test (0 to n_splits-1)
            n_val_splits: Number of folds for train/val split
            val_split_idx: Which fold to use as validation
            split_validation: Whether to create train/val split

        --- Other Options ---
        random_state: Random seed for reproducibility
        exclude_columns: Column names to exclude from features
        exclude_labels: Label values to filter out (classification only)
        sample_n_rows: Subsample training data (int=count, float=fraction)
        label_pipeline: Custom pipeline for label preprocessing
        label_stratify_pipeline: Pipeline for creating stratification target

    Example:
        >>> # Ratio-based splitting with custom pipeline
        >>> config = TableProcessorConfig(
        ...     task_kind="classification",
        ...     test_ratio=0.2,
        ...     val_ratio=0.1,
        ...     pipeline=[
        ...         {"class": "Impute", "params": {"method": "mean"}},
        ...         {"class": "Encode", "params": {"method": "label"}},
        ...         {"class": "Scale", "params": {"method": "standard"}}
        ...     ]
        ... )

        >>> # K-fold splitting
        >>> config = TableProcessorConfig(
        ...     task_kind="regression",
        ...     n_splits=5,
        ...     split_idx=0,
        ...     random_state=42
        ... )
    """
    # Preprocessing pipeline
    pipeline: list[dict] | None = None

    # Task configuration
    task_kind: Literal["classification", "regression"] = "classification"

    # Ratio-based splitting (takes precedence if set)
    test_ratio: float | None = None
    val_ratio: float | None = None

    # K-fold splitting
    n_splits: int = 10
    split_idx: int = 0
    n_val_splits: int = 9
    val_split_idx: int = 0
    split_validation: bool = True

    # Other options
    random_state: int = 0
    exclude_columns: list[str] | None = None
    exclude_labels: list[str] | None = None
    sample_n_rows: int | float | None = None
    label_pipeline: list[dict] | None = None
    label_stratify_pipeline: list[dict] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary (for backward compatibility)."""
        from dataclasses import asdict
        return asdict(self)
