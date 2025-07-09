from typing import Any, Literal

from tabkit.utils import Configuration


class DatasetConfig(Configuration):
    dataset_name: str
    data_source: Literal["openml", "uci", "automm", "disk"]
    random_state: int = 0
    openml_task_id: int | None = None
    openml_dataset_id: int | None = None
    openml_split_idx: int = 0
    uci_dataset_id: int | None = None
    automm_dataset_id: str | None = None
    file_path: str | None = None
    file_type: Literal["csv", "parquet"] = "csv"
    label_col: str | None = None
    split_file_path: str | None = None

    def __post_init__(self):
        if self.data_source not in [
            "openml",
            "uci",
            "automm",
            "disk",
        ]:
            raise ValueError(f"Invalid data source: {self.data_source}")
        if self.data_source == "openml":
            if self.openml_task_id is None and self.openml_dataset_id is None:
                raise ValueError(
                    "openml_task_id and openml_dataset_id must be set for openml data source"
                )
        elif self.data_source == "uci":
            if self.uci_dataset_id is None:
                raise ValueError("uci_dataset_id must be set for uci data source")
        elif self.data_source == "automm":
            if self.automm_dataset_id is None:
                raise ValueError("automm_dataset_id must be set for automm data source")
        elif self.data_source == "disk":
            if self.file_path is None:
                raise ValueError("file_path must be set for disk data source")
            if self.file_type not in ["csv", "parquet"]:
                raise ValueError(
                    "file_type must be either csv or parquet for disk data source"
                )


DEFAULT_PIPELINE = [
    {
        "class": "Impute",
        "params": {
            "method": "most_frequent",
        },
    },
    {
        "class": "Encode",
        "params": {
            "method": "most_frequent",
        },
    },
    {
        "class": "ConvertDatetime",
        "params": {
            "method": "to_timestamp",
        },
    },
]

DEFAULT_LABEL_PIPELINE = [
    {
        "class": "Encode",
        "params": {
            "method": "most_frequent",
        },
    },
]


class TableProcessorConfig(Configuration):
    pipeline: list[dict[str, Any]] | None = None
    task_kind: Literal["classification", "regression"] = "classification"
    n_splits: int = 10
    split_idx: int = 0
    n_val_splits: int = 9
    val_split_idx: int = 0
    random_state: int = 0
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)

    exclude_columns: list[str] | None = None
    exclude_labels: list[str] | None = None
    sample_n_rows: int | float | None = None

    label_pipeline: list[dict[str, Any]] | None = None
    # only used to help with splitting the dataset. Will not affect the actual label column
    label_stratify_pipeline: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.pipeline is None:
            self.pipeline = DEFAULT_PIPELINE
        if self.label_pipeline is None:
            self.label_pipeline = DEFAULT_LABEL_PIPELINE
        if self.label_stratify_pipeline is None:
            self.label_stratify_pipeline = DEFAULT_LABEL_PIPELINE
        if split_idx >= n_splits:
            raise ValueError("split_idx must be less than n_splits")
        if val_split_idx >= n_val_splits:
            raise ValueError("val_split_idx must be less than n_val_splits")
