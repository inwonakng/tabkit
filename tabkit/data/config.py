from dataclasses import dataclass
from typing import Literal

from tabkit.utils import Configuration


class DatasetConfig(Configuration):
    dataset_name: str
    data_source: Literal["openml", "uci", "automm", "disk"]
    random_state: int = 0
    openml_task_id: int | None = None
    openml_dataset_id: int | None = None
    openml_fold_idx: int = 0
    uci_dataset_id: int | None = None
    automm_dataset_id: str | None = None
    grdive_link: str | None = None
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
            if self.openml_task_id is None or self.openml_dataset_id is None:
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


@dataclass
class TableProcessorConfig(Configuration):
    """Configuation for preprocessing the dataset.

    Attributes:
        fold_idx: Which fold to use for train/val/test split.
        n_splits: How many splits to consider for splitting (`K` for `KFold`).
        scale_cont: Whether to scale continuous features.
        cont_scale_method: Type of scaler to apply to contiuous features.
        cont_scale_max_quantiles: Number of quantiles to use if using
        `quantile` strategy.
        bin_cont: Whether to bin continuous features.
        cont_bin_strat: What strategy to use to bin continuous features.
        n_bins: How many bins to use for binning continuous features.
        handle_cat_unseen: How to handle unknown/unseen categorical values. any
        values that are nan or not seein the train split will be replaced.
        cat_unseen_fill: Which value to use if using `constant` strategy for
        `handle_cat_unseen`.
        handle_cat_missing: How to handle missing values in categorical
        features.
        cat_missing_fill: Which value to use if using `constant` strategy for
        `handle_cat_missing`.
        handle_cont_missing: How to handle missing values in continuous
        features.
        cont_missing_fill: Which value to use if using `constant` strategy for
        `handle_cont_missing`.
        cont_strat_n_bins: How many bins to use for stratifying continuous
        features.
        cont_strat_bin_strat: What strategy to use for stratifying continuous
        features.
        sample_n_rows: How much to subsample the dataset. If `None`, no
        subsampling.
        exclude_columns: Which columns to exclude in the dataset. If `None`, all
        columns are included.
        exclude_labels: Which labels to exclude in the dataset. If not `None`,
        the rows will be sampled only from rows with the specified label
        values.
        random_state: Random state to use for reproducibility.
    """

    fold_idx: int = 0
    n_splits: int = 10
    scale_cont: bool = False
    cont_scale_method: Literal["standard", "minmax", "quantile"] | None = "quantile"
    cont_scale_max_quantiles: int = 1000
    bin_cont: bool = False
    cont_bin_strat: Literal["uniform", "quantile", "kmeans", "dtree"] = "quantile"
    n_bins: int = 32
    handle_cat_unseen: Literal["most_frequent", "random", "constant"] = "most_frequent"
    cat_unseen_fill: str | None = "Unknown"
    handle_cat_missing: Literal["most_frequent", "random", "constant"] = "most_frequent"
    cat_missing_fill: str | None = "Missing"
    handle_cont_missing: Literal[
        "mean", "median", "most_frequent", "random", "constant"
    ] = "median"
    cont_missing_fill: float | None = None
    cont_strat_n_bins: int = 8
    cont_strat_bin_strat: Literal["uniform", "quantile"] = "quantile"
    task_kind: Literal["classification", "regression"] = "classification"
    sample_n_rows: int | float | None = None
    exclude_columns: list[str] | None = None
    exclude_labels: list[str] | None = None
    random_state: int = 0
