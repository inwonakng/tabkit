import hashlib
import json
from dataclasses import asdict
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabkit.config import DATA_DIR
from tabkit.utils import setup_logger

from .column_metadata import ColumnMetadata, is_column_categorical
from .data_config import (
    DatasetConfig,
    TableProcessorConfig,
    get_default_label_pipeline_clf,
    get_default_label_pipeline_reg,
)
from .transforms import TRANSFORM_MAP, BaseTransform
from .utils import load_from_disk, load_openml_dataset, load_uci_dataset


def _parse_config(obj: Any, dataclass_type: type, config_name: str) -> Any:
    """Convert input to dataclass instance with robust type checking.

    Args:
        obj: Input object to convert
        dataclass_type: Target dataclass type
        config_name: Name of config (for error messages)

    Returns:
        Instance of dataclass_type

    Raises:
        TypeError: If object cannot be converted to the dataclass type
    """
    from dataclasses import is_dataclass

    # Case 1: Already the correct dataclass type
    if isinstance(obj, dataclass_type):
        return obj

    # Case 2: Dictionary - convert directly
    if isinstance(obj, dict):
        return dataclass_type(**obj)

    # Case 3: Try to convert to dict (handles other dataclasses or dict-like objects)
    try:
        # Try asdict for dataclasses
        if is_dataclass(obj):
            obj_dict = asdict(obj)
        # Try __dict__ for regular objects
        elif hasattr(obj, '__dict__'):
            obj_dict = obj.__dict__
        else:
            raise TypeError(f"Cannot convert {type(obj)} to dict")

        return dataclass_type(**obj_dict)
    except Exception as e:
        raise TypeError(
            f"{config_name} must be a {dataclass_type.__name__} instance, "
            f"a dict, or convertible to dict. Got {type(obj)}. Error: {e}"
        )


def compute_config_hash(config_dict: dict, truncate: int = 16) -> str:
    """
    Compute a deterministic hash from config dictionary for cache directory naming.
    Excludes 'config_name' and 'dataset_name' as these are used for readable naming.
    """
    # Remove metadata fields that shouldn't affect the hash
    hashable_config = {k: v for k, v in config_dict.items() if v is not None}
    # Canonical JSON representation
    canonical_json = json.dumps(hashable_config, sort_keys=True, separators=(",", ":"))
    hash_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return hash_digest[:truncate]


class TableProcessor:
    """
    Preprocesses and splits tabular datasets with automatic caching.

    TableProcessor handles loading, preprocessing, and splitting tabular data into
    train/validation/test sets. All preprocessing is cached based on configuration
    hash for reproducibility and performance.

    Args:
        dataset_config: Configuration for data loading. Use DatasetConfig class or dict.
        config: Configuration for preprocessing and splitting. Use TableProcessorConfig class or dict.
        verbose: Whether to print logging information.

    Configuration:
        For detailed configuration options, see the DatasetConfig and TableProcessorConfig
        classes in `tabkit.data`. Both type-safe config classes and plain dictionaries
        are supported and work identically.

    Example:
        >>> from tabkit import TableProcessor, DatasetConfig, TableProcessorConfig
        >>>
        >>> # Using config classes (recommended - provides type hints and autocomplete)
        >>> dataset_cfg = DatasetConfig(
        ...     data_source="disk",
        ...     file_path="data.csv",
        ...     file_type="csv",
        ...     label_col="target"
        ... )
        >>> processor_cfg = TableProcessorConfig(
        ...     task_kind="classification",
        ...     test_ratio=0.2,
        ...     val_ratio=0.1
        ... )
        >>> processor = TableProcessor(dataset_config=dataset_cfg, config=processor_cfg)
        >>>
        >>> # Or using plain dictionaries (also supported)
        >>> processor = TableProcessor(
        ...     dataset_config={"data_source": "disk", "file_path": "data.csv", "label_col": "target"},
        ...     config={"test_ratio": 0.2, "val_ratio": 0.1}
        ... )
        >>>
        >>> # Prepare data (caches for future runs)
        >>> processor.prepare()
        >>>
        >>> # Get splits
        >>> X_train, y_train = processor.get_split("train")
        >>> X_val, y_val = processor.get_split("val")
        >>> X_test, y_test = processor.get_split("test")

    Attributes:
        config (TableProcessorConfig): Processor configuration
        dataset_config (DatasetConfig): Dataset configuration
        columns_info (list[ColumnMetadata]): Metadata for each feature column
        label_info (ColumnMetadata): Metadata for label column
        save_dir (Path): Cache directory for processed data
        n_samples (int): Total number of samples in dataset
    """

    config: TableProcessorConfig
    dataset_config: DatasetConfig
    dataset_name: str
    save_dir: Path
    logger: Logger

    columns_info: list[ColumnMetadata]
    loadable: list[str]
    label_info: ColumnMetadata
    n_samples: int

    def __init__(
        self,
        dataset_config: dict | Any,
        config: dict | Any | None = None,
        verbose: bool = False,
    ):
        # Convert input to dataclass instance
        self.dataset_config = _parse_config(
            dataset_config, DatasetConfig, "dataset_config"
        )

        if config is None:
            self.config = TableProcessorConfig()
        else:
            self.config = _parse_config(config, TableProcessorConfig, "config")

        # Handle conditional defaults based on task_kind
        if self.config.label_pipeline is None:
            if self.config.task_kind == "classification":
                self.config.label_pipeline = get_default_label_pipeline_clf()
            else:
                self.config.label_pipeline = get_default_label_pipeline_reg()

        # Default stratification pipeline for classification if not provided
        if self.config.label_stratify_pipeline is None:
            if self.config.task_kind == "classification":
                self.config.label_stratify_pipeline = (
                    get_default_label_pipeline_clf()
                )
            else:
                self.config.label_stratify_pipeline = (
                    get_default_label_pipeline_reg()
                )

        # Extract dataset name
        self.dataset_name = self.dataset_config.dataset_name

        # Compute cache directory using hash
        dataset_hash = compute_config_hash(asdict(self.dataset_config))
        config_hash = compute_config_hash(asdict(self.config))

        self.save_dir = DATA_DIR / "data" / dataset_hash / config_hash

        self.logger = setup_logger("TableProcessor", silent=not verbose)
        self.verbose = verbose

    def _instantiate_pipeline(self, config_list) -> list[BaseTransform]:
        pipeline = []
        for step_config in config_list:
            class_name = step_config["class"]
            params = step_config.get("params", {})
            if class_name not in TRANSFORM_MAP:
                raise ValueError(
                    f"Unknown transform class: '{class_name}'. "
                    "Did you forget to register it with register_transform()?"
                )
            pipeline.append(TRANSFORM_MAP[class_name](**params))
        return pipeline

    @property
    def is_cached(self):
        return (
            self.save_dir.exists()
            and (self.save_dir / "dataset_info.json").exists()
            and (self.save_dir / "pipeline.joblib").exists()
            and (self.save_dir / "label_pipeline.joblib").exists()
            and (self.save_dir / "train.parquet").exists()
            and (self.save_dir / "val.parquet").exists()
            and (self.save_dir / "test.parquet").exists()
            and (self.save_dir / "train_idxs.npy").exists()
            and (self.save_dir / "val_idxs.npy").exists()
            and (self.save_dir / "test_idxs.npy").exists()
        )

    @property
    def n_cols(self) -> int:
        return len(self.columns_info)

    @property
    def cat_idx(self) -> list[int]:
        return [
            i for i, c in enumerate(self.columns_info) if c["kind"] == "categorical"
        ]

    @property
    def cont_idx(self) -> list[int]:
        return [i for i, c in enumerate(self.columns_info) if c["kind"] == "continuous"]

    @property
    def col_names(self) -> list[str]:
        return [c["name"] for c in self.columns_info]

    @property
    def col_shapes(self) -> list[int]:
        return [len(c.mapping) if c.is_cont else 1 for c in self.columns_info]

    def _try_stratified_split(
        self,
        X: np.ndarray,
        n_splits: int,
        stratify_target: np.ndarray,
        random_state: int,
        fold_idx: int,
    ):
        if fold_idx >= n_splits:
            raise ValueError(
                f"fold_idx={fold_idx} must be less than n_splits={n_splits}"
            )

        unique_labels, unique_labels_count = np.unique(
            stratify_target, return_counts=True
        )
        if unique_labels.shape[0] < n_splits and unique_labels_count.min() >= n_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            tr_idxs, te_idxs = list(splitter.split(X, stratify_target))[
                fold_idx
            ]
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            tr_idxs, te_idxs = list(splitter.split(X))[fold_idx]
        return tr_idxs, te_idxs

    def _prepare_split_target(
        self,
        y: pd.Series,
        label_info: ColumnMetadata,
        label_stratify_pipeline: list[dict[str, Any]] | None = None,
    ) -> pd.Series:
        labels = y.copy()
        if label_stratify_pipeline is not None:
            label_pipeline = self._instantiate_pipeline(label_stratify_pipeline)
            for t in label_pipeline:
                labels = t.fit_transform(
                    X=labels.to_frame(), metadata=[label_info]
                ).iloc[:, 0]
        return labels

    def _get_splits_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tr_idxs: np.ndarray | None = None,
        te_idxs: np.ndarray | None = None,
        random_state: int = 0,
        n_splits: int = 10,
        fold_idx: int = 0,
        n_val_splits: int = 9,
        split_validation: bool = True,
        sample_n_rows: int | float | None = None,
        val_fold_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        K-fold based splitting for robust cross-validation.

        This method uses K-fold cross-validation to split data into train/val/test sets.
        By varying fold_idx across different runs, you can ensure every sample appears
        in the test set exactly once, enabling comprehensive model evaluation.

        Args:
            X: Feature data
            y: Stratification target (preprocessed labels for splitting)
            tr_idxs: Predefined training indices (optional, from data source)
            te_idxs: Predefined test indices (optional, from data source)
            random_state: Random seed
            n_splits: Number of folds for train/test split
            fold_idx: Which fold to use as test set
            n_val_splits: Number of folds for train/val split
            split_validation: Whether to create validation split
            sample_n_rows: Optional subsampling of training data
            val_fold_idx: Which fold to use as validation set

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # if no predefined splits, do it here.
        if tr_idxs is None or te_idxs is None:
            self.logger.info("No predefined split found, using K-fold splitting")
            tr_idxs, te_idxs = self._try_stratified_split(
                X=X,
                n_splits=n_splits,
                stratify_target=y,
                random_state=random_state,
                fold_idx=fold_idx,
            )

        if sample_n_rows is not None:
            tr_idxs = self._subsample_data(
                tr_idxs=tr_idxs,
                sample_n_rows=sample_n_rows,
                stratify_target=y,
                random_state=random_state,
            )
            self.logger.info("subsampled by `sample_n_rows`")

        if split_validation:
            tr_sub_idxs, val_sub_idxs = self._try_stratified_split(
                X=tr_idxs,
                n_splits=n_val_splits,
                stratify_target=y[tr_idxs],
                random_state=random_state,
                fold_idx=val_fold_idx,
            )
        else:
            tr_sub_idxs = np.arange(len(tr_idxs))
            val_sub_idxs = np.arange(len(tr_idxs))

        self.logger.info("K-fold split complete")
        return (
            tr_idxs[tr_sub_idxs],
            tr_idxs[val_sub_idxs],
            te_idxs,
        )

    def _get_splits_ratio(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float,
        val_ratio: float,
        random_state: int = 0,
        sample_n_rows: int | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ratio-based splitting for simple train/val/test splits.

        This method creates a simple random split based on specified percentages.
        It's faster and more intuitive than K-fold, but doesn't provide full
        dataset coverage across different runs.

        Args:
            X: Feature data
            y: Stratification target (preprocessed labels for splitting)
            test_ratio: Fraction of data to use for test (e.g., 0.2 for 20%)
            val_ratio: Fraction of data to use for validation (e.g., 0.1 for 10%)
            random_state: Random seed
            sample_n_rows: Optional subsampling of training data

        Returns:
            Tuple of (train_indices, val_indices, test_indices)

        Raises:
            ValueError: If test_ratio + val_ratio >= 1.0
        """
        if test_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"test_ratio ({test_ratio}) + val_ratio ({val_ratio}) must be < 1.0"
            )

        n_samples = len(X)
        all_idxs = np.arange(n_samples)

        self.logger.info(
            f"Using ratio-based split: test={test_ratio:.1%}, val={val_ratio:.1%}, "
            f"train={1 - test_ratio - val_ratio:.1%}"
        )

        # First split: separate test set
        try:
            train_val_idxs, test_idxs = train_test_split(
                all_idxs,
                test_size=test_ratio,
                stratify=y,
                random_state=random_state,
            )
        except ValueError as e:
            # Fallback to non-stratified if stratification fails (e.g., too few samples per class)
            self.logger.warning(
                f"Stratified split failed: {e}. Using non-stratified split."
            )
            train_val_idxs, test_idxs = train_test_split(
                all_idxs,
                test_size=test_ratio,
                random_state=random_state,
            )

        # Second split: separate validation from training
        val_ratio_adjusted = val_ratio / (1 - test_ratio)  # Adjust for remaining data

        try:
            train_idxs, val_idxs = train_test_split(
                train_val_idxs,
                test_size=val_ratio_adjusted,
                stratify=y[train_val_idxs],
                random_state=random_state,
            )
        except ValueError as e:
            self.logger.warning(
                f"Stratified split failed: {e}. Using non-stratified split."
            )
            train_idxs, val_idxs = train_test_split(
                train_val_idxs,
                test_size=val_ratio_adjusted,
                random_state=random_state,
            )

        # Optional subsampling
        if sample_n_rows is not None:
            train_idxs = self._subsample_data(
                tr_idxs=train_idxs,
                sample_n_rows=sample_n_rows,
                stratify_target=y,
                random_state=random_state,
            )
            self.logger.info("subsampled by `sample_n_rows`")

        self.logger.info("Ratio-based split complete")
        return train_idxs, val_idxs, test_idxs

    def _get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tr_idxs: np.ndarray | None = None,
        te_idxs: np.ndarray | None = None,
        random_state: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Router method that selects between K-fold and ratio-based splitting.

        PRIORITY: If test_ratio and val_ratio are set in config, uses ratio-based splitting.
                  Otherwise, uses K-fold based splitting.

        For predefined splits (tr_idxs, te_idxs provided by data source):
        - If using ratio mode: Only val_ratio is used to split the predefined train portion
        - If using kfold mode: Only n_val_splits is used to split the predefined train portion

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Check which mode to use
        test_ratio = self.config.test_ratio
        val_ratio = self.config.val_ratio
        use_ratio_mode = test_ratio is not None and val_ratio is not None

        # Handle predefined splits from data source (e.g., OpenML, UCI)
        if tr_idxs is not None and te_idxs is not None:
            self.logger.info("Using predefined train/test split from data source")

            # Only split the predefined training portion into train/val
            if use_ratio_mode:
                # Use ratio to split train into train+val
                val_ratio_adjusted = val_ratio / (1 - test_ratio)
                try:
                    train_idxs, val_idxs = train_test_split(
                        tr_idxs,
                        test_size=val_ratio_adjusted,
                        stratify=y[tr_idxs],
                        random_state=random_state,
                    )
                except ValueError:
                    train_idxs, val_idxs = train_test_split(
                        tr_idxs,
                        test_size=val_ratio_adjusted,
                        random_state=random_state,
                    )
            else:
                # Use K-fold to split train into train+val
                if self.config.split_validation:
                    tr_sub_idxs, val_sub_idxs = self._try_stratified_split(
                        X=tr_idxs,
                        n_splits=self.config.n_val_splits,
                        stratify_target=y[tr_idxs],
                        random_state=random_state,
                        fold_idx=self.config.val_fold_idx,
                    )
                    train_idxs = tr_idxs[tr_sub_idxs]
                    val_idxs = tr_idxs[val_sub_idxs]
                else:
                    train_idxs = tr_idxs
                    val_idxs = tr_idxs

            return train_idxs, val_idxs, te_idxs

        # No predefined splits - use selected mode
        if use_ratio_mode:
            return self._get_splits_ratio(
                X=X,
                y=y,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                random_state=random_state,
                sample_n_rows=self.config.sample_n_rows,
            )
        else:
            return self._get_splits_kfold(
                X=X,
                y=y,
                tr_idxs=None,
                te_idxs=None,
                random_state=random_state,
                n_splits=self.config.n_splits,
                fold_idx=self.config.fold_idx,
                n_val_splits=self.config.n_val_splits,
                split_validation=self.config.split_validation,
                sample_n_rows=self.config.sample_n_rows,
                val_fold_idx=self.config.val_fold_idx,
            )

    def _load_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        tr_idxs, te_idxs = None, None
        if self.dataset_config.data_source == "openml":
            X, y, tr_idxs, te_idxs = load_openml_dataset(
                task_id=self.dataset_config.openml_task_id,
                dataset_id=self.dataset_config.openml_dataset_id,
                fold_idx=self.dataset_config.openml_fold_idx,
                random_state=self.config.random_state,
            )
            self.logger.info("Loaded openml data")
        elif self.dataset_config.data_source == "uci":
            X, y = load_uci_dataset(dataset_id=self.dataset_config.uci_dataset_id)
        elif self.dataset_config.data_source == "automm":
            X, y, tr_idxs, te_idxs = load_automm_dataset(
                dataset_id=self.dataset_config.automm_dataset_id,
            )
        elif self.dataset_config.data_source == "disk":
            X, y, tr_idxs, te_idxs = load_from_disk(
                file_path=self.dataset_config.file_path,
                file_type=self.dataset_config.file_type,
                label_col=self.dataset_config.label_col,
                split_file_path=self.dataset_config.split_file_path,
            )
        else:
            raise ValueError(
                f"Unknown data source {self.dataset_config.data_source}"
            )
        return X, y, tr_idxs, te_idxs

    def _filter_labels(
        self, X: pd.DataFrame, y: pd.Series, exclude_labels: list[str]
    ) -> tuple[pd.DataFrame, pd.Series]:
        X = X[~y.isin(exclude_labels)].reset_index(drop=True).copy()
        y = y[~y.isin(exclude_labels)].reset_index(drop=True).copy()
        return X, y

    def _filter_columns(
        self, X: pd.DataFrame, exclude_columns: list[str]
    ) -> pd.DataFrame:
        missing_cols = [c for c in exclude_columns if c not in X.columns]
        if len(missing_cols) > 0:
            raise ValueError("columns {} are not in the dataset!".format(missing_cols))
        columns_filter = ~X.columns.isin(exclude_columns)
        X = X[X.columns[columns_filter]].reset_index(drop=True).copy()
        return X

    def _subsample_data(
        self,
        tr_idxs: np.ndarray,
        sample_n_rows: int | float,
        stratify_target: pd.Series | None = None,
        random_state: int = 0,
    ) -> np.ndarray:
        if sample_n_rows < 0:
            raise ValueError(f"Invalid sample_n_rows: {sample_n_rows}")
        elif sample_n_rows > 1:
            sample_n_rows = int(sample_n_rows)
        else:
            sample_n_rows = float(sample_n_rows)
        sampled = tr_idxs
        if sample_n_rows < len(tr_idxs):
            _, sampled = train_test_split(
                tr_idxs,
                random_state=random_state,
                test_size=sample_n_rows,
                stratify=stratify_target[tr_idxs],
            )
        return sampled

    def prepare(self, overwrite: bool = False) -> "TableProcessor":
        if self.is_cached and not overwrite:
            # self.logger.info("Loading from cache.")
            self.pipeline = joblib.load(self.save_dir / "pipeline.joblib")
            self.label_pipeline = joblib.load(self.save_dir / "label_pipeline.joblib")
            with open(self.save_dir / "dataset_info.json") as f:
                dataset_info = json.load(f)
            self.columns_info = [
                ColumnMetadata.from_dict(c) for c in dataset_info["columns_info"]
            ]
            self.label_info = ColumnMetadata.from_dict(dataset_info["label_info"])
            self.n_samples = dataset_info["n_samples"]
            return self

        self.logger.info("Preparing data processor for dataset: %s", self.dataset_name)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        X, y, tr_idxs, te_idxs = self._load_data()
        if (
            self.config.exclude_labels
            and self.config.task_kind == "classification"
        ):
            X, y = self._filter_labels(X, y, self.config.exclude_labels)
            self.logger.info("filtered by `exclude_labels`")
        if self.config.exclude_columns:
            X = self._filter_columns(X, self.config.exclude_columns)
            self.logger.info("filtered by `exclude_columns`")

        # if the task is classification and the label is continuous, discretize it.
        if (
            self.config.task_kind == "classification"
            and not is_column_categorical(y)
            and not self.config.label_pipeline
        ):
            self.logger.info(
                "Continuous label detected for classification task. "
                "Applying default quantile discretization."
            )
            self.config.label_pipeline = [
                {
                    "class": "Discretize",
                    "params": {"method": "quantile", "n_bins": 4},
                }
            ]

        # preliminary metadata. this will change as we apply transforms
        columns_info = [ColumnMetadata.from_series(X[col]) for col in X.columns]
        label_info = ColumnMetadata.from_series(y)

        stratify_target = self._prepare_split_target(
            y=y,
            label_info=label_info,
            label_stratify_pipeline=self.config.label_stratify_pipeline,
        )

        train_idx, val_idx, test_idx = self._get_splits(
            X=X,
            y=stratify_target,
            tr_idxs=tr_idxs,
            te_idxs=te_idxs,
            random_state=self.config.random_state,
        )

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        self.pipeline = self._instantiate_pipeline(self.config.pipeline)
        self.label_pipeline = self._instantiate_pipeline(self.config.label_pipeline)

        self.logger.info("Fitting pipeline...")
        for transform in self.pipeline:
            X_train = transform.fit_transform(
                X=X_train,
                y=y_train,
                metadata=columns_info,
                random_state=self.config.random_state,
            )
            X_val = transform.transform(X=X_val)
            X_test = transform.transform(X=X_test)
            columns_info = transform.update_metadata(
                X_new=X_train,
                metadata=columns_info,
            )

        # same deal with labels
        for transform in self.label_pipeline:
            y_train = transform.fit_transform(
                X=y_train.to_frame(),
                y=None,
                metadata=[label_info],
                random_state=self.config.random_state,
            ).iloc[:, 0]
            y_val = transform.transform(y_val.to_frame()).iloc[:, 0]
            y_test = transform.transform(y_test.to_frame()).iloc[:, 0]
            label_info = transform.update_metadata(
                X_new=y_train.to_frame(),
                metadata=[label_info],
            )[0]

        self.columns_info = columns_info
        self.label_info = label_info
        self.n_samples = len(X)

        self.logger.info("Saving processed data and pipeline to cache...")
        X_train[y.name] = y_train
        X_train.to_parquet(self.save_dir / "train.parquet")
        X_val[y.name] = y_val
        X_val.to_parquet(self.save_dir / "val.parquet")
        X_test[y.name] = y_test
        X_test.to_parquet(self.save_dir / "test.parquet")

        # also save the indices
        np.save(self.save_dir / "train_idxs.npy", train_idx)
        np.save(self.save_dir / "val_idxs.npy", val_idx)
        np.save(self.save_dir / "test_idxs.npy", test_idx)

        joblib.dump(self.pipeline, self.save_dir / "pipeline.joblib")
        joblib.dump(self.label_pipeline, self.save_dir / "label_pipeline.joblib")
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # save the raw df and the indices as well.
        df = X.copy()
        df[y.name] = y

        df.to_parquet(self.save_dir / "raw_df.parquet", index=False)

        dataset_info = {
            "columns_info": [c.to_dict() for c in self.columns_info],
            "label_info": self.label_info.to_dict(),
            "n_samples": self.n_samples,
        }
        with open(self.save_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f)

        self.logger.info("Done.")

        return self

    def get_split(
        self, split: Literal["all", "train", "val", "test"] = "all"
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not self.is_cached:
            raise RuntimeError(
                "Processor has not been prepared. Call .prepare() first."
            )
        if split in ["train", "val", "test"]:
            df = pd.read_parquet(self.save_dir / f"{split}.parquet")
        else:
            df_tr = pd.read_parquet(self.save_dir / "train.parquet")
            df_val = pd.read_parquet(self.save_dir / "val.parquet")
            df_te = pd.read_parquet(self.save_dir / "test.parquet")
            df = pd.concat([df_tr, df_val, df_te], ignore_index=True).reset_index(
                drop=True
            )
        y = df[self.label_info.name].copy()
        X = df.drop(columns=[self.label_info.name]).copy()
        return X, y

    def get(self, key: str) -> Any:
        if not self.is_cached:
            raise RuntimeError(
                "Processor has not been prepared. Call .prepare() first."
            )
        # first check if the file exists
        candidates = sorted(self.save_dir.glob(f"{key}.*"))
        if len(candidates) == 0:
            raise ValueError(f"Key {key} not found in cache.")
        if len(candidates) > 1:
            raise ValueError(f"Multiple files found for key {key}: {candidates}")
        file_path = candidates[0]
        if file_path.suffix == ".npy":
            return np.load(file_path)
        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".joblib":
            return joblib.load(file_path)
        elif file_path.suffix == ".json":
            with open(file_path) as f:
                return json.load(f)
        else:
            raise ValueError(
                f"Unsupported file format for key {key}: {file_path.suffix}"
            )

    def get_dataframe(self) -> pd.DataFrame:
        return self.get_df()

    def get_df(self) -> pd.DataFrame:
        return self.get("raw_df")
