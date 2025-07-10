import json
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabkit.config import DATA_DIR
from tabkit.utils import setup_logger

from .column_metadata import ColumnMetadata
from .config import DatasetConfig, TableProcessorConfig
from .transforms import TRANSFORM_MAP, BaseTransform
from .utils import load_from_disk, load_openml_dataset, load_uci_dataset


class TableProcessor:
    config: TableProcessorConfig
    dataset_name: str
    save_dir: Path
    logger: Logger

    columns_info: list[ColumnMetadata]
    loadable: list[str]
    label_info: ColumnMetadata
    n_samples: int

    def __init__(
        self,
        dataset_config: DatasetConfig,
        config: TableProcessorConfig | None = None,
        verbose: bool = False,
    ):
        if config is None:
            config = TableProcessorConfig()
        self.config = config
        self.dataset_config = dataset_config
        self.dataset_name = dataset_config.dataset_name
        self.save_dir = (
            DATA_DIR
            / "data"
            / self.dataset_config.get_unique_name()
            / self.config.get_unique_name()
        )
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
    def col_names(self) -> str:
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
        split_idx: int,
    ):
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
                self.config.split_idx
            ]
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            tr_idxs, te_idxs = list(splitter.split(X))[split_idx]
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

    def _get_splits(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        tr_idxs: np.ndarray | None = None,
        te_idxs: np.ndarray | None = None,
        label_info: ColumnMetadata = None,
        random_state: int = 0,
        n_splits: int = 10,
        split_idx: int = 0,
        n_val_splits: int = 9,
        val_split_idx: int = 0,
        label_stratify_pipeline: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        handles splitting the data and filtering column/labels
        """
        # if no predefined splits, do it here.
        if tr_idxs is None or te_idxs is None:
            unique_labels, unique_labels_count = np.unique(labels, return_counts=True)
            self.logger.info("No predefined split found, splitting data")
            tr_idxs, te_idxs = self._try_stratified_split(
                X=X,
                n_splits=n_splits,
                stratify_target=labels,
                random_state=random_state,
                split_idx=split_idx,
            )

        tr_sub_idxs, val_sub_idxs = self._try_stratified_split(
            X=tr_idxs,
            n_splits=n_val_splits,
            stratify_target=labels[tr_idxs],
            random_state=random_state,
            split_idx=val_split_idx,
        )

        self.logger.info("Split indices using target column")
        return (
            tr_idxs[tr_sub_idxs],
            tr_idxs[val_sub_idxs],
            te_idxs,
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
                task_id=(self.dataset_config.openml_task_id),
                dataset_id=(self.dataset_config.openml_dataset_id),
                split_idx=self.dataset_config.openml_split_idx,
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
            raise ValueError(f"Unknown data source {self.dataset_config.data_source}")
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
        missing_cols = [c for c in self.config.exclude_columns if c not in X.columns]
        if len(missing_cols) > 0:
            raise ValueError("columns {} are not in the dataset!".format(missing_cols))
        columns_filter = ~X.columns.isin(self.config.exclude_columns)
        X = X[X.columns[columns_filter]].reset_index(drop=True).copy()
        return X

    def _subsample_data(
        self,
        tr_idxs: np.ndarray,
        sample_n_rows: int | float,
        stratify_target: pd.Series | None = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series]:
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

    def prepare(self, overwrite: bool = False):
        if self.is_cached and not overwrite:
            print("Loading from cache.")
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

        self.save_dir.mkdir(parents=True, exist_ok=True)

        X, y, tr_idxs, te_idxs = self._load_data()
        if self.config.exclude_labels and self.config.task_kind == "classification":
            X, y = self._filter_labels(X, y, self.config.exclude_labels)
            self.logger.info("filtered by `exclude_labels`")
        if self.config.exclude_columns:
            X = self._filter_columns(X, self.config.exclude_columns)
            self.logger.info("filtered by `exclude_columns`")

        # preliminary metadata. this will change as we apply transforms
        columns_info = [ColumnMetadata.from_series(X[col]) for col in X.columns]
        label_info = ColumnMetadata.from_series(y)

        startify_target = self._prepare_split_target(
            y=y,
            label_info=label_info,
            label_stratify_pipeline=self.config.label_stratify_pipeline,
        )

        if self.config.sample_n_rows is not None:
            tr_idxs = self._subsample_data(
                tr_idxs=tr_idxs,
                sample_n_rows=self.config.sample_n_rows,
                stratify_target=y,
                random_state=self.config.random_state,
            )
            self.logger.info("subsampled by `sample_n_rows`")

        if self.config.split_validation:
            train_idx, val_idx, test_idx = self._get_splits(
                X=X,
                labels=startify_target,
                tr_idxs=tr_idxs,
                te_idxs=te_idxs,
                label_info=label_info,
                random_state=self.config.random_state,
                n_splits=self.config.n_splits,
                split_idx=self.config.split_idx,
                n_val_splits=self.config.n_val_splits,
                val_split_idx=self.config.val_split_idx,
                label_stratify_pipeline=self.config.label_stratify_pipeline,
            )
        else:
            train_idx = tr_idxs
            val_idx = tr_idxs
            test_idx = te_idxs

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
            json.dump(self.config.to_dict(), f, indent=2)

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
