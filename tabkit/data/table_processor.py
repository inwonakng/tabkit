import json
from logging import Logger
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tabkit.config import DATA_DIR
from tabkit.utils import setup_logger

from .config import DatasetConfig, TableProcessorConfig
from .utils import (
    ColumnMetadata,
    apply_bins,
    compute_bins,
    encode_col,
    impute_col,
    load_from_disk,
    load_openml_dataset,
    load_uci_dataset,
    scale_col,
)

"""
This class handles all preprocessing related (including splitting) of the
dataset. Although labels are technically a part of the dataset, we handle them
separately because some of the preprocessing steps require labels -- e.g.
stratified spliting or heuristic-based binning. Once loaded, the data will be
cached in npy files for easy access. The configuration is hashed and the cache
is stored in a directory named after the hash. If the processor is loaded again
with the same configuration, it will load the cached data.
"""


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
            / self.dataset_config.config_name
            / self.config.config_name
        )
        self.logger = setup_logger("TableProcessor", silent=not verbose)
        self.verbose = verbose

    @property
    def is_cached(self):
        return (
            self.save_dir.exists()
            and all(
                (self.save_dir / f"{f}.npy").exists()
                for f in [
                    "train_noval_idxs",
                    "train_idxs",
                    "val_idxs",
                    "test_idxs",
                    "feature_values",
                    "feature_idxs",
                    "labels",
                ]
            )
            and (self.save_dir / "dataset_info.json").exists()
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

    def prepare_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tr_idxs: np.ndarray | None = None,
        te_idxs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if y.dtype.name == "category":
            labels = y.cat.codes.values.astype(int)
        else:
            labels = y.values.astype(float)

        """
        handles splitting the data and filtering column/labels
        """
        # if no predefined splits, do it here.
        if tr_idxs is None or te_idxs is None:
            unique_labels, unique_labels_count = np.unique(labels, return_counts=True)
            self.logger.info("No predefined split found, splitting data")
            if (
                self.config.task_kind == "classification"
                and unique_labels.shape[0] < 10
                and unique_labels_count.min() >= 10
            ):
                splitter = StratifiedKFold(
                    n_splits=10,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
                tr_idxs, te_idxs = list(splitter.split(X, labels))[self.config.fold_idx]
            else:
                splitter = KFold(
                    n_splits=10,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
                tr_idxs, te_idxs = list(splitter.split(X))[self.config.fold_idx]

        # if we want to filter by labels, do it here
        if self.config.exclude_labels:
            if not self.config.task_kind == "classification":
                raise ValueError(
                    "exclude_labels is only supported for classification tasks"
                )
            labels_filter = ~y.isin(self.config.exclude_labels)
            y = y[labels_filter].reset_index(drop=True).copy()
            X = X[labels_filter].reset_index(drop=True).copy()
            # fix labels as well
            labels = y.cat.codes.values.astype(int)
            # need to re-do the splitting
            splitter = StratifiedKFold(
                n_splits=10,
                shuffle=True,
                random_state=self.config.random_state,
            )
            tr_idxs, te_idxs = list(splitter.split(X, y))[self.config.fold_idx]
            self.logger.info("filtered by `exclude_labels`")

        # if we are doing regression, we need to bin the labels first
        stratify_labels = labels[tr_idxs]
        if self.label_info.is_cont:
            bins, _ = compute_bins(
                method=self.config.cont_strat_bin_strat,
                col=labels[tr_idxs],
                n_bins=self.config.cont_strat_n_bins,
                random_state=self.config.random_state,
            )
            stratify_labels = apply_bins(bins, labels[tr_idxs])

        # if we want to filter by columns, do it here
        if self.config.exclude_columns:
            missing_cols = [
                c for c in self.config.exclude_columns if c not in X.columns
            ]
            if len(missing_cols) > 0:
                raise ValueError(
                    "columns {} are not in the dataset!".format(missing_cols)
                )
            columns_filter = ~X.columns.isin(self.config.exclude_columns)
            X = X[X.columns[columns_filter]].reset_index(drop=True).copy()
            self.logger.info("filtered by `exclude_columns`")

        # if we want to subsample, we sample here
        if self.config.sample_n_rows:
            if self.config.sample_n_rows < 0:
                raise ValueError(f"Invalid sample_n_rows: {self.config.sample_n_rows}")
            elif self.config.sample_n_rows > 1:
                sample_n_rows = int(self.config.sample_n_rows)
            else:
                sample_n_rows = float(self.config.sample_n_rows)

            if sample_n_rows < len(tr_idxs):
                _, tr_idxs_ss = train_test_split(
                    tr_idxs,
                    random_state=self.config.random_state,
                    test_size=sample_n_rows,
                    stratify=stratify_labels,
                )
                tr_idxs = tr_idxs_ss
            tr_sub_idxs = np.arange(len(tr_idxs))
            val_sub_idxs = np.arange(len(tr_idxs))

        else:
            # if we want to use validation, split the train once more. notice
            # that we are indexing the tr_idxs by tr_sub_idxs and val_sub_idxs.
            unique_st_labels, unique_st_labels_count = np.unique(
                stratify_labels, return_counts=True
            )
            if unique_st_labels.shape[0] < 9 and unique_st_labels_count.min() >= 9:
                tr_sub_idxs, val_sub_idxs = list(
                    StratifiedKFold(
                        n_splits=9,
                        shuffle=True,
                        random_state=self.config.random_state,
                    ).split(
                        X=tr_idxs,
                        y=stratify_labels,
                    )
                )[self.config.fold_idx]
            else:
                tr_sub_idxs, val_sub_idxs = list(
                    KFold(
                        n_splits=9,
                        shuffle=True,
                        random_state=self.config.random_state,
                    ).split(
                        X=tr_idxs,
                    )
                )[self.config.fold_idx]

            # save the split idxs for later use.
            self.logger.info("Split indices using target column")
        return (
            labels,
            tr_idxs,
            tr_sub_idxs,
            val_sub_idxs,
            te_idxs,
        )

    def prepare(self, overwrite: bool = False):
        """Huge function for preprocessing the dataset. This is where multiple
        soures (openml, raw file, etc) are handled. We first load the data,
        compute split if not pre-defined, and apply any preprocessing (scale
        cont. values, encode categ. values, fill missing values. etc.)

        Args:
            overwrite: if True, will ignore cache and recompute everything.

        Returns:
            None

        Raises:
            ValueError: if types are not known or n_samples is invalid.
        """

        if self.is_cached and not overwrite:
            with open(self.save_dir / "dataset_info.json") as f:
                dataset_info = json.load(f)
                self.columns_info = [
                    ColumnMetadata.from_dict(c) for c in dataset_info["columns_info"]
                ]
                self.loadable = dataset_info["loadable"]
                self.n_samples = dataset_info["n_samples"]
                self.label_info = ColumnMetadata.from_dict(dataset_info["label_info"])
            self.logger.info(f"{self.dataset_name}: Loaded from cache")
            return self

        self.logger.info(f"{self.dataset_name}: Loading from scratch")
        # start by loading dataset details
        self.logger.info("Loaded dataset info")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # load data from source
        tr_idxs, te_idxs = None, None
        if self.dataset_config.data_source == "openml":
            X, y, tr_idxs, te_idxs = load_openml_dataset(
                task_id=(self.dataset_config.openml_task_id),
                dataset_id=(self.dataset_config.openml_dataset_id),
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
            raise ValueError(f"Unknown data source {self.dataset_config.data_source}")

        # TODO: in the future, if we are pretraining, we might want more
        # permutations of the dataset, so we should allow for dynamic target column.

        self.raw_df = X.copy()
        self.raw_df[y.name] = y.copy()

        self.columns_info = [ColumnMetadata.from_series(X[col]) for col in X.columns]
        self.label_info = ColumnMetadata.from_series(y)

        if self.label_info.is_cat or self.label_info.is_bin:
            y = y.fillna(self.config.cat_missing_fill)
            if y.dtype.name != "category":
                y = y.astype("category")
            self.label_info.mapping = y.cat.categories.astype(str).tolist()
        elif self.label_info.is_cont:
            if y.isna().any():
                raise ValueError("we can't handle continuous targets with NaN!")
            if self.config.task_kind == "classification":
                bins, value_mapping = compute_bins(
                    method=self.config.cont_bin_strat,
                    col=y,
                    n_bins=self.config.n_bins,
                    random_state=self.config.random_state,
                    y=y,
                )
                old_y = y.copy()
                y = pd.Series(apply_bins(bins=bins, col=y), dtype="category")
                # in this case, there may be cases there are empty bins. We need to remove that case.
                self.label_info.mapping = y.cat.categories.astype(str).tolist()
        else:
            raise ValueError(
                f"Invalid label kind and dtype: {self.label_info.kind}, {self.label_info.dtype}"
            )

        labels, tr_idxs, tr_sub_idxs, val_sub_idxs, te_idxs = self.prepare_splits(
            X, y, tr_idxs, te_idxs
        )

        # for classification, the labels must be consecutive integerst starting from 0.
        if self.config.task_kind == "classification":
            unique_labels = np.unique(labels)
            if (np.arange(len(unique_labels)) != unique_labels).any():
                raise ValueError(
                    "task is classification but labels are not consecutive: {} case: {}".format(
                        unique_labels.tolist(), parse_which_case
                    )
                )

        # we will be filling this up. This representation makes it much easier to handle with embedding-based models
        feature_idxs = np.zeros(X.shape, dtype=int)
        feature_values = np.ones(X.shape, dtype=float)

        # iterate over the columns and apply appropriate transformations
        for i, col_info in enumerate(self.columns_info):
            if col_info.is_cat or col_info.is_bin:
                X[col_info.name] = impute_col(
                    method=self.config.handle_cat_missing,
                    col=X[col_info.name],
                    tr_idxs=tr_sub_idxs,
                    fill_val=self.config.cat_missing_fill,
                    random_state=self.config.random_state,
                )
                # Clean up categorical features. Remove values we don't see outside of training
                fixed_col, fixed_mapping = encode_col(
                    method=self.config.handle_cat_unseen,
                    col=X[col_info.name],
                    tr_idxs=tr_sub_idxs,
                    fill_val_name=self.config.cat_unseen_fill,
                    random_state=self.config.random_state,
                )
                X[col_info.name] = fixed_col
                self.columns_info[i].mapping = fixed_mapping
                feature_idxs[:, i] = X[col_info.name]
            elif col_info.is_cont:
                # if not already, we need to make sure dtype is correct.
                X[col_info.name] = X[col_info.name].astype(col_info.dtype)
                X[col_info.name] = impute_col(
                    method=self.config.handle_cont_missing,
                    col=X[col_info.name],
                    tr_idxs=tr_sub_idxs,
                    fill_val=self.config.cont_missing_fill,
                    random_state=self.config.random_state,
                )
                if self.config.scale_cont:
                    X[col_info.name] = scale_col(
                        method=self.config.cont_scale_method,
                        col=X[col_info.name],
                        tr_idxs=tr_sub_idxs,
                        max_quantiles=self.config.cont_scale_max_quantiles,
                    )
                feature_values[:, i] = X[col_info.name]

                if self.config.bin_cont:
                    bins, value_mapping = compute_bins(
                        method=self.config.cont_bin_strat,
                        col=X[col_info.name],
                        n_bins=self.config.n_bins,
                        random_state=self.config.random_state,
                        y=y,
                    )
                    self.columns_info[i].mapping = value_mapping

                    feature_idxs[:, i] = apply_bins(
                        bins=bins,
                        col=X[col_info.name],
                    )
            elif col_info.is_date:
                feature_values[:, i] = pd.to_datetime(
                    X[col_info.name],
                    format="mixed",
                    errors="coerce",
                ).astype(int)
            else:
                pass

        self.n_samples = int(feature_idxs.shape[0])

        # save raw_df
        self.raw_df.to_csv(self.save_dir / "raw_df.csv", index=False)
        self.logger.info("Saved raw dataframe")

        dumpable = {
            "train_noval_idxs": (tr_idxs, "int"),
            "train_idxs": (tr_idxs[tr_sub_idxs], "int"),
            "val_idxs": (tr_idxs[val_sub_idxs], "int"),
            "test_idxs": (te_idxs, "int"),
            "feature_values": (feature_values, "float"),
            "feature_idxs": (feature_idxs, "int"),
            "labels": (
                labels,
                "float" if self.label_info.is_cont else "int",
            ),
        }
        self.loadable = list(dumpable.keys())
        for k, (obj, dtype) in dumpable.items():
            if dtype == "int":
                np.save(self.save_dir / f"{k}.npy", obj.astype(int))
            elif dtype == "float":
                np.save(self.save_dir / f"{k}.npy", obj.astype(float))
            elif dtype == "bool":
                np.save(self.save_dir / f"{k}.npy", obj.astype(bool))
            else:
                raise ValueError(f"Invalid dtype: {dtype}")
        self.logger.info("Dumped cache")

        dataset_info = {
            "columns_info": [c.to_dict() for c in self.columns_info],
            "loadable": self.loadable,
            "label_info": self.label_info.to_dict(),
            "n_samples": self.n_samples,
        }
        with open(self.save_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f)
        self.logger.info("Saved dataset info")

        return self

    def get(
        self,
        key: str,
    ) -> np.ndarray | pd.DataFrame:
        if not self.is_cached:
            raise ValueError("Data must be processed first by calling .prepare()")
        if key in ["df", "dataframe"]:
            return pd.read_csv(self.save_dir / "raw_df.csv")
        if key not in self.loadable:
            raise ValueError(f"Invalid key: {key}")
        val = np.load(self.save_dir / f"{key}.npy")
        return val

    def reset(self):
        if self.save_dir.exists():
            self.logger.info("Resetting cache...")
            shutil.rmtree(self.save_dir)
        return self

    def get_split(
        self,
        split: Literal["train", "train_noval", "val", "test", "all"] = "all",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads the train/val/test split in numpy arrays.

        Args:
            split: Name of the split to load. `train_noval` refers
            to the training split including the validation split.

        Returns:
            a tuple of `np.ndarray`s for the features and target if `kind` is `numpy`,

        Raises:
            ValueError if `split` is not one of `[None, "train", "train_noval", "val", "test"]`.
        """
        if not self.is_cached:
            raise ValueError("Data must be processed first by calling .prepare()")
        if split and split not in ["train", "train_noval", "val", "test", "all"]:
            raise ValueError(f"Invalid split: {split}")
        if split == "all":
            split_idxs = np.arange(self.n_samples)
        else:
            split_idxs = self.get(f"{split}_idxs")
        feature_val = self.get("feature_values")
        feature_idx = self.get("feature_idxs")
        X = feature_idx
        if not self.config.bin_cont:
            X = X.astype(float)
            cont_cols = [c.is_cont for c in self.columns_info]
            X[:, cont_cols] = feature_val[:, cont_cols]
        X = X[split_idxs]
        y = self.get("labels")[split_idxs]
        return X, y
