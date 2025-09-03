from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .column_metadata import ColumnMetadata
from .compute_bins import compute_bins


@dataclass
class BaseTransform(ABC):
    """Abstract base class for all preprocessing transforms."""

    def fit(
        self,
        X: pd.DataFrame,
        **kwargs,
    ):
        """Fit the transform on the training data. Should store state in attributes with a trailing underscore."""
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform."""
        raise NotImplementedError

    def update_metadata(
        self, X_new: pd.DataFrame, metadata: list[ColumnMetadata], **kwargs
    ) -> list[ColumnMetadata]:
        return metadata

    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(X, **kwargs)
        return self.transform(X)

    def to_dict(self) -> dict:
        """Serialize the transform's configuration for hashing. assumes the name is registered"""
        return {"class": self.__class__.__name__, "params": asdict(self)}


@dataclass
class Impute(BaseTransform):
    method: str
    fill_value: Any | None = None

    def fit(
        self,
        X: pd.DataFrame,
        *,
        y: pd.Series = None,
        random_state: int | None = None,
        **kwargs,
    ):
        self.imputation_values_ = {}
        for c in X.columns:
            if not X[c].isna().any():
                continue

            if self.method in ["mean", "median"] and not pd.api.types.is_numeric_dtype(
                X[c]
            ):
                continue
            if self.method == "constant":
                self.imputation_values_[c] = self.fill_value
            elif self.method == "most_frequent":
                self.imputation_values_[c] = X[c].mode().iloc[0]
            elif self.method == "mean":
                self.imputation_values_[c] = X[c].mean()
            elif self.method == "median":
                self.imputation_values_[c] = X[c].median()  # Example
            elif self.method == "random":
                self.imputation_values_[c] = (
                    X[c].dropna().sample(1, random_state=random_state).iloc[0]
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for c in X.columns:
            if c not in self.imputation_values_:
                continue
            X_new[c] = X_new[c].fillna(self.imputation_values_.get(c))
        return X_new


@dataclass
class Scale(BaseTransform):
    method: str

    def fit(
        self,
        X: pd.DataFrame,
        *,
        metadata: list[ColumnMetadata],
        y: pd.Series = None,
        **kwargs,
    ):
        self.scalers_ = {}
        self.cont_cols_ = [m.name for m in metadata if m.kind == "continuous"]

        if not self.cont_cols_:
            return self

        if self.method == "standard":
            from sklearn.preprocessing import StandardScaler

            self.scaler_ = StandardScaler()
        elif self.method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler_ = MinMaxScaler()
        elif self.method == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            self.scaler_ = QuantileTransformer(n_quantiles=min(1000, len(X)))
        else:
            raise ValueError(f"Unknown scaler method: {self.method}")

        self.scaler_.fit(X[self.cont_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cont_cols_:
            return X

        X_new = X.copy()
        X_new[self.cont_cols_] = self.scaler_.transform(X[self.cont_cols_])
        return X_new


@dataclass
class Discretize(BaseTransform):
    method: str
    n_bins: int
    # Supervised params
    is_task_regression: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        *,
        metadata: list[ColumnMetadata],
        y: pd.Series = None,
        random_state: int | None = None,
        **kwargs,
    ):
        self.bins_ = {}
        for i, col in enumerate(X.columns):
            if metadata[i].kind != "continuous":
                continue
            # Using your original compute_bins function
            bins, _ = compute_bins(
                method=self.method,
                col=X[col],
                n_bins=self.n_bins,
                y=y,
                is_task_regression=self.is_task_regression,
                random_state=random_state,
            )
            self.bins_[col] = bins
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for c in X.columns:
            if c not in self.bins_:
                continue
            X_new[c] = np.clip(
                np.digitize(X_new[c], self.bins_[c]) - 1,
                0,
                len(self.bins_[c]) - 2,
            )
        return X_new

    def update_metadata(
        self,
        X_new: pd.DataFrame,
        metadata: list[ColumnMetadata],
    ) -> list[ColumnMetadata]:
        """Change the 'kind' and 'mapping' for binned columns."""
        new_metadata = []
        for i, col in enumerate(X_new.columns):
            updated_meta = deepcopy(metadata[i])
            if col in self.bins_:
                bins = self.bins_[col]
                updated_meta.kind = "categorical"
                updated_meta.mapping = [
                    f"[{bins[j]:.4f}, {bins[j + 1]:.4f})" for j in range(len(bins) - 1)
                ]
            new_metadata.append(updated_meta)
        return new_metadata


@dataclass
class Encode(BaseTransform):
    method: str
    fill_val_name: str | None = None

    def fit(
        self,
        X: pd.DataFrame,
        *,
        metadata: list[ColumnMetadata],
        y: pd.Series = None,
        random_state: int | None = None,
        **kwargs,
    ):
        self.encodings_ = {}
        for i, col in enumerate(X.columns):
            if metadata[i].kind not in ["categorical", "binary"]:
                continue
            uniq_tr_val = sorted(X[col].unique().tolist())
            tr_only_mapping = {v: k for k, v in enumerate(uniq_tr_val)}
            if self.method == "constant":
                fill_unseen_val = len(uniq_tr_val)
                uniq_tr_val.append(self.fill_val_name)
            elif self.method in ["most_frequent", "mode"]:
                fill_unseen_val = tr_only_mapping[X[col].mode().iloc[0]]
            elif self.method == "random":
                fill_unseen_val = tr_only_mapping[
                    X[col].sample(1, random_state=random_state).iloc[0]
                ]
            else:
                raise ValueError(f"Encode method [{self.method}] not found")
            self.encodings_[col] = (tr_only_mapping, fill_unseen_val)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for col in X.columns:
            if col not in self.encodings_:
                continue
            mapping, fill_unseen_val = self.encodings_[col]
            X_new[col] = X_new[col].map(mapping).fillna(fill_unseen_val).astype(int)
        return X_new

    def update_metadata(
        self,
        X_new: pd.DataFrame,
        metadata: list[ColumnMetadata],
    ) -> list[ColumnMetadata]:
        new_metadata = []
        for i, col in enumerate(X_new.columns):
            updated_meta = deepcopy(metadata[i])
            if col in self.encodings_:
                mapping, fill_unseen_val = self.encodings_[col]
                updated_meta.kind = "binary" if len(mapping) == 2 else "categorical"
                updated_meta.mapping = [None] * (
                    len(mapping) + (1 if self.method == "constant" else 0)
                )
                for val, idx in mapping.items():
                    updated_meta.mapping[idx] = str(val)
                if self.method == "constant":
                    updated_meta.mapping[-1] = self.fill_val_name
            new_metadata.append(updated_meta)
        return new_metadata


@dataclass
class ConvertDatetime(BaseTransform):
    method: str

    def fit(
        self,
        X: pd.DataFrame,
        *,
        metadata: list[ColumnMetadata],
        y: pd.Series = None,
        **kwargs,
    ):
        self._datetime_columns = []
        for met in metadata:
            if met.kind == "datetime":
                self._datetime_columns.append(met.name)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        self._original_columns = X.columns.tolist()
        self._removed_columns = []
        self._added_columns = []
        for i, c in enumerate(X.columns):
            if c not in self._datetime_columns:
                continue
            X_new[c] = pd.to_datetime(
                X_new[c],
                format="mixed",
                errors="coerce",
            )

            if self.method == "to_timestamp":
                X_new[c] = pd.to_numeric(X_new[c]) // 10**9
            elif self.method == "ignore":
                X_new = X_new.drop(columns=[c])
                self._removed_columns.append(c)
            elif self.method == "decompose":
                X_new[c + "_year"] = X_new[c].dt.year
                X_new[c + "_month"] = X_new[c].dt.month
                X_new[c + "_day"] = X_new[c].dt.day
                X_new[c + "_hour"] = X_new[c].dt.hour
                X_new[c + "_minute"] = X_new[c].dt.minute
                X_new[c + "_second"] = X_new[c].dt.second
                self._added_columns += [
                    c + "_year",
                    c + "_month",
                    c + "_day",
                    c + "_hour",
                    c + "_minute",
                    c + "_second",
                ]
                X_new = X_new.drop(columns=[c])
                self._removed_columns.append(c)
        return X_new

    def update_metadata(
        self,
        X_new: pd.DataFrame,
        metadata: list[ColumnMetadata],
    ) -> list[ColumnMetadata]:
        new_metadata = []
        to_add = []
        for i, met in enumerate(metadata):
            updated_meta = deepcopy(met)
            if met.name in self._datetime_columns:
                if self.method == "to_timestamp":
                    updated_meta.kind = "continuous"
                    updated_meta.dtype = "int"
                elif self.method in ["ignore", "decompose"]:
                    continue
            new_metadata.append(updated_meta)

        if self.method == "decompose":
            for met in metadata:
                if met.name not in self._datetime_columns:
                    continue
                for f in [
                    "_year",
                    "_month",
                    "_day",
                    "_hour",
                    "_minute",
                    "_second",
                ]:
                    new_meta = deepcopy(met)
                    new_meta.name = met.name + f
                    new_meta.dtype = "int"
                    new_meta.kind = "continuous"
                    to_add.append(new_meta)
        new_metadata += to_add
        return new_metadata


TRANSFORM_MAP = {
    "Impute": Impute,
    "Scale": Scale,
    "Discretize": Discretize,
    "Encode": Encode,
    "ConvertDatetime": ConvertDatetime,
}


# for adding custom transforms
def add_transform(cls: type[BaseTransform]) -> type[BaseTransform]:
    TRANSFORM_MAP[cls.__name__] = cls
    return cls
