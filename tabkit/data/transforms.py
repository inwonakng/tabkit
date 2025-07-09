from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

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
        self, metadata: list[ColumnMetadata], **kwargs
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
        for i, c in enumerate(X.columns):
            if metadata[i].kind != "continuous":
                continue
            if self.method == "standard":
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
            elif self.method == "minmax":
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()
            elif self.method == "quantile":
                from sklearn.preprocessing import QuantileTransformer

                scaler = QuantileTransformer(n_quantiles=min(1000, len(X)))
            else:
                raise ValueError(f"Unknown scaler method: {self.method}")
            self.scalers_[c] = scaler.fit(X[[c]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for col in X.columns:
            if c not in X_new.columns:
                continue
            X_new[c] = self.scalers_[c].transform(X_new[[c]])
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
        y: pd.Series = None,
        metadata: list[ColumnMetadata],
        random_state: int | None = None,
        **kwargs,
    ):
        self.bins_ = {}
        for i, c in enumerate(X.columns):
            if metadata[i].kind != "continuous":
                continue
            # Using your original compute_bins function
            bins, _ = compute_bins(
                method=self.method,
                col=X[c],
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
        if not hasattr(self, "value_mappings_"):
            raise RuntimeError(
                "Transform must be fitted before metadata can be updated."
            )
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
        y: pd.Series = None,
        metadata: list[ColumnMetadata],
        random_state: int | None = None,
        **kwargs,
    ):
        self.encodings_ = {}
        for i, col in enumerate(X.columns):
            if metadata[i].kind not in ["categorical", "binary"]:
                continue
            uniq_tr_val = X[col].unique().tolist()
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
                updated_meta.kind = "categorical"
                updated_meta.mapping = [None] * (
                    len(mapping) + (1 if self.method == "constant" else 0)
                )
                for val, idx in mapping.items():
                    updated_meta.mapping[idx] = str(val)
                if self.method == "constant":
                    updated_meta.mapping[-1] = self.fill_val_name
            new_metadata.append(updated_meta)
        return new_metadata


TRANSFORM_MAP = {
    "Impute": Impute,
    "Scale": Scale,
    "Discretize": Discretize,
    "Encode": Encode,
}


def add_transform(cls: type[BaseTransform]) -> type[BaseTransform]:
    METHODS[cls.__name__] = cls
    return cls
