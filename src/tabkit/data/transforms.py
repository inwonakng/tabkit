from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .column_metadata import ColumnMetadata
from .compute_bins import compute_bins


class BaseTransform(ABC):
    """Abstract base class for all preprocessing transforms."""

    # Class-level name attribute - should be set by subclasses
    name: str = None

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

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the fitted transform. Default implementation returns X unchanged."""
        return X

    def update_metadata(
        self, X_new: pd.DataFrame, metadata: list[ColumnMetadata], **kwargs
    ) -> list[ColumnMetadata]:
        return metadata

    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(X, **kwargs)
        return self.transform(X)

class Pipeline:
    """Container for transforms that supports both sequential iteration and name-based access."""

    def __init__(self, transforms: list[BaseTransform] | None = None):
        """Initialize pipeline with optional list of transforms.

        Args:
            transforms: List of transform instances. Names will be auto-assigned if not set.
        """
        self._transforms_list: list[BaseTransform] = []
        self._transforms_dict: dict[str, BaseTransform] = {}
        self._names_list: list[str] = []  # Parallel list to track names in order

        if transforms:
            for transform in transforms:
                self.add(transform)

    def add(self, transform: BaseTransform) -> str:
        """Add a transform to the pipeline.

        Args:
            transform: Transform instance to add

        Returns:
            The assigned unique name for the transform
        """
        # Use the class-level name attribute, fallback to class name
        base_name = transform.name if transform.name is not None else transform.__class__.__name__
        unique_name = self._make_unique_name(base_name)

        self._transforms_list.append(transform)
        self._transforms_dict[unique_name] = transform
        self._names_list.append(unique_name)
        return unique_name

    def _make_unique_name(self, base_name: str) -> str:
        """Generate a unique name by appending numbers if needed.

        Args:
            base_name: The base name to make unique

        Returns:
            A unique name not in the current dict
        """
        if base_name not in self._transforms_dict:
            return base_name

        counter = 2
        while f"{base_name}_{counter}" in self._transforms_dict:
            counter += 1
        return f"{base_name}_{counter}"

    def get(self, name: str) -> BaseTransform:
        """Get a transform by name.

        Args:
            name: Name of the transform

        Returns:
            The transform instance

        Raises:
            KeyError: If name not found
        """
        return self._transforms_dict[name]

    def __getitem__(self, key: str | int) -> BaseTransform:
        """Access transform by name (str) or index (int).

        Args:
            key: Transform name or index

        Returns:
            The transform instance
        """
        if isinstance(key, int):
            return self._transforms_list[key]
        return self._transforms_dict[key]

    def __iter__(self):
        """Iterate over transforms in order."""
        return iter(self._transforms_list)

    def __len__(self):
        """Return number of transforms."""
        return len(self._transforms_list)

    @property
    def names(self) -> list[str]:
        """Get list of transform names in order."""
        return self._names_list

    @property
    def transforms(self) -> list[BaseTransform]:
        """Get list of transforms (for backward compatibility)."""
        return self._transforms_list


@dataclass
class Impute(BaseTransform):
    name = "Impute"

    method: str
    fill_value: Any | None = None

    # Fitted attributes
    imputation_values_: dict[str, Any] = field(default_factory=dict, init=False)

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

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform for Impute - returns unchanged as we can't recover original NaN locations."""
        return X


@dataclass
class Scale(BaseTransform):
    name = "Scale"

    method: str

    # Fitted attributes
    scalers_: dict[str, Any] = field(default_factory=dict, init=False)
    cont_cols_: list[str] = field(default_factory=list, init=False)
    scaler_: Any = field(default=None, init=False)

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

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse scale transformation to recover original continuous values."""
        if not self.cont_cols_:
            return X

        X_new = X.copy()
        X_new[self.cont_cols_] = self.scaler_.inverse_transform(X[self.cont_cols_])
        return X_new


@dataclass
class Discretize(BaseTransform):
    name = "Discretize"

    method: str
    n_bins: int
    # Supervised params
    is_task_regression: bool = False

    # Fitted attributes
    bins_: dict[str, np.ndarray] = field(default_factory=dict, init=False)

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

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse discretization by mapping bin indices to bin midpoints."""
        X_new = X.copy()
        for c in X.columns:
            if c not in self.bins_:
                continue
            bins = self.bins_[c]
            # Map each bin index to its midpoint, handling NaN values in bins
            with np.errstate(invalid='ignore'):
                bin_midpoints = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
            X_new[c] = bin_midpoints[X_new[c].astype(int)]
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
    name = "Encode"

    method: str
    fill_val_name: str | None = None

    # Fitted attributes - stores (mapping, fill_unseen_val) for each column
    encodings_: dict[str, tuple[dict[Any, int], int]] = field(default_factory=dict, init=False)

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

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse encoding by mapping integer codes back to original values."""
        X_new = X.copy()
        for col in X.columns:
            if col not in self.encodings_:
                continue
            mapping, fill_unseen_val = self.encodings_[col]
            # Create reverse mapping: int -> original value
            reverse_mapping = {v: k for k, v in mapping.items()}
            # Handle the fill value if constant method was used
            if self.method == "constant" and self.fill_val_name is not None:
                reverse_mapping[fill_unseen_val] = self.fill_val_name
            X_new[col] = X_new[col].astype(int).map(reverse_mapping)
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
    name = "ConvertDatetime"

    method: str

    # Fitted attributes
    _datetime_columns: list[str] = field(default_factory=list, init=False)
    _original_columns: list[str] = field(default_factory=list, init=False)
    _removed_columns: list[str] = field(default_factory=list, init=False)
    _added_columns: list[str] = field(default_factory=list, init=False)

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
                X_new[c + "_weekday"] = X_new[c].dt.weekday
                self._added_columns += [
                    c + "_year",
                    c + "_month",
                    c + "_day",
                    c + "_hour",
                    c + "_minute",
                    c + "_second",
                    c + "_weekday",
                ]
                X_new = X_new.drop(columns=[c])
                self._removed_columns.append(c)
        return X_new

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse datetime conversion - reconstructs datetime columns from transformed data."""
        X_new = X.copy()

        for c in self._datetime_columns:
            if self.method == "to_timestamp":
                # Convert timestamp back to datetime
                X_new[c] = pd.to_datetime(X_new[c], unit='s')
            elif self.method == "decompose":
                # Reconstruct datetime from decomposed components
                X_new[c] = pd.to_datetime({
                    'year': X_new[c + '_year'],
                    'month': X_new[c + '_month'],
                    'day': X_new[c + '_day'],
                    'hour': X_new[c + '_hour'],
                    'minute': X_new[c + '_minute'],
                    'second': X_new[c + '_second']
                })
                # Drop the decomposed columns for this datetime column only
                columns_to_drop = [
                    c + '_year', c + '_month', c + '_day',
                    c + '_hour', c + '_minute', c + '_second', c + '_weekday'
                ]
                X_new = X_new.drop(columns=columns_to_drop)
            elif self.method == "ignore":
                # Cannot reconstruct ignored columns - they are lost
                pass

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
