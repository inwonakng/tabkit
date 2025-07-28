import warnings
from dataclasses import dataclass
from typing import Literal

import pandas as pd


def is_column_categorical(col: pd.Series, max_cardinalities: int = 20) -> bool:
    if col.dtype.name in ["object", "category"]:
        return True
    else:
        unique_vals = col.unique()
        if unique_vals.shape[0] < max_cardinalities:
            return True
    return False


@dataclass
class ColumnMetadata:
    name: str
    kind: Literal[
        "datetime",
        "categorical",
        "binary",
        "continuous",
    ]
    dtype: Literal["float", "int", "string", "datetime"]
    mapping: list[str] | None = None

    @property
    def is_cat(self) -> bool:
        return self.kind == "categorical"

    @property
    def is_num(self) -> bool:
        return self.dtype in ["float", "int"]

    @property
    def is_date(self) -> bool:
        return self.dtype == "datetime"

    @property
    def is_cont(self) -> bool:
        return self.kind == "continuous"

    @property
    def is_bin(self) -> bool:
        return self.kind == "binary"

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMetadata":
        return cls(
            name=d["name"],
            kind=d["kind"],
            dtype=d["dtype"],
            mapping=d["mapping"] if d["mapping"] else None,
        )

    def to_dict(self) -> dict[str, str | dict[str, str]]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "mapping": self.mapping if self.mapping else {},
        }

    def get_mapping(self, val: str | int) -> str:
        if self.mapping is None:
            raise ValueError("Mapping is not defined for this column.")
        try:
            val = int(val)  # Ensure val is an integer for indexing
            return self.mapping[val]
        except ValueError:
            raise ValueError(
                f"Value must be convertible to an integer, but got {val} with type {type(val)}."
            )

    @classmethod
    def from_series(cls, col: pd.Series) -> "ColumnMetadata":
        """
        This is where we will try to automatically infer the kind and the dtype
        of the column.
        """

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            warnings.simplefilter(action="ignore", category=FutureWarning)
            is_datetime = (
                pd.to_datetime(
                    # we will let the imputer handle NaNs.
                    col[~col.isna()].astype(str),
                    format="mixed",
                    errors="coerce",
                )
                .notna()
                .all()
            )

        is_numeric = pd.to_numeric(col.fillna(-1), errors="coerce").notna().all()
        is_binary = col.nunique() == 2
        is_categorical = is_column_categorical(col)

        kwargs = {"mapping": None}
        # datetime only counts if it's not all just numbers
        if is_datetime:
            kwargs["dtype"] = "datetime"
            kwargs["kind"] = "datetime"
        elif is_binary:
            if is_numeric:
                col_num = pd.to_numeric(col)
                kwargs["dtype"] = "float" if col_num.dtype.kind == "f" else "int"
            else:
                kwargs["dtype"] = "string"
            kwargs["kind"] = "binary"
        elif is_categorical:
            kwargs["dtype"] = "string"
            kwargs["kind"] = "categorical"
        elif is_numeric:
            col_num = pd.to_numeric(col)
            kwargs["dtype"] = "float" if col_num.dtype.kind == "f" else "int"
            kwargs["kind"] = "continuous"
        else:
            raise ValueError(
                f"Cannot determine metadata for col: {col.name}. "
                "It may contain mixed types or unsupported data."
                f"{{is_datetime: {is_datetime}, is_numeric: {is_numeric},"
                f"is_binary: {is_binary}, is_categorical: {is_categorical}}}\nSample: {col.head(5)}"
            )

        return cls(name=col.name, **kwargs)
