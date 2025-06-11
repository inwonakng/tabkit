from typing import Literal, Callable
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

METHODS = {}


def register_scale_method(
    func: Callable[
        [
            pd.Series,
            np.ndarray,
            int,
        ],
        pd.Series,
    ]
) -> None:
    METHODS[func.__name__] = func


@register_scale_method
def standard(
    col: pd.Series,
    tr_idxs: np.ndarray,
    **kwargs,
) -> pd.Series:
    scaler = StandardScaler().fit(col[tr_idxs])
    return scaler.transform(col).reshape(-1)


@register_scale_method
def minmax(
    col: pd.Series,
    tr_idxs: np.ndarray,
    **kwargs,
) -> pd.Series:
    scaler = MinMaxScaler().fit(col[tr_idxs])
    return scaler.transform(col).reshape(-1)


@register_scale_method
def quantile(
    col: pd.Series,
    tr_idxs: np.ndarray,
    max_quantiles: int = 1000,
    **kwargs,
) -> pd.Series:
    scaler = QuantileTransformer(
        n_quantiles=min(max_quantiles, len(tr_idxs)),
    ).fit(col[tr_idxs])
    return scaler.transform(col).reshape(-1)


def scale_col(
    method: str,
    col: pd.Series,
    tr_idxs: np.ndarray,
    max_quantiles: int = 1000,
) -> pd.Series:
    """Scale a column using a given method.

    Args:
        col: Pandas series of values to scale. A single column is expected.
        tr_idxs: Indices of the training set.
        method: method to scale the values by.
        max_quantiles: If using the quantile method, the maximum number of quantiles to use.

    Raises:
        ValueError: If column contains missing values or an unknown method is provided.

    Returns:
        The scaled column.
    """
    if col.isna().any():
        raise ValueError("Column contains missing values")
    col = col.values.reshape(-1, 1)

    if method in METHODS:
        return METHODS[method](
            col=col,
            tr_idxs=tr_idxs,
            max_quantiles=max_quantiles,
        )
    else:
        raise ValueError(f"Scale method [{method}] not found")
