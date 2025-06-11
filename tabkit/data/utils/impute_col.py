from typing import Any, Callable
import pandas as pd
import numpy as np

METHODS = {}


def register_impute_method(
    func: Callable[
        [
            pd.Series,
            np.ndarray,
            Any | None,
            int | None,
        ],
        pd.Series,
    ]
) -> None:
    METHODS[func.__name__] = func


@register_impute_method
def mean(
    col: pd.Series,
    tr_idxs: np.ndarray,
    **kwargs,
) -> Any:
    return col[tr_idxs].mean()


@register_impute_method
def median(
    col: pd.Series,
    tr_idxs: np.ndarray,
    **kwargs,
) -> Any:
    return col[tr_idxs].median()


@register_impute_method
def most_frequent(
    col: pd.Series,
    tr_idxs: np.ndarray,
    **kwargs,
) -> Any:
    return col[tr_idxs].mode().iloc[0]


@register_impute_method
def random(
    col: pd.Series,
    tr_idxs: np.ndarray,
    random_state: int | None = None,
    **kwargs,
) -> Any:
    return col[tr_idxs].sample(1, random_state=random_state).iloc[0]


@register_impute_method
def constant(
    fill_val: Any,
    **kwargs,
) -> Any:
    return fill_val


def impute_col(
    method: str,
    col: pd.Series,
    tr_idxs: np.ndarray,
    fill_val: Any | None = None,
    random_state: int | None = None,
) -> pd.Series:
    if not col.isna().any():
        return col

    if method in METHODS:
        fill_nan_val = METHODS[method](
            col=col,
            tr_idxs=tr_idxs,
            fill_val=fill_val,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Impute method [{method}] not found")
    col = col.fillna(fill_nan_val)
    return col
