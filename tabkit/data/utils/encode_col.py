from typing import Any, Callable

import numpy as np
import pandas as pd

METHODS = {}


def register_encode_method(
    func: Callable[
        [
            pd.Series,
            np.ndarray,
            int,
            int | None,
        ],
        tuple[pd.Series, dict],
    ],
) -> None:
    METHODS[func.__name__] = func


@register_encode_method
def most_frequent(
    col: pd.Series,
    tr_idxs: np.ndarray,
    **kwargs,
) -> Any:
    uniq_tr_val = col[tr_idxs].unique()
    tr_only_mapping = {v: k for k, v in enumerate(uniq_tr_val)}
    fill_unseen_val = tr_only_mapping[col[tr_idxs].mode().iloc[0]]
    return fill_unseen_val


@register_encode_method
def random(
    col: pd.Series,
    tr_idxs: np.ndarray,
    random_state: int | None = None,
    **kwargs,
) -> Any:
    uniq_tr_val = col[tr_idxs].unique()
    tr_only_mapping = {v: k for k, v in enumerate(uniq_tr_val)}
    fill_unseen_val = tr_only_mapping[
        col[tr_idxs].sample(1, random_state=random_state).iloc[0]
    ]
    return fill_unseen_val


def encode_col(
    method: str,
    col: pd.Series,
    tr_idxs: np.ndarray,
    fill_val_name: str | None = None,
    random_state: int | None = None,
) -> tuple[pd.Series, list[str]]:
    # this is the mapping of the original feature values to the indices.
    # We now need to change this to map to the indices of the training set.
    uniq_tr_val = col[tr_idxs].unique().tolist()
    tr_only_mapping = {v: k for k, v in enumerate(uniq_tr_val)}

    if method in METHODS and method != "constant":
        fill_unseen_val = METHODS[method](
            col=col,
            tr_idxs=tr_idxs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Encode method [{method}] not found")

    for v in col.unique():
        if v not in uniq_tr_val:
            tr_only_mapping[v] = fill_unseen_val
    col = col.map(tr_only_mapping)
    fixed_val_mapping = uniq_tr_val
    if method == "constant":
        # if we are setting aside some value instead of using values from the
        # training set, just allocalte another index for the unseen val.
        fixed_val_mapping.append(fill_val_name)
    return col, fixed_val_mapping
