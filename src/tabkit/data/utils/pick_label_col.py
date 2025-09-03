import re

import pandas as pd

from ..column_metadata import ColumnMetadata


def pick_label_col(
    df: pd.DataFrame,
    random_state: int | None = None,
    exclude_kind: list[str] | None = None,
    exclude_dtype: list[str] | None = None,
    exclude_column_names: list[str] | None = None,
    excldue_column_values: list[str] | None = None,
    min_ratio: float = 0.3,
) -> str:
    """
    Given a dataframe, pick the column that can be used as the prediction target.
    like TabuLa, we will asign higher probability to columns that are categorical.
    """

    # first handle default args
    if exclude_kind is None:
        exclude_kind = []
    if exclude_dtype is None:
        exclude_dtype = []
    if exclude_column_names is None:
        exclude_column_names = []
    if excldue_column_values is None:
        excldue_column_values = []

    # first, we will assign a score to each column based on the number of unique values
    # and the number of missing values.
    scores = pd.Series(index=df.columns, dtype=float)
    exclude_col_patterns = [
        re.compile(exclude_col) for exclude_col in exclude_column_names
    ]
    exclude_val_patterns = [
        re.compile(exclude_val) for exclude_val in excldue_column_values
    ]

    for col in df.columns:
        # check if the exclude columns is passed as regex
        if any(bool(pt.match(col)) for pt in exclude_col_patterns):
            scores.pop(col)
            continue
        # check if the column has any of the excluded values
        if (
            df[col]
            .astype(str)
            .apply(lambda x: any(pt.match(str(x)) for pt in exclude_val_patterns))
            .any()
        ):
            scores.pop(col)
            continue

        col_info = ColumnMetadata.from_series(df[col])
        if col_info.kind in exclude_kind:
            # skip columns that are not of the allowed kind
            if col in scores:
                scores.pop(col)
            continue
        if col_info.dtype in exclude_dtype:
            # skip columns that are not of the allowed dtype
            if col in scores:
                scores.pop(col)
            continue

        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        # now we will assign a score based on the data type
        if n_unique == 1:
            # can't have this
            if col in scores:
                scores.pop(col)
        elif col_info.is_cat or col_info.is_bin:
            n_vals = df[col].value_counts(dropna=False)
            if n_missing and col in scores:
                scores.pop(col)
            if n_vals.min() < min_ratio * len(df):
                # if the minority value is less than the min_ratio of the dataset,
                # we will not use this column as a label.
                if col in scores:
                    scores.pop(col)
            else:
                # these are the best
                if col in scores:
                    scores[col] = 0.9
        else:
            if n_missing:
                # we don't know how to handle continuous targets with missing data.
                if col in scores:
                    scores.pop(col)
            else:
                if col in scores:
                    scores[col] = 0.1

    if not (scores == 0).all():
        # now, randomly pick a column based on the scores
        label = scores.sample(weights=scores, random_state=random_state).index[0]
    else:
        label = None

    return label
