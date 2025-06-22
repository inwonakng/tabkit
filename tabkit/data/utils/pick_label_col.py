import re

import pandas as pd

from .column_metadata import ColumnMetadata


def pick_label_col(
    df: pd.DataFrame,
    random_state: int | None = None,
    allowed_kind: list[str] | None = None,
    allowed_dtype: list[str] | None = None,
    exclude_column_names: list[str] | None = None,
    excldue_column_values: list[str] | None = None,
    min_ratio: float = 0.3,
) -> str:
    """
    Given a dataframe, pick the column that can be used as the prediction target.
    like TabuLa, we will asign higher probability to columns that are categorical.
    """

    # first, we will assign a score to each column based on the number of unique values
    # and the number of missing values.
    scores = pd.Series(index=df.columns, dtype=float)
    exclude_col_patterns = (
        [re.compile(exclude_col) for exclude_col in exclude_column_names]
        if exclude_columns
        else []
    )
    exclude_val_patterns = (
        [re.compile(exclude_val) for exclude_val in excldue_column_values]
        if excldue_column_values
        else []
    )

    for col in df.columns:
        if exclude_col_patterns:
            # also check if the exclude columns is passed as regex
            if any(bool(pt.match(col)) for pt in exclude_col_patterns):
                scores.pop(col)
                continue
        if exclude_val_patterns:
            # check if the column has any of the excluded values
            if df[col].str.apply(lambda x: any(pt.match(str(x)) for pt in exclude_val_patterns)).any():
                scores.pop(col)
                continue

        col_info = ColumnMetadata.from_series(df[col])
        if allowed_kind is not None and col_info.kind not in allowed_kind:
            # skip columns that are not of the allowed kind
            scores.pop(col)
            continue
        if allowed_dtype is not None and col_info.dtype not in allowed_dtype:
            # skip columns that are not of the allowed dtype
            scores.pop(col)
            continue

        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        # now we will assign a score based on the data type
        if n_unique == 1:
            # can't have this
            scores.pop(col)
        elif (
            df[col].dtype.name in ["object", "category"]
            or n_unique.min() >= min_ratio * df.shape[0]
        ):
            # these are the best
            scores[col] = 0.9
        else:
            if n_missing:
                # we don't know how to handle continuous targets with missing data.
                scores.pop(col)
            else:
                scores[col] = 0.1

    if not (scores == 0).all():
        # now, randomly pick a column based on the scores
        label = scores.sample(weights=scores, random_state=random_state).index[0]
    else:
        label = None

    return label
