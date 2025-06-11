from typing import Literal
import pandas as pd


def pick_label_col(
    df: pd.DataFrame,
    random_state: int = 0,
) -> str:
    """
    Given a dataframe, pick the column that can be used as the prediction target.
    like TabuLa, we will asign higher probability to columns that are categorical.
    """

    # first, we will assign a score to each column based on the number of unique values
    # and the number of missing values.
    scores = pd.Series(index=df.columns, dtype=float)

    for col in df.columns:
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        # now we will assign a score based on the data type
        if n_unique == 1:
            # can't have this
            scores.pop(col)
        elif df[col].dtype.name in ["object", "category"] or n_unique < 0.3*df.shape[0]:
            # these are the best
            scores[col] = 0.9
        else:
            if n_missing: 
                # we don't know how to handle continuous targets with missing data.
                scores.pop(col)
            else:
                scores[col] = 0.1

    if not (scores==0).all():
        # now, randomly pick a column based on the scores
        label = scores.sample(weights=scores, random_state=random_state).index[0]
    else:
        label = None

    return label
