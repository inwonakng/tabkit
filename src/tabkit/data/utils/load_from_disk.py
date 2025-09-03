import json
from typing import Literal

import pandas as pd

from .pick_label_col import pick_label_col


def load_from_disk(
    file_path: str,
    file_type: Literal["csv", "parquet"] = "csv",
    label_col: str | None = None,
    random_state: int | None = None,
    split_file_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[int] | None, list[int] | None]:
    if file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # sometimes the t4 tables have constant rows.. need to remove.
    for col in df.columns:
        if col.startswith("Unnamed: "):
            df.pop(col)
            continue
        if df[col].nunique() == 1:
            if col == label_col:
                raise Exception(
                    "label {} has only one unique value and will be removed!".format(
                        label_col
                    )
                )
            df.pop(col)
    if label_col is None:
        label_col = pick_label_col(df, random_state=random_state)
    X = df[df.columns[df.columns != label_col]].copy()
    y = df[label_col].copy()

    if split_file_path is not None:
        with open(split_file_path, "r") as f:
            split_data = json.load(f)
        tr_idxs = split_data.get("train", None)
        te_idxs = split_data.get("test", None)
    else:
        tr_idxs, te_idxs = None, None

    return X, y, tr_idxs, te_idxs
