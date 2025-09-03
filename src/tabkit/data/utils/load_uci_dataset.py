from typing import Optional, Literal

import arff
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_uci_dataset(dataset_id: int, target_name: str | None = None) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load and preprocess an UCI dataset.

    This function retrieves a dataset from UCI, processes it, and prepares it
    for machine learning tasks.

    Args:
        dataset_id: The ID of the UCI dataset to load.
        target_name: The name of the target column. If None, assumed to be the
        first column of the targets df.

    Returns:
        A tuple containing:
        - pd.DataFrame: The processed dataset with cleaned column names and encoded categorical variables.
        - pd.Series: The target column.
    """

    dset = fetch_ucirepo(id=dataset_id)
    if target_name is None:
        target_name = dset.data.targets.columns[0]
    X = dset.data.features
    y = dset.data.targets[target_name]
    return X, y
