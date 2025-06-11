import pandas as pd
import numpy as np
from auto_mm_bench.datasets import dataset_registry


def load_automm_dataset(
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    train_dataset = dataset_registry.create(dataset_id, 'train')
    test_dataset = dataset_registry.create(dataset_id, 'test')
    X_tr = train_dataset.data[train_dataset.feature_columns]
    y_tr = train_dataset.data[train_dataset.label_columns[0]]
    X_te = test_dataset.data[train_dataset.feature_columns]
    y_te = test_dataset.data[train_dataset.label_columns[0]]

    X = pd.concat([X_tr, X_te]).reset_index(drop=True)
    y = pd.concat([y_tr, y_te]).reset_index(drop=True)
    tr_idx = np.arange(len(X_tr))
    te_idx = np.arange(len(X_te)) + len(X_tr)

    return X, y, tr_idx, te_idx
