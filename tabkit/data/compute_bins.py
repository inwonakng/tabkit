import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def uniform(
    col: pd.Series,
    n_bins: int,
    **kwargs,
) -> np.ndarray:
    return np.linspace(col.min(), col.max(), n_bins + 1)


def quantile(
    col: pd.Series,
    n_bins: int,
    **kwargs,
) -> np.ndarray:
    return np.unique(np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1)))


def dtree(
    col: pd.Series,
    n_bins: int,
    y: np.ndarray | None = None,
    is_task_regression: bool = False,
    random_state: int | None = None,
    **kwargs,
) -> np.ndarray:
    if is_task_regression:
        dtree = DecisionTreeRegressor(
            max_leaf_nodes=n_bins,
            random_state=random_state,
        )
    else:
        dtree = DecisionTreeClassifier(
            max_leaf_nodes=n_bins,
            random_state=random_state,
        )
    col = col.values
    tree = dtree.fit(col.reshape(-1, 1), col).tree_
    bins = np.unique(
        [col.min(), col.max()]
        + [
            float(tree.threshold[node_id])
            for node_id in range(tree.node_count)
            if tree.children_left[node_id] != tree.children_right[node_id]
        ]
    )
    return bins


def kmeans(
    col: pd.Series,
    n_bins: int,
    kmeans_sample_weight: np.ndarray | None = None,
    random_state: int | None = None,
    **kwargs,
):
    col = col.values
    uniform_edges = np.linspace(col.min(), col.max(), n_bins + 1)
    init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
    # 1D k-means procedure
    km = KMeans(
        n_clusters=n_bins,
        init=init,
        n_init=1,
        random_state=random_state,
    )
    centers = np.unique(
        km.fit(
            col[:, None],
            sample_weight=kmeans_sample_weight,
        ).cluster_centers_[:, 0]
    )
    center_means = (centers[1:] + centers[:-1]) / 2
    bins = np.r_[col.min(), center_means, col.max()]
    return bins


METHODS = {
    "kmeans": kmeans,
    "quantile": quantile,
    "uniform": uniform,
    "dtree": dtree,
}


def compute_bins(
    method: str,
    col: pd.Series | np.ndarray,
    n_bins: int,
    y: np.ndarray | None = None,
    is_task_regression: bool = False,
    kmeans_sample_weight: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Computes the bins for a continuous column using provided method.

    Args:
        col: A pandas series representing the column to process.
        method: method to use for binning.
            if `quantile`, all bins in each feature have the same number of points. Following PLE paper, we drop identical bins.
            if `uniform`, all bins in each feature have identical widths.
            if `kmeans`, values in each bin have the same distance from the bin center.
            if `dtree`, bins are determined by a scikit-learn decision tree regressor.
        n_bins: Number of bins to use for binning.
        y: Only required when `method` is `dtree`. Target values for decision tree regressor.
        is_task_regression: If True, use a decision tree regressor. Otherwise, use a decision tree classifier.
        kmeans_sample_weight: Sample weights for kmeans binning.

    Returns:
        bins: A numpy array that contains the bin edges for a feature.
        value_mapping: A dictionary that maps bin index to a tuple of lower and upper bounds of the bin.
    """
    bins = None
    if np.unique(col).shape[0] == 1:
        print(
            "Current column has only 1 unique value. This means that every value for this feature will fall into the same bin"
        )
        bins = np.array([-np.inf, np.inf])
    elif np.unique(col).shape[0] == 2:
        print(
            "Current column has only 2 unique values. Creating bins for each unique value."
        )
        bins = np.linspace(col.min(), col.max(), 3)
    else:
        if method in METHODS:
            bins = METHODS[method](
                col=col,
                n_bins=n_bins,
                y=y,
                is_task_regression=is_task_regression,
                kmeans_sample_weight=kmeans_sample_weight,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Bin method [{method}] not found")

    # clean bins, remove edges that are too close.
    mask = np.ediff1d(bins, to_end=np.inf) > 1e-20
    bins = bins[mask]
    value_mapping = [
        (float(lower), float(upper))
        for lower, upper in zip(
            bins[:-1],
            bins[1:],
        )
    ]
    return bins, value_mapping

