import numpy as np
from sklearn.neighbors import radius_neighbors_graph
import pandas as pd


def density_filtering(
    pos: pd.DataFrame, radius: float = 0.2, n_neighbors: int = None
) -> pd.Series:
    # Filtering based on number of neighbors in radius
    A = radius_neighbors_graph(pos[["x", "y"]].values, radius=radius)
    N = np.reshape(np.array(A.sum(axis=0)), (-1,))
    if n_neighbors is None:
        n_neighbors = np.quantile(N[N > 0], 0.01)
    return pd.Series(index=pos.index, data=N >= n_neighbors)
