import numpy as np
import pandas as pd
import dask.array as da
from dask_image.ndfilters import convolve


def smooth_average(x, w):
    r = (da.arange(w) - w / 2) / (w / 6)
    gaussian = da.exp(-0.5 * r**2)
    gaussian /= da.sum(gaussian)
    return convolve(x, gaussian, "reflect").compute()


def mean_intensity_center(data):
    T, W, H = data.shape
    # indices = tuple(
    #    np.ogrid[0:T, (W // 2 - 10) : (W // 2 + 10), (H // 2 - 10) : (H // 2 + 10)]
    # )
    # center = data[indices]
    # print(indices)
    center = data[:, (W // 2 - 10) : (W // 2 + 10), :]
    center = center[:, :, (H // 2 - 10) : (H // 2 + 10)]
    average = center.mean(axis=(1, 2))
    smoothed_average = smooth_average(average, min(500, average.shape[0]))
    return pd.DataFrame.from_dict(
        {
            "intensity": smoothed_average,
            "frame": np.arange(smoothed_average.shape[0]),
        }
    )
