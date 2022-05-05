from typing import Any
import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import square
from skimage.feature import peak_local_max
from scipy.ndimage import convolve1d
from dask_image.ndfilters import percentile_filter
import pandas as pd
from math import factorial
import warnings
import logging
import dask.array as da
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar


def lsradialcenterfit(m, b, w):
    """
    Adapted from Matlab code found in https://www.nature.com/articles/nmeth.2071
    Least squares solution to determine the radial symmetry center.
    Inputs m, b, w are defined on a grid.
    w are the weights for each point.
    """
    wm2p1 = w / (m * m + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m * m * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    det = smw * smw - smmw * sw
    assert det != 0
    xc = (smbw * sw - smw * sbw) / det  # relative to image center
    yc = (smbw * smw - smmw * sbw) / det  # relative to image centerc
    assert ~np.isnan(xc)
    assert ~np.isnan(yc)
    return xc, yc


def radialCenter(I):
    Ny, Nx = I.shape
    assert Nx % 2 == 1
    assert Ny % 2 == 1
    nx = Nx // 2
    ny = Ny // 2
    # Nx and Ny must be even
    xm, ym = np.meshgrid(np.arange(-nx, nx) + 0.5, np.arange(-ny, ny) + 0.5)

    assert xm.shape[0] % 2 == 0

    # dIdu = I(1:Ny-1,2:Nx)-I(2:Ny,1:Nx-1);
    # dIdv = I(1:Ny-1,1:Nx-1)-I(2:Ny,2:Nx);

    # dIdu = I[1:, :-1]-I[:-1, 1:]
    # dIdv = I[:-1, :-1]-I[1:, 1:]
    dIdu = I[:-1, 1:] - I[1:, :-1]
    dIdv = I[:-1, :-1] - I[1:, 1:]

    h = np.ones((3, 3))
    h /= np.sum(h)

    fdu = convolve2d(dIdu, h, mode="same")
    fdv = convolve2d(dIdv, h, mode="same")

    dImag2 = fdu**2 + fdv**2
    m = -(fdv + fdu) / (fdu - fdv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unsmoothed_m = -(dIdu + dIdv) / (dIdu - dIdv)
    m[np.isnan(m)] = unsmoothed_m[np.isnan(m)]
    m[np.isnan(m)] = 0.0
    m[np.isinf(m)] = 1 * np.max(m[~np.isinf(m)])
    assert np.sum(np.isnan(m)) == 0
    b = ym - m * xm
    sdI2 = np.sum(dImag2)

    xcentroid = np.sum(dImag2 * xm) / sdI2
    ycentroid = np.sum(dImag2 * ym) / sdI2
    w = dImag2 / np.sqrt(
        (xm - xcentroid) * (xm - xcentroid)
        + (ym - ycentroid) * (ym - ycentroid)
    )

    xc, yc = lsradialcenterfit(m, b, w)
    xc = xc + nx
    yc = yc + ny

    Isub = I - np.min(I)
    px, py = np.meshgrid(Nx, Ny)
    xoffset = px - xc
    yoffset = py - yc
    r2 = xoffset * xoffset + yoffset * yoffset
    sigma = (
        np.sqrt(np.sum(Isub * r2) / np.sum(Isub)) / 2
    )  # % second moment is 2*Gaussian width

    return yc, xc, sigma


def phaser(ROI: np.array):
    """Adapted from https://colab.research.google.com/drive/1Jir3HxTZ-au8L56ZrNHGxfBD0XlDkOMl

    Args:
        ROI (np.array): 2D array on which to run the dubpixel localization.

    Returns:
        tuple: x, y, sigma. in pixels.
    """
    ROIradius = ROI.shape[-1] // 2
    # Perform 2D Fourier transform over the complete ROI
    ROI_F = np.fft.fft2(ROI)

    # We have to calculate the phase angle of array entries [0,1] and [1,0] for
    # the sub-pixel x and y values, respectively
    # This phase angle can be calculated as follows:
    xangle = np.arctan(ROI_F[0, 1].imag / ROI_F[0, 1].real) - np.pi
    # Correct in case it's positive
    if xangle > 0:
        xangle -= 2 * np.pi
    # Calculate position based on the ROI radius
    PositionX = abs(xangle) / (2 * np.pi / (ROIradius * 2 + 1)) + 0.5

    # Do the same for the Y angle and position
    yangle = np.arctan(ROI_F[1, 0].imag / ROI_F[1, 0].real) - np.pi
    if yangle > 0:
        yangle -= 2 * np.pi
    PositionY = abs(yangle) / (2 * np.pi / (ROIradius * 2 + 1)) + 0.5

    return PositionX, PositionY, 1.0


def plus_func(x, n):
    if x < 0:
        return 0
    if x == 0 and n == 0:
        return 0.5
    if x > 0 and n == 0:
        return 1
    else:
        return x**n


def b_splines(x, scale, order):
    x_ = x / scale
    n = order
    b = 0.0
    for k in range(n + 2):
        increment = (
            ((-1) ** k) * (n + 1) / (factorial(n + 1 - k) * factorial(k))
        )
        increment *= plus_func(x_ - k + (n + 1) / 2, n)
        b += increment
    return b


def sliding_window_filter(
    data: da.Array, percentile: float = 10, window_size: int = 100
):
    percent = percentile_filter(
        data, percentile=percentile, size=(window_size, 1, 1), mode="reflect"
    )
    clipped = (data - percent).clip(0.0)
    return clipped


def make_filters(scale, order, L):
    # H0, H1, H2 = 3.0 / 8, 1.0 / 4, 1.0 / 16
    # g1 = np.array([H2, H1, H0, H1, H2])
    # g2 = np.array([H2, 0.0, H1, 0.0, H0, 0.0, H1, 0.0, H2])
    # L = 3  # Le filter est de taille 2*L - 1
    values = [b_splines(x, scale, order) for x in np.arange(L)]
    g1 = np.concatenate([values[1:][::-1], values], axis=0)
    g1 /= np.sum(g1)
    g2 = np.array(
        [0.0 if i % 2 == 1 else g1[i // 2] for i in range(2 * len(g1) - 1)]
    )

    return g1, g2


def SMLM_filtering(data, filter_size, scale):
    g1, g2 = make_filters(scale=scale, order=3, L=filter_size)
    V0 = data
    V1 = convolve1d(convolve1d(V0, g1, axis=1), g1, axis=2)
    assert V1.shape == data.shape
    V2 = convolve1d(convolve1d(V1, g2, axis=1), g2, axis=2)
    return V0, V1, V2


def SMLM_localization(
    data: np.ndarray,
    factor: float = 1.0,
    filter_size: int = 3,
    scale: float = 2.0,
    verbose: bool = False,
    return_all: bool = False,
    subpixel_mode: str = "radial",
    frame_start: int = 0,  # shift frame index by this
):
    logging.debug(
        "Getting localizations on data of shape %d %d %d" % data.shape
    )

    V0, V1, V2 = SMLM_filtering(data, filter_size, scale=scale)
    F1 = V0 - V1
    F2 = V1 - V2

    stdF1 = np.reshape(
        np.std(F1, axis=(1, 2)), (-1, 1, 1)
    )  # 1 value per image
    structure = np.stack(
        [np.zeros((3, 3)), square(3), np.zeros((3, 3))], axis=0
    )
    structure = np.array(structure, dtype=int)

    R_detection = 3
    R_fit = 5
    # Possible de faire marcher ça sans boucle for ?
    # Detection de pics locaux d'intensité
    # CF la thèse de Thunderstorm pour les explications
    logging.debug("Looking for objects")
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        is_peak = np.copy(F2 < -np.inf)  # Initialize all cells with 0
        assert np.sum(is_peak * 1) == 0
        for frame in range(F2.shape[0]):
            peak_idx = peak_local_max(
                F2[frame],
                threshold_abs=factor * stdF1[frame],
                min_distance=2 * R_detection - 1,
                exclude_border=max(R_detection, R_fit) + 1,
            )
            is_peak[frame][tuple(peak_idx.T)] = True

    logging.debug("Found %d objects" % np.sum(is_peak * 1))

    ts, xs, ys = np.where(is_peak)
    ts, xs, ys = list(ts), list(xs), list(ys)
    spots = {}

    subpixel_loc = radialCenter
    if subpixel_mode == "phaser":
        subpixel_loc = phaser

    for i, indices in enumerate(zip(ts, xs, ys)):
        t, x, y = indices

        x_min, x_max = x - R_fit, x + R_fit
        y_min, y_max = y - R_fit, y + R_fit

        img = data[t, x_min : (x_max + 1), y_min : (y_max + 1)]
        if img.shape != (2 * R_fit + 1, 2 * R_fit + 1):
            continue
        try:
            ratio = (
                np.max(F2[t, x_min : (x_max + 1), y_min : (y_max + 1)])
                / stdF1[t, 0, 0]
            )
        except:
            logging.debug(F2[t].shape)
            logging.debug(x_min, x_max)
            logging.debug(y_min, y_max)
            raise
        assert img.shape[0] == 1 + x_max - x_min
        assert img.shape[1] == 1 + y_max - y_min

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            xc, yc, sigma = subpixel_loc(img)
        spot = {}
        spot["frame"] = int(t + frame_start)
        spot["x"] = float(xc + x_min)
        spot["y"] = float(yc + y_min)
        spot["ratio"] = ratio
        spot["sigma"] = sigma
        spot["total_intensity"] = np.sum(img)
        spots[i] = spot
    spots = pd.DataFrame.from_dict(spots, orient="index")
    if return_all:
        return spots, is_peak, F2
    else:
        return spots


def localize_movie(
    movie: Any,  # Dask or numpy
    factor: float = 1.0,
    filter_size: int = 3,
    sliding_filter: bool = False,
    verbose: bool = False,
    subpixel_mode: str = "radial",
    progress_bar: bool = False,
):

    data = movie.astype(float)
    if sliding_filter:
        logging.debug("Sliding filter starts")
        logging.info("Computing sliding filter...")
        data = sliding_window_filter(data)
        logging.info("... Done !")
        logging.debug("Sliding filter done")

    logging.debug("Converted to float")
    # TODO: progress_bar does not work

    slice_size = data.chunksize[0]
    n_slices = data.shape[0] // slice_size
    positions_dfs = []
    for i in range(n_slices + 1):
        start = i * slice_size
        end = min((i + 1) * slice_size, data.shape[0])
        if start >= end:
            continue
        positions_dfs.append(
            delayed(SMLM_localization)(
                data[start:end],
                factor=factor,
                return_all=False,
                verbose=verbose,
                filter_size=filter_size,
                subpixel_mode=subpixel_mode,
                frame_start=start,
            )
        )
    loc_results_del = dd.from_delayed(
        positions_dfs,
        verify_meta=False,
        meta={
            "x": float,
            "y": float,
            "frame": int,
            "sigma": float,
            "ratio": float,
            "total_intensity": float,
        },
    )
    with ProgressBar():
        loc_results = loc_results_del.compute()
    loc_results.set_index(np.arange(loc_results.shape[0]), inplace=True)
    logging.debug(loc_results.head(10))
    logging.debug(loc_results.tail(10))
    return loc_results
