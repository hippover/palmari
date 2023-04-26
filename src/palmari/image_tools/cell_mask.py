import numpy as np
from scipy.ndimage.morphology import binary_dilation
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import closing, square, remove_small_objects, convex_hull_object
from skimage.segmentation import clear_border
from skimage.measure import label
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt


def get_cell_mask(data: np.ndarray):
    data_max = .5*(np.max(data[::10], axis=0) + np.mean(data[::10], axis=0))
    gblur = gaussian(data_max, sigma=4)
    threshold = threshold_otsu(gblur)
    #plt.figure()
    #plt.imshow(gblur)
    bw = closing(gblur > threshold, square(4))
    #plt.figure()
    #plt.imshow(bw)
    extra_width = 5
    bw = binary_dilation(bw, iterations=extra_width)
    #plt.figure()
    #plt.imshow(bw)

    cleared = remove_small_objects(bw, 100)
    #plt.figure()
    #plt.imshow(cleared)
    convex_image = convex_hull_object(cleared, connectivity=2)
    label_image = label(convex_image).astype(int)
    
    return label_image


def tag_localizations_per_cell(pos: pd.DataFrame, labels: np.ndarray, scale: float):
    XY = pos[["x", "y"]].values/scale
    cell = pd.Series(data=np.zeros(pos.shape[0]))
    _, _, _, bins = binned_statistic_2d(x=XY[:, 0], y=XY[:, 1], values=np.ones_like(XY[:,0]), bins=(
        np.arange(labels.shape[0]+1), np.arange(labels.shape[1]+1)))
    flat_labels = np.reshape(labels, (-1,))
    for l in np.unique(labels):
        if l == 0:
            continue
        good_bins = np.where(flat_labels == l)
        cell.loc[np.isin(bins,good_bins)] = l
    return cell
        