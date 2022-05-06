import logging
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import pandas as pd
from dask import delayed
import dask.array as da


def get_optimal_shift(pos1, pos2, L, step):
    # objective : find dx, dy to move pos2 on pos1
    best = (None, None)
    bins_x = np.arange(pos1.min() - L, pos1.max() + L, step)
    bins_y = np.arange(pos2.min() - L, pos2.max() + L, step)

    hist1, _, _ = np.histogram2d(
        pos1[:, 0], pos1[:, 1], bins=(bins_x, bins_y), normed=True
    )

    best_correlation = 0.0

    best = np.zeros(2)

    for dx in np.arange(-L, L, step):
        for dy in np.arange(-L, L, step):
            pos2_d = pos2 - np.array([dx, dy])

            hist2, _, _ = np.histogram2d(
                pos2_d[:, 0], pos2_d[:, 1], bins=(bins_x, bins_y), normed=True
            )
            correlation = np.mean(hist2 * hist1)
            if correlation > best_correlation:
                best_correlation = correlation
                best = np.array([dx, dy])
    return best


def correct_drift(
    pos,
    L=0.2,
    step_size=0.03,
    prog_bar_position=None,
    min_n_locs_per_bin: int = 10000,
    max_n_bins: int = 20,
):
    pos["n_detection"] = np.arange(pos.shape[0])

    if "shift_x" not in pos.columns:
        pos[["shift_x", "shift_y"]] = np.zeros((pos.shape[0], 2))

    bin_size = max(min_n_locs_per_bin, int(pos.shape[0] / max_n_bins))
    bins = np.arange(0, pos.n_detection.max() + 1, bin_size)
    # print("Cut in %d bins" % len(bins))

    if bin_size > pos.shape[0] / 2:
        logging.info("Too few detections to correct drift")
        interp = None
    else:
        pos["n_bin"] = np.digitize(pos.n_detection, bins)
        sorted_bins = sorted(np.unique(pos.n_bin))
        n_steps = np.array(
            [pos.loc[pos.n_bin == b, "n_detection"].min() for b in sorted_bins]
        )

        pos_1 = [
            pos.loc[pos.n_bin == b, ["x", "y"]].copy().values
            for b in sorted_bins[:-1]
        ]
        pos_2 = [
            pos.loc[pos.n_bin == b, ["x", "y"]].copy().values
            for b in sorted_bins[1:]
        ]

        shifts_del = []
        for p1, p2 in tqdm(
            zip(pos_1, pos_2),
            leave=False,
            unit="bins",
            total=len(pos_1),
            position=0 if prog_bar_position is None else prog_bar_position,
            disable=False,  # Disable parce que compliqué en pooling
        ):
            shifts_del.append(
                da.from_delayed(
                    delayed(get_optimal_shift)(p1, p2, L=L, step=step_size),
                    dtype=float,
                    shape=(2,),
                )
            )
        shifts_del = da.stack(shifts_del, axis=0)
        shifts_del = da.concatenate([da.zeros((1, 2)), shifts_del], axis=0)
        shifts_del = da.cumsum(shifts_del, axis=0)
        shifts = shifts_del.compute()

        interp = interp1d(
            n_steps,
            shifts,
            axis=0,
            bounds_error=False,
            kind="quadratic" if pos.n_bin.nunique() > 2 else "linear",
            fill_value=(shifts[0], shifts[-1]),
        )
        interp_values = interp(pos.n_detection.values)
        pos[["x", "y"]] -= interp_values
        pos[["shift_x", "shift_y"]] += interp_values

    frames = np.arange(
        start=pos.n_detection.min(), stop=pos.n_detection.max(), step=1000
    )
    interp_df = pd.DataFrame(index=np.arange(len(frames)))
    interp_df["frame"] = frames
    interp_df["dx"] = 0.0
    interp_df["dy"] = 0.0

    if interp is not None:
        interp_df[["dx", "dy"]] = interp(interp_df["frame"])

    return pos, interp_df
