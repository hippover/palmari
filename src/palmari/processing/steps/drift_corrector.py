from .base import *
from ...tif_tools.correct_drift import correct_drift
from dask_image.ndfilters import gaussian_filter
from .quot_localizer import BaseDetector, MaxLikelihoodLocalizer
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


class CorrelationDriftCorrector(LocProcessor):

    widget_type = {"max_n_bins": "Spinner", "min_n_locs_per_bin": "Spinner"}
    widget_options = {
        "max_n_bins": {
            "min": 2,
            "max": 50,
            "label": "Bins",
            "tooltip": "Maximum number of bins. One shift per bin will be computed.",
        },
        "min_n_locs_per_bin": {
            "min": 100,
            "max": int(1e7),
            "step": 1000,
            "label": "Locs / bin",
            "tooltip": "Minimum number of localizations per bin",
        },
    }

    def __init__(self, min_n_locs_per_bin: int = 10000, max_n_bins: int = 20):
        self.min_n_locs_per_bin = min_n_locs_per_bin
        self.max_n_bins = max_n_bins

    def process(
        self, mov: da.Array, locs: pd.DataFrame, pixel_size: float
    ) -> pd.DataFrame:

        logging.info("Correcting drift iteration #1/2")
        df, shift_1 = correct_drift(
            locs,
            L=0.2,
            step_size=0.03,
            min_n_locs_per_bin=self.min_n_locs_per_bin,
            max_n_bins=self.max_n_bins,
        )
        logging.info("Correcting drift iteration #2/2")
        df, shift_2 = correct_drift(
            df,
            L=0.05,
            step_size=0.01,
            min_n_locs_per_bin=self.min_n_locs_per_bin,
            max_n_bins=self.max_n_bins,
        )

        # this causes a crash
        # plt.figure()
        # plt.plot(df["frame"], df["shift_x"], label="x shift")
        # plt.plot(df["frame"], df["shift_y"], label="y shift")
        # plt.legend()
        # plt.show()
        return df

    @property
    def name(self):
        return "Correlation-based drift corrector"

    @property
    def action_name(self):
        return "Correct drift"


class BeadDriftCorrector(LocProcessor):

    widget_types = {"sigma": "FloatSpinBox"}
    widget_options = {
        "sigma": {
            "min": 2,
            "max": 50,
            "label": "Bead radius (in pixels)",
        },
    }

    def __init__(self, sigma: int = 3):
        self.sigma = sigma
        self._n_frames = (
            100  # Width of the Gaussian kernel along the 'frame' axis
        )

    def process(
        self, mov: da.Array, locs: pd.DataFrame, pixel_size: float
    ) -> pd.DataFrame:

        k = 10
        threshold_mov = (mov / mov.max(axis=(1, 2), keepdims=True)).astype(
            float
        )
        gaussian_mov = gaussian_filter(
            threshold_mov,
            sigma=(self._n_frames, self.sigma, self.sigma),
            mode="nearest",
            truncate=3,
        )[::k]
        detector = BaseDetector(
            k=self.sigma, w=int(10 * self.sigma) + 1, t=10.0
        )
        detections = detector.process(gaussian_mov)

        localizer = MaxLikelihoodLocalizer(
            window_size=int(6 * self.sigma) + 1, sigma=self.sigma
        )
        _pos = localizer.process(gaussian_mov, detections)
        bead_pos = (
            (_pos.sort_values("I0", ascending=False).groupby("frame").first())
            .reset_index()
            .sort_values("frame")
        )
        bead_pos["frame"] = (bead_pos["frame"] - 1) * k + 1

        shift = bead_pos[["x", "y"]].values * pixel_size

        # print("Shift std :")
        # print(_pos[["x", "y"]].std(axis=0))
        # print("Locs std : ")
        # print(locs[["x", "y"]].std(axis=0))
        shift = shift - shift[0]
        t = bead_pos["frame"].values
        plt.figure()
        plt.plot(t, shift[:, 0], label="shift")

        f = interp1d(t, shift, kind="linear", axis=0, fill_value="extrapolate")
        computed_shift = f(locs["frame"])

        plt.plot(locs["frame"], computed_shift[:, 0], label="computed")
        plt.legend()

        locs[["x", "y"]] -= computed_shift
        return locs

    @property
    def name(self):
        return "Bead-based drift corrector"

    @property
    def action_name(self):
        return "Correct drift with bead"
