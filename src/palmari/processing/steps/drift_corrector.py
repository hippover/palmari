from .base import *
from ...tif_tools.correct_drift import correct_drift


class DriftCorrector(LocProcessor):

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

    def process(self, locs: pd.DataFrame) -> pd.DataFrame:

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
        return "Drift corrector"

    @property
    def action_name(self):
        return "Correct drift"
