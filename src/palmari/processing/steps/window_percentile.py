from .base import *
from ...tif_tools.localization import sliding_window_filter


class WindowPercentileFilter(MoviePreProcessor):

    widget_types = {"percentile": "FloatSpinBox", "window_size": "SpinBox"}
    widget_options = {
        "percentile": {
            "step": 1.0,
            "tooltip": "percentile of the pixel intensity values in the window which will be considered as the ground level.",
            "min": 0.0,
            "label": "Percentile",
        },
        "window_size": {
            "step": 50,
            "tooltip": "Size of the windows along which quantiles are computed",
            "label": "Window",
            "min": 100,
        },
    }

    def __init__(self, percentile: float = 3.0, window_size: int = 100):
        self.percentile = percentile
        self.window_size = window_size

    def preprocess(self, mov: da.Array) -> da.Array:

        return sliding_window_filter(
            data=mov, percentile=self.percentile, window_size=self.window_size
        )

    @property
    def name(self):
        return "Local percentile filtering"

    @property
    def action_name(self):
        return "Filter"
