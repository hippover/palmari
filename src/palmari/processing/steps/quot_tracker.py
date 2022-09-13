import pandas as pd
import numpy as np
from .base import Tracker
from typing import Dict
import logging


class MTTTracker(Tracker):
    def __init__(self, params: Dict = {}):
        # Attributes will automatically be detected as parameters of the step and stored/loaded.
        # Parameters must have default values
        self.params = params

    def track(self, locs: pd.DataFrame):
        # This is where the actual tracking happen.
        from ...quot.core import track

        delta_t = self.estimate_delta_t(locs)  # This is a Tracker's method.
        dim = 2
        max_radius = np.sqrt(2 * dim * self.max_diffusivity * delta_t)
        logging.info("Max radius is %.2f" % max_radius)
        locs["n"] = track(
            locs,
            search_radius=max_radius,
            pixel_size_um=1.0,
            frame_interval=delta_t,
            **self.params
        )
        return locs

    @property
    def name(self):
        # This is for printing
        return "MTT tracker (Trackpy)"

    # The following dicts are used when setting the parameters through a graphic interface, using open_in_napari()
    widget_types = {
        "max_diffusivity": "FloatSpinBox",
    }
    # For details about widget types, see https://napari.org/magicgui/
    widget_options = {
        "max_diffusivity": {
            "step": 1.0,
            "tooltip": "Assumed maximum diffusivity (in microns per square second).\nThis is used in conjunction with the Time delta to set the maximal distance between consecutive localizations",
            "label": "D_max (um/s^2)",
            "min": 0.0,
        },
    }
