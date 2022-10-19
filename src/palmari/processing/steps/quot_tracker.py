import pandas as pd
import numpy as np
from .base import Tracker
from typing import Dict
import logging


class DiffusionTracker(Tracker):
    def __init__(
        self,
        max_diffusivity: float = 5.0,
        max_blinks: int = 0,
        # For the diffusion weight matrix
        d_bound_naive: float = 0.1,  # naive estimate of diffusion
        init_cost: float = 50.0,  # Cost of starting a new trajectory when reconnections are available in the search radius
        y_diff: float = 0.9,  # relative importance of the trajectory's history in diffusion estimation
    ):
        # Attributes will automatically be detected as parameters of the step and stored/loaded.
        # Parameters must have default values
        self.max_diffusivity = max_diffusivity
        self.y_diff = y_diff
        self.init_cost = init_cost
        self.d_bound_naive = d_bound_naive
        self.max_blinks = max_blinks

    def track(self, locs: pd.DataFrame):
        # This is where the actual tracking happen.
        from ...quot.core import track

        delta_t = self.estimate_delta_t(locs)  # This is a Tracker's method.
        dim = 2
        max_radius = np.sqrt(2 * dim * self.max_diffusivity * delta_t)
        logging.info("Max radius is %.2f" % max_radius)
        locs["n"] = track(
            locs,
            method="diffusion",
            search_radius=max_radius,
            pixel_size_um=1.0,
            frame_interval=delta_t,
            max_blinks=self.max_blinks,
            y_diff=self.y_diff,
            d_bound_naive=self.d_bound_naive,
            init_cost=self.init_cost,
        )["trajectory"]
        return locs

    @property
    def name(self):
        # This is for printing
        return "Diffusion tracker"

    # The following dicts are used when setting the parameters through a graphic interface, using open_in_napari()
    widget_types = {
        "max_diffusivity": "FloatSpinBox",
        "max_blinks": "SpinBox",
        "d_bound_naive": "FloatSpinBox",
        "init_cost": "FloatSpinBox",
        "y_diff": "FloatSlider",
    }
    # For details about widget types, see https://napari.org/magicgui/
    widget_options = {
        "max_diffusivity": {
            "step": 1.0,
            "tooltip": "Assumed maximum diffusivity (in microns per square second).\nThis is used in conjunction with the Time delta to set the maximal distance between consecutive localizations",
            "label": "D_max (um^2/s)",
            "min": 0.0,
        },
        "max_blinks": {
            "step": 1,
            "tooltip": "Maximum number of tolerated blinks (i.e. number of frames during which a particle can 'disappear').",
            "label": "Max blinks",
            "min": 0,
            "max": 2,
        },
        "y_diff": {
            "step": 0.01,
            "min": 0.0,
            "max": 1.0,
            "label": "Past vs naive",
            "tooltip": "Relative weight of a trajectory's past in the estimation of its diffusivity",
        },
        "d_bound_naive": {
            "step": 0.1,
            "tooltip": "First guess of a trajectory's diffusivity",
            "label": "D_0 (um^2/s)",
            "min": 0.0,
        },
        "init_cost": {
            "step": 10,
            "label": "Initialization cost",
            "min": 1,
            "tooltip": "Cost of initializing a new trajectory when a reconnection is available in the search radius",
        },
    }


class EuclideanTracker(Tracker):
    def __init__(
        self,
        max_diffusivity: float = 5.0,
        max_blinks: int = 0,
        scale: float = 1.0,  # scaling factor between distances and weights
        init_cost: float = 50.0,  # Cost of starting a new trajectory when reconnections are available in the search radius
    ):
        # Attributes will automatically be detected as parameters of the step and stored/loaded.
        # Parameters must have default values
        self.max_diffusivity = max_diffusivity
        self.init_cost = init_cost
        self.max_blinks = max_blinks
        self.scale = scale

    def track(self, locs: pd.DataFrame):
        # This is where the actual tracking happen.
        from ...quot.core import track

        delta_t = self.estimate_delta_t(locs)  # This is a Tracker's method.
        dim = 2
        max_radius = np.sqrt(2 * dim * self.max_diffusivity * delta_t)
        logging.info("Max radius is %.2f" % max_radius)
        locs["n"] = track(
            locs,
            method="euclidean",
            search_radius=max_radius,
            pixel_size_um=1.0,
            frame_interval=delta_t,
            max_blinks=self.max_blinks,
            scale=self.scale,
            init_cost=self.init_cost,
        )["trajectory"]
        return locs

    @property
    def name(self):
        # This is for printing
        return "Euclidean tracker"

    # The following dicts are used when setting the parameters through a graphic interface, using open_in_napari()
    widget_types = {
        "max_diffusivity": "FloatSpinBox",
        "scale": "FloatSpinBox",
        "init_cost": "FloatSpinBox",
        "max_blinks": "SpinBox",
    }
    # For details about widget types, see https://napari.org/magicgui/
    widget_options = {
        "max_diffusivity": {
            "step": 1.0,
            "tooltip": "Assumed maximum diffusivity (in microns per square second).\nThis is used in conjunction with the Time delta to set the maximal distance between consecutive localizations",
            "label": "D_max (um^2/s)",
            "min": 0.0,
        },
        "scale": {
            "step": 0.5,
            "label": "Weight scale",
            "min": 0.1,
            "tooltip": "Scaling factor between distance (in um) and coefficient of the corresponding assignment in the weight matrix",
        },
        "init_cost": {
            "step": 10,
            "label": "Initialization cost",
            "min": 1,
            "tooltip": "Cost of initializing a new trajectory when a reconnection is available in the search radius",
        },
        "max_blinks": {
            "step": 1,
            "tooltip": "Maximum number of tolerated blinks (i.e. number of frames during which a particle can 'disappear').",
            "label": "Max blinks",
            "min": 0,
            "max": 2,
        },
    }
