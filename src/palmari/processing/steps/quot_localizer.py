from .base import Detector, SubpixelLocalizer
from ...quot.core import localize_frame, detect
import pandas as pd
import numpy as np
from typing import Dict
import enum

detection_methods_dict = {
    "log": "Laplacian of Gaussians",
    "llr": "Likelihood ratio",
}


class BaseDetector(Detector):
    def __init__(
        self, k: float = 1.5, w: int = 21, t: float = 1.0, method: str = "llr"
    ):
        self.k = k
        self.w = w
        self.t = t
        self.method = method

    def detect_frame(self, img: np.array) -> pd.DataFrame:
        return detect(img, t=self.t, k=self.k, method=self.method, w=self.w)

    @property
    def name(self):
        return "Log-likelihood detector"

    widget_types = {"t": "FloatSlider", "method": "ComboBox"}
    widget_options = {
        "t": {
            "step": 0.1,
            "min": 0.5,
            "max": 3.0,
            "label": "Threshold",
            "readout": True,
        },
        "k": {
            "step": 0.1,
            "label": "Spot size (px)",
            "min": 0.0,
        },
        "w": {
            "step": 2,
            "label": "Window size (px)",
            "min": 5,
            "tooltip": "Size of the square window used for testing.",
        },
        "method": {
            "label": "Method",
            "choices": [
                ("Log-likelihood ratio", "llr"),
                ("Laplacian of Gaussians", "log"),
            ],
        },
    }


class RadialLocalizer(SubpixelLocalizer):

    cols_dtype = {
        "y": float,
        "x": float,
        "I0": float,
        "bg": float,
        "error_flag": bool,
        "snr": float,
    }

    widget_options = {
        "window_size": {"label": "Window size (px)", "step": 1},
    }

    def __init__(
        self,
        window_size: int = 9,
    ):
        self.window_size = window_size

    @property
    def name(self):
        return "Radial symmetry localizer"

    def localize_frame(
        self, img: np.array, detections: np.array
    ) -> pd.DataFrame:
        return localize_frame(
            img.T,
            detections,
            method="radial_symmetry",
            sigma=1.0,
            window_size=self.window_size,
        )


class MaxLikelihoodLocalizer(SubpixelLocalizer):

    cols_dtype = {
        "x": float,
        "y": float,
        "frame": int,
        "I0": float,
        "bg": float,
        "y_err": float,
        "x_err": float,
        "I0_err": float,
        "bg_err": float,
        "H_det": float,
        "error_flag": int,
        "snr": float,
        "rmse": float,
    }

    widget_types = {"method": "ComboBox"}
    widget_options = {
        "method": {
            "label": "Noise",
            "value": "ls_int_gaussian",
            "choices": [
                ("Gaussian", "ls_int_gaussian"),
                ("Poisson", "poisson_int_gaussian"),
            ],
            "tooltip": "Assumed type of background noise",
        },
        "window_size": {"label": "Window size (px)", "step": 1},
        "sigma": {"label": "Spot size (px)", "step": 0.1},
    }

    def __init__(
        self,
        method: str = "ls_int_gaussian",
        window_size: int = 9,
        sigma: float = 1.5,
    ):
        self._loc_params = {
            "ridge": 0.0001,
            "convergence": 0.0001,
            "divergence": 1.0,
            "max_iter": 10,
            "damp": 0.3,
            "camera_bg": 0.0,
            "camera_gain": 1.0,
        }
        self.method = method
        self.window_size = window_size
        self.sigma = sigma

    @property
    def name(self):
        return "Max-likelihood subpixel localizer"

    def localize_frame(
        self, img: np.array, detections: np.array
    ) -> pd.DataFrame:
        return localize_frame(
            img.T,
            detections,
            method=self.method,
            sigma=self.sigma,
            window_size=self.window_size,
            **self._loc_params
        )
