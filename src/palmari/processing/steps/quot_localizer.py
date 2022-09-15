from .base import Localizer
from ...quot.core import localize_frame, detect
import pandas as pd
import numpy as np
from typing import Dict


class MTTLogLocalizer(MTTLocalizer):
    

class MTTLocalizer(Localizer):

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

    widget_type = {"threshold_factor": "FloatSpinner"}
    widget_options = {
        "method": {
            "label": "Detection method",
            "tooltip": "Method used for detection.\nLarger values mean more conservative detections.\nThe detection algorithm is inspired from ThunderSTORM's, see documentation for details)",
        }
    }

    def __init__(
        self,
        detect_params: Dict = {"method": "llr", "k": 1.5, "w": 21, "t": 16.0},
        loc_params: Dict = {
            "method": "ls_int_gaussian",
            "window_size": 9,
            "sigma": 1.5,
            "ridge": 0.0001,
            "convergence": 0.0001,
            "divergence": 1.0,
            "max_iter": 10,
            "damp": 0.3,
            "camera_bg": 0.0,
            "camera_gain": 1.0,
        },
    ):
        self.detect_params = detect_params
        self.loc_params = loc_params

    def localize_slice(self, img: np.array) -> pd.DataFrame:

        locs = []
        for frame_idx in range(img.shape[0]):
            frame = img[frame_idx].T

            # Find spots in this image frame
            detections = detect(frame, **self.detect_params)

            # Localize spots to subpixel resolution
            locs.append(
                localize_frame(frame, detections, **self.loc_params).assign(
                    frame=frame_idx
                )
            )

        locs = pd.concat(locs, ignore_index=True, sort=False)
        return locs

    @property
    def name(self):
        return "MTT (quot) localizer"
