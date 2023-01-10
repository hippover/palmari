from __future__ import annotations

from .processing.tif_pipeline_widget import TifPipelineWidget
from .processing.tif_pipeline import TifPipeline
from .processing.steps import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from napari import Viewer


class PalmariWidget(TifPipelineWidget):
    def __init__(self, napari_viewer: Viewer):
        tp = TifPipeline.from_dict(
            {
                "name": "palmari-default",
                "detector": {"BaseDetector": {"k": 1.5, "w": 21, "t": 1.0}},
                "localizer": {
                    "MaxLikelihoodLocalizer": {
                        "method": "ls_int_gaussian",
                        "window_size": 9,
                        "sigma": 1.5,
                    }
                },
                "tracker": {
                    "DiffusionTracker": {
                        "max_diffusivity": 5.0,
                        "y_diff": 0.9,
                        "init_cost": 50.0,
                        "d_bound_naive": 0.1,
                    }
                },
            }
        )
        print(tp)
        super().__init__(tif_pipeline=tp, napari_viewer=napari_viewer)
