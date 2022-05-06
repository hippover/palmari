from .base import *
from ...tif_tools.localization import SMLM_localization


class DefaultLocalizer(Localizer):

    widget_type = {"threshold_factor": "FloatSpinner"}
    widget_options = {
        "threshold_factor": {
            "step": 0.05,
            "min": 0.5,
            "max": 3.0,
            "label": "Threshold",
            "tooltip": "Threshold factor used for detection.\nLarger values mean more conservative detections.\nThe detection algorithm is inspired from ThunderSTORM's, see documentation for details)",
        }
    }

    def __init__(self, threshold_factor: float = 1.0):
        self.threshold_factor = threshold_factor

    def localize_slice(self, img: np.array) -> pd.DataFrame:

        return SMLM_localization(img, factor=self.threshold_factor)

    @property
    def name(self):
        return "Default Localizer"
