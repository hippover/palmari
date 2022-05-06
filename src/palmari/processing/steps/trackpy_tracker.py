from .base import *


class TrackpyTracker(Tracker):
    def __init__(self, max_diffusivity: float = 5.0):
        # Attributes will automatically be detected as parameters of the step and stored/loaded.
        # Parameters must have default values
        self.max_diffusivity = max_diffusivity

    def track(self, locs: pd.DataFrame):
        # This is where the actual tracking happen.
        import trackpy as tp

        delta_t = self.estimate_delta_t(locs)  # This is a Tracker's method.
        dim = 2
        max_radius = np.sqrt(2 * dim * self.max_diffusivity * delta_t)
        logging.info("Max radius is %.2f" % max_radius)
        tracks = tp.link(locs, search_range=max_radius, link_strategy="drop")
        locs["n"] = tracks["particle"]
        return locs

    @property
    def name(self):
        # This is for printing
        return "Default tracker (Trackpy)"

    # The following dicts are used when setting the parameters through a graphic interface, using open_in_napari()
    widget_types = {
        "max_diffusivity": "FloatSpinBox",
        "delta_t": "FloatSpinBox",
    }
    # For details about widget types, see https://napari.org/magicgui/
    widget_options = {
        "delta_t": {
            "step": 0.01,
            "tooltip": "time interval between frames (in seconds)",
            "min": 0.0,
            "label": "Time delta (s)",
        },
        "max_diffusivity": {
            "step": 1.0,
            "tooltip": "Assumed maximum diffusivity (in microns per square second).\nThis is used in conjunction with the Time delta to set the maximal distance between consecutive localizations",
            "label": "D_max (um/s^2)",
            "min": 0.0,
        },
    }
