from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from ast import arg
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

from ...data_structure.acquisition import Acquisition
from ...data_structure.experiment import Experiment


class ProcessingStep(ABC):

    widget_options = {}
    widget_types = {}

    @abstractproperty
    def name(self):
        return "Abstract step"

    def __str__(self):
        desc = "%s (%s) :\n" % (self.name, self.__class__.__qualname__)
        for param, value in self.__dict__.items():
            desc += "\t\t %s : %s\n" % (param, value)
        return desc

    @abstractmethod
    def process(self, *args):
        """
        Depending on their types, subclasses might have different effector functions.
        e.g. Trackers'effector function is named ``.track()``
        This function is just meant to be a link towards the effector function
        """
        pass

    def to_dict(self) -> dict:
        params_dict = dict(self.__dict__)
        all_keys = list([key for key in params_dict])
        for key in all_keys:
            if key[0] == "_":
                # Do not include parameters starting with _
                del params_dict[key]

        return {self.__class__.__qualname__: params_dict}

    def update_param(self, param, *args):
        logging.debug("Updating %s" % param)
        value = args[0]
        logging.debug("%s -> %s" % (self.__dict__[param], value))
        self.__dict__[param] = value

    @property
    def is_localizer(self):
        # To be overriden in localizers and detectors
        return False

    @property
    def is_detector(self):
        # To be overriden in detectors
        return False


class Tracker(ProcessingStep):
    @abstractmethod
    def track(self, locs: pd.DataFrame) -> pd.DataFrame:
        locs["n"] = np.arange(locs.shape[0])
        return locs

    @property
    def name(self):
        return "Tracker"

    @property
    def action_name(self):
        return "Track"

    def process(self, *args):
        return self.track(*args)

    def estimate_delta_t(self, locs):
        delta_t = (locs["t"].max() - locs["t"].min()) / (
            locs["frame"].max() - locs["frame"].min()
        )
        logging.info("Tracker %s estimated delta_t = %.3f" % (self, delta_t))
        return delta_t


class MoviePreProcessor(ProcessingStep):
    @abstractmethod
    def preprocess(self, mov: da.Array) -> pd.DataFrame:
        return mov

    @property
    def name(self):
        return "Movie Processor"

    def process(self, *args):
        logging.debug("%s is about to process" % self.name)
        return self.preprocess(*args)


class Detector(ProcessingStep):

    cols_dtype = {
        "x": float,
        "y": float,
        "frame": int,
    }

    @abstractmethod
    def detect_frame(self, img: np.array) -> pd.DataFrame:
        """Detect spots from a temporal slice of an image. Override in subclasses

        Args:
            img (np.array): 3D array [T, X, Y]

        Returns:
            pd.DataFrame: localizations table with the following columns : x, y, frame
        """
        return pd.DataFrame.from_dict(
            {"x": [], "y": [], "frame": []}, orient="columns"
        )

    def movie_detection(self, mov: da.Array):
        slice_size = mov.chunksize[0]
        n_slices = mov.shape[0] // slice_size
        positions_dfs = []
        for i in range(n_slices + 1):
            start = i * slice_size
            end = min((i + 1) * slice_size, mov.shape[0])
            if start >= end:
                continue
            positions_dfs.append(
                delayed(self.detect)(
                    mov[start:end],
                    frame_start=start,
                )
            )
        loc_results_delayed = dd.from_delayed(
            positions_dfs,
            verify_meta=False,
            meta=self.cols_dtype,
        )
        with ProgressBar():
            # with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category="RuntimeWarning")
            loc_results = loc_results_delayed.compute()
        loc_results.set_index(np.arange(loc_results.shape[0]), inplace=True)
        loc_results["detection_index"] = np.arange(loc_results.shape[0])
        return loc_results

    def detect(self, img: np.array, frame_start: int = 0) -> pd.DataFrame:
        detections = []
        for frame_idx in range(img.shape[0]):
            frame = img[frame_idx].T
            data = self.detect_frame(frame)
            if data.shape[0] > 0:
                frame_detections = pd.DataFrame(data=data, columns=["x", "y"])
                frame_detections["frame"] = frame_idx
                # Localize spots to subpixel resolution
                detections.append(frame_detections)

        if len(detections) > 0:
            detections = pd.concat(detections, ignore_index=True, sort=False)
        else:
            detections = pd.DataFrame.from_dict(
                {"x": [], "y": [], "frame": []}
            )
        # Performs checks on the returned pd.DataFrame
        detections["frame"] += frame_start
        for c, v in self.cols_dtype.items():
            assert c in detections.columns, "%s not in columns" % c
        for c in self.cols_dtype:
            if c in detections.columns:
                detections[c] = detections[c].astype(self.cols_dtype[c])
        return detections

    @property
    def name(self):
        return "Abstract Detector"

    @property
    def action_name(self):
        return "Detect spots"

    def process(self, *args):
        return self.movie_detection(*args)

    @property
    def is_localizer(self):
        return True

    @property
    def is_detector(self):
        return True


class SubpixelLocalizer(ProcessingStep):

    cols_dtype = {
        "x": float,
        "y": float,
        "frame": int,
    }

    @abstractmethod
    def localize_frame(
        self, img: np.array, detections: np.array
    ) -> pd.DataFrame:
        """
        Extract localizations from a slice of a .tif file. Override in subclasses

        Args:
            img (np.array): 2D array [X, Y]

            detections (np.array):  2D array [X, Y] center of spots (pixel indices)

        Returns:
            pd.DataFrame: localizations table with the following columns : x, y (in pixel units)
        """
        return pd.DataFrame.from_dict({"x": [], "y": []}, orient="columns")

    def movie_localization(self, mov: da.Array, detections: pd.DataFrame):
        slice_size = mov.chunksize[0]
        n_slices = mov.shape[0] // slice_size
        positions_dfs = []
        for i in range(n_slices + 1):
            start = i * slice_size
            end = min((i + 1) * slice_size, mov.shape[0])
            if start >= end:
                continue
            positions_dfs.append(
                delayed(self.localize)(
                    mov[start:end],
                    frame_start=start,
                    detections=detections.loc[
                        (detections.frame >= start) & (detections.frame < end)
                    ],
                )
            )
        loc_results_delayed = dd.from_delayed(
            positions_dfs,
            verify_meta=False,
            meta=self.cols_dtype,
        )
        with ProgressBar():
            # with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category="RuntimeWarning")
            loc_results = loc_results_delayed.compute()
        loc_results.set_index(np.arange(loc_results.shape[0]), inplace=True)
        return loc_results

    def localize(
        self, img: np.array, detections: pd.DataFrame, frame_start: int = 0
    ) -> pd.DataFrame:
        locs = []
        for i in range(img.shape[0]):
            frame = img[i]
            frame_locs = self.localize_frame(
                frame,
                np.array(
                    detections.loc[
                        detections.frame == i + frame_start, ["x", "y"]
                    ].values,
                    dtype=int,
                ),
            )
            frame_locs["frame"] = i
            locs.append(frame_locs)
        # Performs checks on the returned pd.DataFrame
        locs = pd.concat(locs, ignore_index=True)
        if locs.shape[0] > 0:
            locs["frame"] += frame_start
            for c, v in self.cols_dtype.items():
                assert c in locs.columns, "%s not in columns" % c
            for c in self.cols_dtype:
                if c in locs.columns:
                    locs[c] = locs[c].astype(self.cols_dtype[c])
        return locs

    @property
    def name(self):
        return "Abstract Localizer"

    @property
    def action_name(self):
        return "Localize"

    def process(self, *args):
        return self.movie_localization(*args)

    @property
    def is_localizer(self):
        # To be overriden in localizers
        return True


class LocProcessor(ProcessingStep):
    @abstractmethod
    def process(
        self,
        mov: da.Array,
        locs: pd.DataFrame,
        pixel_size: float,
    ) -> pd.DataFrame:
        return locs

    @property
    def name(self):
        return "Locs processor doing nothing"

    def process(self, *args):
        return self.process(*args)
