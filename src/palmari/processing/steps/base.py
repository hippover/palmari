from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
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
        return {self.__class__.__qualname__: self.__dict__}

    def update_param(self, param, value):
        logging.debug("Updating %s" % param)
        logging.debug("%s -> %s" % (self.__dict__[param], value))
        self.__dict__[param] = value

    @property
    def is_localizer(self):
        return isinstance(self, Localizer)


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


class Localizer(ProcessingStep):

    cols_dtype = {
        "x": float,
        "y": float,
        "frame": int,
        "sigma": float,
        "ratio": float,
        "total_intensity": float,
    }

    @abstractmethod
    def localize_slice(self, img: np.array) -> pd.DataFrame:
        """
        Extract localizations from a slice of a .tif file. Override in subclasses

        Args:
            img (np.array): 3D array [T, X, Y]

        Returns:
            pd.DataFrame: localizations table with the following columns : x, y, frame
        """
        return pd.DataFrame.from_dict(
            {"x": [], "y": [], "frame": []}, orient="columns"
        )

    def movie_localization(self, mov: da.Array):
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

    def localize(self, img: np.array, frame_start: int = 0) -> pd.DataFrame:
        locs = self.localize_slice(img)
        # Performs checks on the returned pd.DataFrame
        locs["frame"] += frame_start
        for c, v in self.cols_dtype.items():
            assert c in locs.columns
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


class LocProcessor(ProcessingStep):
    @abstractmethod
    def process(self, locs: pd.DataFrame) -> pd.DataFrame:
        return locs

    @property
    def name(self):
        return "Locs processor doing nothing"

    def process(self, *args):
        return self.process(*args)
