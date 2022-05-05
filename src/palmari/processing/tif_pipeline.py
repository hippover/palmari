from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import logging
from typing import TYPE_CHECKING, Dict, List, Union
import os
import napari
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
import yaml
import dask_image

from ..data_structure.acquisition import Acquisition
from ..data_structure.experiment import Experiment

from .tif_pipeline_widget import TifPipelineWidget


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
        from ..tif_tools.localization import sliding_window_filter

        return sliding_window_filter(
            data=mov, percentile=self.percentile, window_size=self.window_size
        )

    @property
    def name(self):
        return "Local percentile filtering"

    @property
    def action_name(self):
        return "Filter"


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
        from ..tif_tools.localization import SMLM_localization

        return SMLM_localization(img, factor=self.threshold_factor)

    @property
    def name(self):
        return "Default Localizer"


class LocProcessor(ProcessingStep):
    @abstractmethod
    def process(self, locs: pd.DataFrame) -> pd.DataFrame:
        return locs

    @property
    def name(self):
        return "Locs processor doing nothing"

    def process(self, *args):
        return self.process(*args)


class DriftCorrector(LocProcessor):
    def process(self, locs: pd.DataFrame) -> pd.DataFrame:
        from ..tif_tools.correct_drift import correct_drift

        logging.info("Correcting drift iteration #1/2")
        df, shift_1 = correct_drift(locs, L=0.2, step_size=0.03)
        logging.info("Correcting drift iteration #2/2")
        df, shift_2 = correct_drift(df, L=0.05, step_size=0.01)
        # TODO: should we store the shift information somewhere ?
        plt.figure()
        plt.plot(shift_1.values[:, 0])
        plt.plot(shift_2.values[:, 0])
        return df

    @property
    def name(self):
        return "Drift corrector"

    @property
    def action_name(self):
        return "Correct drift"


class TifPipeline:
    def __init__(
        self,
        name: str,
        movie_preprocessors: List[MoviePreProcessor],
        localizer: Localizer,
        loc_processors: List[LocProcessor],
        tracker: Tracker,
    ):
        self.name = name
        self.movie_preprocessors = movie_preprocessors
        self.localizer = localizer
        self.loc_processors = loc_processors
        self.tracker = tracker

    @classmethod
    def from_dict(cls, p: Dict):
        """
        Instantiate from a dictionnary.
        example Dictionnary :
        ..code python3

            {
                "name":"my_pipeline",
                "movie_preprocessors":[
                    {
                        "MyPreProcessingClass":{"param1":value,"param2":other_value}
                    },
                    {
                        "WindowPercentileFilter":{}
                        # If the parameter's dict is empty, default parameters will be used
                    }
                }],
                "localizer":{
                    "DefaultLocalizer":{"threshold_factor":1.5}
                    },
                "tracker":{
                    "UnknownClass":{"bla":bla}
                    # If the class is not found, this will raise an exception
                    # Similarly, if the class provided does not inherit Tracker, an exception will be raised
                }
                # If some step is not mentioned (e.g. here, there's nothing about localization processing), then
                # if it's movie_preprocessors, then no movie_preprocessors will be used (same for loc_processors)
                # it it's localizer or tracker, then the default classes will be used.
            }
        """

        def instantiate_from_dict(d):
            assert len(d) <= 1, "Only one key can be provided"
            if len(d) == 0:
                return None
            cls = list(d.keys())[0]
            params = d[cls]
            klass = globals()[cls]
            instance = klass(**params)
            return instance

        def load_from_dict(d, key, default_class):
            instance = None
            if key in d:
                try:
                    instance = instantiate_from_dict(d[key])
                except Exception as e:
                    logging.debug(e)
            if instance is None:
                instance = default_class()
            return instance

        def load_list_from_dict(d, key, base_class):
            instances = []
            if key in d:
                items_list = d[key]
                for item in items_list:
                    instance = instantiate_from_dict(item)
                    try:
                        assert isinstance(instance, base_class)
                        instances.append(instance)
                    except AssertionError as e:
                        logging.debug(e)
            return instances

        # LOAD COMPONENT BY COMPONENT
        name = p["name"]
        movie_preprocessors = load_list_from_dict(
            p, "movie_preprocessors", MoviePreProcessor
        )
        localizer = load_from_dict(p, "localizer", DefaultLocalizer)
        loc_processors = load_list_from_dict(p, "loc_processors", LocProcessor)
        tracker = load_from_dict(p, "tracker", TrackpyTracker)
        return cls(
            name=name,
            movie_preprocessors=movie_preprocessors,
            localizer=localizer,
            loc_processors=loc_processors,
            tracker=tracker,
        )

    @classmethod
    def from_yaml(cls, file):
        with open(file, "r") as file:
            params = dict(yaml.safe_load(file))
        return cls.from_dict(params)

    @classmethod
    def default_with_name(cls, name: str):
        return cls.from_dict({"name": name})

    def to_yaml(self, fileName):
        tp_params = self.to_dict()
        yaml.dump(tp_params, open(fileName, "w"))

    def to_dict(self) -> dict:
        res = {
            "name": self.name,
            "localizer": self.localizer.to_dict(),
            "tracker": self.tracker.to_dict(),
        }
        if len(self.movie_preprocessors) > 0:
            res["movie_preprocessors"] = [
                p.to_dict() for p in self.movie_preprocessors
            ]
        if len(self.loc_processors) > 0:
            res["loc_processors"] = [p.to_dict() for p in self.loc_processors]
        return res

    def __str__(self):
        desc = "TIF Processing pipeline\n"
        desc += "-----------------------\n"
        desc += "Movie preprocessing steps :\n"
        for i, step in enumerate(self.movie_preprocessors):
            desc += "%d/%d \t %s\n" % (
                i + 1,
                len(self.movie_preprocessors),
                step,
            )
        desc += "-----------------------\n"
        desc += "Localizer :\n"
        desc += "\t %s\n" % self.localizer
        desc += "-----------------------\n"
        desc += "Localization processing steps :\n"
        for i, step in enumerate(self.loc_processors):
            desc += "- %d/%d \t %s\n" % (i + 1, len(self.loc_processors), step)
        desc += "-----------------------\n"
        desc += "Tracker :\n"
        desc += "\t %s\n" % self.tracker
        return desc

    def movie_preprocessing(self, mov: da.Array) -> da.Array:
        for i, processor in enumerate(self.movie_preprocessors):
            logging.info(
                "Preprocessing step %d / %d : %s"
                % (i, len(self.movie_preprocessors), processor.name)
            )
            mov = processor.preprocess(mov)
        return mov

    def movie_localization(
        self, mov: da.Array, DT: float, pixel_size: float
    ) -> pd.DataFrame:
        locs = self.localizer.movie_localization(mov)
        locs[["x", "y"]] *= pixel_size
        locs["t"] = locs["frame"] * DT
        return locs

    def loc_processing(self, locs: pd.DataFrame) -> pd.DataFrame:
        for i, locproc in enumerate(self.loc_processors):
            logging.info(
                "Processing locs, step %d / %d : %s"
                % (i, len(self.loc_processors), locproc.name)
            )
            locs = locproc.process(locs)
        return locs

    def tracking(self, locs: pd.DataFrame) -> pd.DataFrame:
        return self.tracker.track(locs)

    def _process(
        self,
        acq: Acquisition,
        skip_loc: bool = False,
        skip_tracking: bool = False,
    ):
        # Load the original .tif
        if skip_tracking and not skip_loc:
            skip_tracking = False
            logging.warning(
                "Can't skip tracking after fresh localization. Overriding to skip_tracking = False"
            )
        if not skip_loc:
            mov = acq.image.astype(float)
            mov = self.movie_preprocessing(mov)
            locs = self.movie_localization(
                mov, DT=acq.experiment.DT, pixel_size=acq.experiment.pixel_size
            )
            acq.raw_locs = locs.copy()
            self.mark_as_localized(acq)
        if not skip_tracking:
            locs = self.loc_processing(acq.raw_locs.copy())
            acq.locs = self.tracking(locs)
            self.mark_as_tracked(acq)

    def process(
        self,
        to_process: Union[Acquisition, Experiment],
        force_reprocess: bool = False,
    ):
        logging.debug("Process %s" % to_process)
        if isinstance(to_process, Acquisition):
            self._process(
                to_process,
                skip_loc=(not force_reprocess)
                and self.is_already_localized(to_process),
                skip_tracking=(not force_reprocess)
                and self.is_already_tracked(to_process),
            )
        elif isinstance(to_process, Experiment):
            for acq_file in to_process:
                acq = Acquisition(
                    acq_file, experiment=to_process, tif_pipeline=self
                )
                self.process(acq, force_reprocess=force_reprocess)
        else:
            raise BaseException(
                "to_process shoud be an experiment or an acquisition but is %s"
                % to_process.__class__.__qualname__
            )

    def exp_run_df_path(self, acq: Acquisition):
        run_path = os.path.join(acq.experiment.export_folder, self.name)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        assert os.path.isdir(run_path)
        exp_run_df_path = os.path.join(
            acq.experiment.export_folder, self.name, "run_index.csv"
        )
        return exp_run_df_path

    def is_already_localized(self, acq: Acquisition):
        p = self.exp_run_df_path(acq=acq)
        if os.path.exists(p):
            run_index_df = pd.read_csv(p, index_col=0)
            try:
                return run_index_df.loc[acq.ID, "is_localized"] == True
            except:
                pass
        return False

    def is_already_tracked(self, acq: Acquisition):
        p = self.exp_run_df_path(acq=acq)
        if os.path.exists(p):
            run_index_df = pd.read_csv(p, index_col=0)
            try:
                return run_index_df.loc[acq.ID, "is_tracked"] == True
            except:
                pass
        return False

    def mark_as_localized(self, acq: Acquisition):
        p = self.exp_run_df_path(acq)
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
        else:
            df = pd.DataFrame(
                columns=["is_localized", "is_tracked"], dtype=bool
            )
        df.loc[acq.ID, "is_localized"] = True
        df.to_csv(p, index=True)

    def mark_as_tracked(self, acq: Acquisition):
        p = self.exp_run_df_path(acq)
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
        else:
            df = pd.DataFrame(
                columns=["is_localized", "is_tracked"], dtype=bool
            )
        df.loc[acq.ID, "is_tracked"] = True
        df.to_csv(p, index=True)

    def open_in_napari(self, acq: Acquisition = None, tif_file: str = None):
        napari.Viewer()
        viewer = napari.current_viewer()
        widget = TifPipelineWidget(self, viewer)

        if acq is not None:
            widget.pixel_size = acq.experiment.pixel_size
            widget.delta_t = acq.experiment.DT
            viewer.add_image(
                data=acq.image,
                name=acq.tif_file.split(os.path.sep)[-1],
                scale=(1.0, widget.pixel_size, widget.pixel_size),
            )
        elif tif_file is not None:
            viewer.add_image(
                data=dask_image.imread.imread(
                    tif_file,
                    nframes=300,
                )
            )

        viewer.window.add_dock_widget(
            widget=widget,
            name="PALM pipeline : %s" % self.name,
            area="right",
        )
        widget.rescale_image_layers()
