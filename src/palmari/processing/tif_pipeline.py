from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import logging
from typing import TYPE_CHECKING, Dict, List, Union
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
import yaml

from ..data_structure.acquisition import Acquisition
from ..data_structure.experiment import Experiment
from .steps import *


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
        self.last_storage_path = None

    @classmethod
    def from_dict(cls, p: Dict):
        """
        Instantiate from a dictionnary.
        Here's an example Dictionnary which could be passed as an argument :

        .. code-block:: python3

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
        with open(file, "r") as f:
            params = dict(yaml.safe_load(f))
        res = cls.from_dict(params)
        res.last_storage_path = str(file)
        return res

    @classmethod
    def default_with_name(cls, name: str):
        return cls.from_dict({"name": name})

    def to_yaml(self, fileName):
        tp_params = self.to_dict()
        yaml.dump(tp_params, open(fileName, "w"))
        self.last_storage_path = str(fileName)

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
            # Actually process
            self._process(
                to_process,
                skip_loc=(not force_reprocess)
                and self.is_already_localized(to_process),
                skip_tracking=(not force_reprocess)
                and self.is_already_tracked(to_process),
            )
        elif isinstance(to_process, Experiment):

            # Actually process, acquisition after acquisition

            for i, acq_file in enumerate(to_process):
                acq = Acquisition(
                    acq_file, experiment=to_process, tif_pipeline=self
                )
                if i == 0:
                    pipeline_export_path_for_exp = self.exp_params_path(acq)
                    logging.info(
                        "Saving pipeline to %s" % pipeline_export_path_for_exp
                    )
                    self.to_yaml(pipeline_export_path_for_exp)
                self.process(acq, force_reprocess=force_reprocess)
        else:
            raise BaseException(
                "to_process shoud be an experiment or an acquisition but is %s"
                % to_process.__class__.__qualname__
            )

    def exp_run_df_path(self, acq: Acquisition) -> str:
        run_path = os.path.join(acq.experiment.export_folder, self.name)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        assert os.path.isdir(run_path)
        exp_run_df_path = os.path.join(
            acq.experiment.export_folder, self.name, "run_index.csv"
        )
        return exp_run_df_path

    def exp_params_path(self, acq: Acquisition) -> str:
        run_path = os.path.join(acq.experiment.export_folder, self.name)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        assert os.path.isdir(run_path)
        exp_params_path = os.path.join(
            acq.experiment.export_folder, self.name, "%s.yaml" % self.name
        )
        return exp_params_path

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
