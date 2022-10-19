from __future__ import annotations
import logging
from typing import Dict, List, Type, Union
import os
import pandas as pd
import pandas as pd
import dask.array as da
import yaml

from ..data_structure.acquisition import Acquisition
from ..data_structure.experiment import Experiment
from .steps import *


class TifPipeline:
    def __init__(
        self,
        name: str,
        movie_preprocessors: List[MoviePreProcessor],
        detector: Detector,
        localizer: SubpixelLocalizer,
        loc_processors: List[LocProcessor],
        tracker: Tracker,
    ):
        self.name = name
        self.movie_preprocessors = movie_preprocessors
        self.detector = detector
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
                    raise
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
        detector = load_from_dict(p, "detector", BaseDetector)
        localizer = load_from_dict(p, "localizer", MaxLikelihoodLocalizer)
        loc_processors = load_list_from_dict(p, "loc_processors", LocProcessor)
        tracker = load_from_dict(p, "tracker", ConservativeTracker)
        return cls(
            name=name,
            movie_preprocessors=movie_preprocessors,
            detector=detector,
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

    @property
    def available_steps(self):
        if not hasattr(self, "_available_steps"):
            all_globals = globals().items()
            base_classes = {
                "movie_preprocessors": MoviePreProcessor,
                "detector": Detector,
                "localizer": SubpixelLocalizer,
                "loc_processors": LocProcessor,
                "tracker": Tracker,
            }
            self._available_steps = {}
            for step_type in base_classes:
                self._available_steps[step_type] = []

            for name, obj in all_globals:
                try:
                    for step_type, base_class in base_classes.items():
                        if (
                            issubclass(obj, base_class)
                            and obj is not base_class
                        ):
                            self._available_steps[step_type].append(
                                (name, obj)
                            )
                            break
                except TypeError as e:
                    pass

        return self._available_steps

    def to_yaml(self, fileName):
        tp_params = self.to_dict()
        yaml.dump(tp_params, open(fileName, "w"))
        self.last_storage_path = str(fileName)

    def to_dict(self) -> dict:
        res = {
            "name": self.name,
            "detector": self.detector.to_dict(),
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

    def contains_class(self, step):
        return self.index_of(step) is not None

    def step_type_of(self, step):
        d = self.available_steps
        for step_type, steps in d.items():
            for step_tuple in steps:
                if step_tuple[0] == step:
                    return step_type
        return None

    def step_class_of(self, step):
        d = self.available_steps
        for step_type, steps in d.items():
            for step_tuple in steps:
                if step_tuple[0] == step:
                    return step_tuple[1]
        return None

    def index_of(self, step):
        d = self.to_dict()
        for step_type, steps in d.items():
            if step_type == "name":
                continue
            if isinstance(steps, list):
                # Si possible d'avoir plusieurs items, on trouve le rang
                for step_dict in steps:
                    for i, option in enumerate(step_dict):
                        if step == option:
                            return step_type
            else:
                assert isinstance(steps, dict), steps
                # Sinon, on renvoie zéro parce qu'il n'y a forcément qu'un step
                if step in steps:
                    return 0
        return None

    def can_be_removed(self, step):
        return self.contains_class(step) and (
            self.step_type_of(step)
            in ["movie_preprocessors", "loc_processors"]
        )

    def is_mandatory(self, step):
        return self.step_type_of(step) not in [
            "movie_preprocessors",
            "loc_processors",
        ]

    def has_alternatives_to(self, step):
        step_type = self.step_type_of(step)
        if step_type is None:
            return False
        else:
            return len(self.available_steps[step_type]) > 1

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
        desc += "Detector :\n"
        desc += "\t %s\n" % self.detector
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
        detections = self.detector.movie_detection(mov)
        locs = self.localizer.movie_localization(mov, detections)
        locs[["x", "y"]] *= pixel_size
        locs["t"] = locs["frame"] * DT
        return locs

    def loc_processing(
        self,
        mov: da.Array,  # Movie
        locs: pd.DataFrame,  # Previous localizations, with x and y in micrometers
        pixel_size: float = 1.0,  # Pixel size in micrometers
    ) -> pd.DataFrame:
        for i, locproc in enumerate(self.loc_processors):
            logging.info(
                "Processing locs, step %d / %d : %s"
                % (i, len(self.loc_processors), locproc.name)
            )
            locs = locproc.process(mov, locs, pixel_size=pixel_size)
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
        original_mov = acq.image.astype(float)
        if not skip_loc:
            mov = self.movie_preprocessing(original_mov)
            locs = self.movie_localization(
                mov, DT=acq.experiment.DT, pixel_size=acq.experiment.pixel_size
            )
            acq.raw_locs = locs.copy()
            self.mark_as_localized(acq)
        if not skip_tracking:
            acq.experiment.pixel_size
            locs = self.loc_processing(original_mov, acq.raw_locs.copy())
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
