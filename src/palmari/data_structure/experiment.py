from __future__ import annotations
import json
import logging
from glob import glob
import secrets
from subprocess import call
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from typing import TYPE_CHECKING

from .acquisition import Acquisition


class Experiment:
    def __init__(self, data_folder: str, export_folder: str):
        self.data_folder = data_folder
        self.export_folder = export_folder
        self.check_export_folder_and_load_info()
        self.index_path = os.path.join(self.export_folder, "index.csv")
        self.look_for_updates()
        logging.info(self)

    def __str__(self):
        desc = "Data : %s\n" % self.data_folder
        desc += "Export : %s\n" % self.export_folder
        desc += "Files : %d\n" % self.index_df.shape[0]
        n_cols = len(self.index_df.columns) - 2
        desc += "Information columns : %d\n" % n_cols
        for col in self.index_df.columns:
            if col not in ["file", "ID"]:
                desc += "\t- %s\n" % (col)
        desc += "Exposure : %.3f second(s)\n" % self.DT
        desc += "Pixel size : %.3f micron(s)\n" % self.pixel_size
        return desc

    def __iter__(self):
        return iter(self.index_df.file.tolist())

    def __getitem__(self, item):
        if item in self.index_df.ID.tolist():
            file = self.index_df.loc[self.index_df.ID == item, "file"].values[
                0
            ]
            return file
        elif type(item) == int:
            if item < self.index_df.shape[0]:
                return self.index_df["file"].values[item]
        else:
            raise "Invalid item %s" % item

    def __len__(self):
        return self.index_df.shape[0]

    @classmethod
    def from_single_tif(cls, tif_file: str, export_folder: str):
        data_folder = os.path.split(tif_file)[0]
        return Experiment(data_folder=data_folder, export_folder=export_folder)

    """
    def to_tracksets(self, tif_pipeline: TifPipeline) -> TrackSets:
        tracksets = []
        included_files = []
        for tif_file in self:
            acq = Acquisition(
                tif_file=tif_file, experiment=self, tif_pipeline=tif_pipeline
            )
            if not acq.is_processed:
                logging.debug("Not including %s" % tif_file)
                continue
            included_files.append(tif_file)
            tracksets.append(
                TrackSet(locs_path=acq.locs_path, origin_file=tif_file)
            )
        return TrackSets(
            tracksets=tracksets,
            root_folder=self.export_folder,
            index_df=self.index_df.loc[
                self.index_df.file.isin(included_files)
            ],
        )
    """

    def check_export_folder_and_load_info(self):
        json_path = os.path.join(self.export_folder, "exp_info.json")
        if not os.path.isdir(self.export_folder):
            # Create a json file with experiment-level information
            logging.info("Create %s" % self.export_folder)
            os.mkdir(self.export_folder)

        if not os.path.exists(json_path):
            logging.info("JSON does not exist, create %s" % json_path)
            pixel_size = 0.097
            try:
                pixel_size = float(
                    input("Pixel size, in microns (default is 0.097) : ")
                )
            except BaseException as e:
                print(e)
                print(
                    "Incorrect value, keeping default : %.2f um" % pixel_size
                )

            DT = 0.03
            try:
                DT = float(
                    input(
                        "Time interval between successive frames, in seconds (default is 0.03) : "
                    )
                )
            except BaseException as e:
                print(e)
                print("Incorrect value, keeping default : %.2f s" % DT)

            file_pattern = "*.tif"
            try:
                file_pattern = str(input("File pattern (default is *.tif) : "))
            except BaseException as e:
                print(e)
                print("Incorrect value, keeping default : %.2f s" % DT)

            exp_info = {
                "data_path": self.data_folder,
                "pixel_size": pixel_size,
                "file_pattern": file_pattern,
                "DT": DT,
            }
            json.dump(
                exp_info,
                open(json_path, "w"),
            )
        else:
            # Read the JSON file and check that path of origin data is the same
            logging.info("JSON already exists : %s" % json_path)
            exp_info = json.load(open(json_path, "r"))

            if not os.path.samefile(self.data_folder, exp_info["data_path"]):
                logging.debug("Data folder is %s" % self.data_folder)
                logging.debug(
                    "Data path in JSON in %s" % exp_info["data_path"]
                )
        self.DT = exp_info["DT"]
        self.pixel_size = exp_info["pixel_size"]
        self.file_pattern = exp_info["file_pattern"]

    @property
    def index_df(self) -> pd.DataFrame:
        if not hasattr(self, "_index_df"):
            if not os.path.exists(self.index_path):
                self._index_df = pd.DataFrame(columns=["file", "ID"])
            else:
                self._index_df = pd.read_csv(self.index_path)
        return self._index_df

    @index_df.setter
    def index_df(self, v: pd.DataFrame):
        self._index_df = v
        self.save_index()

    def save_index(self):
        """
        Just saves index_df
        """
        self.index_df.to_csv(self.index_path, index=False)

    @property
    def custom_fields(self) -> dict:
        """
        Override this in a subclass of Experience to meet your needs
        keys of the dict are column names
        values are used to fill the columns, using the TIF file name of each acquisition
        values can be :

        - string : True if the file name contains that string
        - int : the i-th part of the file name, when split using the filesystem separator
        - callable : callable(filename)

        for instance

        .. code-block:: python3

            {"condition":get_condition_from_name}

        """
        return {}

    """
    @property
    def available_processing_runs(self) -> List[TifProcessingRun]:
        if not hasattr(self, "_processing_runs"):
            os.chdir(self.export_folder)
            run_param_files = glob("*/run_params.json", recursive=True)
            logging.info("%d available processing runs" % len(run_param_files))
            self._processing_runs = [
                TifProcessingRun.from_json(f, self) for f in run_param_files
            ]
        return self._processing_runs
    """

    @property
    def all_files(self) -> List[str]:
        """Return all files indexed in self.index_df

        Returns:
            List[Acquisition]: all files indexed in self.index_df
        """
        if not hasattr(self, "_all_files"):
            # Call it with no filters to retrieve all files
            self._all_files = self.files_with_filters()
            logging.info(
                "Querying all files : Found %d files" % len(self._all_files)
            )
            if len(self._all_files) > 0:
                logging.info(
                    "The first queried file is %s" % self._all_files[0]
                )
        return self._all_files

    """
    def files_with_filters(
        self, filters: dict = {}, only_processed_with_run: TifProcessingRun = None
    ) -> List[str]:
        # filters = {"cell_type":["neuron","platelet"],
        #            "replicate":"Experience0"}
        cond = ~self.index_df.file.isnull()
        for column, value in filters.items():
            try:
                if isinstance(value, list):
                    cond = cond & self.index_df[column].isin(value)
                else:
                    cond = cond & (self.index_df[column] == value)
            except KeyError:
                logging.debug("%s is not a column" % column)
                logging.debug("Use one of %s" % ", ".join(self.index_df.columns))
                raise
        files = self.index_df.loc[cond].file.tolist()
        if only_processed_with_run is not None:
            files = [
                f
                for f in files
                if Acquisition(
                    f, experiment=self, tif_pipeline=only_processed_with_run
                ).is_processed
            ]
        return files
    """

    def add_new_roi_to_index(self, f: str):
        row = pd.DataFrame.from_dict(
            {"file": [f], "ID": [secrets.token_hex(8)]}, orient="columns"
        )
        self.index_df = pd.concat(
            [self.index_df, row], axis=0, ignore_index=True
        )
        logging.info("Added row for %s" % f)
        logging.info(str(row))

    def remove_old_roi_from_index(self, f: str):
        assert f in self.index_df["file"].tolist()
        self.index_df = self.index_df.loc[self.index_df.file != f]
        logging.info("Removed in index the row %s" % f)

    def scan_folder(self) -> list:
        os.chdir(self.data_folder)
        roi_files = glob("**/%s" % self.file_pattern, recursive=True)
        roi_files = [
            f for f in roi_files if os.path.getsize(f) > 1e7
        ]  # Don't consider files of less than 10Mb
        # TODO: improve filtering here
        return roi_files

    def look_for_updates(self):
        """
        Look if the index dataframe matches
        the reality of present/absent files
        And computes custom columns if needed
        """
        # D'abord, on vérifie qu'on a une ligne par fichier .tif
        roi_files = self.scan_folder()
        logging.info("look for updates : found %d files" % len(roi_files))
        for f in roi_files:
            if f not in self.index_df["file"].tolist():
                self.add_new_roi_to_index(f)

        # Ensuite on vérifie que tous les fichiers référencés dans l'index
        # sont encore là
        if self.index_df.shape[0] == 0:
            return
        for f in self.index_df["file"].tolist():
            if f not in roi_files:
                self.remove_old_roi_from_index(f)

        self.look_for_new_columns()

    def get_ID_of_acq(self, acquisition: Acquisition):
        ID = self.index_df.loc[
            self.index_df.file == acquisition.tif_file, "ID"
        ].values[0]
        logging.debug("ID of file %s is %s" % (acquisition.tif_file, ID))
        return str(ID)

    def look_for_new_columns(self, overwrite=False):
        """
        Computes custom columns

        Args:
            overwrite (bool, optional): Whether to overwrite pre-existing values. Defaults to False.
        """
        for column, value in self.custom_fields.items():

            if callable(value):
                values = self.index_df["file"].apply(value)
            elif type(value) is int:
                try:
                    s = self.index_df["file"].str.split(
                        pat=os.path.sep, expand=True
                    )
                    i = value
                    if i < 0:
                        i = s.shape[1] + i
                    values = s[i]
                except KeyError as e:
                    logging.debug(
                        self.index_df["file"].str.split(
                            pat=os.path.sep, expand=True
                        )
                    )
                    logging.debug(e)
                    raise
            elif type(value) is str:

                def last_part_that_contains(f):
                    parts = f.lower().split(os.path.sep)
                    for p in parts[::-1]:
                        if value in p:
                            return p

                values = self.index_df["file"].map(last_part_that_contains)
            else:
                values = pd.Series(index=self.index_df.index)
                values[:] = value
            if column in self.index_df.columns and not overwrite:
                values.loc[
                    ~self.index_df[column].isnull()
                ] = self.index_df.loc[~self.index_df[column].isnull(), column]
            self.index_df[column] = values

    # AGGREGATE STATS

    @property
    def runs_stats(self) -> pd.DataFrame:
        if not hasattr(self, "_runs_stats"):
            runs = self.available_processing_runs
            logging.info(
                "Concatenating stats from %d sets of parameters" % len(runs)
            )
            self._runs_stats = pd.concat(
                [run.stats_df for run in runs], axis=0
            )

        return self._runs_stats

    def parameter_influence_on_stats(
        self, param_name: str, stats: list = ["n_locs"], mode: str = "hist"
    ):
        """Plots the influence of a processing parameter on some statistics.

        Args:
            param_name (str): The parameter whose influence is studied. it should be one returned by acquisition.basic_stats
            stats (list, optional): Statistics to compare. Defaults to ["n_locs"].
            mode (str, optional): Plotting mode. supported : "bars" and "hist". Defaults to "hist".
        """
        stat_cols = [c for c in self.runs_stats.columns if c != "file"]
        df = self.runs_stats
        pv = df.pivot_table(
            columns=param_name,
            index="file",
            values=stat_cols,
            aggfunc="median",
        )
        param_values = df[param_name].unique().tolist()

        if mode == "bars":
            w = 1.0 / (1 + len(stats))
            for s in stats:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for i, v in enumerate(param_values):
                    label = None
                    if type(v) is float:
                        label = "%s = %.2f" % (param_name, v)
                    else:
                        label = "%s = %s" % (param_name, v)
                    ax.bar(
                        np.arange(pv.shape[0])
                        - w * (len(stats) - 1) / 2
                        + i * w,
                        height=pv[(s, v)],
                        width=w,
                        tick_label=pv.index,
                        label=label,
                    )
                if self.runs_stats[s].max() > 100 * self.runs_stats[s].min():
                    plt.yscale("log")
        elif mode == "hist":
            for s in stats:
                fig = plt.figure(figsize=(4, 2))
                ax = fig.add_subplot(111)
                log = self.runs_stats[s].max() > 100 * self.runs_stats[s].min()
                title = s
                if log:
                    title = "log_10(%s)" % title
                ax.set_title(title)
                for i, v in enumerate(param_values):
                    label = None
                    if type(v) is float:
                        label = "%s = %.2f" % (param_name, v)
                    else:
                        label = "%s = %s" % (param_name, v)
                    H = pv[(s, v)]
                    if log:
                        H = np.log10(1 + H.astype(float).values)
                    ax.hist(H, bins=20, label=label, histtype="step")
                ax.legend()
