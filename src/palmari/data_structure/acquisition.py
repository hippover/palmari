from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Callable, List
import os
import warnings
import pandas as pd
import numpy as np
import trackpy as tp
import logging
import napari
from glob import glob
import dask_image.imread
import dask.array as da
import dask_image.ndfilters
from skimage.filters import sato, threshold_local


from ..tif_tools.intensity import mean_intensity_center
from ..tif_tools.correct_drift import correct_drift
from ..tif_tools.localization import localize_movie

if TYPE_CHECKING:
    from .experiment import Experiment
    from ..processing.tif_pipeline import TifPipeline


class Acquisition:
    """
    An acquisition corresponds to a PALM movie.
    It is part of an :py:class:`Experiment`, and bound to a :py:class:`TifPipeline` with which it is processed.
    """

    def __init__(
        self,
        tif_file,
        experiment: Experiment,
        tif_pipeline: TifPipeline,
    ):
        self.tif_file = tif_file
        self.tif_pipeline = tif_pipeline
        self.experiment = experiment
        self.export_root = os.path.join(
            self.experiment.export_folder,
            self.tif_pipeline.name,
            ".".join(self.tif_file.split(".")[:-1]),
        )
        logging.debug(
            "export_root : file %s -> %s" % (self.tif_file, self.export_root)
        )
        export_parent_folder = os.sep.join(self.export_root.split(os.sep)[:-1])
        os.makedirs(export_parent_folder, exist_ok=True)

    @property
    def image(self) -> da.Array:
        """
        The actual movie, loaded with Dask.

        Returns:
            da.Array: the movie.
        """
        if not hasattr(self, "_image"):
            logging.info("Loading tif file %s" % self.tif_file)
            # self._image = tifffile.imread(
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._image = dask_image.imread.imread(
                    os.path.join(self.experiment.data_folder, self.tif_file),
                    nframes=300,
                )
        return self._image

    def get_property(self, col: str) -> Any:
        """Access an acquisition's property, read from its experiment's index table.

        Args:
            col (str): name of the index table column to look into

        Returns:
            Any: value of the corresponding row x column in the experiment's index table
        """
        df = self.experiment.index_df
        return df.loc[df.file == self.tif_file, col].values[0]

    @property
    def ID(self) -> str:
        return self.get_property("ID")

    @property
    def locs(self) -> pd.DataFrame:
        if not hasattr(self, "_locs"):
            try:
                self._locs = pd.read_csv(self.locs_path, index_col=0)
            except FileNotFoundError:
                print(
                    "This acquisition wasn't localized. Or perhaps with another Pipeline ?"
                )
                return None
        return self._locs

    @locs.setter
    def locs(self, value):
        value.to_csv(self.locs_path)
        self._locs = value

    @property
    def locs_path(self) -> str:
        if not hasattr(self, "_locs_path"):
            self._locs_path = self.export_root + ".locs"
        return self._locs_path

    @property
    def raw_locs(self) -> pd.DataFrame:
        if not hasattr(self, "_raw_locs"):
            loaded = False
            raw_locs_path = self.raw_locs_path
            if os.path.exists(raw_locs_path):
                self._raw_locs = pd.read_csv(raw_locs_path)
                loaded = True
            if not loaded:
                self.localize()
        return self._raw_locs

    @raw_locs.setter
    def raw_locs(self, value):
        value.to_csv(self.raw_locs_path)
        self._raw_locs = value

    @property
    def raw_locs_path(self) -> str:
        if not hasattr(self, "_raw_locs_path"):
            self._raw_locs_path = self.export_root + ".raw_locs"
        return self._raw_locs_path

    @property
    def intensity(self) -> pd.DataFrame:
        if not hasattr(self, "_intensity"):
            loaded = False
            intensity_path = self.intensity_path
            if os.path.exists(intensity_path):
                self._intensity = pd.read_csv(intensity_path)
                loaded = True
            if not loaded:
                self.compute_intensity()
        return self._intensity

    @property
    def intensity_path(self) -> str:
        if not hasattr(self, "_intensity_path"):
            self._intensity_path = self.export_root + ".int"
        return self._intensity_path

    @property
    def tubeness(self) -> np.array:
        if not hasattr(self, "_tubeness"):
            loaded = False
            tubeness_path = self.tubeness_path
            if os.path.exists(tubeness_path):
                self._tubeness = np.load(tubeness_path)
                loaded = True
            if not loaded:
                self.compute_tubeness()
        return self._tubeness

    @property
    def tubeness_path(self) -> str:
        if not hasattr(self, "_tubeness_path"):
            self._tubeness_path = self.export_root + ".tubeness"
        return self._tubeness_path

    @property
    def is_processed(self) -> bool:
        return (
            self.is_localized and self.is_tracked
        )  # and self.intensity_is_computed

    @property
    def is_localized(self) -> bool:
        return os.path.exists(self.raw_locs_path)

    @property
    def intensity_is_computed(self) -> bool:
        return os.path.exists(self.intensity_path)

    @property
    def tubeness_is_computed(self) -> bool:
        return os.path.exists(self.tubeness_path)

    @property
    def drift_is_corrected(self) -> bool:
        return os.path.exists(self.locs_path)

    @property
    def is_tracked(self):
        if not self.drift_is_corrected:
            return False
        locs_path = self.locs_path
        df = pd.read_csv(locs_path, nrows=10)
        return "n" in df.columns

    @property
    def polygon_folder(self) -> str:
        if not hasattr(self, "_acq_poly_path"):
            base = self.experiment.export_folder
            acq_ID = self.get_property("ID")
            poly_path = os.path.join(base, "polygons")
            if not os.path.exists(poly_path):
                os.mkdir(poly_path)
            self._acq_poly_path = os.path.join(poly_path, acq_ID)
            if not os.path.exists(self._acq_poly_path):
                os.mkdir(self._acq_poly_path)
        return self._acq_poly_path

    @property
    def polygon_files(self) -> List:
        return glob(os.path.join(self.polygon_folder, "*.polygon"))

    def add_columns_to_loc(self, columns: pd.DataFrame, save: bool = False):

        for c in columns.columns:
            if c in self._locs.columns:
                del self._locs[c]
            self._locs = self._locs.merge(
                columns[[c]], left_index=True, right_index=True, how="left"
            )
        if save:
            self._save_locs()

    def add_traj_cols_to_locs(self, traj_columns: pd.DataFrame):
        """
        traj_columns is a dataframe whose index corresponds to the 'n' column (traj ID)
        We merge it with the locs dataframe
        """
        to_delete_cols = [
            c for c in traj_columns.columns if c in self._locs.columns
        ]
        for col in to_delete_cols:
            del self._locs[col]
        self._locs = self._locs.merge(
            traj_columns, left_on="n", right_index=True, how="left"
        )

    def compute_intensity(self):
        if not self.intensity_is_computed:
            self._intensity = mean_intensity_center(self.image)
            self._save_intensity()
        else:
            logging.info("%s : intensity is already computed" % self.tif_file)

    def _save_intensity(self):
        storage_path = self.intensity_path
        self._intensity.to_csv(storage_path, index=False)
        logging.info(
            "Stored Intensity of ROI %s at path %s"
            % (self.tif_file, storage_path)
        )

    def compute_tubeness(self):
        if not self.tubeness_is_computed:
            self._compute_tubeness()
            self._save_tubeness()
        else:
            logging.info("%s : tubeness was already processed" % self.tif_file)

    def _compute_tubeness(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = self.image[::10].mean(axis=0)
            mean_intensity = mean.compute()
            tubeness = sato(
                np.log10(mean_intensity), sigmas=3.0, black_ridges=False
            )
            tubeness_threshold = threshold_local(tubeness, 31)
            tubeness_mask = tubeness > tubeness_threshold
            self._tubeness = tubeness_mask

    def _save_tubeness(self):
        storage_path = self.tubeness_path
        np.save(storage_path, self._tubeness)
        logging.info("Saved tubeness at %s" % storage_path)

    def localize(self):
        if not self.is_localized:
            logging.info("Running localization for file %s" % self.tif_file)
            self._localize(**self.tif_pipeline.params.loc_params)
            self._save_raw_locs()
        else:
            logging.info(
                "Localization step has already been done for file %s"
                % self.tif_file
            )

    def _localize(
        self, threshold, sliding_window_filter, subpixel_mode, **args
    ):

        data = self.image
        self._raw_locs = localize_movie(
            data,
            progress_bar=True,
            verbose=False,
            sliding_filter=sliding_window_filter,
            factor=threshold,
            subpixel_mode=subpixel_mode,
        )
        self._raw_locs["t"] = self._raw_locs["frame"] * self.experiment.DT
        self._raw_locs[
            ["x", "y"]
        ] *= self.experiment.pixel_size  # x, y are in um
        N = self._raw_locs.shape[0]
        T = data.shape[0]
        logging.info("==== %d locs\t %.2f / frame" % (N, N / T))

    def _save_raw_locs(self):
        storage_path = self.raw_locs_path
        self._raw_locs.to_csv(storage_path, index=False)
        logging.info(
            "Stored raw locs of ROI %s at path %s"
            % (self.tif_file, storage_path)
        )

    def correct_drift(self):
        if not self.drift_is_corrected:
            if self.tif_pipeline.params.loc_params["correct_drift"] == True:
                self._correct_drift()
            else:
                self._locs = self.raw_locs
            self._save_locs()
        else:
            logging.info(
                "Drift correction step has already been done for file %s"
                % self.tif_file
            )

    def _correct_drift(self):
        logging.info("Correcting drift iteration #1/2")
        df, self._shift_1 = correct_drift(self.raw_locs, L=0.2, step_size=0.03)
        logging.info("Correcting drift iteration #2/2")
        self._locs, self._shift_2 = correct_drift(df, L=0.05, step_size=0.01)

    def _save_locs(self):
        storage_path = self.locs_path
        self._locs.to_csv(storage_path, index=False)
        logging.info(
            "Stored locs of ROI %s at path %s" % (self.tif_file, storage_path)
        )

    def track(self):
        if not self.is_tracked:
            logging.info("Running tracking for file %s" % self.tif_file)
            self._track(**self.tif_pipeline.params.track_params)
            self._save_locs()
        else:
            logging.info(
                "Tracking was already done for file %s" % self.tif_file
            )

    def _track(self, mode, max_diffusivity):

        if mode == "trackpy":
            self._track_trackpy(max_diffusivity)
        else:
            raise "Unknown tracking mode %s" % mode

    def _track_trackpy(self, max_diffusivity):
        max_radius = np.sqrt(2 * max_diffusivity * self.experiment.DT)
        logging.info("Starting to track %d localisations" % self.locs.shape[0])
        logging.info("Max radius is %.2f" % max_radius)

        tracks = tp.link(
            self.locs, search_range=max_radius, link_strategy="drop"
        )
        self._locs["n"] = tracks["particle"]
        traj_length = self._locs.n.value_counts()
        n_long_trajs = (traj_length >= 7).sum()
        logging.info(
            "Found %d long trajectories for file %s"
            % (n_long_trajs, self.tif_file)
        )

    # EXPLOITATION OF TRACKS AND LOCS

    def trajectories_list(
        self,
        min_length: int = 7,
        return_indices: bool = True,
        filter: Callable = None,
    ):
        """
        Returns a list of trajectories whose length is above a given threshold, possibly filtered according to the locs they're based on

        Args:
            min_length (int, optional): Defaults to 7.
            return_indices (bool, optional): Whether to return indices ('n' column of the locs DataFrame) along with coordinates. Defaults to True.
            filter (Callable, optional): Callable, which takes as input the locs DataFrame and returns a boolean Series with the same index. Defaults to None.

        Returns:
            _type_: either a list of trajectories, or a tuple containing this same list and the list of indices
        """
        if filter is not None:
            df = pd.DataFrame(index=self.locs.index)
            df["n"] = self.locs.n
            df["cond"] = filter(self.locs)
            # The filter is applied point by point
            # Then we make sure to only select trajectories fo which all points pass the filter
            cond = df.groupby("n")["cond"].min()
            cond = self.locs.n.isin(cond[cond].index.tolist())
        else:
            cond = ~self.locs.x.isnull()
        traj_length = self.locs.loc[cond].n.value_counts()
        good_trajs = traj_length.loc[traj_length >= min_length].index.tolist()
        indices = []
        trajs = []
        for n, traj in (
            self.locs.loc[self.locs.n.isin(good_trajs)]
            .sort_values(["n", "t"])
            .groupby("n")
        ):
            indices.append(n)
            trajs.append(traj[["x", "y"]].values)
        to_return = (trajs,)
        if return_indices:
            to_return = to_return + (indices,)
        return to_return

    def get_traj(self, n: int):
        return (
            self.locs.loc[self.locs.n == n].sort_values("t")[["x", "y"]].values
        )

    def basic_stats(self):
        stats = {}
        stats["file"] = self.tif_file
        stats["n_frames"] = self.locs.frame.max()
        stats["duration"] = stats["n_frames"] * self.experiment.DT
        # the above is not exactly true, but is much faster than what's below
        # self.image.shape[0]
        if not self.is_processed:
            return stats
        stats["n_locs"] = self.locs.shape[0]
        traj_length = self.locs.n.value_counts()
        for L in [4, 7, 10, 15, 25]:
            stats["n_trajs_%d" % L] = (traj_length >= L).sum()
        return stats

    # VIEWER FUNCTIONS

    def view(
        self,
        min_traj_length=1,
        polygon_ID_col: str = None,
        short_for_tests: bool = False,
        contrast_limits: tuple = (100, 500),
    ):
        viewer = napari.viewer.Viewer(title=os.path.dirname(self.tif_file))
        viewer.add_image(
            self.image[:1000] if short_for_tests else self.image,
            name=self.tif_file.split("/")[-1],
            multiscale=False,
            contrast_limits=contrast_limits,
            scale=(1, self.experiment.pixel_size, self.experiment.pixel_size),
        )
        if self.locs is not None:
            self.view_points(viewer, polygon_ID_col=polygon_ID_col)
            self.view_tracks(
                viewer,
                min_length=min_traj_length,
                polygon_ID_col=polygon_ID_col,
            )
            self.view_polygons(viewer)

    def view_points(
        self, viewer=None, polygon_ID_col: str = None, subsample: int = None
    ):
        if viewer is None:
            viewer = napari.viewer.Viewer()
        locs = self.locs.copy()
        if subsample is not None and subsample < locs.shape[0]:
            locs = locs.sample(subsample)

        points = locs[["frame", "x", "y"]].copy()
        # points[["x","y"]] /= self.pixel_size

        props = {"frame": points["frame"].values}
        for c in ["abs_density", "norm_density"]:
            if c in locs.columns:
                props[c] = locs[c].values
                logging.info("Adding property %s" % c)
        if polygon_ID_col is not None and polygon_ID_col in locs.columns:
            props[polygon_ID_col] = pd.Categorical(
                locs[polygon_ID_col].fillna("no")
            ).codes
            logging.info("Adding polygon tags")

        viewer.add_points(
            points,
            size=0.15,
            face_color="frame",
            symbol="x",
            properties=props,
            edge_width=0.03,
            # edge_color="#dd3333",
            name="locs (adjusted)",
        )

        points = self.raw_locs.loc[locs.index, ["frame", "x", "y"]].copy()
        # points[["x","y"]] /= self.pixel_size
        viewer.add_points(
            points,
            size=0.15,
            symbol="x",
            face_color="frame",
            edge_width=0.03,
            # edge_color="#3333dd",
            name="locs (unadjusted)",
            properties=props,
            visible=False,
        )

    # FOR THE VIEWER

    def view_tracks(
        self, viewer=None, min_length=1, polygon_ID_col: str = None
    ):
        if viewer is None:
            viewer = napari.viewer.Viewer()

        traj_length = self.locs.n.value_counts()
        good_trajs = traj_length.loc[traj_length >= min_length].index.tolist()

        tracks = self.locs.loc[
            self.locs.n.isin(good_trajs),
        ].copy()
        # tracks = self.compute_tracks_displacement(tracks)
        # tracks[["x","y"]] /= self.pixel_size
        tracks["t"] = tracks["frame"].astype(int)
        props = {"time": tracks["t"].values}
        if polygon_ID_col is not None and polygon_ID_col in tracks.columns:
            props[polygon_ID_col] = pd.Categorical(
                tracks[polygon_ID_col].fillna("no")
            ).codes
        viewer.add_points(
            tracks[["t", "x", "y"]],
            size=0.1,
            symbol="x",
            # face_color="#ffffff00",
            edge_width=0.03,
            edge_color="time",
            name="... used in tracks",
            properties=props,
        )
        track_properties = {
            "length": np.clip(
                pd.merge(
                    tracks,
                    tracks.groupby("n")[["x"]]
                    .count()
                    .rename(columns={"x": "length"}),
                    left_on="n",
                    right_index=True,
                )["length"].values,
                a_min=7,
                a_max=30,
            ),
            # "log_D": np.log10(
            #    pd.merge(
            #        tracks[["n"]],
            #        tracks.loc[~tracks.is_last].groupby("n")[["dr_2"]].mean(),
            #        left_on="n",
            #        right_index=True,
            #    )["dr_2"].values
            # ),
            # "log_dr": np.log10(tracks["dr_2"].values),
        }
        track_properties.update(props)

        viewer.add_tracks(
            tracks[["n", "t", "x", "y"]],
            tail_width=0.2,
            name="Tracks",
            tail_length=10,
            head_length=10,
            properties=track_properties,
        )

    def view_polygons(self, viewer: napari.Viewer = None):
        if viewer is None:
            viewer = napari.viewer.Viewer()
        polygons = []

        for p in self.polygon_files:
            polygon_dict = json.load(open(p, "r"))
            if len(polygon_dict["x"]) < 3:
                logging.debug("%s a moins de 3 points, on le supprime" % p)
                os.remove(p)
                continue
            synapse_ID = os.path.split(p)[1].split(".")[0]
            polygon_path = np.stack(
                [(x, y) for x, y in zip(polygon_dict["x"], polygon_dict["y"])]
            )
            polygons.append(polygon_path)

        if len(polygons) > 0:
            viewer.add_shapes(
                data=polygons,
                shape_type="polygon",
                face_color_cycle=["#AA77CC44", "#77AACC44", "#CC77AA44"],
                edge_width=0.1,
                edge_color="red",
            )
